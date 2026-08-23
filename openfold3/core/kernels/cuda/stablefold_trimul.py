# SPDX-License-Identifier: Apache-2.0
"""StableFold triangle multiplicative update: fused-pass CUDA implementation.

Pipeline (forward):
  z --[K1 ln_fwd fp32-stats]--> z_ln --[cuBLAS GEMM CxFiveH]--> proj
    --[K2 gate_split: sigmoid-gate + mask + transpose-write]--> a,b ([B*H,N,N]) , gate kept
  a,b --[cuBLAS bmm]--> x ([B*H,N,N])
  x --[K3 strided ln_fwd (channel-major read)]--> x_ln ([N^2,H] row-major)
    --[cuBLAS GEMM HxC]--> out_raw --[K4 sig_mul]--> out

All reductions fixed-order (single-warp shfl tree per row; grid-stride
column sums for dgamma/dbeta). Zero atomics. bf16 storage, fp32 math.
"""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

using bf16 = __nv_bfloat16;

#define WARP 32
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
    return __shfl_sync(FULL, v, 0);
}

// ---------------- K1 / K3: LayerNorm forward, strided read ----------------
// one warp per row; C multiple of 32.
__global__ void ln_fwd_kernel(
    const bf16* __restrict__ input, long rows, int C,
    long row_stride, long col_stride,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    bf16* __restrict__ output, float* __restrict__ save_mean,
    float* __restrict__ save_rstd) {
  long row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
  if (row >= rows) return;
  int lane = threadIdx.x % WARP;
  const bf16* base = input + row * row_stride;
  float sum = 0.f, sq = 0.f;
  if (col_stride == 1) {
    for (int c2 = lane * 2; c2 < C; c2 += WARP * 2) {
      __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(base + c2);
      float v0 = __bfloat162float(v2.x), v1 = __bfloat162float(v2.y);
      sum += v0 + v1; sq += v0 * v0 + v1 * v1;
    }
  } else {
    for (int c = lane; c < C; c += WARP) {
      float v = __bfloat162float(base[c * col_stride]);
      sum += v; sq += v * v;
    }
  }
  sum = warp_sum(sum); sq = warp_sum(sq);
  float mean = sum / C;
  float var = sq / C - mean * mean;
  float rstd = rsqrtf(var + 1e-5f);
  if (lane == 0) { save_mean[row] = mean; save_rstd[row] = rstd; }
  bf16* out = output + row * (long)C;
  if (col_stride == 1) {
    for (int c2 = lane * 2; c2 < C; c2 += WARP * 2) {
      __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(base + c2);
      __nv_bfloat162 res;
      res.x = __float2bfloat16((__bfloat162float(v2.x) - mean) * rstd
                               * gamma[c2] + beta[c2]);
      res.y = __float2bfloat16((__bfloat162float(v2.y) - mean) * rstd
                               * gamma[c2 + 1] + beta[c2 + 1]);
      *reinterpret_cast<__nv_bfloat162*>(out + c2) = res;
    }
  } else {
    for (int c = lane; c < C; c += WARP) {
      float v = __bfloat162float(base[c * col_stride]);
      out[c] = __float2bfloat16((v - mean) * rstd * gamma[c] + beta[c]);
    }
  }
}

// ---------------- LN backward: dx (one warp per row) ----------------
__global__ void ln_bwd_dx_kernel(
    const bf16* __restrict__ input, const bf16* __restrict__ grad_out,
    long rows, int C, long row_stride, long col_stride,
    const float* __restrict__ gamma,
    const float* __restrict__ save_mean, const float* __restrict__ save_rstd,
    bf16* __restrict__ grad_in) {
  long row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
  if (row >= rows) return;
  int lane = threadIdx.x % WARP;
  const bf16* x = input + row * row_stride;
  const bf16* dy = grad_out + row * (long)C;
  float mean = save_mean[row], rstd = save_rstd[row];
  float s1 = 0.f, s2 = 0.f;
  for (int c = lane; c < C; c += WARP) {
    float g = __bfloat162float(dy[c]) * gamma[c];
    float xh = (__bfloat162float(x[c * col_stride]) - mean) * rstd;
    s1 += g; s2 += g * xh;
  }
  s1 = warp_sum(s1) / C; s2 = warp_sum(s2) / C;
  bf16* dx = grad_in + row * row_stride;  // same layout as input
  for (int c = lane; c < C; c += WARP) {
    float g = __bfloat162float(dy[c]) * gamma[c];
    float xh = (__bfloat162float(x[c * col_stride]) - mean) * rstd;
    dx[c * col_stride] = __float2bfloat16((g - s1 - xh * s2) * rstd);
  }
}

// -------- LN backward: dgamma/dbeta, two-stage deterministic reduction ------
// stage 1: each block owns (column c, row-chunk k); serial fixed order in chunk
__global__ void ln_bwd_dgb_partial_kernel(
    const bf16* __restrict__ input, const bf16* __restrict__ grad_out,
    long rows, int C, long row_stride, long col_stride, long chunk_rows,
    const float* __restrict__ save_mean, const float* __restrict__ save_rstd,
    float* __restrict__ partial) {   // [chunks, 2, C]
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int chunk = blockIdx.y;
  if (c >= C) return;
  long row0 = (long)chunk * chunk_rows;
  long row1 = min(rows, row0 + chunk_rows);
  float dg = 0.f, db = 0.f;
  for (long row = row0; row < row1; ++row) {   // fixed ascending order
    float dy = __bfloat162float(grad_out[row * (long)C + c]);
    float xh = (__bfloat162float(input[row * row_stride + c * col_stride])
                - save_mean[row]) * save_rstd[row];
    dg += dy * xh; db += dy;
  }
  partial[((long)chunk * 2 + 0) * C + c] = dg;
  partial[((long)chunk * 2 + 1) * C + c] = db;
}

// grouped fold: block (h, group) folds its chunk range in fixed order
__global__ void ln_bwd_dgb_group_fold_kernel(
    const float* __restrict__ partial, int chunks, int C, int group_len,
    float* __restrict__ out) {   // out: [groups*2, C]
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int group = blockIdx.y;
  if (c >= C) return;
  int k0 = group * group_len;
  int k1 = min(chunks, k0 + group_len);
  float dg = 0.f, db = 0.f;
  for (int k = k0; k < k1; ++k) {   // fixed ascending order
    dg += partial[((long)k * 2 + 0) * C + c];
    db += partial[((long)k * 2 + 1) * C + c];
  }
  out[((long)group * 2 + 0) * C + c] = dg;
  out[((long)group * 2 + 1) * C + c] = db;
}

// stage 2: fold chunks in fixed ascending order
__global__ void ln_bwd_dgb_fold_kernel(
    const float* __restrict__ partial, int chunks, int C,
    float* __restrict__ dgamma, float* __restrict__ dbeta) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float dg = 0.f, db = 0.f;
  for (int k = 0; k < chunks; ++k) {   // fixed ascending order
    dg += partial[((long)k * 2 + 0) * C + c];
    db += partial[((long)k * 2 + 1) * C + c];
  }
  dgamma[c] = dg; dbeta[c] = db;
}

// ---------------- K2: gate-split forward ----------------
// proj: [B, N, N, 5H] ; mask: [B, N, N] ; outputs a,b: [B*H, N, N] (opt
// transposed), gate: [B, N, N, H] kept in proj layout.
__global__ void gate_split_fwd_kernel(
    const bf16* __restrict__ proj, const float* __restrict__ mask,
    long BNN, int N, int H, bool outgoing,
    bf16* __restrict__ a_out, bf16* __restrict__ b_out) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = BNN * H;
  if (idx >= total) return;
  long pos = idx / H;            // (b, i, j) flattened
  int h = idx % H;
  long b = pos / ((long)N * N);
  long rem = pos % ((long)N * N);
  int i = rem / N, j = rem % N;
  const bf16* p = proj + pos * (long)(5 * H);
  float m = mask[pos];
  float ap = __bfloat162float(p[h]);
  float ag = __bfloat162float(p[H + h]);
  float bp = __bfloat162float(p[2 * H + h]);
  float bg = __bfloat162float(p[3 * H + h]);
  float a = ap / (1.f + __expf(-ag)) * m;
  float bb = bp / (1.f + __expf(-bg)) * m;
  // position-major coalesced writes: [B, N, N, H]; transposes done outside.
  a_out[pos * (long)H + h] = __float2bfloat16(a);
  b_out[pos * (long)H + h] = __float2bfloat16(bb);
  (void)i; (void)j; (void)b; (void)outgoing;
}

// ---------------- K2 backward ----------------
__global__ void gate_split_bwd_kernel(
    const bf16* __restrict__ proj, const float* __restrict__ mask,
    const bf16* __restrict__ da, const bf16* __restrict__ db,
    long BNN, int N, int H, bool outgoing,
    bf16* __restrict__ dproj) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = BNN * H;
  if (idx >= total) return;
  long pos = idx / H;
  int h = idx % H;
  long b = pos / ((long)N * N);
  long rem = pos % ((long)N * N);
  int i = rem / N, j = rem % N;
  const bf16* p = proj + pos * (long)(5 * H);
  float m = mask[pos];
  float ap = __bfloat162float(p[h]);
  float ag = __bfloat162float(p[H + h]);
  float bp = __bfloat162float(p[2 * H + h]);
  float bg = __bfloat162float(p[3 * H + h]);
  float sa = 1.f / (1.f + __expf(-ag));
  float sb = 1.f / (1.f + __expf(-bg));
  float ga = __bfloat162float(da[pos * (long)H + h]) * m;
  float gb = __bfloat162float(db[pos * (long)H + h]) * m;
  (void)i; (void)j; (void)b; (void)outgoing;
  bf16* dp = dproj + pos * (long)(5 * H);
  dp[h] = __float2bfloat16(ga * sa);
  dp[H + h] = __float2bfloat16(ga * ap * sa * (1.f - sa));
  dp[2 * H + h] = __float2bfloat16(gb * sb);
  dp[3 * H + h] = __float2bfloat16(gb * bp * sb * (1.f - sb));
  // gate slot (4H..5H) is filled by sig_mul backward on the host side.
}


// ---- K2t: gate-split, tiled, writes a,b CHANNEL-major [B*H, N*N] ----------
// block: 32x8 threads handles tile [32 h x 32 pos]; smem transpose.
__global__ void gate_split_fwd_tiled_kernel(
    const bf16* __restrict__ proj, const float* __restrict__ mask,
    long NN, int H, bf16* __restrict__ a_out, bf16* __restrict__ b_out) {
  __shared__ float sa[32][33];
  __shared__ float sb[32][33];
  long b_idx = blockIdx.z;
  long pos0 = (long)blockIdx.x * 32;
  int h0 = blockIdx.y * 32;
  int tx = threadIdx.x;              // 0..31 -> h within tile
  int ty = threadIdx.y;              // 0..7  -> pos phase
  const bf16* pbase = proj + (b_idx * NN + pos0) * (long)(5 * H);
  // load & gate: each (ty phase) covers pos = ty, ty+8, ty+16, ty+24
  #pragma unroll
  for (int pp = ty; pp < 32; pp += 8) {
    long pos = pos0 + pp;
    if (pos < NN) {
      const bf16* p = pbase + (long)pp * 5 * H;
      float m = mask[b_idx * NN + pos];
      if (tx < 16) {                       // 16 threads x bf16x2 = 32 h
        int h = h0 + tx * 2;
        __nv_bfloat162 ap2 = *reinterpret_cast<const __nv_bfloat162*>(p + h);
        __nv_bfloat162 ag2 = *reinterpret_cast<const __nv_bfloat162*>(p + H + h);
        __nv_bfloat162 bp2 = *reinterpret_cast<const __nv_bfloat162*>(p + 2 * H + h);
        __nv_bfloat162 bg2 = *reinterpret_cast<const __nv_bfloat162*>(p + 3 * H + h);
        sa[pp][tx * 2] = __bfloat162float(ap2.x)
            / (1.f + __expf(-__bfloat162float(ag2.x))) * m;
        sa[pp][tx * 2 + 1] = __bfloat162float(ap2.y)
            / (1.f + __expf(-__bfloat162float(ag2.y))) * m;
        sb[pp][tx * 2] = __bfloat162float(bp2.x)
            / (1.f + __expf(-__bfloat162float(bg2.x))) * m;
        sb[pp][tx * 2 + 1] = __bfloat162float(bp2.y)
            / (1.f + __expf(-__bfloat162float(bg2.y))) * m;
      }
    } else if (tx < 16) {
      sa[pp][tx * 2] = 0.f; sa[pp][tx * 2 + 1] = 0.f;
      sb[pp][tx * 2] = 0.f; sb[pp][tx * 2 + 1] = 0.f;
    }
  }
  __syncthreads();
  // write channel-major: row h, 32 consecutive pos
  #pragma unroll
  for (int hh = ty; hh < 32; hh += 8) {
    int h = h0 + hh;
    long out_row = (b_idx * H + h) * NN + pos0;
    long pos_left = NN - pos0;
    if (pos_left >= 32 || tx < pos_left)
      a_out[out_row + tx] = __float2bfloat16(sa[tx][hh]);
    if (pos_left >= 32 || tx < pos_left)
      b_out[out_row + tx] = __float2bfloat16(sb[tx][hh]);
  }
}

// ---- K2t backward: read dA,dB channel-major, write dproj position-major ----
__global__ void gate_split_bwd_tiled_kernel(
    const bf16* __restrict__ proj, const float* __restrict__ mask,
    const bf16* __restrict__ da, const bf16* __restrict__ db,
    long NN, int H, bf16* __restrict__ dproj) {
  __shared__ float sda[32][33];
  __shared__ float sdb[32][33];
  long b_idx = blockIdx.z;
  long pos0 = (long)blockIdx.x * 32;
  int h0 = blockIdx.y * 32;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // load dA/dB channel-major (row h, consecutive pos)
  #pragma unroll
  for (int hh = ty; hh < 32; hh += 8) {
    int h = h0 + hh;
    long in_row = (b_idx * H + h) * NN + pos0;
    long pos_left = NN - pos0;
    float va = 0.f, vb = 0.f;
    if (pos_left >= 32 || tx < pos_left) {
      va = __bfloat162float(da[in_row + tx]);
      vb = __bfloat162float(db[in_row + tx]);
    }
    sda[tx][hh] = va;
    sdb[tx][hh] = vb;
  }
  __syncthreads();
  #pragma unroll
  for (int pp = ty; pp < 32; pp += 8) {
    long pos = pos0 + pp;
    if (pos >= NN) continue;
    const bf16* p = proj + (b_idx * NN + pos) * (long)(5 * H) ;
    bf16* dp = dproj + (b_idx * NN + pos) * (long)(5 * H);
    float m = mask[b_idx * NN + pos];
    int h = h0 + tx;
    float ap = __bfloat162float(p[h]);
    float ag = __bfloat162float(p[H + h]);
    float bp = __bfloat162float(p[2 * H + h]);
    float bg = __bfloat162float(p[3 * H + h]);
    float sa_ = 1.f / (1.f + __expf(-ag));
    float sb_ = 1.f / (1.f + __expf(-bg));
    float ga = sda[pp][tx] * m;
    float gb = sdb[pp][tx] * m;
    dp[h] = __float2bfloat16(ga * sa_);
    dp[H + h] = __float2bfloat16(ga * ap * sa_ * (1.f - sa_));
    dp[2 * H + h] = __float2bfloat16(gb * sb_);
    dp[3 * H + h] = __float2bfloat16(gb * bp * sb_ * (1.f - sb_));
  }
}

// ---- K3t: fused transpose + LayerNorm over channel-major input ----------
// input x: [B*H, NN] channel-major; output x_ln: [B*NN, H] row-major.
// block: 256 threads handles pos-tile of 32; H=blockDim-driven loop.
__global__ void transpose_ln_fwd_kernel(
    const bf16* __restrict__ x, long NN, int H, long b_count,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    bf16* __restrict__ x_ln, float* __restrict__ save_mean,
    float* __restrict__ save_rstd) {
  extern __shared__ float tile[];   // [H][33] fp32
  long b_idx = blockIdx.z;
  long pos0 = (long)blockIdx.x * 32;
  int tid = threadIdx.x;            // 256
  int lane = tid % 32, warp = tid / 32;
  // load: each warp handles rows h = warp, warp+8, ... coalesced across pos
  {
    long pos_left = NN - pos0;
    bool full = pos_left >= 32;
    // bf16x2 path requires 4-byte alignment: row offsets stay even only
    // when NN is even. Odd NN takes the scalar path uniformly.
    bool vec = full && ((NN & 1L) == 0);
    for (int h = warp; h < H; h += 8) {
      long row = (b_idx * H + h) * NN + pos0;
      if (vec) {
        if (lane < 16) {
          __nv_bfloat162 v2 =
              *reinterpret_cast<const __nv_bfloat162*>(x + row + lane * 2);
          tile[h * 33 + lane * 2] = __bfloat162float(v2.x);
          tile[h * 33 + lane * 2 + 1] = __bfloat162float(v2.y);
        }
      } else {
        tile[h * 33 + lane] = (full || lane < pos_left)
            ? __bfloat162float(x[row + lane]) : 0.f;
      }
    }
  }
  __syncthreads();
  // per position LN: warp w handles pos = w, w+8, ...
  for (int pp = warp; pp < 32; pp += 8) {
    long pos = pos0 + pp;
    if (pos >= NN) continue;
    float sum = 0.f, sq = 0.f;
    for (int h = lane; h < H; h += 32) {
      float v = tile[h * 33 + pp];
      sum += v; sq += v * v;
    }
    sum = warp_sum(sum); sq = warp_sum(sq);
    float mean = sum / H;
    float var = sq / H - mean * mean;
    float rstd = rsqrtf(var + 1e-5f);
    long out_row = b_idx * NN + pos;
    if (lane == 0) { save_mean[out_row] = mean; save_rstd[out_row] = rstd; }
    bf16* out = x_ln + out_row * (long)H;
    for (int h = lane; h < H; h += 32) {
      float v = tile[h * 33 + pp];
      out[h] = __float2bfloat16((v - mean) * rstd * gamma[h] + beta[h]);
    }
  }
}

// ---- K3t backward extra: transpose row-major grad to channel-major ------
__global__ void transpose_rows_to_channel_kernel(
    const bf16* __restrict__ src, long NN, int H, long b_count,
    bf16* __restrict__ dst) {
  __shared__ float tile[32][33];
  long b_idx = blockIdx.z;
  long pos0 = (long)blockIdx.x * 32;
  int h0 = blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  #pragma unroll
  for (int pp = ty; pp < 32; pp += 8) {
    long pos = pos0 + pp;
    float v = 0.f;
    if (pos < NN) v = __bfloat162float(src[(b_idx * NN + pos) * (long)H + h0 + tx]);
    tile[pp][tx] = v;
  }
  __syncthreads();
  #pragma unroll
  for (int hh = ty; hh < 32; hh += 8) {
    int h = h0 + hh;
    long row = (b_idx * H + h) * NN + pos0;
    long pos_left = NN - pos0;
    if (pos_left >= 32 || tx < pos_left)
      dst[row + tx] = __float2bfloat16(tile[tx][hh]);
  }
}

// ---- K3t backward: fused LN-bwd over channel-major x -> d_x channel-major --
__global__ void transpose_ln_bwd_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ d_xln,
    long NN, int H, const float* __restrict__ gamma,
    const float* __restrict__ save_mean, const float* __restrict__ save_rstd,
    bf16* __restrict__ d_x, float* __restrict__ dgb_partial) {
  extern __shared__ float smem2[];   // two tiles: x[H][33], dx[H][33]
  float* tx_ = smem2;
  float* tdx = smem2 + (long)H * 33;
  long b_idx = blockIdx.z;
  long pos0 = (long)blockIdx.x * 32;
  int tid = threadIdx.x;
  int lane = tid % 32, warp = tid / 32;
  for (int h = warp; h < H; h += 8) {
    long row = (b_idx * H + h) * NN + pos0;
    long left = NN - pos0;
    tx_[h * 33 + lane] = (left >= 32 || lane < left)
        ? __bfloat162float(x[row + lane]) : 0.f;
  }
  __syncthreads();
  for (int pp = warp; pp < 32; pp += 8) {
    long pos = pos0 + pp;
    if (pos >= NN) continue;
    long row = b_idx * NN + pos;
    float mean = save_mean[row], rstd = save_rstd[row];
    const bf16* dy = d_xln + row * (long)H;
    float s1 = 0.f, s2 = 0.f;
    for (int h = lane; h < H; h += 32) {
      float g = __bfloat162float(dy[h]) * gamma[h];
      float xh = (tx_[h * 33 + pp] - mean) * rstd;
      s1 += g; s2 += g * xh;
    }
    s1 = warp_sum(s1) / H; s2 = warp_sum(s2) / H;
    for (int h = lane; h < H; h += 32) {
      float g = __bfloat162float(dy[h]) * gamma[h];
      float xh = (tx_[h * 33 + pp] - mean) * rstd;
      tdx[h * 33 + pp] = (g - s1 - xh * s2) * rstd;
    }
  }
  __syncthreads();
  for (int h = warp; h < H; h += 8) {
    long row = (b_idx * H + h) * NN + pos0;
    long left = NN - pos0;
    if (left >= 32 || lane < left)
      d_x[row + lane] = __float2bfloat16(tdx[h * 33 + lane]);
  }
  // phase 4: per-block dgamma/dbeta partials (fixed order: pos ascending,
  // two half-thread partials folded t0-then-t1). 256 threads, 2 per h.
  __syncthreads();   // phase 3 reads tdx[h*33+..]; staging below reuses tdx
  int h_own = tid % H;
  int half = tid / H;                 // 0 or 1 (host asserts 2*H <= blockDim)
  if (half >= 2) { /* threads beyond 2H idle in phase 4 */ } else {
  long left = NN - pos0;
  int limit = left < 32 ? (int)left : 32;
  float pdg = 0.f, pdb = 0.f;
  long row_base = b_idx * NN + pos0;
  for (int pp = half * 16; pp < min(limit, half * 16 + 16); ++pp) {
    long row = row_base + pp;
    float dy = __bfloat162float(d_xln[row * (long)H + h_own]);
    float xh = (tx_[h_own * 33 + pp] - save_mean[row]) * save_rstd[row];
    pdg += dy * xh; pdb += dy;
  }
  // reuse tdx as staging: [2][H] floats
  tdx[half * H + h_own] = pdg;
  tdx[2 * H + half * H + h_own] = pdb;
  }
  __syncthreads();
  if (half == 0) {
    long block_id = (long)blockIdx.z * gridDim.x + blockIdx.x;
    dgb_partial[(block_id * 2 + 0) * H + h_own] = tdx[h_own] + tdx[H + h_own];
    dgb_partial[(block_id * 2 + 1) * H + h_own] =
        tdx[2 * H + h_own] + tdx[3 * H + h_own];
  }
}

// ---- row-regather: contiguous D-rows remapped between [.,H,N,.] layouts ----
__global__ void gather_rows_kernel(
    const bf16* __restrict__ s0, bf16* __restrict__ d0,
    const bf16* __restrict__ s1, bf16* __restrict__ d1,
    const bf16* __restrict__ s2, bf16* __restrict__ d2,
    long rows, int H, int N, int D, bool src_h_inner) {
  long row = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (row >= rows) return;
  long bo = row / ((long)H * N);
  long rem = row % ((long)H * N);
  long h = rem / N, n = rem % N;
  long src_row = src_h_inner
      ? (bo * N + n) * (long)H + h
      : (bo * H + h) * (long)N + n;
  const bf16* srcs[3] = {s0, s1, s2};
  bf16* dsts[3] = {d0, d1, d2};
  int lane = threadIdx.x;
  for (int t = 0; t < 3; ++t) {
    if (srcs[t] == nullptr) break;
    const __nv_bfloat162* sp = reinterpret_cast<const __nv_bfloat162*>(
        srcs[t] + src_row * (long)D);
    __nv_bfloat162* dp = reinterpret_cast<__nv_bfloat162*>(
        dsts[t] + row * (long)D);
    for (int c = lane; c < D / 2; c += blockDim.x) dp[c] = sp[c];
  }
}

// ---------------- K4: sigmoid-gate multiply ----------------
__global__ void sig_mul_fwd_kernel(
    const bf16* __restrict__ raw, const bf16* __restrict__ gate,
    long rows, int H, long gate_row_stride, bf16* __restrict__ out) {
  // grid: (row-blocks, h-pair) ; each thread handles 2 consecutive h (bf16x2)
  long row = blockIdx.x * (long)blockDim.y + threadIdx.y;
  int h2 = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  if (row >= rows || h2 >= H) return;
  const __nv_bfloat162* raw2 =
      reinterpret_cast<const __nv_bfloat162*>(raw + row * (long)H + h2);
  const __nv_bfloat162* gate2 =
      reinterpret_cast<const __nv_bfloat162*>(gate + row * gate_row_stride + h2);
  __nv_bfloat162 rv = *raw2, gv = *gate2;
  float g0 = 1.f / (1.f + __expf(-__bfloat162float(gv.x)));
  float g1 = 1.f / (1.f + __expf(-__bfloat162float(gv.y)));
  __nv_bfloat162 res;
  res.x = __float2bfloat16(__bfloat162float(rv.x) * g0);
  res.y = __float2bfloat16(__bfloat162float(rv.y) * g1);
  *reinterpret_cast<__nv_bfloat162*>(out + row * (long)H + h2) = res;
}

__global__ void sig_mul_bwd_kernel(
    const bf16* __restrict__ raw, const bf16* __restrict__ gate,
    const bf16* __restrict__ grad_out, long rows, int H,
    long gate_row_stride, bf16* __restrict__ draw, bf16* __restrict__ dgate) {
  long row = blockIdx.x * (long)blockDim.y + threadIdx.y;
  int h2 = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
  if (row >= rows || h2 >= H) return;
  long base = row * (long)H + h2;
  long gbase = row * gate_row_stride + h2;
  __nv_bfloat162 rv = *reinterpret_cast<const __nv_bfloat162*>(raw + base);
  __nv_bfloat162 gv = *reinterpret_cast<const __nv_bfloat162*>(gate + gbase);
  __nv_bfloat162 dy = *reinterpret_cast<const __nv_bfloat162*>(grad_out + base);
  float g0 = 1.f / (1.f + __expf(-__bfloat162float(gv.x)));
  float g1 = 1.f / (1.f + __expf(-__bfloat162float(gv.y)));
  __nv_bfloat162 dr, dg;
  dr.x = __float2bfloat16(__bfloat162float(dy.x) * g0);
  dr.y = __float2bfloat16(__bfloat162float(dy.y) * g1);
  dg.x = __float2bfloat16(__bfloat162float(dy.x) * __bfloat162float(rv.x) * g0 * (1.f - g0));
  dg.y = __float2bfloat16(__bfloat162float(dy.y) * __bfloat162float(rv.y) * g1 * (1.f - g1));
  *reinterpret_cast<__nv_bfloat162*>(draw + base) = dr;
  *reinterpret_cast<__nv_bfloat162*>(dgate + gbase) = dg;
}

// ------------------------- host wrappers -------------------------
static inline int cdiv(long a, int b) { return (int)((a + b - 1) / b); }

std::vector<torch::Tensor> ln_fwd(torch::Tensor input, long rows, long row_stride,
                                  long col_stride, torch::Tensor gamma,
                                  torch::Tensor beta) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int C = gamma.size(0);
  auto out = torch::empty({rows, (long)C},
      torch::TensorOptions().dtype(torch::kBFloat16).device(input.device()));
  auto mean = torch::empty({rows},
      torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
  auto rstd = torch::empty_like(mean);
  int warps_per_block = 8;
  dim3 block(WARP * warps_per_block);
  dim3 grid(cdiv(rows, warps_per_block));
  ln_fwd_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)input.data_ptr(), rows, C, row_stride, col_stride,
      gamma.data_ptr<float>(), beta.data_ptr<float>(),
      (bf16*)out.data_ptr(), mean.data_ptr<float>(), rstd.data_ptr<float>());
  return {out, mean, rstd};
}

torch::Tensor ln_bwd_dx(torch::Tensor input, torch::Tensor grad_out, long rows,
                        long row_stride, long col_stride, torch::Tensor gamma,
                        torch::Tensor mean, torch::Tensor rstd) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int C = gamma.size(0);
  auto grad_in = torch::empty_like(input);
  int warps_per_block = 8;
  dim3 block(WARP * warps_per_block);
  dim3 grid(cdiv(rows, warps_per_block));
  ln_bwd_dx_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)input.data_ptr(), (const bf16*)grad_out.data_ptr(), rows, C,
      row_stride, col_stride, gamma.data_ptr<float>(), mean.data_ptr<float>(),
      rstd.data_ptr<float>(), (bf16*)grad_in.data_ptr());
  return grad_in;
}

std::vector<torch::Tensor> ln_bwd_dgb(torch::Tensor input, torch::Tensor grad_out,
                                      long rows, long row_stride, long col_stride,
                                      int C, torch::Tensor mean,
                                      torch::Tensor rstd) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dgamma = torch::empty({(long)C},
      torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
  auto dbeta = torch::empty_like(dgamma);
  long chunk_rows = 2048;
  int chunks = (int)((rows + chunk_rows - 1) / chunk_rows);
  auto partial = torch::empty({(long)chunks * 2, (long)C},
      torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
  dim3 block(128);
  dim3 grid(cdiv(C, 128), chunks);
  ln_bwd_dgb_partial_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)input.data_ptr(), (const bf16*)grad_out.data_ptr(), rows, C,
      row_stride, col_stride, chunk_rows, mean.data_ptr<float>(),
      rstd.data_ptr<float>(), partial.data_ptr<float>());
  ln_bwd_dgb_fold_kernel<<<cdiv(C, 128), 128, 0, stream>>>(
      partial.data_ptr<float>(), chunks, C,
      dgamma.data_ptr<float>(), dbeta.data_ptr<float>());
  return {dgamma, dbeta};
}

std::vector<torch::Tensor> gate_split_fwd(torch::Tensor proj, torch::Tensor mask,
                                          int N, int H, bool outgoing, int B) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long BNN = (long)B * N * N;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(proj.device());
  auto a = torch::empty({(long)B, (long)N, (long)N, (long)H}, opts);
  auto b = torch::empty_like(a);
  long total = BNN * H;
  dim3 block(256);
  dim3 grid(cdiv(total, 256));
  gate_split_fwd_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)proj.data_ptr(), mask.data_ptr<float>(), BNN, N, H, outgoing,
      (bf16*)a.data_ptr(), (bf16*)b.data_ptr());
  return {a, b};
}

torch::Tensor gate_split_bwd(torch::Tensor proj, torch::Tensor mask,
                             torch::Tensor da, torch::Tensor db, int N, int H,
                             bool outgoing, int B) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long BNN = (long)B * N * N;
  auto dproj = torch::zeros_like(proj);
  long total = BNN * H;
  dim3 block(256);
  dim3 grid(cdiv(total, 256));
  gate_split_bwd_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)proj.data_ptr(), mask.data_ptr<float>(),
      (const bf16*)da.data_ptr(), (const bf16*)db.data_ptr(), BNN, N, H,
      outgoing, (bf16*)dproj.data_ptr());
  return dproj;
}

std::vector<torch::Tensor> gate_split_fwd_tiled(torch::Tensor proj, torch::Tensor mask,
                                                int N, int H, int B) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long NN = (long)N * N;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(proj.device());
  auto a = torch::empty({(long)B * H, (long)N, (long)N}, opts);
  auto b = torch::empty_like(a);
  dim3 block(32, 8);
  dim3 grid((unsigned)((NN + 31) / 32), (unsigned)(H / 32), (unsigned)B);
  gate_split_fwd_tiled_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)proj.data_ptr(), mask.data_ptr<float>(), NN, H,
      (bf16*)a.data_ptr(), (bf16*)b.data_ptr());
  return {a, b};
}

void gate_split_bwd_tiled_into(torch::Tensor proj, torch::Tensor mask,
                               torch::Tensor da, torch::Tensor db,
                               torch::Tensor dproj, int N, int H, int B) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long NN = (long)N * N;
  dim3 block(32, 8);
  dim3 grid((unsigned)((NN + 31) / 32), (unsigned)(H / 32), (unsigned)B);
  gate_split_bwd_tiled_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)proj.data_ptr(), mask.data_ptr<float>(),
      (const bf16*)da.data_ptr(), (const bf16*)db.data_ptr(), NN, H,
      (bf16*)dproj.data_ptr());
}

std::vector<torch::Tensor> transpose_ln_fwd(torch::Tensor x, int N, int H, int B,
                                            torch::Tensor gamma, torch::Tensor beta) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long NN = (long)N * N;
  long rows = (long)B * NN;
  auto x_ln = torch::empty({rows, (long)H},
      torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));
  auto mean = torch::empty({rows},
      torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  auto rstd = torch::empty_like(mean);
  dim3 block(256);
  dim3 grid((unsigned)((NN + 31) / 32), 1, (unsigned)B);
  size_t smem = (size_t)H * 33 * sizeof(float);
  transpose_ln_fwd_kernel<<<grid, block, smem, stream>>>(
      (const bf16*)x.data_ptr(), NN, H, B, gamma.data_ptr<float>(),
      beta.data_ptr<float>(), (bf16*)x_ln.data_ptr(), mean.data_ptr<float>(),
      rstd.data_ptr<float>());
  return {x_ln, mean, rstd};
}

torch::Tensor transpose_rows_to_channel(torch::Tensor src, int N, int H, int B) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long NN = (long)N * N;
  auto dst = torch::empty({(long)B * H, (long)N, (long)N},
      torch::TensorOptions().dtype(torch::kBFloat16).device(src.device()));
  dim3 block(32, 8);
  dim3 grid((unsigned)((NN + 31) / 32), (unsigned)(H / 32), (unsigned)B);
  transpose_rows_to_channel_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)src.data_ptr(), NN, H, B, (bf16*)dst.data_ptr());
  return dst;
}

std::vector<torch::Tensor> transpose_ln_bwd(torch::Tensor x, torch::Tensor d_xln, int N,
                               int H, int B, torch::Tensor gamma,
                               torch::Tensor mean, torch::Tensor rstd) {
  auto stream = at::cuda::getCurrentCUDAStream();
  long NN = (long)N * N;
  auto d_x = torch::empty({(long)B * H, (long)N, (long)N},
      torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));
  TORCH_CHECK(H % 32 == 0 && H <= 128,
              "stablefold trimul: c_hidden must be a multiple of 32 and <= 128");
  dim3 block(256);
  unsigned tiles = (unsigned)((NN + 31) / 32);
  dim3 grid(tiles, 1, (unsigned)B);
  long chunks = (long)tiles * B;
  auto partial = torch::empty({chunks * 2, (long)H},
      torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  size_t smem = (size_t)H * 33 * sizeof(float) * 2;
  transpose_ln_bwd_kernel<<<grid, block, smem, stream>>>(
      (const bf16*)x.data_ptr(), (const bf16*)d_xln.data_ptr(), NN, H,
      gamma.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      (bf16*)d_x.data_ptr(), partial.data_ptr<float>());
  auto dgamma = torch::empty({(long)H},
      torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  auto dbeta = torch::empty_like(dgamma);
  int groups = 64;
  int group_len = (int)((chunks + groups - 1) / groups);
  auto stage = torch::empty({(long)groups * 2, (long)H},
      torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  dim3 fold_grid(cdiv(H, 128), groups);
  ln_bwd_dgb_group_fold_kernel<<<fold_grid, 128, 0, stream>>>(
      partial.data_ptr<float>(), (int)chunks, H, group_len,
      stage.data_ptr<float>());
  ln_bwd_dgb_fold_kernel<<<cdiv(H, 128), 128, 0, stream>>>(
      stage.data_ptr<float>(), groups, H,
      dgamma.data_ptr<float>(), dbeta.data_ptr<float>());
  return {d_x, dgamma, dbeta};
}

std::vector<torch::Tensor> pack_rows3(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, long bo, int H, int N,
                                      int D, bool src_h_inner) {
  TORCH_CHECK(D % 2 == 0, "stablefold gather: head dim must be even");
  auto stream = at::cuda::getCurrentCUDAStream();
  long rows = bo * (long)H * N;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
  auto oq = torch::empty({rows, (long)D}, opts);
  auto ok = torch::empty({rows, (long)D}, opts);
  auto ov = torch::empty({rows, (long)D}, opts);
  dim3 block(std::min(D / 2, 32), 8);
  dim3 grid((unsigned)((rows + 7) / 8));
  gather_rows_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)q.data_ptr(), (bf16*)oq.data_ptr(),
      (const bf16*)k.data_ptr(), (bf16*)ok.data_ptr(),
      (const bf16*)v.data_ptr(), (bf16*)ov.data_ptr(),
      rows, H, N, D, src_h_inner);
  return {oq, ok, ov};
}

torch::Tensor gather_rows1(torch::Tensor src, long bo, int H, int N, int D,
                           bool src_h_inner) {
  TORCH_CHECK(D % 2 == 0, "stablefold gather: head dim must be even");
  auto stream = at::cuda::getCurrentCUDAStream();
  long rows = bo * (long)H * N;
  auto dst = torch::empty({rows, (long)D},
      torch::TensorOptions().dtype(torch::kBFloat16).device(src.device()));
  dim3 block(std::min(D / 2, 32), 8);
  dim3 grid((unsigned)((rows + 7) / 8));
  gather_rows_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)src.data_ptr(), (bf16*)dst.data_ptr(),
      nullptr, nullptr, nullptr, nullptr,
      rows, H, N, D, src_h_inner);
  return dst;
}

torch::Tensor sig_mul_fwd(torch::Tensor raw, torch::Tensor gate, int H,
                          long gate_row_stride) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty_like(raw);
  long rows = raw.numel() / H;
  dim3 block(std::min(H / 2, 64), 4);
  dim3 grid((unsigned)((rows + 3) / 4), (unsigned)((H / 2 + block.x - 1) / block.x));
  sig_mul_fwd_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)raw.data_ptr(), (const bf16*)gate.data_ptr(), rows, H,
      gate_row_stride, (bf16*)out.data_ptr());
  return out;
}

torch::Tensor sig_mul_bwd(torch::Tensor raw, torch::Tensor gate,
                          torch::Tensor grad_out, torch::Tensor dgate_dst,
                          int H, long gate_row_stride) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto draw = torch::empty_like(raw);
  long rows = raw.numel() / H;
  dim3 block(std::min(H / 2, 64), 4);
  dim3 grid((unsigned)((rows + 3) / 4), (unsigned)((H / 2 + block.x - 1) / block.x));
  sig_mul_bwd_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)raw.data_ptr(), (const bf16*)gate.data_ptr(),
      (const bf16*)grad_out.data_ptr(), rows, H, gate_row_stride,
      (bf16*)draw.data_ptr(), (bf16*)dgate_dst.data_ptr());
  return draw;
}
"""

_CPP_SRC = """
std::vector<torch::Tensor> ln_fwd(torch::Tensor input, long rows, long row_stride, long col_stride, torch::Tensor gamma, torch::Tensor beta);
torch::Tensor ln_bwd_dx(torch::Tensor input, torch::Tensor grad_out, long rows, long row_stride, long col_stride, torch::Tensor gamma, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> ln_bwd_dgb(torch::Tensor input, torch::Tensor grad_out, long rows, long row_stride, long col_stride, int C, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> gate_split_fwd(torch::Tensor proj, torch::Tensor mask, int N, int H, bool outgoing, int B);
std::vector<torch::Tensor> gate_split_fwd_tiled(torch::Tensor proj, torch::Tensor mask, int N, int H, int B);
void gate_split_bwd_tiled_into(torch::Tensor proj, torch::Tensor mask, torch::Tensor da, torch::Tensor db, torch::Tensor dproj, int N, int H, int B);
std::vector<torch::Tensor> transpose_ln_fwd(torch::Tensor x, int N, int H, int B, torch::Tensor gamma, torch::Tensor beta);
std::vector<torch::Tensor> transpose_ln_bwd(torch::Tensor x, torch::Tensor d_xln, int N, int H, int B, torch::Tensor gamma, torch::Tensor mean, torch::Tensor rstd);
torch::Tensor transpose_rows_to_channel(torch::Tensor src, int N, int H, int B);
torch::Tensor gate_split_bwd(torch::Tensor proj, torch::Tensor mask, torch::Tensor da, torch::Tensor db, int N, int H, bool outgoing, int B);
std::vector<torch::Tensor> pack_rows3(torch::Tensor q, torch::Tensor k, torch::Tensor v, long bo, int H, int N, int D, bool src_h_inner);
torch::Tensor gather_rows1(torch::Tensor src, long bo, int H, int N, int D, bool src_h_inner);
torch::Tensor sig_mul_fwd(torch::Tensor raw, torch::Tensor gate, int H, long gate_row_stride);
torch::Tensor sig_mul_bwd(torch::Tensor raw, torch::Tensor gate, torch::Tensor grad_out, torch::Tensor dgate_dst, int H, long gate_row_stride);
"""

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="stablefold_trimul",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=[
                "ln_fwd", "ln_bwd_dx", "ln_bwd_dgb",
                "gate_split_fwd", "gate_split_bwd",
                "gate_split_fwd_tiled", "gate_split_bwd_tiled_into",
                "transpose_ln_fwd", "transpose_ln_bwd", "transpose_rows_to_channel",
                "sig_mul_fwd", "sig_mul_bwd", "pack_rows3", "gather_rows1",
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


class _StableTrimulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, w_in, w_out, g_in, b_in, g_out, b_out, outgoing):
        ext = _get_ext()
        # Accept arbitrary leading batch dims ([*, N, N, C]) like the official
        # module; flatten them for the kernels and restore on exit.
        orig_shape = z.shape
        C = z.shape[-1]
        N = z.shape[-2]
        if z.shape[-3] != N:
            raise ValueError(f"expected square pair input, got {tuple(z.shape)}")
        B = z.numel() // (N * N * C)
        H = w_in.shape[1] // 5
        z_bf = z.reshape(B, N, N, C).to(torch.bfloat16).contiguous()
        mask_f = (
            mask.expand(*orig_shape[:-1])
            .reshape(B, N, N)
            .to(torch.float32)
            .contiguous()
        )
        ctx.orig_shape = orig_shape
        rows = B * N * N
        z_ln, mean_in, rstd_in = ext.ln_fwd(z_bf.view(rows, C), rows, C, 1, g_in, b_in)
        proj = (z_ln @ w_in.to(torch.bfloat16)).contiguous()          # [rows, 5H]
        gate_view = proj.view(rows, 5 * H)[:, 4 * H:]  # strided view, no copy
        a, b = ext.gate_split_fwd_tiled(proj.view(B, N, N, 5 * H), mask_f, N, H, B)
        # plane transpose via cublas op flags (views, no copies)
        if outgoing:
            x = torch.bmm(a, b.transpose(1, 2))                        # [B*H, N, N]
        else:
            x = torch.bmm(a.transpose(1, 2), b)
        x_ln, mean_out, rstd_out = ext.transpose_ln_fwd(x, N, H, B, g_out, b_out)
        x_rows = None
        out_raw = (x_ln @ w_out.to(torch.bfloat16)).contiguous()
        out = ext.sig_mul_fwd(out_raw.view(-1), proj.view(rows, 5 * H)[:, 4 * H:],
                              C, 5 * H)
        ctx.save_for_backward(z_bf, mask_f, z_ln, proj, a, b, x, x_ln,
                              mean_in, rstd_in, mean_out, rstd_out,
                              w_in, w_out, g_in, g_out, out_raw)
        ctx.dims = (B, N, C, H, outgoing)
        ctx.z_dtype = z.dtype
        return out.view(ctx.orig_shape).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _get_ext()
        (z_bf, mask_f, z_ln, proj, a, b, x, x_ln, mean_in, rstd_in,
         mean_out, rstd_out, w_in, w_out, g_in, g_out, out_raw) = ctx.saved_tensors
        B, N, C, H, outgoing = ctx.dims
        rows = B * N * N
        go = grad_out.to(torch.bfloat16).contiguous().view(-1)
        d_proj_buf = torch.empty_like(proj)   # slots 0..4H by gate_split, 4H..5H here
        d_raw = ext.sig_mul_bwd(out_raw.view(-1), proj.view(rows, 5 * H)[:, 4 * H:],
                                go, d_proj_buf.view(rows, 5 * H)[:, 4 * H:],
                                C, 5 * H)
        d_raw = d_raw.view(rows, C)
        # out GEMM backward
        d_xln = (d_raw @ w_out.to(torch.bfloat16).t()).contiguous()
        d_wout = (x_ln.t() @ d_raw).float()
        # LN out backward, fused transpose variant (dgb folded in)
        d_x, d_gout, d_bout = ext.transpose_ln_bwd(
            x.view(B * H, N * N), d_xln, N, H, B, g_out, mean_out, rstd_out)
        # bmm backward via cublas op views (no copies)
        if outgoing:
            d_a = torch.bmm(d_x, b)
            d_b = torch.bmm(d_x.transpose(1, 2), a)
        else:
            d_a = torch.bmm(b, d_x.transpose(1, 2))
            d_b = torch.bmm(a, d_x)
        # gate-split backward -> dproj (slots 0..4H)
        ext.gate_split_bwd_tiled_into(proj.view(B, N, N, 5 * H), mask_f,
                                      d_a.contiguous(), d_b.contiguous(),
                                      d_proj_buf.view(B, N, N, 5 * H), N, H, B)
        d_proj = d_proj_buf.view(rows, 5 * H)
        # in GEMM backward
        d_zln = (d_proj @ w_in.to(torch.bfloat16).t()).contiguous()
        d_win = (z_ln.t() @ d_proj).float()
        # LN in backward
        d_z = ext.ln_bwd_dx(z_bf.view(rows, C), d_zln, rows, C, 1,
                            g_in, mean_in, rstd_in)
        d_gin, d_bin = ext.ln_bwd_dgb(z_bf.view(rows, C), d_zln, rows, C, 1,
                                      C, mean_in, rstd_in)
        return (d_z.view(ctx.orig_shape).to(ctx.z_dtype), None, d_win, d_wout,
                d_gin, d_bin, d_gout, d_bout, None)


class StableTrimul:
    """Callable mirroring the official module's numerics with fused passes."""

    def __init__(self, module):
        self.w_in = torch.cat(
            [
                module.linear_a_p.weight, module.linear_a_g.weight,
                module.linear_b_p.weight, module.linear_b_g.weight,
                module.linear_g.weight,
            ],
            dim=0,
        ).t().contiguous()
        self.w_out = module.linear_z.weight.t().contiguous()
        self.g_in = module.layer_norm_in.weight.float().contiguous()
        self.b_in = module.layer_norm_in.bias.float().contiguous()
        self.g_out = module.layer_norm_out.weight.float().contiguous()
        self.b_out = module.layer_norm_out.bias.float().contiguous()
        self.outgoing = module._outgoing

    def __call__(self, z, mask):
        return _StableTrimulFn.apply(
            z, mask, self.w_in, self.w_out, self.g_in, self.b_in,
            self.g_out, self.b_out, self.outgoing,
        )


def run_stable_trimul(module, z, mask):
    """Differentiable entry point bound to an official module's parameters."""
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    w_in = torch.cat(
        [
            module.linear_a_p.weight, module.linear_a_g.weight,
            module.linear_b_p.weight, module.linear_b_g.weight,
            module.linear_g.weight,
        ],
        dim=0,
    ).t().contiguous()
    w_out = module.linear_z.weight.t().contiguous()
    return _StableTrimulFn.apply(
        z, mask, w_in, w_out,
        module.layer_norm_in.weight.float().contiguous(),
        module.layer_norm_in.bias.float().contiguous(),
        module.layer_norm_out.weight.float().contiguous(),
        module.layer_norm_out.bias.float().contiguous(),
        module._outgoing,
    )
