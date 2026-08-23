# SPDX-License-Identifier: Apache-2.0
"""StableFold triangle multiplicative update: fused-kernel CUDA implementation.

Forward (3 launches):
  z --[trimul_in_fwd: LN(fp32 stats) + 5-way projection GEMM (mma.sync,
       persistent col-sliced CTAs, weights resident in smem) + sigmoid-gate
       split + channel-major write]--> a, b ([B*H,N,N]), raw gate
  a,b --[cuBLAS bmm (op-flag views, no copies)]--> x
  x --[trimul_out_fwd: transpose-LN + out-projection GEMM + sigmoid gate mul,
       cp.async double-buffered tile pipeline]--> out
z_ln / proj / out_raw never round-trip HBM on the forward; training saves
z_ln, gate, x_ln^T and LN stats only (activation ~630MB/call lighter at
N=640 than materializing proj).

Backward: trimul_gate_split_bwd recomputes proj tiles on tensor cores from
the saved z_ln with the exact forward accumulation order; sig_mul_bwd_out
reconstructs out_raw from the forward output (r*sig = out identity, so
d_gate = dy*out*(1-sig)).  Remaining GEMMs are cuBLAS; LN backwards are the
fixed-order kernels below.

All reductions fixed-order, zero atomics, bitwise replay-deterministic.
Contract (explicit reject, no silent specialization): c_hidden == c_z,
C % 32 == 0, C <= 128, arbitrary N and batch.  Legacy pack/gather helpers
remain for benchmark compatibility; production attention owns its fused QKV
pack in the forward AOT extension and does not import this JIT module.
"""
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

__device__ __forceinline__ float warp_sum32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);
    return v;                                 // butterfly: all lanes hold sum
}

__device__ __forceinline__ void mma_16816(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const bf16* p) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// ---------------------------------------------------------------------------
// trimul_in: fused LN + projection GEMM + gate split.
//   z:    [rows, C] bf16 (rows = B*N*N, C % 16 == 0, C <= 128)
//   wT:   [S, 160, C] bf16, S = H/32; col = wc*80 + p*16 + j maps to
//         projection p of channel h = s*32 + wc*16 + j  (p: ap,ag,bp,bg,g)
//   grid: (S, min(tiles, 2*sm/S)) persistent;  block 256 (8 warps)
//   warp tile: 32 rows x 80 cols -> acc[2][10][4]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_in_fwd_kernel(
    const bf16* __restrict__ z, const float* __restrict__ w_ap,
    const float* __restrict__ w_ag, const float* __restrict__ w_bp,
    const float* __restrict__ w_bg, const float* __restrict__ w_g,
    const float* __restrict__ mask,
    const float* __restrict__ ln_gamma, const float* __restrict__ ln_beta,
    long rows, long NN, int C, int H,
    bf16* __restrict__ a_out, bf16* __restrict__ b_out,
    bf16* __restrict__ gate_out, float* __restrict__ save_mean,
    float* __restrict__ save_rstd, bf16* __restrict__ zln_out) {
  extern __shared__ bf16 smem[];
  const int Cp = C + 8;                       // padded k stride
  bf16* wbuf = smem;                          // [160][Cp]
  bf16* zbuf = smem + 160 * Cp;               // [128][Cp]
  bf16* a_st = zbuf + 128 * Cp;               // [32 ch][136 rows]
  bf16* b_st = a_st + 32 * 136;               // [32 ch][136 rows]
  bf16* g_st = b_st + 32 * 136;               // [128 rows][40 ch-pad]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int slice = blockIdx.x;
  const long tiles = (rows + 127) >> 7;

  // ---- once per CTA: raw fp32 weights -> permuted bf16 smem slice ----
  // col = wc*80 + p*16 + j -> projection p of channel h = slice*32 + wc*16 + j
  {
    const float* wsrc[5] = {w_ap, w_ag, w_bp, w_bg, w_g};
    int chunks_per_col = C >> 3;              // 8 values per uint4 store
    for (int i = tid; i < 160 * chunks_per_col; i += 256) {
      int col = i / chunks_per_col, q = i % chunks_per_col;
      int wcg = col / 80, rem = col % 80;
      const float* src = wsrc[rem >> 4]
          + (long)(slice * 32 + wcg * 16 + (rem & 15)) * C + q * 8;
      union { bf16 b[8]; uint4 u; } pk;
      #pragma unroll
      for (int t = 0; t < 8; ++t) pk.b[t] = __float2bfloat16(src[t]);
      *reinterpret_cast<uint4*>(wbuf + col * Cp + q * 8) = pk.u;
    }
  }
  const int c0 = lane * 4;
  const bool cok = c0 < C;
  const float invC = 1.0f / (float)C;

  // ---- persistent tile loop ----
  for (long tile = blockIdx.y; tile * 128 < rows; tile += gridDim.y) {
    const long row0 = tile * 128;

    // ---- phase 1: LN, batched loads first (16 latencies -> 1) ----
    {
      float gam[4] = {0.f, 0.f, 0.f, 0.f}, bet[4] = {0.f, 0.f, 0.f, 0.f};
      if (cok) {
        float4 gv = *reinterpret_cast<const float4*>(ln_gamma + c0);
        float4 bv = *reinterpret_cast<const float4*>(ln_beta + c0);
        gam[0] = gv.x; gam[1] = gv.y; gam[2] = gv.z; gam[3] = gv.w;
        bet[0] = bv.x; bet[1] = bv.y; bet[2] = bv.z; bet[3] = bv.w;
      }
      uint2 raw[16];
      #pragma unroll
      for (int it = 0; it < 16; ++it) {
        long row = row0 + warp * 16 + it;
        raw[it] = (row < rows && cok)
            ? *reinterpret_cast<const uint2*>(z + row * (long)C + c0)
            : make_uint2(0, 0);
      }
      #pragma unroll
      for (int it = 0; it < 16; ++it) {
        int r = warp * 16 + it;
        long row = row0 + r;
        float v[4];
        const bf16* pv = reinterpret_cast<const bf16*>(&raw[it]);
        #pragma unroll
        for (int t = 0; t < 4; ++t) v[t] = __bfloat162float(pv[t]);
        float sum = warp_sum32(v[0] + v[1] + v[2] + v[3]);
        float sq = warp_sum32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
                              + v[3] * v[3]);
        float mean = sum * invC;
        float rstd = rsqrtf(sq * invC - mean * mean + 1e-5f);
        if (lane == 0 && slice == 0 && row < rows) {
          save_mean[row] = mean; save_rstd[row] = rstd;
        }
        if (cok) {
          bool live = row < rows;
          __nv_bfloat162 p0 = __floats2bfloat162_rn(
              live ? (v[0] - mean) * rstd * gam[0] + bet[0] : 0.f,
              live ? (v[1] - mean) * rstd * gam[1] + bet[1] : 0.f);
          __nv_bfloat162 p1 = __floats2bfloat162_rn(
              live ? (v[2] - mean) * rstd * gam[2] + bet[2] : 0.f,
              live ? (v[3] - mean) * rstd * gam[3] + bet[3] : 0.f);
          uint2 packed;
          packed.x = *reinterpret_cast<unsigned*>(&p0);
          packed.y = *reinterpret_cast<unsigned*>(&p1);
          *reinterpret_cast<uint2*>(zbuf + r * Cp + c0) = packed;
          if (zln_out != nullptr && slice == 0 && live) {
            *reinterpret_cast<uint2*>(zln_out + row * (long)C + c0) = packed;
          }
        }
      }
    }
    __syncthreads();                          // S1: zbuf ready (wbuf on tile 0)

    // ---- phase 2: GEMM via ldmatrix + mma ----
    const int wr = warp & 3, wc = warp >> 2;
    const int rbase = wr * 32;
    float acc[2][10][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 10; ++j)
        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const bf16* za0 = zbuf + (rbase + (lane & 15)) * Cp + 8 * (lane >> 4);
    const bf16* za1 = za0 + 16 * Cp;
    const bf16* wb0 = wbuf + (wc * 80 + (lane >> 4) * 8 + (lane & 7)) * Cp
                    + 8 * ((lane >> 3) & 1);
    auto gemm_step = [&](int k0) {
      unsigned afrag[2][4];
      ldmatrix_x4(afrag[0][0], afrag[0][1], afrag[0][2], afrag[0][3], za0 + k0);
      ldmatrix_x4(afrag[1][0], afrag[1][1], afrag[1][2], afrag[1][3], za1 + k0);
      #pragma unroll
      for (int jp = 0; jp < 5; ++jp) {        // B: two 16x8 tiles per x4
        unsigned b00, b01, b10, b11;
        ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * Cp + k0);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          mma_16816(acc[i][jp * 2][0], acc[i][jp * 2][1],
                    acc[i][jp * 2][2], acc[i][jp * 2][3],
                    afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                    b00, b01);
          mma_16816(acc[i][jp * 2 + 1][0], acc[i][jp * 2 + 1][1],
                    acc[i][jp * 2 + 1][2], acc[i][jp * 2 + 1][3],
                    afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                    b10, b11);
        }
      }
    };
    if (C == 128) {                           // compile-time lane, same source
      #pragma unroll
      for (int k0 = 0; k0 < 128; k0 += 16) gemm_step(k0);
    } else {
      for (int k0 = 0; k0 < C; k0 += 16) gemm_step(k0);
    }
    // no barrier: S3 below separates these zbuf reads from next tile's LN
    // writes, and S1 separated last tile's stage readers from this stage.

    // ---- phase 3: register sigmoid-gate epilogue -> stage ----
    const int n0 = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int r_lo = rbase + i * 16 + (lane >> 2);
      long row_lo = row0 + r_lo, row_hi = row_lo + 8;
      float m_lo = row_lo < rows ? mask[row_lo] : 0.f;
      float m_hi = row_hi < rows ? mask[row_hi] : 0.f;
      #pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
          int ch32 = wc * 16 + jp * 8 + n0 + q;
          float ap0 = acc[i][jp][q],     ag0 = acc[i][2 + jp][q];
          float bp0 = acc[i][4 + jp][q], bg0 = acc[i][6 + jp][q];
          float g0  = acc[i][8 + jp][q];
          float ap1 = acc[i][jp][2 + q],     ag1 = acc[i][2 + jp][2 + q];
          float bp1 = acc[i][4 + jp][2 + q], bg1 = acc[i][6 + jp][2 + q];
          float g1  = acc[i][8 + jp][2 + q];
          a_st[ch32 * 136 + r_lo] = __float2bfloat16(
              __fdividef(ap0, 1.f + __expf(-ag0)) * m_lo);
          b_st[ch32 * 136 + r_lo] = __float2bfloat16(
              __fdividef(bp0, 1.f + __expf(-bg0)) * m_lo);
          g_st[r_lo * 40 + ch32] = __float2bfloat16(g0);
          a_st[ch32 * 136 + r_lo + 8] = __float2bfloat16(
              __fdividef(ap1, 1.f + __expf(-ag1)) * m_hi);
          b_st[ch32 * 136 + r_lo + 8] = __float2bfloat16(
              __fdividef(bp1, 1.f + __expf(-bg1)) * m_hi);
          g_st[(r_lo + 8) * 40 + ch32] = __float2bfloat16(g1);
        }
      }
    }

    __syncthreads();                          // S3: stage complete

    // ---- phase 4: cooperative coalesced global writes ----
    // one 64-bit div per tile; per-chunk batch index via carry propagation
    // (tile is 128 rows, NN >= 144 -> at most one boundary crossing)
    const long bidx0 = row0 / NN;
    const long pos0 = row0 - bidx0 * NN;
    for (int id = tid; id < 32 * 16; id += 256) {
      int ch32 = id >> 4, r = (id & 15) * 8;
      int h = slice * 32 + ch32;
      long pos = pos0 + r;
      long bidx = bidx0;
      while (pos >= NN) { pos -= NN; ++bidx; }
      long addr = (bidx * H + h) * NN + pos;
      bool fast = (row0 + r + 7 < rows) && (pos + 7 < NN) && ((addr & 7) == 0);
      if (fast) {
        *reinterpret_cast<uint4*>(a_out + addr) =
            *reinterpret_cast<const uint4*>(a_st + ch32 * 136 + r);
        *reinterpret_cast<uint4*>(b_out + addr) =
            *reinterpret_cast<const uint4*>(b_st + ch32 * 136 + r);
      } else {
        for (int t = 0; t < 8; ++t) {
          if (row0 + r + t >= rows) break;
          long p = pos + t;
          long bi = bidx;
          while (p >= NN) { p -= NN; ++bi; }
          long ad = (bi * H + h) * NN + p;
          a_out[ad] = a_st[ch32 * 136 + r + t];
          b_out[ad] = b_st[ch32 * 136 + r + t];
        }
      }
    }
    for (int id = tid; id < 128 * 4; id += 256) {
      int r = id >> 2, part = id & 3;
      long row = row0 + r;
      if (row >= rows) continue;
      *reinterpret_cast<uint4*>(gate_out + row * (long)H + slice * 32
                                + part * 8) =
          *reinterpret_cast<const uint4*>(g_st + r * 40 + part * 8);
    }
    // no tail barrier: next iteration's S1 orders these stage reads before
    // the following stage overwrite.
  }
}

std::vector<torch::Tensor> trimul_in_fwd(torch::Tensor z, torch::Tensor w_ap,
                                       torch::Tensor w_ag, torch::Tensor w_bp,
                                       torch::Tensor w_bg, torch::Tensor w_g,
                                       torch::Tensor mask, torch::Tensor gamma,
                                       torch::Tensor beta, long B, long NN,
                                       int C, int H, bool save_zln) {
  TORCH_CHECK(C % 16 == 0 && C <= 128, "trimul_in: C must be mult of 16, <=128");
  TORCH_CHECK(H % 32 == 0, "trimul_in: H must be mult of 32");
  for (auto* w : {&w_ap, &w_ag, &w_bp, &w_bg, &w_g}) {
    TORCH_CHECK(w->dtype() == torch::kFloat && w->is_contiguous(),
                "trimul_in: weights must be contiguous fp32");
  }
  long rows = B * NN;
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(z.device());
  auto a = torch::empty({B * (long)H, NN}, opts);
  auto b = torch::empty_like(a);
  auto gate = torch::empty({rows, (long)H}, opts);
  auto fopts = torch::TensorOptions().dtype(torch::kFloat).device(z.device());
  auto mean = torch::empty({rows}, fopts);
  auto rstd = torch::empty({rows}, fopts);
  auto zln = torch::empty({save_zln ? rows : 0, (long)C}, opts);
  int Cp = C + 8;
  size_t smem = ((size_t)160 * Cp + 128 * Cp
                 + 2 * 32 * 136 + 128 * 40) * sizeof(bf16);
  cudaFuncSetAttribute(trimul_in_fwd_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int device = 0; cudaGetDevice(&device);
  int sm_count = 132;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  int S = H / 32;
  long y = std::max(1L, std::min(tiles, (long)(2 * sm_count) / S));
  TORCH_CHECK(y <= 65535, "trimul_in: grid.y overflow");
  dim3 grid((unsigned)S, (unsigned)y);
  trimul_in_fwd_kernel<<<grid, dim3(256), smem, stream>>>(
      (const bf16*)z.data_ptr(), w_ap.data_ptr<float>(),
      w_ag.data_ptr<float>(), w_bp.data_ptr<float>(), w_bg.data_ptr<float>(),
      w_g.data_ptr<float>(),
      mask.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
      rows, NN, C, H, (bf16*)a.data_ptr(), (bf16*)b.data_ptr(),
      (bf16*)gate.data_ptr(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      save_zln ? (bf16*)zln.data_ptr() : nullptr);
  return {a, b, gate, mean, rstd, zln};
}

__device__ __forceinline__ void cp_async16(bf16* dst, const bf16* src) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const bf16* p) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// ---------------------------------------------------------------------------
// trimul_out: fused transpose-LN + out GEMM + gate mul.
//   x:    [B*H, NN] bf16 channel-major (bmm output)
//   wc:   [C, H] bf16 = linear_z.weight, k(H)-contiguous
//   gate: [rows, C] bf16 raw gate logits (trimul_in_fwd output)
//   out:  [rows, C] bf16 = (x_ln @ wc^T) * sigmoid(gate)
//   training also writes: out_raw [rows, C], xlnT [H, rows], mean/rstd [rows]
//   grid: 1-D persistent (2*sm CTAs); block 256; 128-position tiles
//   warp tile: 32 pos x C/2 cols
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_out_fwd_kernel(
    const bf16* __restrict__ x, const float* __restrict__ wz,
    const bf16* __restrict__ gate,
    const float* __restrict__ ln_gamma, const float* __restrict__ ln_beta,
    long rows, long NN, int C, int H,
    bf16* __restrict__ out, bf16* __restrict__ xlnT,
    float* __restrict__ save_mean, float* __restrict__ save_rstd) {
  extern __shared__ bf16 smem[];
  const int Pp = 136;                          // padded pos stride in xbuf
  const int Hp = H + 8;                        // padded k stride in wbuf
  bf16* xb[2] = {smem, smem + (long)H * Pp};   // double-buffered [H][Pp]
  bf16* wbuf = smem + 2 * (long)H * Pp;        // [C][Hp]
  float* gbuf = reinterpret_cast<float*>(wbuf + (long)C * Hp);  // gamma|beta [2H]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const bool training = xlnT != nullptr;

  // ---- once per CTA: raw fp32 linear_z.weight [C,H] -> bf16 smem ----
  {
    int chunks = H >> 3;
    for (int i = tid; i < C * chunks; i += 256) {
      int c = i / chunks, q = i % chunks;
      const float* src = wz + (long)c * H + q * 8;
      union { bf16 b[8]; uint4 u; } pk;
      #pragma unroll
      for (int t = 0; t < 8; ++t) pk.b[t] = __float2bfloat16(src[t]);
      reinterpret_cast<uint4*>(wbuf + c * Hp)[q] = pk.u;
    }
    for (int i = tid; i < H; i += 256) {
      gbuf[i] = ln_gamma[i];
      gbuf[H + i] = ln_beta[i];
    }
  }
  const float invH = 1.0f / (float)H;

  // async phase T: x planes -> xb[buf] [h][pos], straddle/tail-guarded
  auto issue_T = [&](long tile, int buf) {
    const long row0 = tile * 128;
    const long bidx0 = row0 / NN;
    const long pos0 = row0 - bidx0 * NN;
    bf16* xbuf = xb[buf];
    for (int id = tid; id < H * 16; id += 256) {
      int h = id >> 4, r8 = (id & 15) * 8;
      long pos = pos0 + r8;
      long bi = bidx0;
      while (pos >= NN) { pos -= NN; ++bi; }
      if (row0 + r8 + 7 < rows && pos + 7 < NN && (NN & 7) == 0) {
        cp_async16(xbuf + h * Pp + r8, x + (bi * H + h) * NN + pos);
      } else {
        bf16 tmp[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
          long p = pos + t;
          long b2 = bi;
          while (p >= NN) { p -= NN; ++b2; }
          tmp[t] = (row0 + r8 + t < rows)
              ? x[(b2 * H + h) * NN + p] : __float2bfloat16(0.f);
        }
        *reinterpret_cast<uint4*>(xbuf + h * Pp + r8) =
            *reinterpret_cast<const uint4*>(tmp);
      }
    }
    cp_async_commit();
  };

  // ---- persistent pipelined tile loop ----
  int cur = 0;
  if (blockIdx.x * 128 < rows) issue_T(blockIdx.x, cur);
  for (long tile = blockIdx.x; tile * 128 < rows; tile += gridDim.x) {
    const long row0 = tile * 128;
    bf16* xbuf = xb[cur];
    cp_async_wait0();
    __syncthreads();                           // xb[cur] filled for all warps

    // ---- phase LN: per-position stats, lane-pair per position ----
    // warp w owns positions [w*16, w*16+16); lane = 2*pos_local + h_parity
    {
      int p = warp * 16 + (lane >> 1);
      int hpar = lane & 1;
      long row = row0 + p;
      float sum = 0.f, sq = 0.f;
      for (int h = hpar; h < H; h += 2) {
        float v = __bfloat162float(xbuf[h * Pp + p]);
        sum += v; sq += v * v;
      }
      sum += __shfl_xor_sync(FULL, sum, 1);
      sq += __shfl_xor_sync(FULL, sq, 1);
      float mean = sum * invH;
      float rstd = rsqrtf(sq * invH - mean * mean + 1e-5f);
      if (training && hpar == 0 && row < rows) {
        save_mean[row] = mean; save_rstd[row] = rstd;
      }
      bool live = row < rows;
      for (int h = hpar; h < H; h += 2) {
        float v = __bfloat162float(xbuf[h * Pp + p]);
        xbuf[h * Pp + p] = __float2bfloat16(
            live ? (v - mean) * rstd * gbuf[h] + gbuf[H + h] : 0.f);
      }
    }
    __syncthreads();

    long ntile = tile + gridDim.x;
    if (ntile * 128 < rows) issue_T(ntile, cur ^ 1);

    // ---- training: coalesced x_ln^T write-out from normalized xbuf ----
    if (training) {
      for (int id = tid; id < H * 16; id += 256) {
        int h = id >> 4, r8 = (id & 15) * 8;
        long row = row0 + r8;
        if (row + 7 < rows && (rows & 7) == 0) {
          *reinterpret_cast<uint4*>(xlnT + (long)h * rows + row) =
              *reinterpret_cast<const uint4*>(xbuf + h * Pp + r8);
        } else {
          for (int t = 0; t < 8 && row + t < rows; ++t)
            xlnT[(long)h * rows + row + t] = xbuf[h * Pp + r8 + t];
        }
      }
    }

    // ---- phase GEMM: out_raw = x_ln @ wc^T via ldmatrix(.trans) + mma ----
    const int wr = warp & 3, wc2 = warp >> 2;
    const int rbase = wr * 32;
    const int halfC = C >> 1;
    const int cbase = wc2 * halfC;
    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // A(.trans): row = h, col = pos;  B: wbuf rows = c, k contiguous
    const bf16* xa0 = xbuf + ((lane & 7) + ((lane >> 4) << 3)) * Pp
                    + rbase + (((lane >> 3) & 1) << 3);
    const bf16* xa1 = xa0 + 16;
    const bf16* wb0 = wbuf + (cbase + (lane >> 4) * 8 + (lane & 7)) * Hp
                    + 8 * ((lane >> 3) & 1);
    auto gemm_all = [&](int ksteps, int ntp) {
      for (int ks = 0; ks < ksteps; ++ks) {
        int k0 = ks * 16;
        unsigned afrag[2][4];
        ldmatrix_x4_trans(afrag[0][0], afrag[0][1], afrag[0][2], afrag[0][3],
                          xa0 + k0 * Pp);
        ldmatrix_x4_trans(afrag[1][0], afrag[1][1], afrag[1][2], afrag[1][3],
                          xa1 + k0 * Pp);
        for (int jp = 0; jp < ntp; ++jp) {
          unsigned b00, b01, b10, b11;
          ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * Hp + k0);
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma_16816(acc[i][jp * 2][0], acc[i][jp * 2][1],
                      acc[i][jp * 2][2], acc[i][jp * 2][3],
                      afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                      b00, b01);
            mma_16816(acc[i][jp * 2 + 1][0], acc[i][jp * 2 + 1][1],
                      acc[i][jp * 2 + 1][2], acc[i][jp * 2 + 1][3],
                      afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                      b10, b11);
          }
        }
      }
    };
    if (C == 128 && H == 128) gemm_all(8, 4);
    else                      gemm_all(H >> 4, C >> 5);

    // ---- phase epilogue: gate-mul, direct paired global writes ----
    const int n0 = (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int p_lo = rbase + i * 16 + (lane >> 2);
      #pragma unroll 2
      for (int half = 0; half < 2; ++half) {   // p_lo and p_lo + 8
        long row = row0 + p_lo + half * 8;
        if (row >= rows) continue;
        const int qsel = half * 2;             // acc[..][qsel], [..][qsel+1]
        for (int j = 0; j < 8; ++j) {
          int c = cbase + j * 8 + n0;
          if (c >= cbase + halfC) break;
          float r0 = acc[i][j][qsel], r1 = acc[i][j][qsel + 1];
          unsigned gpair = *reinterpret_cast<const unsigned*>(
              gate + row * (long)C + c);
          const bf16* gp = reinterpret_cast<const bf16*>(&gpair);
          float g0 = __bfloat162float(gp[0]), g1 = __bfloat162float(gp[1]);
          float s0 = __fdividef(r0, 1.f + __expf(-g0));
          float s1 = __fdividef(r1, 1.f + __expf(-g1));
          __nv_bfloat162 po = __floats2bfloat162_rn(s0, s1);
          *reinterpret_cast<unsigned*>(out + row * (long)C + c) =
              *reinterpret_cast<unsigned*>(&po);
        }
      }
    }
    cur ^= 1;                                  // next tile reads the async buf
  }
}

std::vector<torch::Tensor> trimul_out_fwd(torch::Tensor x, torch::Tensor wz,
                                       torch::Tensor gate, torch::Tensor gamma,
                                       torch::Tensor beta, long B, long NN,
                                       int C, int H, bool training) {
  TORCH_CHECK(C % 32 == 0 && C <= 128, "trimul_out: C must be mult of 32, <=128");
  TORCH_CHECK(H % 32 == 0 && H <= 128, "trimul_out: H must be mult of 32, <=128");
  TORCH_CHECK(wz.dtype() == torch::kFloat && wz.is_contiguous(),
              "trimul_out: weight must be contiguous fp32");
  long rows = B * NN;
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());
  auto o = torch::empty({rows, (long)C}, opts);
  auto xln = torch::empty({training ? (long)H : 0, rows}, opts);
  auto fopts = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
  auto mean = torch::empty({training ? rows : 0}, fopts);
  auto rstd = torch::empty({training ? rows : 0}, fopts);
  size_t smem = ((size_t)2 * H * 136 + (size_t)C * (H + 8)) * sizeof(bf16)
              + (size_t)2 * H * sizeof(float);
  cudaFuncSetAttribute(trimul_out_fwd_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int device = 0; cudaGetDevice(&device);
  int sm_count = 132;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  long g = std::max(1L, std::min(tiles, (long)(2 * sm_count)));
  trimul_out_fwd_kernel<<<dim3((unsigned)g), dim3(256), smem, stream>>>(
      (const bf16*)x.data_ptr(), wz.data_ptr<float>(),
      (const bf16*)gate.data_ptr(), gamma.data_ptr<float>(),
      beta.data_ptr<float>(), rows, NN, C, H,
      (bf16*)o.data_ptr(),
      training ? (bf16*)xln.data_ptr() : nullptr,
      training ? mean.data_ptr<float>() : nullptr,
      training ? rstd.data_ptr<float>() : nullptr);
  return {o, xln, mean, rstd};
}

// ---------------------------------------------------------------------------
// trimul_gate_split_bwd: recompute-proj gate-split backward.
//   z_ln: [rows, C] bf16 (saved by K_in forward)
//   wT4:  [S, 128, C] bf16, S = H/32; col = wc*64 + p*16 + j -> projection
//         p in {ap,ag,bp,bg} of channel h = s*32 + wc*16 + j
//   d_a, d_b: [B*H, NN] bf16 channel-major (bmm grads)
//   d_proj5: [rows, 5H] bf16, slots [p*H + h] for p < 4 written here
//   grid (S, 2*sm/S) persistent; block 256; warp tile 32 rows x 64 cols
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_gate_split_bwd_kernel(
    const bf16* __restrict__ zln, const float* __restrict__ w_ap,
    const float* __restrict__ w_ag, const float* __restrict__ w_bp,
    const float* __restrict__ w_bg,
    const float* __restrict__ mask,
    const bf16* __restrict__ d_a, const bf16* __restrict__ d_b,
    long rows, long NN, int C, int H,
    bf16* __restrict__ d_proj5) {
  extern __shared__ bf16 smem[];
  const int Cp = C + 8;
  bf16* wbuf = smem;                           // [128][Cp]
  bf16* zbuf = wbuf + 128 * Cp;                // [128][Cp]
  bf16* da_st = zbuf + 128 * Cp;               // [32 ch][136 rows]
  bf16* db_st = da_st + 32 * 136;              // [32 ch][136 rows]
  bf16* dp_st = db_st + 32 * 136;              // [128 rows][72] two-block wave

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int slice = blockIdx.x;
  const long fiveH = 5 * (long)H;

  // ---- once per CTA: raw fp32 4-projection slice -> permuted bf16 smem ----
  // col = wc*64 + p*16 + j -> projection p of channel h = slice*32 + wc*16 + j
  {
    const float* wsrc[4] = {w_ap, w_ag, w_bp, w_bg};
    int chunks = C >> 3;
    for (int i = tid; i < 128 * chunks; i += 256) {
      int col = i / chunks, q = i % chunks;
      int wcg = col / 64, rem = col % 64;
      const float* src = wsrc[rem >> 4]
          + (long)(slice * 32 + wcg * 16 + (rem & 15)) * C + q * 8;
      union { bf16 b[8]; uint4 u; } pk;
      #pragma unroll
      for (int t = 0; t < 8; ++t) pk.b[t] = __float2bfloat16(src[t]);
      *reinterpret_cast<uint4*>(wbuf + col * Cp + q * 8) = pk.u;
    }
  }

  // async z_ln tile load (dense [rows, C], no straddle concerns)
  auto issue_z = [&](long tile, bool first) {
    const long row0 = tile * 128;
    int chunks = C >> 3;
    for (int id = tid; id < 128 * chunks; id += 256) {
      int r = id / chunks, q = id % chunks;
      long row = row0 + r;
      if (row < rows) {
        cp_async16(zbuf + r * Cp + q * 8, zln + row * (long)C + q * 8);
      } else if (first) {                      // zero-fill dead rows once
        *reinterpret_cast<uint4*>(zbuf + r * Cp + q * 8) = make_uint4(0,0,0,0);
      }
    }
    cp_async_commit();
  };

  long tile0 = blockIdx.y;
  if (tile0 * 128 < rows) issue_z(tile0, true);

  for (long tile = tile0; tile * 128 < rows; tile += gridDim.y) {
    const long row0 = tile * 128;
    const long bidx0 = row0 / NN;
    const long pos0 = row0 - bidx0 * NN;

    // ---- stage d_a / d_b (channel-major reads, same pattern as fwd writes) --
    for (int id = tid; id < 32 * 16; id += 256) {
      int ch32 = id >> 4, r = (id & 15) * 8;
      int h = slice * 32 + ch32;
      long pos = pos0 + r;
      long bi = bidx0;
      while (pos >= NN) { pos -= NN; ++bi; }
      long addr = (bi * H + h) * NN + pos;
      if (row0 + r + 7 < rows && pos + 7 < NN && (addr & 7) == 0) {
        *reinterpret_cast<uint4*>(da_st + ch32 * 136 + r) =
            *reinterpret_cast<const uint4*>(d_a + addr);
        *reinterpret_cast<uint4*>(db_st + ch32 * 136 + r) =
            *reinterpret_cast<const uint4*>(d_b + addr);
      } else {
        for (int t = 0; t < 8; ++t) {
          long row = row0 + r + t;
          bf16 va = __float2bfloat16(0.f), vb = va;
          if (row < rows) {
            long p = pos + t;
            long b2 = bi;
            while (p >= NN) { p -= NN; ++b2; }
            long ad = (b2 * H + h) * NN + p;
            va = d_a[ad]; vb = d_b[ad];
          }
          da_st[ch32 * 136 + r + t] = va;
          db_st[ch32 * 136 + r + t] = vb;
        }
      }
    }
    cp_async_wait0();
    __syncthreads();                           // S1: zbuf + da/db staged

    // ---- GEMM: recompute proj slice (bitwise = forward accumulators) ----
    const int wr = warp & 3, wc = warp >> 2;
    const int rbase = wr * 32;
    float acc[2][8][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const bf16* za0 = zbuf + (rbase + (lane & 15)) * Cp + 8 * (lane >> 4);
    const bf16* za1 = za0 + 16 * Cp;
    const bf16* wb0 = wbuf + (wc * 64 + (lane >> 4) * 8 + (lane & 7)) * Cp
                    + 8 * ((lane >> 3) & 1);
    auto gemm_all = [&](int ksteps) {
      for (int ks = 0; ks < ksteps; ++ks) {
        int k0 = ks * 16;
        unsigned afrag[2][4];
        ldmatrix_x4(afrag[0][0], afrag[0][1], afrag[0][2], afrag[0][3],
                    za0 + k0);
        ldmatrix_x4(afrag[1][0], afrag[1][1], afrag[1][2], afrag[1][3],
                    za1 + k0);
        #pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          unsigned b00, b01, b10, b11;
          ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * Cp + k0);
          #pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma_16816(acc[i][jp * 2][0], acc[i][jp * 2][1],
                      acc[i][jp * 2][2], acc[i][jp * 2][3],
                      afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                      b00, b01);
            mma_16816(acc[i][jp * 2 + 1][0], acc[i][jp * 2 + 1][1],
                      acc[i][jp * 2 + 1][2], acc[i][jp * 2 + 1][3],
                      afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                      b10, b11);
          }
        }
      }
    };
    if (C == 128) gemm_all(8);
    else          gemm_all(C >> 4);

    // prefetch next tile's z_ln behind the epilogue (zbuf reads are done
    // per-warp after its GEMM; the S_a barrier below orders all warps)
    // NOTE: issued after S_a so no warp still reads zbuf.

    // ---- two-wave epilogue: (d_ap, d_ag) then (d_bp, d_bg) ----
    const int n0 = (lane & 3) * 2;
    #pragma unroll
    for (int wave = 0; wave < 2; ++wave) {
      __syncthreads();                         // S_a / S_b: previous readers done
      if (wave == 1) {
        long ntile = tile + gridDim.y;
        if (ntile * 128 < rows) issue_z(ntile, false);
      }
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        int r_lo = rbase + i * 16 + (lane >> 2);
        long row_lo = row0 + r_lo, row_hi = row_lo + 8;
        float m_lo = row_lo < rows ? mask[row_lo] : 0.f;
        float m_hi = row_hi < rows ? mask[row_hi] : 0.f;
        #pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
          #pragma unroll
          for (int q = 0; q < 2; ++q) {
            int ch32 = wc * 16 + jp * 8 + n0 + q;
            // wave 0: p=ap(0),g=ag(2); wave 1: p=bp(4),g=bg(6)
            float pv0 = acc[i][wave * 4 + jp][q];
            float gv0 = acc[i][wave * 4 + 2 + jp][q];
            float pv1 = acc[i][wave * 4 + jp][2 + q];
            float gv1 = acc[i][wave * 4 + 2 + jp][2 + q];
            const bf16* dsrc = (wave == 0 ? da_st : db_st) + ch32 * 136;
            float d0 = __bfloat162float(dsrc[r_lo]);
            float d1 = __bfloat162float(dsrc[r_lo + 8]);
            float s0 = __fdividef(1.f, 1.f + __expf(-gv0));
            float s1 = __fdividef(1.f, 1.f + __expf(-gv1));
            // d_p = dy * sig(g) * m ; d_g = dy * p * sig'(g) * m
            dp_st[r_lo * 72 + ch32] =
                __float2bfloat16(d0 * s0 * m_lo);
            dp_st[r_lo * 72 + 32 + ch32] =
                __float2bfloat16(d0 * pv0 * s0 * (1.f - s0) * m_lo);
            dp_st[(r_lo + 8) * 72 + ch32] =
                __float2bfloat16(d1 * s1 * m_hi);
            dp_st[(r_lo + 8) * 72 + 32 + ch32] =
                __float2bfloat16(d1 * pv1 * s1 * (1.f - s1) * m_hi);
          }
        }
      }
      __syncthreads();                         // S_b / S_c: dp_st staged
      // coop write: two H-blocks (p = 2*wave, 2*wave+1) of d_proj5
      for (int id = tid; id < 128 * 2 * 4; id += 256) {
        int r = id >> 3, blk = (id >> 2) & 1, q = id & 3;
        long row = row0 + r;
        if (row >= rows) continue;
        long col = (long)(2 * wave + blk) * H + slice * 32 + q * 8;
        *reinterpret_cast<uint4*>(d_proj5 + row * fiveH + col) =
            *reinterpret_cast<const uint4*>(dp_st + r * 72 + blk * 32 + q * 8);
      }
    }
    __syncthreads();                           // dp_st readers done before next
  }
}

// ---------------------------------------------------------------------------
// sig_mul backward from out: d_raw = dy*sig(g); d_gate = dy*out*(1-sig(g)).
// d_gate written into a strided destination (the [4H:5H) slot of d_proj5).
// ---------------------------------------------------------------------------
__global__ void sig_mul_bwd_out_kernel(
    const bf16* __restrict__ out, const bf16* __restrict__ gate,
    const bf16* __restrict__ grad_out, long total, int H,
    long dgate_stride, bf16* __restrict__ draw, bf16* __restrict__ dgate) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  long row = idx / H; int h = idx % H;
  float gv = __bfloat162float(gate[row * (long)H + h]);
  float g = __fdividef(1.f, 1.f + __expf(-gv));
  float dy = __bfloat162float(grad_out[idx]);
  float o = __bfloat162float(out[idx]);
  draw[idx] = __float2bfloat16(dy * g);
  dgate[row * dgate_stride + h] = __float2bfloat16(dy * o * (1.f - g));
}

torch::Tensor trimul_gate_split_bwd(torch::Tensor zln, torch::Tensor w_ap,
                          torch::Tensor w_ag, torch::Tensor w_bp,
                          torch::Tensor w_bg,
                          torch::Tensor mask, torch::Tensor d_a,
                          torch::Tensor d_b, torch::Tensor d_proj5,
                          long B, long NN, int C, int H) {
  TORCH_CHECK(C % 16 == 0 && C <= 128, "trimul_gate_split_bwd: C must be mult of 16, <=128");
  TORCH_CHECK(H % 32 == 0, "trimul_gate_split_bwd: H must be mult of 32");
  long rows = B * NN;
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  int Cp = C + 8;
  size_t smem = ((size_t)128 * Cp + 128 * Cp + 2 * 32 * 136 + 128 * 72)
              * sizeof(bf16);
  cudaFuncSetAttribute(trimul_gate_split_bwd_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int device = 0; cudaGetDevice(&device);
  int sm_count = 132;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  int S = H / 32;
  long y = std::max(1L, std::min(tiles, (long)(2 * sm_count) / S));
  TORCH_CHECK(y <= 65535, "trimul_gate_split_bwd: grid.y overflow");
  trimul_gate_split_bwd_kernel<<<dim3((unsigned)S, (unsigned)y), dim3(256), smem,
                       stream>>>(
      (const bf16*)zln.data_ptr(), w_ap.data_ptr<float>(),
      w_ag.data_ptr<float>(), w_bp.data_ptr<float>(), w_bg.data_ptr<float>(),
      mask.data_ptr<float>(), (const bf16*)d_a.data_ptr(),
      (const bf16*)d_b.data_ptr(), rows, NN, C, H,
      (bf16*)d_proj5.data_ptr());
  return d_proj5;
}

torch::Tensor sig_mul_bwd_out(torch::Tensor out, torch::Tensor gate,
                              torch::Tensor grad_out, torch::Tensor dgate_dst,
                              int H, long dgate_stride) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto draw = torch::empty_like(out);
  long total = out.numel();
  sig_mul_bwd_out_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
      (const bf16*)out.data_ptr(), (const bf16*)gate.data_ptr(),
      (const bf16*)grad_out.data_ptr(), total, H, dgate_stride,
      (bf16*)draw.data_ptr(), (bf16*)dgate_dst.data_ptr());
  return draw;
}

static inline int cdiv(long a, int b) { return (int)((a + b - 1) / b); }

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

"""

_CPP_SRC = """
std::vector<torch::Tensor> trimul_in_fwd(torch::Tensor z, torch::Tensor w_ap, torch::Tensor w_ag, torch::Tensor w_bp, torch::Tensor w_bg, torch::Tensor w_g, torch::Tensor mask, torch::Tensor gamma, torch::Tensor beta, long B, long NN, int C, int H, bool save_zln);
std::vector<torch::Tensor> trimul_out_fwd(torch::Tensor x, torch::Tensor wz, torch::Tensor gate, torch::Tensor gamma, torch::Tensor beta, long B, long NN, int C, int H, bool training);
torch::Tensor trimul_gate_split_bwd(torch::Tensor zln, torch::Tensor w_ap, torch::Tensor w_ag, torch::Tensor w_bp, torch::Tensor w_bg, torch::Tensor mask, torch::Tensor d_a, torch::Tensor d_b, torch::Tensor d_proj5, long B, long NN, int C, int H);
torch::Tensor sig_mul_bwd_out(torch::Tensor out, torch::Tensor gate, torch::Tensor grad_out, torch::Tensor dgate_dst, int H, long dgate_stride);
torch::Tensor ln_bwd_dx(torch::Tensor input, torch::Tensor grad_out, long rows, long row_stride, long col_stride, torch::Tensor gamma, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> ln_bwd_dgb(torch::Tensor input, torch::Tensor grad_out, long rows, long row_stride, long col_stride, int C, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> transpose_ln_bwd(torch::Tensor x, torch::Tensor d_xln, int N, int H, int B, torch::Tensor gamma, torch::Tensor mean, torch::Tensor rstd);
std::vector<torch::Tensor> pack_rows3(torch::Tensor q, torch::Tensor k, torch::Tensor v, long bo, int H, int N, int D, bool src_h_inner);
torch::Tensor gather_rows1(torch::Tensor src, long bo, int H, int N, int D, bool src_h_inner);
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
                "trimul_in_fwd", "trimul_out_fwd", "trimul_gate_split_bwd",
                "sig_mul_bwd_out", "ln_bwd_dx", "ln_bwd_dgb",
                "transpose_ln_bwd", "pack_rows3", "gather_rows1",
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


class _StableTrimulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, w_ap, w_ag, w_bp, w_bg, w_g, w_z,
                g_in, b_in, g_out, b_out, outgoing):
        ext = _get_ext()
        orig_shape = z.shape
        C = z.shape[-1]
        N = z.shape[-2]
        if z.shape[-3] != N:
            raise ValueError(f"expected square pair input, got {tuple(z.shape)}")
        B = z.numel() // (N * N * C)
        H = w_ap.shape[0]
        if H != C:
            raise ValueError("stable trimul requires c_hidden == c_z")
        z_bf = z.reshape(B, N, N, C).to(torch.bfloat16).contiguous()
        mask_f = (mask.expand(*orig_shape[:-1]).reshape(B, N, N)
                  .to(torch.float32).contiguous())
        rows = B * N * N
        NN = N * N
        training = any(ctx.needs_input_grad)
        mask_flat = mask_f.view(-1)

        a, b, gate, mean_in, rstd_in, z_ln = ext.trimul_in_fwd(
            z_bf.view(rows, C), w_ap, w_ag, w_bp, w_bg, w_g,
            mask_flat, g_in, b_in, B, NN, C, H, training)
        av = a.view(B * H, N, N)
        bv = b.view(B * H, N, N)
        if outgoing:
            x = torch.bmm(av, bv.transpose(1, 2))
        else:
            x = torch.bmm(av.transpose(1, 2), bv)
        out, xlnT, mean_out, rstd_out = ext.trimul_out_fwd(
            x.view(B * H, NN), w_z, gate, g_out, b_out, B, NN, C, H, training)

        ctx.save_for_backward(z_bf, mask_f, z_ln, gate, a, b, x, xlnT,
                              mean_in, rstd_in, mean_out, rstd_out,
                              w_ap, w_ag, w_bp, w_bg, w_g, w_z,
                              g_in, g_out, out)
        ctx.dims = (B, N, C, H, outgoing)
        ctx.orig_shape = orig_shape
        ctx.z_dtype = z.dtype
        return out.view(orig_shape).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _get_ext()
        (z_bf, mask_f, z_ln, gate, a, b, x, xlnT, mean_in, rstd_in,
         mean_out, rstd_out, w_ap, w_ag, w_bp, w_bg, w_g, w_z,
         g_in, g_out, out) = ctx.saved_tensors
        B, N, C, H, outgoing = ctx.dims
        rows = B * N * N
        NN = N * N
        go = grad_out.to(torch.bfloat16).contiguous().view(rows, C)

        # gate slot [4H:5H) of d_proj5 + d_raw, reconstructing out_raw from out
        d_proj5 = torch.empty(rows, 5 * H, dtype=torch.bfloat16,
                              device=z_bf.device)
        d_raw = ext.sig_mul_bwd_out(
            out.view(rows, C), gate, go,
            d_proj5.view(rows, 5 * H)[:, 4 * H:], H, 5 * H)
        # out GEMM backward (w_z is [C, H]; no transposes materialized)
        d_xln = (d_raw @ w_z.to(torch.bfloat16)).contiguous()
        d_wz = (d_raw.t() @ xlnT.t()).float()                # [C, H]
        # LN out backward (fused transpose variant, dgb folded)
        d_x, d_gout, d_bout = ext.transpose_ln_bwd(
            x.view(B * H, NN), d_xln, N, H, B, g_out, mean_out, rstd_out)
        # bmm backward via cublas op views
        av = a.view(B * H, N, N)
        bv = b.view(B * H, N, N)
        if outgoing:
            d_a = torch.bmm(d_x, bv)
            d_b = torch.bmm(d_x.transpose(1, 2), av)
        else:
            d_a = torch.bmm(bv, d_x.transpose(1, 2))
            d_b = torch.bmm(av, d_x)
        # recompute-proj gate-split backward -> slots [0:4H)
        ext.trimul_gate_split_bwd(z_ln, w_ap, w_ag, w_bp, w_bg,
                                  mask_f.view(-1),
                                  d_a.view(B * H, NN), d_b.view(B * H, NN),
                                  d_proj5, B, NN, C, H)
        # in GEMM backward: W5 [5H, C] built once, only here
        w5 = torch.cat([w_ap, w_ag, w_bp, w_bg, w_g], 0).to(torch.bfloat16)
        d_zln = (d_proj5 @ w5).contiguous()
        d_w5 = (d_proj5.t() @ z_ln).float()                  # [5H, C]
        # LN in backward
        d_z = ext.ln_bwd_dx(z_bf.view(rows, C), d_zln, rows, C, 1,
                            g_in, mean_in, rstd_in)
        d_gin, d_bin = ext.ln_bwd_dgb(z_bf.view(rows, C), d_zln, rows, C, 1,
                                      C, mean_in, rstd_in)
        return (d_z.view(ctx.orig_shape).to(ctx.z_dtype), None,
                d_w5[0 * H:1 * H], d_w5[1 * H:2 * H], d_w5[2 * H:3 * H],
                d_w5[3 * H:4 * H], d_w5[4 * H:5 * H], d_wz,
                d_gin, d_bin, d_gout, d_bout, None)


def run_stable_trimul(module, z, mask):
    """Differentiable entry point bound to an official module's parameters."""
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    return _StableTrimulFn.apply(
        z, mask,
        module.linear_a_p.weight, module.linear_a_g.weight,
        module.linear_b_p.weight, module.linear_b_g.weight,
        module.linear_g.weight, module.linear_z.weight,
        module.layer_norm_in.weight.float(),
        module.layer_norm_in.bias.float(),
        module.layer_norm_out.weight.float(),
        module.layer_norm_out.bias.float(),
        module._outgoing,
    )


class StableTrimul:
    """Callable mirroring the official module's numerics with fused passes."""

    def __init__(self, module):
        self.module = module

    def __call__(self, z, mask):
        return run_stable_trimul(self.module, z, mask)
