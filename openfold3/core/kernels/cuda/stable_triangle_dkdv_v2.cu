// SPDX-License-Identifier: Apache-2.0
// Streaming dK/dV v2 — the v9-family rewrite of the resident dkdv kernel.
//
// One CTA owns one (64-key block, head, outer) and its dK/dV accumulators
// outright: dK[k]/dV[k] depend only on column k of dS, so key-block
// ownership is contention-free by construction (zero atomics, no
// workspace). The kernel streams query tiles in fixed ascending order,
// recomputes S^T = K@Q^T directly in the [key, query] accumulator layout
// (P^T and dS^T then follow elementwise from LSE/delta), and accumulates
//   dV += P^T  @ dO   and   dK += dS^T @ Q, scaled once at the epilogue.
//
// Generality by construction: HeadDim templated {16,32,64}; heads and
// sequence runtime (tail rows clamped for loads, contributions zeroed).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kKeyRowsPerWarp = 16;
constexpr int kKeyBlock = kWarpsPerBlock * kKeyRowsPerWarp;  // 64 keys
constexpr int kQueryTile = 32;

struct Acc {
  float v[4];
};

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_m16n8k16(
    Acc& d, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile, int ld, int column_start, int lane,
    uint32_t (&frag)[4]) {
  const int row = lane & 15;
  const int column = column_start + ((lane >> 4) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + column);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// B for K@Q^T (contraction over dims) and for the epilogue products'
// contraction over queries via the transposed 16-row load.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile, int ld, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b_rowtrans(
    const __nv_bfloat16* tile, int ld, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 15);
  const uint32_t addr = shared_u32(tile + row * ld + dim_start);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

__device__ __forceinline__ void cp_async_16(
    __nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
      "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l)) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

template <int HeadDim, int StaticSequence, bool TileMajorGrid>
__global__ __launch_bounds__(kThreads) void stable_triangle_dkdv_v2_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ delta,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out_dk,
    __nv_bfloat16* __restrict__ out_dv,
    int runtime_sequence,
    int heads,
    int batch,
    float scale) {
  const int sequence =
      StaticSequence == 0 ? runtime_sequence : StaticSequence;
  constexpr bool kExact = StaticSequence != 0;
  constexpr int kStride = HeadDim + 8;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  // Prefer key tiles on grid.x whenever z can represent every (batch, outer)
  // task.  This preserves the cache-friendly legacy ordering for all normal
  // shapes while retaining an outer-major fallback for larger B*N.  The two
  // layouts run identical tiles and reductions.
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / sequence;
  const int outer = outer_task - batch_index * sequence;
  const int key_block_start =
      (TileMajorGrid ? blockIdx.x : blockIdx.z) * kKeyBlock;
  const int head = blockIdx.y;
  const int warp_key = key_block_start + warp * kKeyRowsPerWarp;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  // Per-warp K and V slices [16 x D]; shared Q and dO tiles [32 x D].
  __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kKeyRowsPerWarp * kStride;
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kKeyBlock * kStride + warp * kKeyRowsPerWarp * kStride;
  __nv_bfloat16* q_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      2 * kKeyBlock * kStride;
  constexpr int kQBufferElements = 2 * kQueryTile * kStride;  // q + do

  const int64_t head_base =
      ((static_cast<int64_t>(batch_index) * sequence + outer) * heads + head) *
      static_cast<int64_t>(sequence) * HeadDim;
  const int64_t stats_base =
      ((static_cast<int64_t>(batch_index) * sequence + outer) * heads + head) *
      sequence;
  const int64_t mask_base =
      (static_cast<int64_t>(batch_index) * sequence + outer) * sequence;
  const int64_t bias_base =
      (static_cast<int64_t>(batch_index) * heads + head) *
      static_cast<int64_t>(sequence) * sequence;

  // Stage this warp's 16 K rows (clamped tails; contributions zeroed later).
  for (int element = lane; element < kKeyRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gk = kExact || warp_key + row < sequence ? warp_key + row
                                                        : sequence - 1;
    const int64_t src = head_base + static_cast<int64_t>(gk) * HeadDim + col;
    sk[row * kStride + col] = k[src];
    sv[row * kStride + col] = v[src];
  }
  __syncwarp();

  uint32_t k_frags[HeadDim / 16][4];
  uint32_t v_frags[HeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < HeadDim / 16; ++chain) {
    ldmatrix_a16x16(sk, kStride, chain * 16, lane, k_frags[chain]);
    ldmatrix_a16x16(sv, kStride, chain * 16, lane, v_frags[chain]);
  }

  Acc dv_acc[HeadDim / 8];
  Acc dk_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    dv_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
    dk_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const int key_row_low = warp_key + group;
  const int key_row_high = key_row_low + 8;

  // Two-stage cp.async pipeline over the query tiles.
  constexpr int kChunksPerRow =
      HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
  const auto issue_query_tile = [&](int query_start, int buffer) {
    __nv_bfloat16* bq = q_buffers + buffer * kQBufferElements;
    __nv_bfloat16* bdo = bq + kQueryTile * kStride;
    for (int chunk = tid; chunk < kQueryTile * kChunksPerRow;
         chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * 8;
      const int gq = kExact || query_start + row < sequence
          ? query_start + row
          : sequence - 1;
      const int64_t src = head_base + static_cast<int64_t>(gq) * HeadDim +
          col;
      cp_async_16(bq + row * kStride + col, q + src);
      cp_async_16(bdo + row * kStride + col, d_output + src);
    }
    cp_async_commit_group();
  };
  issue_query_tile(0, 0);
#pragma unroll 1
  for (int query_start = 0; query_start < sequence;
       query_start += kQueryTile) {
    const int buffer = (query_start / kQueryTile) & 1;
    __nv_bfloat16* sq = q_buffers + buffer * kQBufferElements;
    __nv_bfloat16* sdo = sq + kQueryTile * kStride;
    cp_async_wait_all();
    __syncthreads();
    if (query_start + kQueryTile < sequence) {
      issue_query_tile(query_start + kQueryTile, buffer ^ 1);
    }

    // S^T tile [16 keys x 32 queries]: A = K rows, B = Q dim-pairs.
    Acc score[kQueryTile / 8];
    Acc dp[kQueryTile / 8];
#pragma unroll
    for (int column = 0; column < kQueryTile / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        uint32_t b_frag[2];
        ldmatrix_b_dimpair(
            sq, kStride, column * 8, chain * 16, lane, b_frag);
        mma_m16n8k16(score[column], k_frags[chain], b_frag);
        ldmatrix_b_dimpair(
            sdo, kStride, column * 8, chain * 16, lane, b_frag);
        mma_m16n8k16(dp[column], v_frags[chain], b_frag);
      }
    }

    // Elementwise: P^T = exp(S^T*scale + mask + bias - LSE[q]);
    // dS^T = P^T * (dP^T - delta[q]). Column index = query.
#pragma unroll
    for (int column = 0; column < kQueryTile / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key_row = half == 0 ? key_row_low : key_row_high;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query = query_start + column * 8 + thread_in_group * 2 +
              pair;
          const int index = half * 2 + pair;
          float d_score = 0.0F;
          float probability = 0.0F;
          if (kExact || (key_row < sequence && query < sequence)) {
            const int64_t stats = stats_base + query;
            const float value = score[column].v[index] * scale +
                mask[mask_base + key_row] +
                __bfloat162float(
                    bias[bias_base +
                         static_cast<int64_t>(query) * sequence + key_row]);
            const float lse = logsumexp[stats];
            probability = lse == -std::numeric_limits<float>::infinity()
                ? 0.0F
                : __expf(value - lse);
            d_score = probability * (dp[column].v[index] - delta[stats]);
          }
          score[column].v[index] = d_score;
          dp[column].v[index] = probability;
        }
      }
    }

    // dV += P^T @ dO; dK += dS^T @ Q — contraction over the 32 queries.
#pragma unroll
    for (int chunk = 0; chunk < kQueryTile / 16; ++chunk) {
      uint32_t p_frag[4];
      uint32_t ds_frag[4];
      p_frag[0] = pack_bf16_pair(
          dp[chunk * 2].v[0], dp[chunk * 2].v[1]);
      p_frag[1] = pack_bf16_pair(
          dp[chunk * 2].v[2], dp[chunk * 2].v[3]);
      p_frag[2] = pack_bf16_pair(
          dp[chunk * 2 + 1].v[0], dp[chunk * 2 + 1].v[1]);
      p_frag[3] = pack_bf16_pair(
          dp[chunk * 2 + 1].v[2], dp[chunk * 2 + 1].v[3]);
      ds_frag[0] = pack_bf16_pair(
          score[chunk * 2].v[0], score[chunk * 2].v[1]);
      ds_frag[1] = pack_bf16_pair(
          score[chunk * 2].v[2], score[chunk * 2].v[3]);
      ds_frag[2] = pack_bf16_pair(
          score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
      ds_frag[3] = pack_bf16_pair(
          score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        uint32_t b_frag[2];
        ldmatrix_b_rowtrans(
            sdo, kStride, chunk * 16, tile * 8, lane, b_frag);
        mma_m16n8k16(dv_acc[tile], p_frag, b_frag);
        ldmatrix_b_rowtrans(
            sq, kStride, chunk * 16, tile * 8, lane, b_frag);
        mma_m16n8k16(dk_acc[tile], ds_frag, b_frag);
      }
    }
  }

  // Epilogue: dK scaled once; both stored BF16, tail keys skipped.
  const int64_t out_base = head_base +
      static_cast<int64_t>(warp_key) * HeadDim;
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = group + half * 8;
      if (!kExact && warp_key + row >= sequence) {
        continue;
      }
      const int column = tile * 8 + thread_in_group * 2;
      out_dv[out_base + static_cast<int64_t>(row) * HeadDim + column] =
          __float2bfloat16_rn(dv_acc[tile].v[half * 2]);
      out_dv[out_base + static_cast<int64_t>(row) * HeadDim + column + 1] =
          __float2bfloat16_rn(dv_acc[tile].v[half * 2 + 1]);
      out_dk[out_base + static_cast<int64_t>(row) * HeadDim + column] =
          __float2bfloat16_rn(dk_acc[tile].v[half * 2] * scale);
      out_dk[out_base + static_cast<int64_t>(row) * HeadDim + column + 1] =
          __float2bfloat16_rn(dk_acc[tile].v[half * 2 + 1] * scale);
    }
  }
}

template <int HeadDim, int StaticSequence>
void launch(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv,
    int sequence,
    int heads,
    int batch,
    cudaStream_t stream) {
  constexpr int kStride = HeadDim + 8;
  constexpr int kSharedBytes =
      (2 * kKeyBlock * kStride + 4 * kQueryTile * kStride) *
      static_cast<int>(sizeof(__nv_bfloat16));
  const float scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
  const int key_blocks = (sequence + kKeyBlock - 1) / kKeyBlock;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * sequence;
  const bool tile_major_grid = outer_tasks <= 65535;
  const dim3 grid = tile_major_grid
      ? dim3(
            static_cast<unsigned int>(key_blocks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(outer_tasks))
      : dim3(
            static_cast<unsigned int>(outer_tasks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(key_blocks));
  const cudaError_t attribute_status = tile_major_grid
      ? cudaFuncSetAttribute(
            stable_triangle_dkdv_v2_kernel<
                HeadDim, StaticSequence, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSharedBytes)
      : cudaFuncSetAttribute(
            stable_triangle_dkdv_v2_kernel<
                HeadDim, StaticSequence, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSharedBytes);
  TORCH_CHECK(
      attribute_status == cudaSuccess,
      "failed to opt in streaming dK/dV shared memory: ",
      cudaGetErrorString(attribute_status));
  if (tile_major_grid) {
    stable_triangle_dkdv_v2_kernel<HeadDim, StaticSequence, true>
        <<<grid, kThreads, kSharedBytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
            logsumexp.data_ptr<float>(),
            delta.data_ptr<float>(),
            mask.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
            sequence,
            heads,
            batch,
            scale);
  } else {
    stable_triangle_dkdv_v2_kernel<HeadDim, StaticSequence, false>
        <<<grid, kThreads, kSharedBytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
            logsumexp.data_ptr<float>(),
            delta.data_ptr<float>(),
            mask.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
            sequence,
            heads,
            batch,
            scale);
  }
}

template <int HeadDim>
void dispatch_sequence(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv,
    int sequence,
    int heads,
    int batch,
    cudaStream_t stream) {
  // These lanes only remove bounds/address arithmetic for the three official
  // training crops. They use the exact same tiles and reduction contract as
  // the runtime-N fallback, which remains the functional production path for
  // every other device-feasible sequence length.
  switch (sequence) {
    case 384:
      launch<HeadDim, 384>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dk, out_dv, sequence, heads, batch, stream);
      break;
    case 640:
      launch<HeadDim, 640>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dk, out_dv, sequence, heads, batch, stream);
      break;
    case 768:
      launch<HeadDim, 768>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dk, out_dv, sequence, heads, batch, stream);
      break;
    default:
      launch<HeadDim, 0>(q, k, v, d_output, logsumexp, delta, mask, bias,
                         out_dk, out_dv, sequence, heads, batch, stream);
      break;
  }
}

void stable_triangle_dkdv_v2_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
  const int64_t batch64 = q.size(0);
  const int64_t sequence64 = q.size(1);
  const int64_t heads64 = q.size(2);
  const int64_t head_dim64 = q.size(4);
  TORCH_CHECK(
      batch64 > 0 && batch64 <= std::numeric_limits<int>::max() &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          heads64 > 0 && heads64 <= 65535 && head_dim64 > 0 &&
          head_dim64 <= std::numeric_limits<int>::max() &&
          batch64 * sequence64 <= 2147483647LL &&
          (sequence64 + kKeyBlock - 1) / kKeyBlock <= 65535,
      "batch/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  TORCH_CHECK(q.size(3) == sequence && q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/d_output/out_dk/out_dv must be contiguous BF16 tensors "
        "matching q on one device");
  };
  check_q_like(k);
  check_q_like(v);
  check_q_like(d_output);
  check_q_like(out_dk);
  check_q_like(out_dv);
  const auto check_row_tensor = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kFloat32 && tensor.is_contiguous() &&
            tensor.numel() == q.numel() / head_dim,
        "logsumexp/delta must be contiguous FP32 row tensors on q.device");
  };
  check_row_tensor(logsumexp);
  check_row_tensor(delta);
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() &&
          mask.dtype() == torch::kFloat32 && mask.is_contiguous() &&
          mask.sizes() ==
              torch::IntArrayRef({batch, sequence, 1, 1, sequence}),
      "mask must be contiguous FP32 [B,N,1,1,N] on q.device");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
      "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (head_dim) {
    case 16:
      dispatch_sequence<16>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dk,
                 out_dv, sequence, heads, batch, stream);
      break;
    case 32:
      dispatch_sequence<32>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dk,
                 out_dv, sequence, heads, batch, stream);
      break;
    case 64:
      dispatch_sequence<64>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dk,
                 out_dv, sequence, heads, batch, stream);
      break;
    case 128:
      dispatch_sequence<128>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dk,
                 out_dv, sequence, heads, batch, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported head_dim for dkdv v2: ", head_dim);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dkdv v2 launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "stable_triangle_dkdv_v2_out",
      &stable_triangle_dkdv_v2_out,
      "Streaming dK/dV v2 (key-block ownership, H/D/N general)");
}
