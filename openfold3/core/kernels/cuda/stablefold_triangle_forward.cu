// SPDX-License-Identifier: Apache-2.0
// StableFold Triangle Attention forward with online softmax.
//
// FA2-style online softmax: each warp owns 16 query rows and streams KV in
// 64-key tiles; scores never touch HBM. Deterministic by construction in the
// forward (a single CTA owns each row; the KV tile order is fixed), replay
// bitwise vs itself. Accuracy contract is the error-ratio bound vs fp64, NOT
// bitwise vs PyTorch (see PRODUCTION_MANIFEST.md).
//
// Goals: correctness, replay stability, and arbitrary N. Perf work (cp.async double
// buffering, register-layout P reuse, WGMMA) comes after the gate is green.
//
// Internal layout contract used by the public StableFold facade:
//   q, k, v : [1, N, H=4, N, D=32] BF16, contiguous
//   mask    : [1, N, 1, 1, N] FP32 additive (0 or -inf-like)
//   bias    : [1, 1, H, N, N] BF16 additive
//   out     : [1, N, H, N, D] BF16
//   lse     : [1, N, H, N] FP32 (m + log l, natural log)

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
constexpr int kRowsPerWarp = 16;
constexpr int kBlockM = kWarpsPerBlock * kRowsPerWarp;  // 64 query rows / CTA
constexpr int kBlockN = 64;                             // keys per streamed tile

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l)) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

struct Acc {
  float v[4];
};

__device__ __forceinline__ void mma_m16n8k16(
    Acc& d, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(
    __nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
      "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all_committed() {
  asm volatile("cp.async.wait_group 0;\n");
}

// A-operand fragment of a 16xD row-major shared tile at (row 0, col dim_start).
__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile, int ld, int dim_start, int lane,
    uint32_t (&frag)[4]) {
  const int row = lane & 15;
  const int col = dim_start + ((lane >> 4) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// B-operand for QK^T: contraction over D, n = 8 keys. From the row-major
// [keys, D] tile this is the NON-transposed load whose pairs run along the
// contiguous dim axis (byte-identical twin proven in the fused-dP gate).
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile, int ld, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(tile + key * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

// B-operand for PV: contraction over 16 keys, n = 8 dims. From the row-major
// [keys, D] tile this is the 16-key-row TRANSPOSED load (byte-identical twin
// proven in the compact_cta phase-3 gate).
__device__ __forceinline__ void ldmatrix_b_keytrans(
    const __nv_bfloat16* tile, int ld, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 15);
  const uint32_t addr = shared_u32(tile + key * ld + dim_start);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

// Reduce a value across the 4 lanes of one accumulator row quad.
__device__ __forceinline__ float quad_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 1, kWarpSize));
  value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 2, kWarpSize));
  return value;
}

__device__ __forceinline__ float quad_sum(float value) {
  value += __shfl_xor_sync(0xFFFFFFFFU, value, 1, kWarpSize);
  value += __shfl_xor_sync(0xFFFFFFFFU, value, 2, kWarpSize);
  return value;
}

// ExactTiles is the production lane: under the facade's pad-to-tile
// contract every tile divides the sequence exactly, so all tail guards
// fold away at compile time with sequence still a runtime parameter.
template <int HeadDim, bool TileMajorGrid, bool ExactTiles>
__global__ __launch_bounds__(kThreads) void stablefold_triangle_forward_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    float scale) {
  constexpr int kKVStride = HeadDim + 8;
  static_assert(HeadDim % 16 == 0 && (kKVStride * 2) % 16 == 0);
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  // Keep query tiles on grid.x whenever the CUDA z dimension can hold all
  // (batch, outer) tasks.  This is the cache-friendly layout used by the
  // original single-prefix kernel, generalized to every batch/sequence that
  // fits the hardware grid.  The alternate layout is a functional fallback
  // for larger B*N; both execute the same tiles and reduction order.
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / outer_count;
  const int outer = outer_task - batch_index * outer_count;
  const int query_block =
      (TileMajorGrid ? blockIdx.x : blockIdx.z) * kBlockM;
  const int head = blockIdx.y;
  // Tail-CTA warps whose rows all fall beyond the sequence stay resident:
  // they must keep participating in the cooperative K/V staging loop and its
  // __syncthreads(). Their scores are forced to -inf and their stores are
  // row-guarded, so their state is inert.
  const int warp_query = query_block + warp * kRowsPerWarp;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  // Per-warp Q tile [16 x D], then shared K tile, V tile, per-warp P tile.
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kRowsPerWarp * kKVStride;
  __nv_bfloat16* kv_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kBlockM * kKVStride;

  const int64_t head_base =
      ((static_cast<int64_t>(batch_index) * outer_count + outer) * heads +
       head) *
      static_cast<int64_t>(sequence) * HeadDim;
  const int64_t bias_base =
      (static_cast<int64_t>(batch_index) * heads + head) *
      static_cast<int64_t>(sequence) * sequence;
  const int64_t mask_base =
      (static_cast<int64_t>(batch_index) * outer_count + outer) * sequence;

  // Stage this warp's Q rows (guarded rows load row 0; their stores are
  // guarded at the epilogue, and their softmax state is discarded).
  for (int element = lane; element < kRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gq = ExactTiles || warp_query + row < sequence
        ? warp_query + row
        : 0;
    sq[row * kKVStride + col] =
        q[head_base + static_cast<int64_t>(gq) * HeadDim + col];
  }
  __syncwarp();

  uint32_t q_frags[HeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < HeadDim / 16; ++chain) {
    ldmatrix_a16x16(sq, kKVStride, chain * 16, lane, q_frags[chain]);
  }

  // Online softmax state: two accumulator rows per lane (r and r+8).
  float m_state[2] = {-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()};
  float l_state[2] = {0.0F, 0.0F};
  Acc o_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    o_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const int key_tiles = (sequence + kBlockN - 1) / kBlockN;
  // Two-stage cp.async pipeline over the K/V tiles: tile t+1 copies while
  // tile t computes. Guarded tail rows stage key 0 and are neutralized by
  // the -inf score mask below. 16-byte chunks; HeadDim*2B = 4 chunks/row.
  constexpr int kChunksPerRow =
      HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
  constexpr int kChunkElements = 16 / static_cast<int>(sizeof(__nv_bfloat16));
  const auto issue_tile = [&](int key_tile, int buffer) {
    __nv_bfloat16* bk = kv_buffers + buffer * 2 * kBlockN * kKVStride;
    __nv_bfloat16* bv = bk + kBlockN * kKVStride;
    const int key_start = key_tile * kBlockN;
    for (int chunk = threadIdx.x; chunk < kBlockN * kChunksPerRow;
         chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * kChunkElements;
      const int gk = ExactTiles || key_start + row < sequence
          ? key_start + row
          : 0;
      const int64_t src = head_base + static_cast<int64_t>(gk) * HeadDim + col;
      cp_async_16(bk + row * kKVStride + col, k + src);
      cp_async_16(bv + row * kKVStride + col, v + src);
    }
    cp_async_commit();
  };
  issue_tile(0, 0);
#pragma unroll 1
  for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const int key_start = key_tile * kBlockN;
    const int buffer = key_tile & 1;
    __nv_bfloat16* sk = kv_buffers + buffer * 2 * kBlockN * kKVStride;
    __nv_bfloat16* sv = sk + kBlockN * kKVStride;
    cp_async_wait_all_committed();
    __syncthreads();
    if (key_tile + 1 < key_tiles) {
      issue_tile(key_tile + 1, buffer ^ 1);
    }

    // Scores for this warp's 16 rows x 64 keys: 8 key-columns of 16x8 HMMA.
    Acc score[kBlockN / 8];
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        uint32_t k_frag[2];
        ldmatrix_b_dimpair(
            sk, kKVStride, column * 8, chain * 16, lane, k_frag);
        mma_m16n8k16(score[column], q_frags[chain], k_frag);
      }
    }

    // scale + mask + bias in FP32, tail keys forced to -inf.
    float tile_max[2] = {-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp_query + group + half * 8;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int key = key_start + column * 8 + thread_in_group * 2 + pair;
          float value = score[column].v[half * 2 + pair];
          if (ExactTiles || (key < sequence && row < sequence)) {
            value = value * scale + mask[mask_base + key] +
                __bfloat162float(
                    bias[bias_base +
                         static_cast<int64_t>(row) * sequence + key]);
          } else {
            value = -std::numeric_limits<float>::infinity();
          }
          score[column].v[half * 2 + pair] = value;
          tile_max[half] = fmaxf(tile_max[half], value);
        }
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      tile_max[half] = quad_max(tile_max[half]);
    }

    // Online rescale.
    float rescale[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float m_new = fmaxf(m_state[half], tile_max[half]);
      rescale[half] =
          m_state[half] == -std::numeric_limits<float>::infinity()
          ? 0.0F
          : __expf(m_state[half] - m_new);
      m_state[half] = m_new;
      l_state[half] *= rescale[half];
    }
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        o_acc[tile].v[half * 2] *= rescale[half];
        o_acc[tile].v[half * 2 + 1] *= rescale[half];
      }
    }

    // P = exp(S - m), accumulate l, stage P to shared for the PV HMMA.
    float row_sum[2] = {0.0F, 0.0F};
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float p0 = score[column].v[half * 2];
        float p1 = score[column].v[half * 2 + 1];
        p0 = p0 == -std::numeric_limits<float>::infinity()
            ? 0.0F
            : __expf(p0 - m_state[half]);
        p1 = p1 == -std::numeric_limits<float>::infinity()
            ? 0.0F
            : __expf(p1 - m_state[half]);
        row_sum[half] += p0 + p1;
        score[column].v[half * 2] = p0;
        score[column].v[half * 2 + 1] = p1;
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      l_state[half] += quad_sum(row_sum[half]);
    }

    // O += P @ V: pack adjacent probability accumulator columns straight
    // into A fragments (identical bytes to the former shared round-trip),
    // ascending 16-key chunks -> fixed order, deterministic.
#pragma unroll
    for (int chunk = 0; chunk < kBlockN / 16; ++chunk) {
      uint32_t p_frag[4];
      p_frag[0] = pack_bf16_pair(
          score[chunk * 2].v[0], score[chunk * 2].v[1]);
      p_frag[1] = pack_bf16_pair(
          score[chunk * 2].v[2], score[chunk * 2].v[3]);
      p_frag[2] = pack_bf16_pair(
          score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
      p_frag[3] = pack_bf16_pair(
          score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        uint32_t v_frag[2];
        ldmatrix_b_keytrans(
            sv, kKVStride, chunk * 16, tile * 8, lane, v_frag);
        mma_m16n8k16(o_acc[tile], p_frag, v_frag);
      }
    }
  }

  // Epilogue: O / l, LSE = m + log l.
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = warp_query + group + half * 8;
    if (!ExactTiles && row >= sequence) {
      continue;
    }
    const float inv_l = l_state[half] == 0.0F ? 0.0F : 1.0F / l_state[half];
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int dim = tile * 8 + thread_in_group * 2 + pair;
        out[head_base + static_cast<int64_t>(row) * HeadDim + dim] =
            __float2bfloat16_rn(o_acc[tile].v[half * 2 + pair] * inv_l);
      }
    }
    if (thread_in_group == 0) {
      lse[((static_cast<int64_t>(batch_index) * outer_count + outer) * heads +
           head) * sequence + row] =
          m_state[half] + __logf(l_state[half]);
    }
  }
}

template <int HeadDim>
constexpr int shared_bytes() {
  return (kBlockM * (HeadDim + 8) + 4 * kBlockN * (HeadDim + 8)) *
      static_cast<int>(sizeof(__nv_bfloat16));
}

template <int HeadDim>
void launch_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    cudaStream_t stream) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
  const int query_blocks = (sequence + kBlockM - 1) / kBlockM;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * outer_count;
  const bool tile_major_grid = outer_tasks <= 65535;
  const dim3 grid = tile_major_grid
      ? dim3(
            static_cast<unsigned int>(query_blocks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(outer_tasks))
      : dim3(
            static_cast<unsigned int>(outer_tasks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(query_blocks));
  const bool exact_tiles =
      sequence % kBlockM == 0 && sequence % kBlockN == 0;
#define LAUNCH_STABLEFOLD_FORWARD(TileMajor, Exact)                         \
  do {                                                                      \
    const cudaError_t status = cudaFuncSetAttribute(                        \
        stablefold_triangle_forward_kernel<HeadDim, TileMajor, Exact>,      \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                        \
        shared_bytes<HeadDim>());                                           \
    TORCH_CHECK(                                                            \
        status == cudaSuccess,                                              \
        "failed to opt in StableFold forward shared memory: ",              \
        cudaGetErrorString(status));                                        \
    stablefold_triangle_forward_kernel<HeadDim, TileMajor, Exact>           \
        <<<grid, kThreads, shared_bytes<HeadDim>(), stream>>>(              \
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),          \
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),          \
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),          \
            mask.data_ptr<float>(),                                        \
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),       \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),              \
            lse.data_ptr<float>(),                                         \
            sequence,                                         \
            outer_count,                                                      \
            heads,                                                         \
            batch,                                                         \
            scale);                                                        \
  } while (false)
  if (tile_major_grid) {
    if (exact_tiles) {
      LAUNCH_STABLEFOLD_FORWARD(true, true);
    } else {
      LAUNCH_STABLEFOLD_FORWARD(true, false);
    }
  } else {
    if (exact_tiles) {
      LAUNCH_STABLEFOLD_FORWARD(false, true);
    } else {
      LAUNCH_STABLEFOLD_FORWARD(false, false);
    }
  }
#undef LAUNCH_STABLEFOLD_FORWARD
}

void stablefold_triangle_forward_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
  const int64_t batch64 = q.size(0);
  const int64_t outer64 = q.size(1);
  const int64_t heads64 = q.size(2);
  const int64_t sequence64 = q.size(3);
  const int64_t head_dim64 = q.size(4);
  TORCH_CHECK(
      batch64 > 0 && batch64 <= std::numeric_limits<int>::max() &&
          outer64 > 0 && outer64 <= std::numeric_limits<int>::max() &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          heads64 > 0 && heads64 <= 65535 && head_dim64 > 0 &&
          head_dim64 <= std::numeric_limits<int>::max() &&
          batch64 * outer64 <= 2147483647LL &&
          (sequence64 + kBlockM - 1) / kBlockM <= 65535,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/out must be contiguous BF16 tensors matching q on one device");
  };
  check_q_like(k);
  check_q_like(v);
  check_q_like(out);
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() &&
          mask.dtype() == torch::kFloat32 && mask.is_contiguous() &&
          mask.sizes() ==
              torch::IntArrayRef({batch, outer_count, 1, 1, sequence}),
      "mask must be contiguous FP32 [B,O,1,1,N] on q.device");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
      "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
  TORCH_CHECK(
      lse.is_cuda() && lse.device() == q.device() &&
          lse.dtype() == torch::kFloat32 && lse.is_contiguous() &&
          lse.sizes() ==
              torch::IntArrayRef({batch, outer_count, heads, sequence}),
      "lse must be contiguous FP32 [B,O,H,N] on q.device");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (head_dim) {
    case 16:
      launch_forward<16>(
          q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
          stream);
      break;
    case 32:
      launch_forward<32>(
          q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
          stream);
      break;
    case 64:
      launch_forward<64>(
          q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
          stream);
      break;
    case 128:
      launch_forward<128>(
          q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
          stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "stablefold_triangle_forward_out",
      &stablefold_triangle_forward_out,
      "StableFold Triangle Attention forward (online softmax)");
}
