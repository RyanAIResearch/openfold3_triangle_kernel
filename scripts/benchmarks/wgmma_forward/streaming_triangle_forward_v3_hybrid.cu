// Streaming Triangle Attention forward v3 — TMA + WGMMA / mma.sync hybrid.
//
// QK^T rides the verified round-3 recipe: K tiles arrive by TMA bulk copy
// (single thread issues, mbarrier signals, zero staging work for the other
// 127 threads) into swizzled shared, consumed by warpgroup wgmma with the
// probe-verified descriptors (LBO=0, SBO=0, per-D swizzle mode; sub-tile
// offset key_group*8*D*2 + dim_half*32 bytes). PV keeps the v1-proven path:
// V staged row-major by cp.async, read through ldmatrix (hardware layout
// conversion at read time), multiplied by mma.sync from register-packed P.
//
// Shape generality by construction: HeadDim is a template (swizzle mode
// derives from it), heads and sequence are runtime values; TMA's tensor map
// handles arbitrary-N tails natively (OOB boxes read as garbage keys which
// the -inf score guard neutralizes, same as v1).
//
// Contract: replay-bitwise vs self; expected bitwise == v1 (wgmma proved
// bit-identical to the chained mma.sync in the round-2 gates).

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;  // one warpgroup
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kRowsPerWarp = 16;
constexpr int kBlockM = kWarpsPerBlock * kRowsPerWarp;
constexpr int kBlockN = 64;
constexpr int kWgmmaN = 8;
constexpr int kWgmmaK = 16;

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

__device__ __forceinline__ void wgmma_m64n8k16(
    Acc& acc, const uint32_t (&a)[4], uint64_t descriptor_b,
    int accumulate) {
  asm volatile(
      "{\n"
      "  .reg .pred use_acc;\n"
      "  setp.ne.b32 use_acc, %9, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "  {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, use_acc, 1, 1, 0;\n"
      "}\n"
      : "+f"(acc.v[0]), "+f"(acc.v[1]), "+f"(acc.v[2]), "+f"(acc.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(descriptor_b),
        "r"(accumulate));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_zero() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(shared_u32(bar)),
      "r"(count));
}

__device__ __forceinline__ void mbarrier_expect_tx(
    uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          shared_u32(bar)),
      "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "  wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "  @!done bra wait;\n"
      "}\n" ::"r"(shared_u32(bar)),
      "r"(parity));
}

__device__ __forceinline__ void tma_load_2d_slice(
    const CUtensorMap* map, __nv_bfloat16* dst, uint64_t* bar,
    int coord_key, int coord_plane) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(0),
      "r"(coord_key),
      "r"(coord_plane),
      "r"(shared_u32(bar)));
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

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l)) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

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

template <int HeadDim>
__global__ __launch_bounds__(kThreads) void
streaming_triangle_forward_v3_kernel(
    const __grid_constant__ CUtensorMap k_map,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int sequence,
    int heads,
    float scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int kQStride = HeadDim + 8;
  constexpr int kVStride = HeadDim + 8;
  constexpr int kDimHalves = HeadDim / kWgmmaK;
  constexpr int kKBufferElements = kBlockN * HeadDim;  // TMA box, swizzled
  constexpr int kVBufferElements = kBlockN * kVStride;

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int query_block = blockIdx.x * kBlockM;
  const int head = blockIdx.y;
  const int outer = blockIdx.z;
  const int warp_query = query_block + warp * kRowsPerWarp;

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kRowsPerWarp * kQStride;
  __nv_bfloat16* k_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kBlockM * kQStride;
  __nv_bfloat16* v_buffers = k_buffers + 2 * kKBufferElements;
  __shared__ __align__(8) uint64_t k_barriers[2];

  const int64_t head_base =
      (static_cast<int64_t>(outer) * heads + head) *
      static_cast<int64_t>(sequence) * HeadDim;
  const int64_t bias_base = static_cast<int64_t>(head) *
      static_cast<int64_t>(sequence) * sequence;
  const int64_t mask_base = static_cast<int64_t>(outer) * sequence;
  const int plane = outer * heads + head;

  if (tid == 0) {
    mbarrier_init(&k_barriers[0], 1);
    mbarrier_init(&k_barriers[1], 1);
  }

  for (int element = lane; element < kRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gq = warp_query + row < sequence ? warp_query + row : 0;
    sq[row * kQStride + col] =
        q[head_base + static_cast<int64_t>(gq) * HeadDim + col];
  }
  __syncthreads();  // covers mbarrier_init visibility too

  uint32_t q_frags[kDimHalves][4];
#pragma unroll
  for (int chain = 0; chain < kDimHalves; ++chain) {
    ldmatrix_a16x16(sq, kQStride, chain * kWgmmaK, lane, q_frags[chain]);
  }

  float m_state[2] = {-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()};
  float l_state[2] = {0.0F, 0.0F};
  Acc o_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    o_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const auto issue_k = [&](int key_tile, int buffer) {
    if (tid == 0) {
      mbarrier_expect_tx(
          &k_barriers[buffer],
          kKBufferElements * static_cast<int>(sizeof(__nv_bfloat16)));
      tma_load_2d_slice(
          &k_map, k_buffers + buffer * kKBufferElements,
          &k_barriers[buffer], key_tile * kBlockN, plane);
    }
  };
  const auto issue_v = [&](int key_tile, int buffer) {
    __nv_bfloat16* bv = v_buffers + buffer * kVBufferElements;
    constexpr int kChunksPerRow =
        HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    for (int chunk = tid; chunk < kBlockN * kChunksPerRow;
         chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * 8;
      const int gk = key_tile * kBlockN + row;
      const int source_key = gk < sequence ? gk : 0;
      cp_async_16(
          bv + row * kVStride + col,
          v + head_base + static_cast<int64_t>(source_key) * HeadDim + col);
    }
    cp_async_commit_group();
  };

  const int key_tiles = (sequence + kBlockN - 1) / kBlockN;
  issue_k(0, 0);
  issue_v(0, 0);
#pragma unroll 1
  for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const int key_start = key_tile * kBlockN;
    const int buffer = key_tile & 1;
    const uint32_t parity = (key_tile >> 1) & 1;
    __nv_bfloat16* sb_k = k_buffers + buffer * kKBufferElements;
    __nv_bfloat16* sv = v_buffers + buffer * kVBufferElements;

    mbarrier_wait_parity(&k_barriers[buffer], parity);
    cp_async_wait_all();
    __syncthreads();
    proxy_fence();

    // QK^T via warpgroup MMA over the TMA-swizzled K buffer. Descriptor
    // recipe from the probe: mode by HeadDim, sub-tile offset
    // key_group * 8 rows * rowbytes + dim_half * 32 bytes.
    constexpr uint64_t kMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
    Acc score[kBlockN / 8];
    wgmma_fence();
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      const uint64_t tile_base =
          (static_cast<uint64_t>(shared_u32(sb_k)) +
           static_cast<uint64_t>(column) * 8 * HeadDim *
               sizeof(__nv_bfloat16));
#pragma unroll
      for (int chain = 0; chain < kDimHalves; ++chain) {
        const uint64_t descriptor =
            ((tile_base + chain * 32) >> 4) | (kMode << 62);
        wgmma_m64n8k16(
            score[column], q_frags[chain], descriptor, chain != 0);
      }
    }
    wgmma_commit();
    // Overlap window: stage the NEXT tile's K (TMA) and V (cp.async) while
    // the QK warpgroup MMAs run.
    if (key_tile + 1 < key_tiles) {
      issue_k(key_tile + 1, buffer ^ 1);
      issue_v(key_tile + 1, buffer ^ 1);
    }
    wgmma_wait_zero();

    float tile_max[2] = {-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp_query + group + half * 8;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int key = key_start + column * 8 + thread_in_group * 2 +
              pair;
          float value = score[column].v[half * 2 + pair];
          if (key < sequence && row < sequence) {
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

    // PV via the v1-proven path: register-packed P, ldmatrix keytrans V.
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
            sv, kVStride, chunk * 16, tile * 8, lane, v_frag);
        mma_m16n8k16(o_acc[tile], p_frag, v_frag);
      }
    }
    __syncthreads();  // all reads of this buffer done before its reuse
  }

  const int64_t lse_head_base =
      (static_cast<int64_t>(outer) * heads + head) * sequence;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = warp_query + group + half * 8;
    if (row >= sequence) {
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
      lse[lse_head_base + row] = m_state[half] + __logf(l_state[half]);
    }
  }
#endif
}

CUtensorMap make_k_map(const torch::Tensor& k, int head_dim) {
  CUtensorMap tensor_map;
  const int sequence = static_cast<int>(k.size(1));
  const int heads = static_cast<int>(k.size(2));
  const cuuint64_t global_dim[3] = {
      static_cast<cuuint64_t>(head_dim),
      static_cast<cuuint64_t>(sequence),
      static_cast<cuuint64_t>(
          static_cast<int64_t>(sequence) * heads)};
  const cuuint64_t global_stride[2] = {
      static_cast<cuuint64_t>(head_dim * sizeof(__nv_bfloat16)),
      static_cast<cuuint64_t>(
          static_cast<int64_t>(head_dim) * sequence *
          sizeof(__nv_bfloat16))};
  const cuuint32_t box_dim[3] = {
      static_cast<cuuint32_t>(head_dim), kBlockN, 1};
  const cuuint32_t element_stride[3] = {1, 1, 1};
  const CUtensorMapSwizzle swizzle =
      head_dim == 16 ? CU_TENSOR_MAP_SWIZZLE_32B
                     : (head_dim == 32 ? CU_TENSOR_MAP_SWIZZLE_64B
                                       : CU_TENSOR_MAP_SWIZZLE_128B);
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      k.data_ptr(),
      global_dim,
      global_stride,
      box_dim,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return tensor_map;
}

template <int HeadDim>
void launch_v3(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    int sequence,
    int heads,
    cudaStream_t stream) {
  const CUtensorMap k_map = make_k_map(k, HeadDim);
  const float scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
  const int shared_bytes =
      (kBlockM * (HeadDim + 8) + 2 * kBlockN * HeadDim +
       2 * kBlockN * (HeadDim + 8)) *
      static_cast<int>(sizeof(__nv_bfloat16));
  const dim3 grid(
      (sequence + kBlockM - 1) / kBlockM,
      static_cast<unsigned int>(heads),
      sequence);
  cudaFuncSetAttribute(
      streaming_triangle_forward_v3_kernel<HeadDim>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  streaming_triangle_forward_v3_kernel<HeadDim>
      <<<grid, kThreads, shared_bytes, stream>>>(
          k_map,
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          mask.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          lse.data_ptr<float>(),
          sequence,
          heads,
          scale);
}

void streaming_triangle_forward_v3_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
  const int sequence = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(4));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (head_dim) {
    case 32:
      launch_v3<32>(q, k, v, mask, bias, out, lse, sequence, heads, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "v3 hybrid currently instantiates D=32; D=16/64 await "
                  "their swizzle-mode probe verification");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v3 launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "streaming_triangle_forward_v3_out",
      &streaming_triangle_forward_v3_out,
      "Streaming forward v3: TMA+WGMMA QK^T, mma.sync PV hybrid");
}
