// Backward ss-wgmma probe: ONE m64n32k16 with BOTH operands from smem
// descriptors (A = K tile, B = Q tile with trans-b) must reproduce the
// dkdv-style mma.sync chain (A-frags from K rows, B dim-pairs from Q)
// bitwise. Both tiles arrive by TMA with the production swizzle; the
// A-side descriptor fields and imm-trans-a are runtime-scanned, B-side
// seeded with the cracked law (LBO=64, SBO=HeadDim/16B, trans-b=1).

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kKeys = 64;
constexpr int kQueries = 32;
constexpr int kHeadDim = 32;

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

__device__ __forceinline__ int swz_col(int row, int col) {
  constexpr int kAtoms = kHeadDim / 8;
  const int atom = (col >> 3) ^ (((row * kHeadDim) >> 6) & (kAtoms - 1));
  return atom * 8 + (col & 7);
}

// A-frag over the swizzled compact K tile: 16 rows of this warp.
__device__ __forceinline__ void ldmatrix_a16x16_sw(
    const __nv_bfloat16* tile, int row_start, int col_start, int lane,
    uint32_t (&frag)[4]) {
  const int row = row_start + (lane & 15);
  const int col = col_start + (((lane >> 4) & 1) << 3);
  const uint32_t addr = shared_u32(tile + row * kHeadDim + swz_col(row, col));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// B dim-pairs over the swizzled compact Q tile (dkdv consumer semantics).
__device__ __forceinline__ void ldmatrix_b_dimpair_sw(
    const __nv_bfloat16* tile, int query_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int query = query_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr =
      shared_u32(tile + query * kHeadDim + swz_col(query, col));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

#define SS_WGMMA_BODY(TA, TB)                                             \
  asm volatile(                                                           \
      "{\n"                                                               \
      "  .reg .pred use_acc;\n"                                           \
      "  setp.ne.b32 use_acc, %18, 0;\n"                                  \
      "  wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "           \
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"       \
      "  %16, %17, use_acc, 1, 1, " #TA ", " #TB ";\n"                    \
      "}\n"                                                               \
      : "+f"(o[0].v[0]), "+f"(o[0].v[1]), "+f"(o[0].v[2]),                \
        "+f"(o[0].v[3]), "+f"(o[1].v[0]), "+f"(o[1].v[1]),                \
        "+f"(o[1].v[2]), "+f"(o[1].v[3]), "+f"(o[2].v[0]),                \
        "+f"(o[2].v[1]), "+f"(o[2].v[2]), "+f"(o[2].v[3]),                \
        "+f"(o[3].v[0]), "+f"(o[3].v[1]), "+f"(o[3].v[2]),                \
        "+f"(o[3].v[3])                                                   \
      : "l"(descriptor_a), "l"(descriptor_b), "r"(accumulate))

__device__ __forceinline__ void ss_wgmma(
    Acc (&o)[4], uint64_t descriptor_a, uint64_t descriptor_b,
    int accumulate, int trans_a, int trans_b) {
  if (trans_a) {
    if (trans_b) { SS_WGMMA_BODY(1, 1); } else { SS_WGMMA_BODY(1, 0); }
  } else {
    if (trans_b) { SS_WGMMA_BODY(0, 1); } else { SS_WGMMA_BODY(0, 0); }
  }
}

__global__ void ss_bwd_probe_kernel(
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap q_map,
    float* __restrict__ out_wgmma,      // [64 x 32]
    float* __restrict__ out_reference,  // [64 x 32]
    uint64_t a_lbo, uint64_t a_sbo, uint64_t b_lbo, uint64_t b_sbo,
    uint64_t mode, int trans_a, int trans_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* q_tile = k_tile + kKeys * kHeadDim;
  __shared__ __align__(8) uint64_t barrier;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (tid == 0) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
            shared_u32(&barrier)));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            shared_u32(&barrier)),
        "r"(static_cast<uint32_t>(
            (kKeys + kQueries) * kHeadDim * sizeof(__nv_bfloat16))));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_u32(k_tile)),
        "l"(reinterpret_cast<uint64_t>(&k_map)), "r"(0), "r"(0),
        "r"(shared_u32(&barrier)));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_u32(q_tile)),
        "l"(reinterpret_cast<uint64_t>(&q_map)), "r"(0), "r"(0),
        "r"(shared_u32(&barrier)));
  }
  __syncthreads();
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "  wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0;\n"
      "  @!done bra wait;\n"
      "}\n" ::"r"(shared_u32(&barrier)));
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  // Reference: dkdv-style chain — A-frags from this warp's K rows,
  // B dim-pairs from Q, m16n8k16 per (column, chain).
  Acc reference[kQueries / 8];
#pragma unroll
  for (int column = 0; column < kQueries / 8; ++column) {
    reference[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }
  uint32_t k_frags[kHeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < kHeadDim / 16; ++chain) {
    ldmatrix_a16x16_sw(k_tile, warp * 16, chain * 16, lane, k_frags[chain]);
  }
#pragma unroll
  for (int column = 0; column < kQueries / 8; ++column) {
#pragma unroll
    for (int chain = 0; chain < kHeadDim / 16; ++chain) {
      uint32_t b_frag[2];
      ldmatrix_b_dimpair_sw(q_tile, column * 8, chain * 16, lane, b_frag);
      mma_m16n8k16(reference[column], k_frags[chain], b_frag);
    }
  }

  // Candidate: chained ss m64n32k16 over both descriptors.
  Acc o[4];
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    o[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int chain = 0; chain < kHeadDim / 16; ++chain) {
    const uint64_t descriptor_a =
        ((static_cast<uint64_t>(shared_u32(k_tile)) + chain * 32) >> 4) |
        (a_lbo << 16) | (a_sbo << 32) | (mode << 62);
    const uint64_t descriptor_b =
        ((static_cast<uint64_t>(shared_u32(q_tile)) + chain * 32) >> 4) |
        (b_lbo << 16) | (b_sbo << 32) | (mode << 62);
    const int accumulate = chain != 0;
    ss_wgmma(o, descriptor_a, descriptor_b, accumulate, trans_a, trans_b);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  const int row_low = warp * 16 + (lane >> 2);
  const int col0 = (lane & 3) * 2;
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    const int col = tile * 8 + col0;
    out_wgmma[row_low * kQueries + col] = o[tile].v[0];
    out_wgmma[row_low * kQueries + col + 1] = o[tile].v[1];
    out_wgmma[(row_low + 8) * kQueries + col] = o[tile].v[2];
    out_wgmma[(row_low + 8) * kQueries + col + 1] = o[tile].v[3];
    out_reference[row_low * kQueries + col] = reference[tile].v[0];
    out_reference[row_low * kQueries + col + 1] = reference[tile].v[1];
    out_reference[(row_low + 8) * kQueries + col] = reference[tile].v[2];
    out_reference[(row_low + 8) * kQueries + col + 1] = reference[tile].v[3];
  }
#endif
}

CUtensorMap make_map(const torch::Tensor& source, int rows) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[2] = {
      static_cast<cuuint64_t>(kHeadDim), static_cast<cuuint64_t>(rows)};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(kHeadDim) * sizeof(__nv_bfloat16)};
  const cuuint32_t box_dim[2] = {
      static_cast<cuuint32_t>(kHeadDim), static_cast<cuuint32_t>(rows)};
  const cuuint32_t element_stride[2] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data_ptr(),
      global_dim, global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "map encode failed: ",
              static_cast<int>(result));
  return tensor_map;
}

std::vector<torch::Tensor> probe_ss(
    const torch::Tensor& k,   // [64, 32] bf16
    const torch::Tensor& q,   // [32, 32] bf16
    int64_t a_lbo, int64_t a_sbo, int64_t b_lbo, int64_t b_sbo,
    int64_t mode, int64_t trans_a, int64_t trans_b) {
  TORCH_CHECK(k.is_cuda() && k.dtype() == torch::kBFloat16 &&
              k.is_contiguous() && k.size(0) == kKeys &&
              k.size(1) == kHeadDim);
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 &&
              q.is_contiguous() && q.size(0) == kQueries &&
              q.size(1) == kHeadDim);
  const CUtensorMap k_map = make_map(k, kKeys);
  const CUtensorMap q_map = make_map(q, kQueries);
  auto out_w = torch::zeros({kKeys, kQueries},
                            k.options().dtype(torch::kFloat32));
  auto out_r = torch::zeros({kKeys, kQueries},
                            k.options().dtype(torch::kFloat32));
  const int shared_bytes = (kKeys + kQueries) * kHeadDim * 2;
  cudaFuncSetAttribute(
      ss_bwd_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  ss_bwd_probe_kernel<<<1, kThreads, shared_bytes,
                        at::cuda::getCurrentCUDAStream()>>>(
      k_map, q_map, out_w.data_ptr<float>(), out_r.data_ptr<float>(),
      static_cast<uint64_t>(a_lbo), static_cast<uint64_t>(a_sbo),
      static_cast<uint64_t>(b_lbo), static_cast<uint64_t>(b_sbo),
      static_cast<uint64_t>(mode), static_cast<int>(trans_a),
      static_cast<int>(trans_b));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "probe launch failed");
  return {out_w, out_r};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("probe_ss", &probe_ss,
             "backward ss-wgmma descriptor probe vs the dkdv mma.sync chain");
}
