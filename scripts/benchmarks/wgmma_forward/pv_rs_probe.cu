// PV wgmma-rs probe: find the descriptor encoding that makes chained
// m64n32k16 (A=P from registers, B=V from TMA-swizzled shared) reproduce
// the production PV path (ldmatrix keytrans + mma.sync) bitwise.
//
// Both paths run in ONE kernel over identical inputs: the reference stages
// V padded row-major (stride D+8) exactly like v4's cp.async path; the
// candidate reads a TMA SW64 V tile through a runtime-parameterized
// descriptor. Host compares the two outputs exactly.

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kHeadDim = 32;
constexpr int kKeys = 64;
constexpr int kVStride = kHeadDim + 8;

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

// m64n32k16 with A from registers; imm-trans-b is an immediate, so two
// bodies selected by a uniform runtime flag.
#define PV_WGMMA_BODY(TRANS)                                              \
  asm volatile(                                                           \
      "{\n"                                                               \
      "  .reg .pred use_acc;\n"                                           \
      "  setp.ne.b32 use_acc, %21, 0;\n"                                  \
      "  wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "           \
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"       \
      "  {%16,%17,%18,%19}, %20, use_acc, 1, 1, " #TRANS ";\n"            \
      "}\n"                                                               \
      : "+f"(o[0].v[0]), "+f"(o[0].v[1]), "+f"(o[0].v[2]),                \
        "+f"(o[0].v[3]), "+f"(o[1].v[0]), "+f"(o[1].v[1]),                \
        "+f"(o[1].v[2]), "+f"(o[1].v[3]), "+f"(o[2].v[0]),                \
        "+f"(o[2].v[1]), "+f"(o[2].v[2]), "+f"(o[2].v[3]),                \
        "+f"(o[3].v[0]), "+f"(o[3].v[1]), "+f"(o[3].v[2]),                \
        "+f"(o[3].v[3])                                                   \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(descriptor),      \
        "r"(accumulate))

__device__ __forceinline__ void pv_wgmma(
    Acc (&o)[4], const uint32_t (&a)[4], uint64_t descriptor,
    int accumulate, int trans_b) {
  if (trans_b) {
    PV_WGMMA_BODY(1);
  } else {
    PV_WGMMA_BODY(0);
  }
}

__global__ void pv_rs_probe_kernel(
    const __grid_constant__ CUtensorMap v_map,
    const __nv_bfloat16* __restrict__ p_global,   // [64 x 64] row-major
    const __nv_bfloat16* __restrict__ v_global,   // [64 x 32] row-major
    float* __restrict__ out_wgmma,                // [64 x 32]
    float* __restrict__ out_reference,            // [64 x 32]
    int box_bytes,
    uint64_t descriptor_lbo,
    uint64_t descriptor_sbo,
    uint64_t descriptor_mode,
    uint64_t chunk_stride_bytes,
    int trans_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* p_stage = v_tile + kKeys * kHeadDim;
  __nv_bfloat16* v_padded = p_stage + kKeys * kKeys;
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
        "r"(static_cast<uint32_t>(box_bytes)));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_u32(v_tile)),
        "l"(reinterpret_cast<uint64_t>(&v_map)),
        "r"(0),
        "r"(0),
        "r"(shared_u32(&barrier)));
  }
  for (int element = tid; element < kKeys * kKeys; element += kThreads) {
    p_stage[element] = p_global[element];
  }
  for (int element = tid; element < kKeys * kHeadDim; element += kThreads) {
    const int key = element / kHeadDim;
    const int dim = element - key * kHeadDim;
    v_padded[key * kVStride + dim] = v_global[element];
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

  uint32_t p_frags[4][4];
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    ldmatrix_a16x16(p_stage, kKeys, chunk * 16, lane, p_frags[chunk]);
  }

  // Reference: the production PV path over the padded row-major V.
  Acc o_ref[kHeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < kHeadDim / 8; ++tile) {
    o_ref[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
#pragma unroll
    for (int tile = 0; tile < kHeadDim / 8; ++tile) {
      uint32_t v_frag[2];
      ldmatrix_b_keytrans(
          v_padded, kVStride, chunk * 16, tile * 8, lane, v_frag);
      mma_m16n8k16(o_ref[tile], p_frags[chunk], v_frag);
    }
  }

  // Candidate: chained m64n32k16 rs over the TMA tile.
  Acc o_w[4];
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    o_w[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int chunk = 0; chunk < 4; ++chunk) {
    const uint64_t descriptor =
        ((static_cast<uint64_t>(shared_u32(v_tile)) +
          chunk * chunk_stride_bytes) >>
         4) |
        (descriptor_lbo << 16) | (descriptor_sbo << 32) |
        (descriptor_mode << 62);
    const uint32_t(&a)[4] = p_frags[chunk];
    const int accumulate = chunk != 0;
    Acc(&o)[4] = o_w;
    pv_wgmma(o, a, descriptor, accumulate, trans_b);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  // Both layouts: rows warp*16 + lane/4 (+8); column group g covers dims
  // g*8 + (lane%4)*2 (+1); v[{0,1,2,3}] = (low row c0,c1; high row c0,c1).
  const int row_low = warp * 16 + (lane >> 2);
  const int col0 = (lane & 3) * 2;
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    const int col = tile * 8 + col0;
    out_wgmma[row_low * kHeadDim + col] = o_w[tile].v[0];
    out_wgmma[row_low * kHeadDim + col + 1] = o_w[tile].v[1];
    out_wgmma[(row_low + 8) * kHeadDim + col] = o_w[tile].v[2];
    out_wgmma[(row_low + 8) * kHeadDim + col + 1] = o_w[tile].v[3];
    out_reference[row_low * kHeadDim + col] = o_ref[tile].v[0];
    out_reference[row_low * kHeadDim + col + 1] = o_ref[tile].v[1];
    out_reference[(row_low + 8) * kHeadDim + col] = o_ref[tile].v[2];
    out_reference[(row_low + 8) * kHeadDim + col + 1] = o_ref[tile].v[3];
  }
#endif
}

CUtensorMap make_map(const torch::Tensor& source) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[2] = {
      static_cast<cuuint64_t>(source.size(1)),
      static_cast<cuuint64_t>(source.size(0))};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(source.size(1) * sizeof(__nv_bfloat16))};
  const cuuint32_t box_dim[2] = {
      static_cast<cuuint32_t>(kHeadDim), static_cast<cuuint32_t>(kKeys)};
  const cuuint32_t element_stride[2] = {1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data_ptr(),
      global_dim, global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return tensor_map;
}

std::vector<torch::Tensor> probe_pv(
    const torch::Tensor& p,   // [64, 64] bf16
    const torch::Tensor& v,   // [64, 32] bf16
    int64_t lbo, int64_t sbo, int64_t mode, int64_t chunk_stride,
    int64_t trans_b) {
  TORCH_CHECK(p.is_cuda() && p.dtype() == torch::kBFloat16 &&
              p.is_contiguous() && p.size(0) == 64 && p.size(1) == 64);
  TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kBFloat16 &&
              v.is_contiguous() && v.size(0) == 64 && v.size(1) == 32);
  const CUtensorMap v_map = make_map(v);
  auto out_w = torch::zeros({64, 32}, p.options().dtype(torch::kFloat32));
  auto out_r = torch::zeros({64, 32}, p.options().dtype(torch::kFloat32));
  const int box_bytes = kKeys * kHeadDim * 2;
  const int shared_bytes =
      (kKeys * kHeadDim + kKeys * kKeys + kKeys * kVStride) * 2;
  cudaFuncSetAttribute(
      pv_rs_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  pv_rs_probe_kernel<<<1, kThreads, shared_bytes,
                       at::cuda::getCurrentCUDAStream()>>>(
      v_map,
      reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      out_w.data_ptr<float>(), out_r.data_ptr<float>(), box_bytes,
      static_cast<uint64_t>(lbo), static_cast<uint64_t>(sbo),
      static_cast<uint64_t>(mode), static_cast<uint64_t>(chunk_stride),
      static_cast<int>(trans_b));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "probe launch failed");
  return {out_w, out_r};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("probe_pv", &probe_pv,
             "PV wgmma-rs descriptor probe vs the production mma.sync path");
}
