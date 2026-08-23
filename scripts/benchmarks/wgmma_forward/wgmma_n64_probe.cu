// m64n64k16 descriptor probe: find the LBO/SBO encoding that makes ONE wide
// wgmma reproduce eight chained m64n8k16 calls over the v3-layout K tile.
//
// Faithful to v3: K [64 keys x 32 dims] arrives by TMA (SW64, same map as
// make_k_map), A [64 x 16] staged via ldmatrix. The n64k16 B tile is the
// dim-half slice (base_offset selects chain*32B). Descriptor fields are
// RUNTIME parameters; the host scans candidates against a fp32 reference.

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void wgmma_n64_probe_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const __nv_bfloat16* __restrict__ a_global,  // [64 x 16] row-major
    float* __restrict__ accumulator_out,         // [64 x 64] row-major
    int box_bytes,
    uint64_t descriptor_lbo,
    uint64_t descriptor_sbo,
    uint64_t descriptor_mode,
    uint64_t base_offset_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
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
        "::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_u32(tile)),
        "l"(reinterpret_cast<uint64_t>(&tensor_map)),
        "r"(0),
        "r"(0),
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

  __shared__ __align__(16) __nv_bfloat16 a_stage[64 * 16];
  for (int element = tid; element < 64 * 16; element += kThreads) {
    a_stage[element] = a_global[element];
  }
  __syncthreads();
  uint32_t a_frag[4];
  {
    const int row = warp * 16 + (lane & 15);
    const int col = (lane >> 4) << 3;
    const uint32_t addr = shared_u32(a_stage + row * 16 + col);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"(addr));
  }

  const uint64_t descriptor =
      ((static_cast<uint64_t>(shared_u32(tile)) + base_offset_bytes) >> 4) |
      (descriptor_lbo << 16) | (descriptor_sbo << 32) |
      (descriptor_mode << 62);

  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.0F;
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  asm volatile(
      "{\n"
      "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "   %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "  {%32,%33,%34,%35}, %36, 0, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
        "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
        "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]),
        "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
        "+f"(d[30]), "+f"(d[31])
      : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
        "l"(descriptor));
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  // m64n64 accumulator layout: rows warp*16 + lane/4 (+8); 8 column groups,
  // cols group*8 + (lane%4)*2 (+1); d[group*4 + {0,1,2,3}] mirrors the
  // per-group m64n8 quad (c0,c1 low row; c2,c3 high row).
  const int row_low = warp * 16 + (lane >> 2);
  const int col0 = (lane & 3) * 2;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    const int col = group * 8 + col0;
    accumulator_out[row_low * 64 + col] = d[group * 4 + 0];
    accumulator_out[row_low * 64 + col + 1] = d[group * 4 + 1];
    accumulator_out[(row_low + 8) * 64 + col] = d[group * 4 + 2];
    accumulator_out[(row_low + 8) * 64 + col + 1] = d[group * 4 + 3];
  }
#endif
}

CUtensorMap make_map(
    const torch::Tensor& source, int box_rows, int box_cols,
    int swizzle_mode) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[2] = {
      static_cast<cuuint64_t>(source.size(1)),
      static_cast<cuuint64_t>(source.size(0))};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(source.size(1) * sizeof(__nv_bfloat16))};
  const cuuint32_t box_dim[2] = {
      static_cast<cuuint32_t>(box_cols), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element_stride[2] = {1, 1};
  const CUtensorMapSwizzle swizzles[4] = {
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, source.data_ptr(),
      global_dim, global_stride, box_dim, element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzles[swizzle_mode],
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return tensor_map;
}

torch::Tensor probe_n64(
    const torch::Tensor& a,        // [64, 16] bf16
    const torch::Tensor& k_tile,   // [64, kcols] bf16 (v3 layout source)
    int64_t swizzle_mode,
    int64_t lbo, int64_t sbo, int64_t mode, int64_t base_offset) {
  TORCH_CHECK(a.is_cuda() && a.dtype() == torch::kBFloat16 &&
              a.is_contiguous() && a.size(0) == 64 && a.size(1) == 16);
  TORCH_CHECK(k_tile.is_cuda() && k_tile.dtype() == torch::kBFloat16 &&
              k_tile.is_contiguous() && k_tile.size(0) == 64);
  const int kcols = static_cast<int>(k_tile.size(1));
  const CUtensorMap tensor_map =
      make_map(k_tile, 64, kcols, static_cast<int>(swizzle_mode));
  auto acc = torch::zeros(
      {64, 64}, a.options().dtype(torch::kFloat32));
  const int shared_bytes = 64 * kcols * 2;
  cudaFuncSetAttribute(
      wgmma_n64_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  wgmma_n64_probe_kernel<<<1, kThreads, shared_bytes,
                           at::cuda::getCurrentCUDAStream()>>>(
      tensor_map,
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      acc.data_ptr<float>(), shared_bytes,
      static_cast<uint64_t>(lbo), static_cast<uint64_t>(sbo),
      static_cast<uint64_t>(mode), static_cast<uint64_t>(base_offset));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "probe launch failed");
  return acc;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("probe_n64", &probe_n64,
             "m64n64k16 descriptor probe over a TMA-swizzled K tile");
}
