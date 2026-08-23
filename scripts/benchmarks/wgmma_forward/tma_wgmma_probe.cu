// TMA + WGMMA pairing probe (round 3, rung 1 and 2).
//
// Rung 1: TMA-load one [rows x cols] BF16 tile into shared with a chosen
// swizzle mode, then dump the RAW shared bytes back to global so the host
// can infer the exact physical layout (no device-side layout assumptions).
//
// Rung 2: run one wgmma.mma_async m64n8k16 with a RUNTIME-constructed B
// descriptor (base/LBO/SBO/mode passed from the host) over that shared
// tile, dump the accumulator, and let the host search the descriptor
// parameter space against a reference product. Zero recompiles per trial.

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

// ---------------- Rung 1: TMA load + raw dump ----------------

__global__ void tma_dump_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    __nv_bfloat16* __restrict__ raw_out,
    int shared_elements,
    int box_x,
    int box_y) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __shared__ __align__(8) uint64_t barrier;

  if (threadIdx.x == 0) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(
            shared_u32(&barrier)));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            shared_u32(&barrier)),
        "r"(static_cast<uint32_t>(
            box_x * box_y * static_cast<int>(sizeof(__nv_bfloat16)))));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(shared_u32(tile)),
        "l"(reinterpret_cast<uint64_t>(&tensor_map)),
        "r"(0),
        "r"(0),
        "r"(shared_u32(&barrier)));
  }
  __syncthreads();
  uint32_t parity = 0;
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "  wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "  @!done bra wait;\n"
      "}\n" ::"r"(shared_u32(&barrier)),
      "r"(parity));
  __syncthreads();
  for (int element = threadIdx.x; element < shared_elements;
       element += kThreads) {
    raw_out[element] = tile[element];
  }
#endif
}

// ---------------- Rung 2: runtime-descriptor wgmma ----------------

__global__ void wgmma_probe_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    const __nv_bfloat16* __restrict__ a_global,  // [64 x 16] row-major
    float* __restrict__ accumulator_out,         // [64 x 8] row-major
    int box_x,
    int box_y,
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
        "r"(static_cast<uint32_t>(
            box_x * box_y * static_cast<int>(sizeof(__nv_bfloat16)))));
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

  // A fragment: [64 rows x 16 k] — per warp 16 rows, standard layout.
  uint32_t a_frag[4];
  {
    const int row = warp * 16 + (lane & 15);
    const int col = ((lane >> 4) << 3);
    const __nv_bfloat16* src = a_global + row * 16 + col;
    // Pack from global directly (probe-only; correctness is what matters).
    uint32_t pair0, pair1, pair2, pair3;
    pair0 = *reinterpret_cast<const uint32_t*>(src);
    pair1 = *reinterpret_cast<const uint32_t*>(src + 2);
    pair2 = *reinterpret_cast<const uint32_t*>(src + 4);
    pair3 = *reinterpret_cast<const uint32_t*>(src + 6);
    // ldmatrix-equivalent packing for row-major A: a0=(r,c0c1) a1=(r+?,..)
    // Use ldmatrix to avoid layout mistakes: stage rows to shared first.
    (void)pair0;
    (void)pair1;
    (void)pair2;
    (void)pair3;
  }
  __shared__ __align__(16) __nv_bfloat16 a_stage[64 * 16];
  for (int element = tid; element < 64 * 16; element += kThreads) {
    a_stage[element] = a_global[element];
  }
  __syncthreads();
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

  float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  asm volatile(
      "{\n"
      "  wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "  {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
      "}\n"
      : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
      : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
        "l"(descriptor));
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  const int row_low = warp * 16 + (lane >> 2);
  const int col = (lane & 3) * 2;
  accumulator_out[row_low * 8 + col] = acc0;
  accumulator_out[row_low * 8 + col + 1] = acc1;
  accumulator_out[(row_low + 8) * 8 + col] = acc2;
  accumulator_out[(row_low + 8) * 8 + col + 1] = acc3;
#endif
}

// ---------------- Host side ----------------

CUtensorMap make_tensor_map(
    const torch::Tensor& source,  // [rows, cols] BF16 contiguous
    int box_rows,
    int box_cols,
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
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_SWIZZLE_128B};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      2,
      source.data_ptr(),
      global_dim,
      global_stride,
      box_dim,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzles[swizzle_mode],
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ",
              static_cast<int>(result));
  return tensor_map;
}

torch::Tensor tma_dump(
    const torch::Tensor& source, int box_rows, int box_cols,
    int swizzle_mode) {
  TORCH_CHECK(source.is_cuda() && source.dtype() == torch::kBFloat16 &&
              source.is_contiguous() && source.dim() == 2);
  const CUtensorMap tensor_map =
      make_tensor_map(source, box_rows, box_cols, swizzle_mode);
  const int shared_elements = box_rows * box_cols;
  auto raw = torch::empty(
      {shared_elements}, source.options());
  const int shared_bytes =
      shared_elements * static_cast<int>(sizeof(__nv_bfloat16));
  cudaFuncSetAttribute(
      tma_dump_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  tma_dump_kernel<<<1, kThreads, shared_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
      tensor_map,
      reinterpret_cast<__nv_bfloat16*>(raw.data_ptr()),
      shared_elements,
      box_cols,
      box_rows);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tma_dump launch failed");
  return raw;
}

torch::Tensor wgmma_probe(
    const torch::Tensor& b_source,  // [rows=keys, cols=k-dims] BF16
    const torch::Tensor& a_source,  // [64, 16] BF16
    int box_rows,
    int box_cols,
    int swizzle_mode,
    int64_t lbo,
    int64_t sbo,
    int64_t mode,
    int64_t base_offset_bytes) {
  const CUtensorMap tensor_map =
      make_tensor_map(b_source, box_rows, box_cols, swizzle_mode);
  auto accumulator = torch::zeros(
      {64, 8}, a_source.options().dtype(torch::kFloat32));
  const int shared_bytes =
      box_rows * box_cols * static_cast<int>(sizeof(__nv_bfloat16));
  cudaFuncSetAttribute(
      wgmma_probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  wgmma_probe_kernel<<<1, kThreads, shared_bytes,
                       at::cuda::getCurrentCUDAStream()>>>(
      tensor_map,
      reinterpret_cast<const __nv_bfloat16*>(a_source.data_ptr()),
      accumulator.data_ptr<float>(),
      box_cols,
      box_rows,
      static_cast<uint64_t>(lbo),
      static_cast<uint64_t>(sbo),
      static_cast<uint64_t>(mode),
      static_cast<uint64_t>(base_offset_bytes));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "wgmma_probe launch failed");
  return accumulator;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("tma_dump", &tma_dump, "TMA tile load + raw shared dump");
  module.def("wgmma_probe", &wgmma_probe,
             "wgmma with runtime B descriptor over a TMA-loaded tile");
}
