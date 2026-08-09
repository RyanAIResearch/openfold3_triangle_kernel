// SPDX-License-Identifier: Apache-2.0
// Streaming line delta precompute: delta[o,h,q] = sum_d dO[o,h,q,d]*O[o,h,q,d].
//
// One warp owns one row; each lane holds one product (D=32 == warp size) and
// the sum runs down a fixed XOR-shuffle tree. Single writer per output
// element, zero atomics, replay-deterministic by construction.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;

__global__ __launch_bounds__(kThreads) void streaming_triangle_delta_kernel(
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ output,
    float* __restrict__ delta,
    int64_t rows,
    int head_dim) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock +
      (threadIdx.x / kWarpSize);
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x & (kWarpSize - 1);
  // Any head_dim: each lane accumulates its strided slice serially (fixed
  // ascending order), then the fixed XOR tree combines the 32 lane sums.
  float value = 0.0F;
  for (int dim = lane; dim < head_dim; dim += kWarpSize) {
    const int64_t base = row * head_dim + dim;
    value += __bfloat162float(d_output[base]) *
        __bfloat162float(output[base]);
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, offset, kWarpSize);
  }
  if (lane == 0) {
    delta[row] = value;
  }
}

void streaming_triangle_delta_out(
    const torch::Tensor& d_output,
    const torch::Tensor& output,
    torch::Tensor& delta) {
  TORCH_CHECK(d_output.is_cuda() && d_output.dtype() == torch::kBFloat16);
  TORCH_CHECK(d_output.dim() >= 1 && d_output.size(-1) > 0);
  TORCH_CHECK(
      output.is_cuda() && output.device() == d_output.device() &&
          output.dtype() == torch::kBFloat16 &&
          output.sizes() == d_output.sizes());
  TORCH_CHECK(d_output.is_contiguous() && output.is_contiguous());
  const int head_dim = static_cast<int>(d_output.size(-1));
  TORCH_CHECK(
      delta.is_cuda() && delta.device() == d_output.device() &&
          delta.dtype() == torch::kFloat32 && delta.is_contiguous());
  const int64_t rows = d_output.numel() / head_dim;
  TORCH_CHECK(delta.numel() == rows);
  const int64_t blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const c10::cuda::CUDAGuard device_guard(d_output.device());
  // Resolve the runtime kernel handle once per calling thread/device.  This
  // also gives Compute Sanitizer a concrete function handle before launch,
  // while keeping the steady-state training path free of repeated API calls.
  static thread_local int resolved_device = -1;
  const int device_index = d_output.get_device();
  if (resolved_device != device_index) {
    cudaFuncAttributes attributes{};
    const cudaError_t attribute_status = cudaFuncGetAttributes(
        &attributes, streaming_triangle_delta_kernel);
    TORCH_CHECK(
        attribute_status == cudaSuccess,
        "failed to resolve streaming delta kernel attributes: ",
        cudaGetErrorString(attribute_status));
    resolved_device = device_index;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  streaming_triangle_delta_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
      delta.data_ptr<float>(),
      rows,
      head_dim);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "streaming_triangle_delta_out",
      &streaming_triangle_delta_out,
      "delta = rowsum(dO * O), warp-owned rows, fixed shuffle tree");
}
