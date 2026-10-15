// SPDX-License-Identifier: Apache-2.0
// TurboPairFormer classic and shadow-residual delta precompute.
// Delta is the FP32 per-query correction used by softmax backward:
//   classic: delta = sum_channel(dO * O)
//   shadow:  delta = sum_channel(dO * (O + residual))
// The default SR shadow route uses the forward's BF16 output-rounding
// residual. Add O and residual in FP32 before multiplying by dO.
//
// Input layout: dO/O/residual [B, O, N, H, D] BF16.
// Output layout: delta [B, O, H, N] FP32, passed as a flat row array.
// B = batch, O = outer planes, N = queries, H = heads, D = head channels.
// One warp owns one (batch, outer, head, query) row; a block owns eight rows.
// Each lane accumulates its channels in ascending order, then a fixed
// XOR-shuffle tree combines the 32 lane sums. Lane 0 writes the row result.
// No shared-memory staging or atomic accumulation is needed.
// Repeated runs with the same inputs, binary, and device produce bitwise
// identical delta values through this fixed accumulation order.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr long long kMaxGridX = 2147483647LL;

__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_delta_kernel(
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ output,
    float* __restrict__ delta,
    int64_t rows,
    int sequence,
    int heads,
    int head_dim) {
  // A. Assign one row to this warp; all its lanes take the same tail branch.
  const int64_t row_index =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x / kWarpSize);
  if (row_index >= rows) {
    return;
  }
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  // B. Decode a [B, O, H, N] row and locate its [B, O, N, H, D] input.
  // batch_outer_index flattens B and O; row_in_plane flattens H and N.
  const int64_t rows_per_plane = static_cast<int64_t>(heads) * sequence;
  const int64_t batch_outer_index = row_index / rows_per_plane;
  const int64_t row_in_plane = row_index - batch_outer_index * rows_per_plane;
  const int64_t head_index = row_in_plane / sequence;
  const int64_t query_index = row_in_plane - head_index * sequence;
  const int64_t io_row_base =
      ((batch_outer_index * sequence + query_index) * heads + head_index) * head_dim;
  // C. Keep a private FP32 partial sum in each lane's register. Lane i
  // visits channels i, i+32, ...; lanes with no channels retain zero.
  float row_sum = 0.0F;
  for (int channel = lane_id; channel < head_dim; channel += kWarpSize) {
    const int64_t element_index = io_row_base + channel;
    row_sum += __bfloat162float(d_output[element_index]) * __bfloat162float(output[element_index]);
  }
  // D. Combine lane sums in the fixed 16, 8, 4, 2, 1 XOR-shuffle order.
  // All 32 lanes participate; afterward each lane has the warp's row sum.
#pragma unroll
  for (int shuffle_offset = kWarpSize / 2; shuffle_offset > 0; shuffle_offset /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFFU, row_sum, shuffle_offset, kWarpSize);
  }
  // E. One lane writes the FP32 result; the register sum is no longer needed.
  if (lane_id == 0) {
    delta[row_index] = row_sum;
  }
}

// Shadow variant: use O plus its saved BF16 rounding residual to approximate
// the forward result before output rounding, without reconstructing scores.
// Keep the addition inside dO * (O + residual); distributing the product
// would change FP32 rounding. Row ownership and shuffle order match classic.
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_delta_shadow_kernel(
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ shadow_residual,
    float* __restrict__ delta,
    int64_t rows,
    int sequence,
    int heads,
    int head_dim) {
  // A. Assign and decode the same stats row as in the classic kernel.
  const int64_t row_index =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x / kWarpSize);
  if (row_index >= rows) {
    return;
  }
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int64_t rows_per_plane = static_cast<int64_t>(heads) * sequence;
  const int64_t batch_outer_index = row_index / rows_per_plane;
  const int64_t row_in_plane = row_index - batch_outer_index * rows_per_plane;
  const int64_t head_index = row_in_plane / sequence;
  const int64_t query_index = row_in_plane - head_index * sequence;
  const int64_t io_row_base =
      ((batch_outer_index * sequence + query_index) * heads + head_index) * head_dim;
  // B. Convert BF16 inputs to FP32, add the residual, then accumulate dO
  // times that corrected output in ascending channel order within each lane.
  float row_sum = 0.0F;
  for (int channel = lane_id; channel < head_dim; channel += kWarpSize) {
    const int64_t element_index = io_row_base + channel;
    row_sum += __bfloat162float(d_output[element_index])
               * (__bfloat162float(output[element_index])
                  + __bfloat162float(shadow_residual[element_index]));
  }
  // C. Reduce the 32 lane sums with the same fixed XOR tree.
#pragma unroll
  for (int shuffle_offset = kWarpSize / 2; shuffle_offset > 0; shuffle_offset /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFFU, row_sum, shuffle_offset, kWarpSize);
  }
  // D. Only lane 0 writes this row's FP32 delta.
  if (lane_id == 0) {
    delta[row_index] = row_sum;
  }
}

void turbopairformer_triangle_delta_out(
    const torch::Tensor& d_output,
    const torch::Tensor& output,
    torch::Tensor& delta) {
  // Validate contiguous BF16 inputs and the caller-provided FP32 row output.
  TORCH_CHECK(d_output.is_cuda() && d_output.dtype() == torch::kBFloat16);
  // Public layout [B, O, N, H, D]: sequence and heads come from the shape.
  TORCH_CHECK(d_output.dim() == 5 && d_output.size(-1) > 0);
  TORCH_CHECK(
      output.is_cuda() && output.device() == d_output.device() && output.dtype() == torch::kBFloat16
      && output.sizes() == d_output.sizes());
  TORCH_CHECK(d_output.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(
      d_output.size(-1) <= std::numeric_limits<int>::max(),
      "head dimension exceeds the CUDA int indexing contract");
  const int head_dim = static_cast<int>(d_output.size(-1));
  const int sequence = static_cast<int>(d_output.size(2));
  const int heads = static_cast<int>(d_output.size(3));
  TORCH_CHECK(
      delta.is_cuda() && delta.device() == d_output.device() && delta.dtype() == torch::kFloat32
      && delta.is_contiguous());
  const int64_t rows = d_output.numel() / head_dim;
  TORCH_CHECK(delta.numel() == rows);
  // Eight warps produce up to eight rows per block, including a short tail.
  const int64_t blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  TORCH_CHECK(blocks <= kMaxGridX, "delta launch exceeds the CUDA grid-x contract");
  const c10::cuda::CUDAGuard device_guard(d_output.device());
  // Cache the last device resolved by this host thread. Repeated calls on
  // that device skip the attribute query; switching devices resolves again.
  // The concrete kernel handle also supports Compute Sanitizer before launch.
  static thread_local int resolved_device = -1;
  const int device_index = d_output.get_device();
  if (resolved_device != device_index) {
    cudaFuncAttributes attributes{};
    const cudaError_t attribute_status =
        cudaFuncGetAttributes(&attributes, turbopairformer_triangle_delta_kernel);
    TORCH_CHECK(
        attribute_status == cudaSuccess,
        "failed to resolve TurboPairFormer delta kernel attributes: ",
        cudaGetErrorString(attribute_status));
    resolved_device = device_index;
  }
  // Launch on the current stream for d_output's device; no workspace is used.
  auto stream = at::cuda::getCurrentCUDAStream();
  turbopairformer_triangle_delta_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()), delta.data_ptr<float>(), rows,
      sequence, heads, head_dim);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

void turbopairformer_triangle_delta_shadow_out(
    const torch::Tensor& d_output,
    const torch::Tensor& output,
    const torch::Tensor& shadow_residual,
    torch::Tensor& delta) {
  // Validate the residual first, then the matching BF16 inputs and FP32 delta.
  TORCH_CHECK(
      shadow_residual.is_cuda() && shadow_residual.device() == output.device()
          && shadow_residual.dtype() == torch::kBFloat16 && shadow_residual.is_contiguous()
          && shadow_residual.sizes() == output.sizes(),
      "shadow_residual must be contiguous BF16 matching output");
  TORCH_CHECK(d_output.is_cuda() && d_output.dtype() == torch::kBFloat16);
  TORCH_CHECK(d_output.dim() == 5 && d_output.size(-1) > 0);
  TORCH_CHECK(
      output.is_cuda() && output.device() == d_output.device() && output.dtype() == torch::kBFloat16
      && output.sizes() == d_output.sizes());
  TORCH_CHECK(d_output.is_contiguous() && output.is_contiguous());
  const int head_dim = static_cast<int>(d_output.size(-1));
  const int sequence = static_cast<int>(d_output.size(2));
  const int heads = static_cast<int>(d_output.size(3));
  TORCH_CHECK(
      delta.is_cuda() && delta.device() == d_output.device() && delta.dtype() == torch::kFloat32
      && delta.is_contiguous());
  const int64_t rows = d_output.numel() / head_dim;
  TORCH_CHECK(delta.numel() == rows);
  // The shadow kernel keeps the classic kernel's eight-row block mapping.
  const int64_t blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  TORCH_CHECK(blocks <= kMaxGridX, "delta launch exceeds the CUDA grid-x contract");
  const c10::cuda::CUDAGuard device_guard(d_output.device());
  // This entry point launches the shadow kernel directly on the current stream.
  auto stream = at::cuda::getCurrentCUDAStream();
  turbopairformer_triangle_delta_shadow_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(shadow_residual.data_ptr()), delta.data_ptr<float>(),
      rows, sequence, heads, head_dim);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "turbopairformer_triangle_delta_out", &turbopairformer_triangle_delta_out,
      "delta = rowsum(dO * O), warp-owned rows, fixed shuffle tree");
  module.def(
      "turbopairformer_triangle_delta_shadow_out", &turbopairformer_triangle_delta_shadow_out,
      "delta = rowsum(dO * (O + shadow residual)), probability-consistent");
}
