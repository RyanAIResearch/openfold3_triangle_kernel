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
//
// The shadow route also has a vectorized kernel for the common case
// (D a multiple of 8, D <= 64, 16-byte aligned inputs). It gives each row
// 2, 4, or 8 lanes that read 16 bytes per tensor, and it performs exactly
// the same FP32 operations, in the same order, as the warp-per-row shadow
// kernel. The warp-per-row kernel stays as the reference definition and
// serves every other input. See the comment above the vectorized kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
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

// -----------------------------------------------------------------------------
// Vectorized shadow delta: a few lanes per row, 16-byte loads
// -----------------------------------------------------------------------------
// The warp-per-row shadow kernel above defines the result. Its compiled
// arithmetic is (ascending channel loop, FP32, round to nearest):
//   lane l:  partial = +0
//            partial = fma(dO[l],    O[l]    + R[l],    partial)
//            partial = fma(dO[l+32], O[l+32] + R[l+32], partial)  if l+32 < D
//   tree:    add partials of lanes that differ in index bit 4, then bit 3,
//            bit 2, bit 1, bit 0 (XOR shuffles 16, 8, 4, 2, 1).
// A lane without channels keeps partial = +0 and still enters the tree, so
// for D < 32 the tree adds +0 to some partials (this turns a -0 into +0).
//
// Here lane j of a row owns the 16-byte chunk of channels 8j..8j+7, kept in
// eight register slots. For channel c = 8j + slot:
//   bits 2-0 of c = slot,  bits 4-3 of c = j & 3,  bit 5 of c = j >> 2.
// The same FP32 operations then map to:
//   bit 5  (8-lane rows, D > 32) lane j+4 continues the FMA chain that lane j
//          started: it receives lane j's partial and applies its own channel.
//          A missing upper chunk leaves the received partial untouched.
//   bit 4  shuffle XOR 2 within the row. In a 2-lane row (D <= 16) the
//          partner chunk never exists, so the partner's value is +0.
//   bit 3  shuffle XOR 1 within the row.
//   bits 2-0  additions between register slots: +4, then +2, then +1.
// A missing chunk in a 2- or 4-lane row holds +0 in every slot, exactly like
// a lane without channels in the warp-per-row kernel. Exchanging the two
// operands of one FP32 addition never changes its bits, so it does not
// matter which lane of a pair performs an addition.
//
// Every operation uses an explicit round-to-nearest intrinsic
// (__fadd_rn, __fmaf_rn). Plain operators would let the compiler contract or
// simplify them; for example, folding x + 0.0F to x keeps a -0 that the tree
// above turns into +0.
//
// Rows are visited in input memory order [B, O, N, H], so consecutive row
// groups of a warp read one contiguous span. The writer lane scatters the
// result to its [B, O, H, N] delta position.
template <int RowLanes>
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_delta_shadow_vectorized_kernel(
    const __nv_bfloat16* __restrict__ d_output,
    const __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ shadow_residual,
    float* __restrict__ delta,
    int64_t rows,
    int sequence,
    int heads,
    int head_dim) {
  static_assert(RowLanes == 2 || RowLanes == 4 || RowLanes == 8, "rows use 2, 4, or 8 lanes");
  constexpr int kChunkChannels = 8;  // 16 bytes of BF16 per lane and tensor.
  constexpr int kRowsPerWarp = kWarpSize / RowLanes;
  // Lane 4 holds the finished row in 8-lane rows (it completes the FMA
  // chain); every lane of a 2- or 4-lane row ends with the same result.
  constexpr int kWriterLane = RowLanes == 8 ? 4 : 0;

  // A. Assign a row and a 16-byte channel chunk to this lane. All lanes stay
  // active through the shuffles; a lane past the last row reads row 0 and
  // does not write.
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int64_t warp_index = (static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x) / kWarpSize;
  const int64_t memory_row = warp_index * kRowsPerWarp + lane_id / RowLanes;
  const int row_lane = lane_id % RowLanes;
  const bool row_is_valid = memory_row < rows;
  const int64_t loaded_row = row_is_valid ? memory_row : 0;
  const int chunk_channel_start = row_lane * kChunkChannels;
  const bool chunk_exists = chunk_channel_start < head_dim;

  // B. Load the chunk of dO, O, and the residual, and form this lane's
  // partial products. For 8-lane rows, only lanes 0-3 (channels 0..31) start
  // from +0 here; lanes 4-7 continue from those partials in step C.
  float partial[kChunkChannels];
  float upper_d_output[kChunkChannels];
  float upper_corrected_output[kChunkChannels];
#pragma unroll
  for (int slot = 0; slot < kChunkChannels; ++slot) {
    partial[slot] = 0.0F;
    upper_d_output[slot] = 0.0F;
    upper_corrected_output[slot] = 0.0F;
  }
  if (chunk_exists) {
    const int64_t element_offset = loaded_row * head_dim + chunk_channel_start;
    const uint4 d_output_bits = *reinterpret_cast<const uint4*>(d_output + element_offset);
    const uint4 output_bits = *reinterpret_cast<const uint4*>(output + element_offset);
    const uint4 residual_bits = *reinterpret_cast<const uint4*>(shadow_residual + element_offset);
    const __nv_bfloat162* d_output_pairs = reinterpret_cast<const __nv_bfloat162*>(&d_output_bits);
    const __nv_bfloat162* output_pairs = reinterpret_cast<const __nv_bfloat162*>(&output_bits);
    const __nv_bfloat162* residual_pairs = reinterpret_cast<const __nv_bfloat162*>(&residual_bits);
#pragma unroll
    for (int pair = 0; pair < kChunkChannels / 2; ++pair) {
      const float2 d_output_values = __bfloat1622float2(d_output_pairs[pair]);
      const float2 output_values = __bfloat1622float2(output_pairs[pair]);
      const float2 residual_values = __bfloat1622float2(residual_pairs[pair]);
      const float d_output_slot[2] = {d_output_values.x, d_output_values.y};
      const float corrected_slot[2] = {
          __fadd_rn(output_values.x, residual_values.x),
          __fadd_rn(output_values.y, residual_values.y)};
#pragma unroll
      for (int element = 0; element < 2; ++element) {
        const int slot = pair * 2 + element;
        if (RowLanes == 8 && row_lane >= 4) {
          upper_d_output[slot] = d_output_slot[element];
          upper_corrected_output[slot] = corrected_slot[element];
        } else {
          partial[slot] = __fmaf_rn(d_output_slot[element], corrected_slot[element], 0.0F);
        }
      }
    }
  }

  // C. Bit 5 (8-lane rows only): lane j+4 applies channel c+32 to the
  // partial of channel c from lane j. Lanes 0-3 keep values that no longer
  // matter; the XOR 2 and XOR 1 shuffles below never mix the two halves.
  if constexpr (RowLanes == 8) {
#pragma unroll
    for (int slot = 0; slot < kChunkChannels; ++slot) {
      const float lower_partial = __shfl_xor_sync(0xFFFFFFFFU, partial[slot], 4, kWarpSize);
      if (row_lane >= 4) {
        partial[slot] = chunk_exists
                            ? __fmaf_rn(upper_d_output[slot], upper_corrected_output[slot], lower_partial)
                            : lower_partial;
      }
    }
  }

  // D. Bit 4: add the chunk 16 channels away. In 2-lane rows that chunk
  // does not exist and contributes +0, as an empty lane does in the
  // warp-per-row tree.
#pragma unroll
  for (int slot = 0; slot < kChunkChannels; ++slot) {
    const float partner =
        RowLanes == 2 ? 0.0F : __shfl_xor_sync(0xFFFFFFFFU, partial[slot], 2, kWarpSize);
    partial[slot] = __fadd_rn(partial[slot], partner);
  }

  // E. Bit 3: add the neighbouring chunk (8 channels away).
#pragma unroll
  for (int slot = 0; slot < kChunkChannels; ++slot) {
    partial[slot] =
        __fadd_rn(partial[slot], __shfl_xor_sync(0xFFFFFFFFU, partial[slot], 1, kWarpSize));
  }

  // F. Bits 2, 1, 0: finish inside the lane with slot pairs 4, 2, then 1 apart.
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    partial[slot] = __fadd_rn(partial[slot], partial[slot + 4]);
  }
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    partial[slot] = __fadd_rn(partial[slot], partial[slot + 2]);
  }
  const float row_sum = __fadd_rn(partial[0], partial[1]);

  // G. One lane per row writes delta at its [B, O, H, N] position.
  if (row_is_valid && row_lane == kWriterLane) {
    const int64_t head_index = memory_row % heads;
    const int64_t batch_outer_query = memory_row / heads;
    const int64_t query_index = batch_outer_query % sequence;
    const int64_t batch_outer_index = batch_outer_query / sequence;
    delta[(batch_outer_index * heads + head_index) * sequence + query_index] = row_sum;
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
  const c10::cuda::CUDAGuard device_guard(d_output.device());
  // This entry point launches a shadow kernel directly on the current stream.
  auto stream = at::cuda::getCurrentCUDAStream();

  // Prefer the vectorized kernel when every lane can read whole, aligned
  // 16-byte chunks. It produces the same bits as the warp-per-row kernel.
  const auto is_16_byte_aligned = [](const torch::Tensor& tensor) {
    return reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0;
  };
  const bool vectorized = head_dim % 8 == 0 && head_dim <= 64 && is_16_byte_aligned(d_output)
                          && is_16_byte_aligned(output) && is_16_byte_aligned(shadow_residual);
  if (vectorized) {
    const int row_lanes = head_dim <= 16 ? 2 : (head_dim <= 32 ? 4 : 8);
    const int64_t rows_per_block = kThreads / row_lanes;
    const int64_t vectorized_blocks = (rows + rows_per_block - 1) / rows_per_block;
    TORCH_CHECK(
        vectorized_blocks <= kMaxGridX, "delta launch exceeds the CUDA grid-x contract");
    const auto* d_output_data = reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr());
    const auto* output_data = reinterpret_cast<const __nv_bfloat16*>(output.data_ptr());
    const auto* residual_data = reinterpret_cast<const __nv_bfloat16*>(shadow_residual.data_ptr());
    const auto launch = [&](auto kernel) {
      kernel<<<static_cast<unsigned int>(vectorized_blocks), kThreads, 0, stream>>>(
          d_output_data, output_data, residual_data, delta.data_ptr<float>(), rows, sequence,
          heads, head_dim);
    };
    if (row_lanes == 2) {
      launch(turbopairformer_triangle_delta_shadow_vectorized_kernel<2>);
    } else if (row_lanes == 4) {
      launch(turbopairformer_triangle_delta_shadow_vectorized_kernel<4>);
    } else {
      launch(turbopairformer_triangle_delta_shadow_vectorized_kernel<8>);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
    return;
  }

  // Otherwise keep the classic kernel's eight-row block mapping.
  const int64_t blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  TORCH_CHECK(blocks <= kMaxGridX, "delta launch exceeds the CUDA grid-x contract");
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
