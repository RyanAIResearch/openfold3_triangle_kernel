// TurboPairFormer triangle multiplication: fused projections and backward helpers.
//
// The Python facade performs the triangle matrix contraction with torch.bmm
// (cuBLAS on the CUDA path). This file fuses the operations before and after it:
//   trimul_in_fwd  : LayerNorm(z), five projections, masked A/B gates.
//   trimul_out_fwd : LayerNorm(contraction result), output projection, final gate.
// Backward helpers recompute projection values, differentiate the gates and
// LayerNorm, and fold weight-gradient partials in a fixed chunk order.
//
// Layout names used throughout this file:
//   B = flattened batch, NN = N*N pair positions, rows = B*NN.
//   C = input/output channels; H = projected channels. Projection entry points
//       require C == H and support 32, 64, 96, or 128 channels.
//   z, zln, gate, output : [rows, channels], contiguous BF16.
//   A, B, contraction x : [B*H, NN], channel-major BF16.
//   d_proj5             : [rows, 5*H], ordered ap/ag/bp/bg/final-gate slots.
// Weights and LayerNorm parameters enter as FP32. Projection kernels round
// weights and normalized activations to BF16, accumulate MMA results in FP32,
// and round stored activation/gradient values to BF16. LayerNorm statistics
// and parameter-gradient partials use FP32. Keep all reduction orders and
// rounding points when changing these stages.
//
// Reading guide:
//   - Each public wrapper validates layouts and launches
//     on the current CUDA stream with the input's device selected.
//   - trimul_in_fwd_kernel and trimul_out_fwd_kernel describe the forward path.
//   - trimul_gate_split_bwd_kernel reconstructs only the four A/B projections;
//     sig_mul_bwd_out_kernel writes the fifth projection's gradient.
//   - LayerNorm backward and fixed folds are separate kernels below.

#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>

using bf16 = __nv_bfloat16;

#define WARP 32
#define FULL 0xffffffffu

namespace {

// -----------------------------------------------------------------------------
// Host validation: sizes, layouts, devices, overlap, and launch capacity
// -----------------------------------------------------------------------------

constexpr int64_t kMaxGridX = 2147483647LL;
constexpr int64_t kMaxGridYZ = 65535LL;
constexpr int kTrimulInMaxDynamicSharedBytes = 105984;
constexpr int kTrimulOutMaxDynamicSharedBytes = 105472;
constexpr int kTrimulGateBwdMaxDynamicSharedBytes = 105472;

int64_t checked_positive(int64_t value, const char* op, const char* name) {
  TORCH_CHECK(value > 0, op, ": ", name, " must be positive, got ", value);
  return value;
}

int64_t checked_mul(int64_t lhs, int64_t rhs, const char* op, const char* expression) {
  TORCH_CHECK(lhs >= 0 && rhs >= 0, op, ": ", expression, " operands must be non-negative");
  TORCH_CHECK(
      lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs, op, ": ", expression,
      " overflows int64");
  return lhs * rhs;
}

int64_t ceil_div_positive(int64_t value, int64_t divisor) {
  return 1 + (value - 1) / divisor;
}

void check_channel(int channel, const char* op, const char* name) {
  TORCH_CHECK(
      channel == 32 || channel == 64 || channel == 96 || channel == 128, op, ": ", name,
      " must be one of {32, 64, 96, 128}, got ", channel);
}

void check_channels(int C, int H, const char* op) {
  check_channel(C, op, "C");
  check_channel(H, op, "H");
  TORCH_CHECK(C == H, op, ": C and H must be equal, got C=", C, ", H=", H);
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    const char* op,
    const char* name) {
  TORCH_CHECK(tensor.defined(), op, ": ", name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), op, ": ", name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, op, ": ", name, " has the wrong dtype");
}

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    const char* op,
    const char* name) {
  check_cuda_tensor(tensor, dtype, op, name);
  TORCH_CHECK(tensor.is_contiguous(), op, ": ", name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& primary,
    const char* op,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == primary.device(), op, ": ", name,
      " must be on the same CUDA device as the primary input");
}

void check_shape(
    const torch::Tensor& tensor,
    std::initializer_list<int64_t> expected,
    const char* op,
    const char* name) {
  TORCH_CHECK(
      tensor.dim() == static_cast<int64_t>(expected.size()), op, ": ", name,
      " has wrong rank; got shape ", tensor.sizes());
  int64_t dimension = 0;
  for (int64_t extent : expected) {
    TORCH_CHECK(
        tensor.size(dimension) == extent, op, ": ", name, " has wrong shape; got ", tensor.sizes());
    ++dimension;
  }
}

void check_alignment(
    const torch::Tensor& tensor,
    uintptr_t alignment,
    const char* op,
    const char* name) {
  auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TORCH_CHECK(
      address % alignment == 0, op, ": ", name, " data pointer must be ", alignment,
      "-byte aligned");
}

int64_t check_pair_extent(int64_t B, int64_t NN, int H, const char* op) {
  checked_positive(B, op, "B");
  checked_positive(NN, op, "NN");
  TORCH_CHECK(B <= kMaxGridYZ, op, ": B exceeds the production grid-z limit");
  int64_t rows = checked_mul(B, NN, op, "B*NN");
  int64_t values = checked_mul(rows, H, op, "B*NN*H");
  TORCH_CHECK(
      ceil_div_positive(rows, 8) <= kMaxGridX, op,
      ": LayerNorm grid-x exceeds the production limit");
  TORCH_CHECK(
      ceil_div_positive(values, 256) <= kMaxGridX, op,
      ": sigmoid grid-x exceeds the production limit");
  int64_t transpose_tiles = ceil_div_positive(NN, 32);
  TORCH_CHECK(transpose_tiles <= kMaxGridX, op, ": transpose grid-x exceeds the production limit");
  TORCH_CHECK(
      checked_mul(transpose_tiles, B, op, "ceil(NN/32)*B") <= kMaxGridX, op,
      ": transpose reduction chunks exceed signed int32");
  return rows;
}

void check_no_overlap(
    const torch::Tensor& output,
    std::initializer_list<const torch::Tensor*> inputs) {
  at::assert_no_internal_overlap(output);
  for (const torch::Tensor* input : inputs)
    at::assert_no_overlap(output, *input);
}

void check_dynamic_shared_memory_limit(int required_bytes, int device, const char* op) {
  int optin_limit = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(
      required_bytes <= optin_limit, op, ": requires ", required_bytes,
      " bytes of dynamic shared memory, but device limit is ", optin_limit);
}

}  // namespace

// Down-shuffle reduction followed by lane-0 broadcast. LayerNorm backward
// uses this tree; the forward butterfly helper below has a different tree.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL, v, off);
  return __shfl_sync(FULL, v, 0);
}

// Fixed-order fold for FP32 split-K weight-gradient partials.  Each thread
// exclusively owns one output element and consumes logical chunks in ascending
// order.  The accumulator is fresh on the first wave and carries the exact
// ordered prefix across later waves; no atomics or inter-CTA reduction occur.
template <bool Initialize>
__global__ void trimul_wgrad_fixed_fold_kernel(
    const float* __restrict__ partials,
    int chunks,
    long elements,
    float* __restrict__ accumulator) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= elements)
    return;
  float sum = 0.f;
  if constexpr (!Initialize) {
    sum = accumulator[idx];
  }
#pragma unroll 1
  for (int chunk = 0; chunk < chunks; ++chunk) {
    sum = __fadd_rn(sum, partials[(long)chunk * elements + idx]);
  }
  accumulator[idx] = sum;
}

// -----------------------------------------------------------------------------
// LayerNorm backward: input gradient and fixed-order parameter reductions
// -----------------------------------------------------------------------------
// One warp owns one row. Sum g and g*x_hat over channels, where
// g = grad_out * gamma, then dx = (g - mean(g) - x_hat*mean(g*x_hat))*rstd.
// The public wrapper requires the contiguous row_stride=C, col_stride=1 layout.

__global__ void ln_bwd_dx_kernel(
    const bf16* __restrict__ input,
    const bf16* __restrict__ grad_out,
    long rows,
    int C,
    long row_stride,
    long col_stride,
    const float* __restrict__ gamma,
    const float* __restrict__ save_mean,
    const float* __restrict__ save_rstd,
    bf16* __restrict__ grad_in) {
  long row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
  if (row >= rows)
    return;
  int lane_id = threadIdx.x % WARP;
  const bf16* x = input + row * row_stride;
  const bf16* dy = grad_out + row * (long)C;
  float mean = save_mean[row], rstd = save_rstd[row];
  float mean_scaled_gradient = 0.f, mean_scaled_gradient_times_normalized = 0.f;
  for (int c = lane_id; c < C; c += WARP) {
    float scaled_gradient = __bfloat162float(dy[c]) * gamma[c];
    float normalized_value = (__bfloat162float(x[c * col_stride]) - mean) * rstd;
    mean_scaled_gradient += scaled_gradient;
    mean_scaled_gradient_times_normalized += scaled_gradient * normalized_value;
  }
  mean_scaled_gradient = warp_sum(mean_scaled_gradient) / C;
  mean_scaled_gradient_times_normalized = warp_sum(mean_scaled_gradient_times_normalized) / C;
  bf16* dx = grad_in + row * row_stride;  // same layout as input
  for (int c = lane_id; c < C; c += WARP) {
    float scaled_gradient = __bfloat162float(dy[c]) * gamma[c];
    float normalized_value = (__bfloat162float(x[c * col_stride]) - mean) * rstd;
    dx[c * col_stride] = __float2bfloat16(
        (scaled_gradient - mean_scaled_gradient
         - normalized_value * mean_scaled_gradient_times_normalized)
        * rstd);
  }
}

// -------- LN backward: dgamma/dbeta, two-stage deterministic reduction ------
// Stage 1: each thread owns one channel in a row chunk and visits rows in order.
__global__ void ln_bwd_dgb_partial_kernel(
    const bf16* __restrict__ input,
    const bf16* __restrict__ grad_out,
    long rows,
    int C,
    long row_stride,
    long col_stride,
    long chunk_rows,
    const float* __restrict__ save_mean,
    const float* __restrict__ save_rstd,
    float* __restrict__ partial) {  // [chunks, 2, C]
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int chunk = blockIdx.y;
  if (c >= C)
    return;
  long row0 = (long)chunk * chunk_rows;
  long row1 = min(rows, row0 + chunk_rows);
  float gamma_gradient_sum = 0.f, beta_gradient_sum = 0.f;
  for (long row = row0; row < row1; ++row) {  // fixed ascending order
    float dy = __bfloat162float(grad_out[row * (long)C + c]);
    float normalized_value =
        (__bfloat162float(input[row * row_stride + c * col_stride]) - save_mean[row])
        * save_rstd[row];
    gamma_gradient_sum += dy * normalized_value;
    beta_gradient_sum += dy;
  }
  partial[((long)chunk * 2 + 0) * C + c] = gamma_gradient_sum;
  partial[((long)chunk * 2 + 1) * C + c] = beta_gradient_sum;
}

// Optional grouped fold: each thread owns one channel and folds its assigned
// contiguous chunk range. transpose_ln_bwd uses this extra level before the root.
__global__ void ln_bwd_dgb_group_fold_kernel(
    const float* __restrict__ partial,
    int chunks,
    int C,
    int group_len,
    float* __restrict__ out) {  // out: [groups*2, C]
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int group = blockIdx.y;
  if (c >= C)
    return;
  int k0 = group * group_len;
  int k1 = min(chunks, k0 + group_len);
  float gamma_gradient_sum = 0.f, beta_gradient_sum = 0.f;
  for (int k = k0; k < k1; ++k) {  // fixed ascending order
    gamma_gradient_sum += partial[((long)k * 2 + 0) * C + c];
    beta_gradient_sum += partial[((long)k * 2 + 1) * C + c];
  }
  out[((long)group * 2 + 0) * C + c] = gamma_gradient_sum;
  out[((long)group * 2 + 1) * C + c] = beta_gradient_sum;
}

// stage 2: fold chunks in fixed ascending order
__global__ void ln_bwd_dgb_fold_kernel(
    const float* __restrict__ partial,
    int chunks,
    int C,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  float gamma_gradient_sum = 0.f, beta_gradient_sum = 0.f;
  for (int k = 0; k < chunks; ++k) {  // fixed ascending order
    gamma_gradient_sum += partial[((long)k * 2 + 0) * C + c];
    beta_gradient_sum += partial[((long)k * 2 + 1) * C + c];
  }
  dgamma[c] = gamma_gradient_sum;
  dbeta[c] = beta_gradient_sum;
}

// -----------------------------------------------------------------------------
// LayerNorm backward for a channel-major contraction result
// -----------------------------------------------------------------------------
// Read x from [B*H, NN] and d_xln from [B*NN, H]. Shared tiles let warps
// normalize across H while preserving coalesced channel-major reads/writes.
// The second tile first holds dx, then is reused for dgamma/dbeta partials.

__global__ void transpose_ln_bwd_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ d_xln,
    long NN,
    int H,
    const float* __restrict__ gamma,
    const float* __restrict__ save_mean,
    const float* __restrict__ save_rstd,
    bf16* __restrict__ d_x,
    float* __restrict__ dgb_partial) {
  // A. Stage 32 pair positions for every channel; the extra column pads rows.
  extern __shared__ float shared_ln_storage[];  // two tiles: x[H][33], dx[H][33]
  float* shared_input_values = shared_ln_storage;
  float* shared_gradient_values = shared_ln_storage + (long)H * 33;
  long batch_index = blockIdx.z;
  long tile_position_start = (long)blockIdx.x * 32;
  int thread_id = threadIdx.x;
  int lane_id = thread_id % 32, warp_id = thread_id / 32;
  for (int h = warp_id; h < H; h += 8) {
    long row = (batch_index * H + h) * NN + tile_position_start;
    long left = NN - tile_position_start;
    shared_input_values[h * 33 + lane_id] =
        (left >= 32 || lane_id < left) ? __bfloat162float(x[row + lane_id]) : 0.f;
  }
  __syncthreads();
  // B. Each warp computes channel reductions for its assigned pair positions.
  for (int tile_position = warp_id; tile_position < 32; tile_position += 8) {
    long pos = tile_position_start + tile_position;
    if (pos >= NN)
      continue;
    long row = batch_index * NN + pos;
    float mean = save_mean[row], rstd = save_rstd[row];
    const bf16* dy = d_xln + row * (long)H;
    float mean_scaled_gradient = 0.f, mean_scaled_gradient_times_normalized = 0.f;
    for (int h = lane_id; h < H; h += 32) {
      float scaled_gradient = __bfloat162float(dy[h]) * gamma[h];
      float normalized_value = (shared_input_values[h * 33 + tile_position] - mean) * rstd;
      mean_scaled_gradient += scaled_gradient;
      mean_scaled_gradient_times_normalized += scaled_gradient * normalized_value;
    }
    mean_scaled_gradient = warp_sum(mean_scaled_gradient) / H;
    mean_scaled_gradient_times_normalized = warp_sum(mean_scaled_gradient_times_normalized) / H;
    for (int h = lane_id; h < H; h += 32) {
      float scaled_gradient = __bfloat162float(dy[h]) * gamma[h];
      float normalized_value = (shared_input_values[h * 33 + tile_position] - mean) * rstd;
      shared_gradient_values[h * 33 + tile_position] =
          (scaled_gradient - mean_scaled_gradient
           - normalized_value * mean_scaled_gradient_times_normalized)
          * rstd;
    }
  }
  __syncthreads();
  // C. Publish dx and write it back in the original channel-major layout.
  for (int h = warp_id; h < H; h += 8) {
    long row = (batch_index * H + h) * NN + tile_position_start;
    long left = NN - tile_position_start;
    if (left >= 32 || lane_id < left)
      d_x[row + lane_id] = __float2bfloat16(shared_gradient_values[h * 33 + lane_id]);
  }
  // D. Reuse the dx tile only after all its readers finish. Two active threads
  // per channel visit positions 0..15 and 16..31, then combine in that order.
  __syncthreads();
  int owned_channel = thread_id % H;
  int half = thread_id / H;  // Only halves 0 and 1 participate; other threads idle.
  if (half >= 2) {           /* Threads beyond 2H are idle for parameter gradients. */
  } else {
    long left = NN - tile_position_start;
    int limit = left < 32 ? (int)left : 32;
    float partial_gamma_sum = 0.f, partial_beta_sum = 0.f;
    long row_base = batch_index * NN + tile_position_start;
    for (int tile_position = half * 16; tile_position < min(limit, half * 16 + 16);
         ++tile_position) {
      long row = row_base + tile_position;
      float dy = __bfloat162float(d_xln[row * (long)H + owned_channel]);
      float normalized_value =
          (shared_input_values[owned_channel * 33 + tile_position] - save_mean[row])
          * save_rstd[row];
      partial_gamma_sum += dy * normalized_value;
      partial_beta_sum += dy;
    }
    // Two [2,H] regions: gamma partials first, then beta partials.
    shared_gradient_values[half * H + owned_channel] = partial_gamma_sum;
    shared_gradient_values[2 * H + half * H + owned_channel] = partial_beta_sum;
  }
  __syncthreads();
  if (half == 0) {
    long block_id = (long)blockIdx.z * gridDim.x + blockIdx.x;
    dgb_partial[(block_id * 2 + 0) * H + owned_channel] =
        shared_gradient_values[owned_channel] + shared_gradient_values[H + owned_channel];
    dgb_partial[(block_id * 2 + 1) * H + owned_channel] =
        shared_gradient_values[2 * H + owned_channel]
        + shared_gradient_values[3 * H + owned_channel];
  }
}

// -----------------------------------------------------------------------------
// Forward reduction and matrix-instruction helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(FULL, v, off);
  return v;  // butterfly: all lanes hold sum
}

// A warp computes a 16x8 output tile over 16 channels. Each thread owns
// four FP32 accumulator values; A/B registers each pack two BF16 operands.
__device__ __forceinline__ void mma_16816(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    unsigned a0,
    unsigned a1,
    unsigned a2,
    unsigned a3,
    unsigned b0,
    unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0,
    unsigned& r1,
    unsigned& r2,
    unsigned& r3,
    const bf16* p) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// ---------------------------------------------------------------------------
// Input forward: LayerNorm + five projections + masked A/B gates.
// One block owns a 32-channel slice and repeatedly processes 128 pair rows.
// The five raw FP32 [H,C] weights are staged once as a permuted BF16 slice:
//   column = half*80 + projection*16 + channel_in_half,
//   projection order = ap, ag, bp, bg, final-gate logits.
// Each warp computes 32 rows x 80 projection columns. Its accumulator array
// indexes [16-row half][8-column fragment][four values owned by this thread].
// A/B are stored channel-major; final gate logits retain the row-major layout.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_in_fwd_kernel(
    const bf16* __restrict__ z,
    const float* __restrict__ w_ap,
    const float* __restrict__ w_ag,
    const float* __restrict__ w_bp,
    const float* __restrict__ w_bg,
    const float* __restrict__ w_g,
    const float* __restrict__ mask,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    long rows,
    long NN,
    int C,
    int H,
    bf16* __restrict__ a_out,
    bf16* __restrict__ b_out,
    bf16* __restrict__ gate_out,
    float* __restrict__ save_mean,
    float* __restrict__ save_rstd,
    bf16* __restrict__ zln_out) {
  // Shared layout: resident weights | normalized Z | A stage | B stage | gate.
  // A/B staging transposes the register results for channel-major stores.
  extern __shared__ bf16 shared_storage[];
  const int input_row_stride = C + 8;     // padded k stride
  bf16* shared_weights = shared_storage;  // [160, input_row_stride]
  bf16* shared_normalized_input =
      shared_storage + 160 * input_row_stride;  // [128, input_row_stride]
  bf16* shared_a_tile = shared_normalized_input + 128 * input_row_stride;  // [32 ch][136 rows]
  bf16* shared_b_tile = shared_a_tile + 32 * 136;                          // [32 ch][136 rows]
  bf16* shared_gate_logits = shared_b_tile + 32 * 136;                     // [128 rows][40 ch-pad]

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id >> 5, lane_id = thread_id & 31;
  const int channel_slice = blockIdx.x;

  // A. Round and stage this block's five projection slices once.
  {
    const float* wsrc[5] = {w_ap, w_ag, w_bp, w_bg, w_g};
    int chunks_per_col = C >> 3;  // 8 values per uint4 store
    for (int i = thread_id; i < 160 * chunks_per_col; i += 256) {
      int col = i / chunks_per_col, q = i % chunks_per_col;
      int wcg = col / 80, rem = col % 80;
      const float* src =
          wsrc[rem >> 4] + (long)(channel_slice * 32 + wcg * 16 + (rem & 15)) * C + q * 8;
      union {
        bf16 b[8];
        uint4 u;
      } packed_weights;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        packed_weights.b[t] = __float2bfloat16(src[t]);
      *reinterpret_cast<uint4*>(shared_weights + col * input_row_stride + q * 8) = packed_weights.u;
    }
  }
  const int lane_channel_start = lane_id * 4;
  const bool has_channels = lane_channel_start < C;
  const float inverse_channels = 1.0f / (float)C;

  // ---- persistent tile loop ----
  for (long tile = blockIdx.y; tile * 128 < rows; tile += gridDim.y) {
    const long row0 = tile * 128;

    // B. Load this warp's rows, compute FP32 LayerNorm statistics, and stage
    // normalized BF16 Z. Only channel slice 0 writes the optional saved Z.
    {
      float gamma_values[4] = {0.f, 0.f, 0.f, 0.f}, beta_values[4] = {0.f, 0.f, 0.f, 0.f};
      if (has_channels) {
        float4 gv = *reinterpret_cast<const float4*>(ln_gamma + lane_channel_start);
        float4 bv = *reinterpret_cast<const float4*>(ln_beta + lane_channel_start);
        gamma_values[0] = gv.x;
        gamma_values[1] = gv.y;
        gamma_values[2] = gv.z;
        gamma_values[3] = gv.w;
        beta_values[0] = bv.x;
        beta_values[1] = bv.y;
        beta_values[2] = bv.z;
        beta_values[3] = bv.w;
      }
      // Load all 16 rows before normalizing them so their memory waits can
      // overlap; preserve this batched-load schedule when editing the loop.
      uint2 raw[16];
#pragma unroll
      for (int it = 0; it < 16; ++it) {
        long row = row0 + warp_id * 16 + it;
        raw[it] = (row < rows && has_channels)
                      ? *reinterpret_cast<const uint2*>(z + row * (long)C + lane_channel_start)
                      : make_uint2(0, 0);
      }
#pragma unroll
      for (int it = 0; it < 16; ++it) {
        int r = warp_id * 16 + it;
        long row = row0 + r;
        float v[4];
        const bf16* pv = reinterpret_cast<const bf16*>(&raw[it]);
#pragma unroll
        for (int t = 0; t < 4; ++t)
          v[t] = __bfloat162float(pv[t]);
        float sum = warp_sum32(v[0] + v[1] + v[2] + v[3]);
        float sq = warp_sum32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        float mean = sum * inverse_channels;
        float rstd = rsqrtf(sq * inverse_channels - mean * mean + 1e-5f);
        if (lane_id == 0 && channel_slice == 0 && row < rows) {
          save_mean[row] = mean;
          save_rstd[row] = rstd;
        }
        if (has_channels) {
          bool live = row < rows;
          __nv_bfloat162 p0 = __floats2bfloat162_rn(
              live ? (v[0] - mean) * rstd * gamma_values[0] + beta_values[0] : 0.f,
              live ? (v[1] - mean) * rstd * gamma_values[1] + beta_values[1] : 0.f);
          __nv_bfloat162 p1 = __floats2bfloat162_rn(
              live ? (v[2] - mean) * rstd * gamma_values[2] + beta_values[2] : 0.f,
              live ? (v[3] - mean) * rstd * gamma_values[3] + beta_values[3] : 0.f);
          uint2 packed;
          packed.x = *reinterpret_cast<unsigned*>(&p0);
          packed.y = *reinterpret_cast<unsigned*>(&p1);
          *reinterpret_cast<uint2*>(
              shared_normalized_input + r * input_row_stride + lane_channel_start) = packed;
          if (zln_out != nullptr && channel_slice == 0 && live) {
            *reinterpret_cast<uint2*>(zln_out + row * (long)C + lane_channel_start) = packed;
          }
        }
      }
    }
    __syncthreads();  // S1: normalized Z and the resident weight slice are ready.

    // C. Multiply normalized Z by the five weights, accumulating in FP32.
    const int warp_row_group = warp_id & 3, warp_column_group = warp_id >> 2;
    const int warp_row_start = warp_row_group * 32;
    float projection_accumulators[2][10][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 10; ++j)
        projection_accumulators[i][j][0] = projection_accumulators[i][j][1] =
            projection_accumulators[i][j][2] = projection_accumulators[i][j][3] = 0.f;

    const bf16* za0 = shared_normalized_input + (warp_row_start + (lane_id & 15)) * input_row_stride
                      + 8 * (lane_id >> 4);
    const bf16* za1 = za0 + 16 * input_row_stride;
    const bf16* wb0 =
        shared_weights
        + (warp_column_group * 80 + (lane_id >> 4) * 8 + (lane_id & 7)) * input_row_stride
        + 8 * ((lane_id >> 3) & 1);
    auto gemm_step = [&](int k0) {
      unsigned input_fragments[2][4];
      ldmatrix_x4(
          input_fragments[0][0], input_fragments[0][1], input_fragments[0][2],
          input_fragments[0][3], za0 + k0);
      ldmatrix_x4(
          input_fragments[1][0], input_fragments[1][1], input_fragments[1][2],
          input_fragments[1][3], za1 + k0);
#pragma unroll
      for (int jp = 0; jp < 5; ++jp) {  // B: two 16x8 tiles per x4
        unsigned b00, b01, b10, b11;
        ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * input_row_stride + k0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          mma_16816(
              projection_accumulators[i][jp * 2][0], projection_accumulators[i][jp * 2][1],
              projection_accumulators[i][jp * 2][2], projection_accumulators[i][jp * 2][3],
              input_fragments[i][0], input_fragments[i][1], input_fragments[i][2],
              input_fragments[i][3], b00, b01);
          mma_16816(
              projection_accumulators[i][jp * 2 + 1][0], projection_accumulators[i][jp * 2 + 1][1],
              projection_accumulators[i][jp * 2 + 1][2], projection_accumulators[i][jp * 2 + 1][3],
              input_fragments[i][0], input_fragments[i][1], input_fragments[i][2],
              input_fragments[i][3], b10, b11);
        }
      }
    };
    if (C == 128) {  // This branch has a fixed eight-step unrolled contraction.
#pragma unroll
      for (int k0 = 0; k0 < 128; k0 += 16)
        gemm_step(k0);
    } else {
      for (int k0 = 0; k0 < C; k0 += 16)
        gemm_step(k0);
    }
    // S2 below finishes all normalized-Z reads before the next iteration
    // can replace Z. S1 already finished the previous output-stage readers.

    // D. Form A = ap*sigmoid(ag)*mask and B = bp*sigmoid(bg)*mask.
    // Round A/B and the raw final-gate logits to BF16 in shared staging.
    const int n0 = (lane_id & 3) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int r_lo = warp_row_start + i * 16 + (lane_id >> 2);
      long row_lo = row0 + r_lo, row_hi = row_lo + 8;
      float m_lo = row_lo < rows ? mask[row_lo] : 0.f;
      float m_hi = row_hi < rows ? mask[row_hi] : 0.f;
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int channel_in_slice = warp_column_group * 16 + jp * 8 + n0 + q;
          float ap0 = projection_accumulators[i][jp][q],
                ag0 = projection_accumulators[i][2 + jp][q];
          float bp0 = projection_accumulators[i][4 + jp][q],
                bg0 = projection_accumulators[i][6 + jp][q];
          float g0 = projection_accumulators[i][8 + jp][q];
          float ap1 = projection_accumulators[i][jp][2 + q],
                ag1 = projection_accumulators[i][2 + jp][2 + q];
          float bp1 = projection_accumulators[i][4 + jp][2 + q],
                bg1 = projection_accumulators[i][6 + jp][2 + q];
          float g1 = projection_accumulators[i][8 + jp][2 + q];
          shared_a_tile[channel_in_slice * 136 + r_lo] =
              __float2bfloat16(__fdividef(ap0, 1.f + __expf(-ag0)) * m_lo);
          shared_b_tile[channel_in_slice * 136 + r_lo] =
              __float2bfloat16(__fdividef(bp0, 1.f + __expf(-bg0)) * m_lo);
          shared_gate_logits[r_lo * 40 + channel_in_slice] = __float2bfloat16(g0);
          shared_a_tile[channel_in_slice * 136 + r_lo + 8] =
              __float2bfloat16(__fdividef(ap1, 1.f + __expf(-ag1)) * m_hi);
          shared_b_tile[channel_in_slice * 136 + r_lo + 8] =
              __float2bfloat16(__fdividef(bp1, 1.f + __expf(-bg1)) * m_hi);
          shared_gate_logits[(r_lo + 8) * 40 + channel_in_slice] = __float2bfloat16(g1);
        }
      }
    }

    __syncthreads();  // S2: all staged outputs are ready for cooperative stores.

    // E. Write A/B in channel-major order and gate logits in row-major order.
    // Advance batch indices when a 128-row tile crosses an NN boundary.
    // Vector stores require a full aligned span inside one batch plane;
    // scalar stores handle short tails and any intervening batch boundaries.
    // Derive the tile's batch/position once, then advance at NN boundaries
    // instead of repeating a 64-bit division for each group of stores.
    const long tile_batch_start = row0 / NN;
    const long tile_position_start = row0 - tile_batch_start * NN;
    for (int id = thread_id; id < 32 * 16; id += 256) {
      int channel_in_slice = id >> 4, r = (id & 15) * 8;
      int h = channel_slice * 32 + channel_in_slice;
      long pos = tile_position_start + r;
      long bidx = tile_batch_start;
      while (pos >= NN) {
        pos -= NN;
        ++bidx;
      }
      long addr = (bidx * H + h) * NN + pos;
      bool fast = (row0 + r + 7 < rows) && (pos + 7 < NN) && ((addr & 7) == 0);
      if (fast) {
        *reinterpret_cast<uint4*>(a_out + addr) =
            *reinterpret_cast<const uint4*>(shared_a_tile + channel_in_slice * 136 + r);
        *reinterpret_cast<uint4*>(b_out + addr) =
            *reinterpret_cast<const uint4*>(shared_b_tile + channel_in_slice * 136 + r);
      } else {
        for (int t = 0; t < 8; ++t) {
          if (row0 + r + t >= rows)
            break;
          long p = pos + t;
          long bi = bidx;
          while (p >= NN) {
            p -= NN;
            ++bi;
          }
          long ad = (bi * H + h) * NN + p;
          a_out[ad] = shared_a_tile[channel_in_slice * 136 + r + t];
          b_out[ad] = shared_b_tile[channel_in_slice * 136 + r + t];
        }
      }
    }
    for (int id = thread_id; id < 128 * 4; id += 256) {
      int r = id >> 2, part = id & 3;
      long row = row0 + r;
      if (row >= rows)
        continue;
      *reinterpret_cast<uint4*>(gate_out + row * (long)H + channel_slice * 32 + part * 8) =
          *reinterpret_cast<const uint4*>(shared_gate_logits + r * 40 + part * 8);
    }
    // The next iteration's S1 finishes these output-stage reads before its
    // epilogue overwrites the same staging buffers; no tail barrier is needed.
  }
}

std::vector<torch::Tensor> trimul_in_fwd(
    torch::Tensor z,
    torch::Tensor w_ap,
    torch::Tensor w_ag,
    torch::Tensor w_bp,
    torch::Tensor w_bg,
    torch::Tensor w_g,
    torch::Tensor mask,
    torch::Tensor gamma,
    torch::Tensor beta,
    long B,
    long NN,
    int C,
    int H,
    bool save_zln) {
  constexpr const char* op = "trimul_in_fwd";
  check_channels(C, H, op);
  int64_t rows = check_pair_extent(B, NN, H, op);
  check_cuda_contiguous(z, at::kBFloat16, op, "z");
  check_shape(z, {rows, C}, op, "z");
  check_alignment(z, 8, op, "z");
  for (auto named_weight :
       {std::pair<const char*, const torch::Tensor*>("w_ap", &w_ap),
        std::pair<const char*, const torch::Tensor*>("w_ag", &w_ag),
        std::pair<const char*, const torch::Tensor*>("w_bp", &w_bp),
        std::pair<const char*, const torch::Tensor*>("w_bg", &w_bg),
        std::pair<const char*, const torch::Tensor*>("w_g", &w_g)}) {
    check_cuda_contiguous(*named_weight.second, at::kFloat, op, named_weight.first);
    check_same_device(*named_weight.second, z, op, named_weight.first);
    check_shape(*named_weight.second, {H, C}, op, named_weight.first);
  }
  check_cuda_contiguous(mask, at::kFloat, op, "mask");
  check_same_device(mask, z, op, "mask");
  check_shape(mask, {rows}, op, "mask");
  TORCH_CHECK(!mask.requires_grad(), op, ": mask must not require gradients");
  for (auto named_parameter :
       {std::pair<const char*, const torch::Tensor*>("gamma", &gamma),
        std::pair<const char*, const torch::Tensor*>("beta", &beta)}) {
    check_cuda_contiguous(*named_parameter.second, at::kFloat, op, named_parameter.first);
    check_same_device(*named_parameter.second, z, op, named_parameter.first);
    check_shape(*named_parameter.second, {C}, op, named_parameter.first);
    check_alignment(*named_parameter.second, 16, op, named_parameter.first);
  }
  const c10::cuda::CUDAGuard device_guard(z.device());
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(z.device());
  auto a = torch::empty({B * (long)H, NN}, opts);
  auto b = torch::empty_like(a);
  auto gate = torch::empty({rows, (long)H}, opts);
  auto fopts = torch::TensorOptions().dtype(torch::kFloat).device(z.device());
  auto mean = torch::empty({rows}, fopts);
  auto rstd = torch::empty({rows}, fopts);
  auto zln = torch::empty({save_zln ? rows : 0, (long)C}, opts);
  int input_row_stride = C + 8;
  size_t shared_bytes =
      ((size_t)160 * input_row_stride + 128 * input_row_stride + 2 * 32 * 136 + 128 * 40)
      * sizeof(bf16);
  int device = z.get_device();
  check_dynamic_shared_memory_limit(kTrimulInMaxDynamicSharedBytes, device, op);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      trimul_in_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kTrimulInMaxDynamicSharedBytes));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(sm_count > 0, op, ": device reports no streaming multiprocessors");
  int S = H / 32;
  long y = std::max(1L, std::min(tiles, (long)(2 * sm_count) / S));
  TORCH_CHECK(y <= kMaxGridYZ, "trimul_in: grid.y overflow");
  dim3 grid((unsigned)S, (unsigned)y);
  trimul_in_fwd_kernel<<<grid, dim3(256), shared_bytes, stream>>>(
      (const bf16*)z.data_ptr(), w_ap.data_ptr<float>(), w_ag.data_ptr<float>(),
      w_bp.data_ptr<float>(), w_bg.data_ptr<float>(), w_g.data_ptr<float>(), mask.data_ptr<float>(),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), rows, NN, C, H, (bf16*)a.data_ptr(),
      (bf16*)b.data_ptr(), (bf16*)gate.data_ptr(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      save_zln ? (bf16*)zln.data_ptr() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b, gate, mean, rstd, zln};
}

__device__ __forceinline__ void cp_async16(bf16* dst, const bf16* src) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    unsigned& r0,
    unsigned& r1,
    unsigned& r2,
    unsigned& r3,
    const bf16* p) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// ---------------------------------------------------------------------------
// Output forward: normalize the contraction result, project, and apply the gate.
//   x:    [B*H, NN] BF16 channel-major output of the Python matrix contraction.
//   wz:   [C, H] FP32 output weights, rounded to BF16 in shared memory.
//   gate: [rows, C] BF16 raw gate logits from trimul_in_fwd.
//   out:  [rows, C] BF16 = (LayerNorm(x) @ wz^T) * sigmoid(gate).
// Training also saves normalized xlnT [H, rows] in BF16 and mean/rstd [rows]
// in FP32. The ungated projection stays in registers; it is not saved separately.
// Each 256-thread block visits 128-position tiles in a persistent 1-D grid.
// Eight warps split each tile into four row groups and two output-channel halves.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_out_fwd_kernel(
    const bf16* __restrict__ x,
    const float* __restrict__ wz,
    const bf16* __restrict__ gate,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    long rows,
    long NN,
    int C,
    int H,
    bf16* __restrict__ out,
    bf16* __restrict__ xlnT,
    float* __restrict__ save_mean,
    float* __restrict__ save_rstd) {
  extern __shared__ bf16 shared_storage[];
  const int position_stride = 136;      // 128 positions plus padding
  const int weight_row_stride = H + 8;  // H channels plus padding
  // Two [H, position_stride] input tiles, then weights [C, weight_row_stride]
  // and FP32 LayerNorm parameters. Normalize each input tile in place.
  bf16* shared_input_buffers[2] = {shared_storage, shared_storage + (long)H * position_stride};
  bf16* shared_weights = shared_storage + 2 * (long)H * position_stride;
  float* shared_ln_parameters =
      reinterpret_cast<float*>(shared_weights + (long)C * weight_row_stride);  // gamma|beta [2H]

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id >> 5, lane_id = thread_id & 31;
  const bool training = xlnT != nullptr;

  // A. Stage weights and LayerNorm parameters once for this block's tile loop.
  {
    int chunks = H >> 3;
    for (int i = thread_id; i < C * chunks; i += 256) {
      int c = i / chunks, q = i % chunks;
      const float* src = wz + (long)c * H + q * 8;
      union {
        bf16 b[8];
        uint4 u;
      } packed_weights;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        packed_weights.b[t] = __float2bfloat16(src[t]);
      reinterpret_cast<uint4*>(shared_weights + c * weight_row_stride)[q] = packed_weights.u;
    }
    for (int i = thread_id; i < H; i += 256) {
      shared_ln_parameters[i] = ln_gamma[i];
      shared_ln_parameters[H + i] = ln_beta[i];
    }
  }
  const float inverse_hidden_channels = 1.0f / (float)H;

  // Load a channel-major tile into one input buffer. Aligned groups of eight
  // BF16 values use cp.async; scalar loads handle tails and batch boundaries.
  auto stage_input_tile = [&](long tile, int buf) {
    const long row0 = tile * 128;
    const long tile_batch_start = row0 / NN;
    const long tile_position_start = row0 - tile_batch_start * NN;
    bf16* shared_input_tile = shared_input_buffers[buf];
    for (int id = thread_id; id < H * 16; id += 256) {
      int h = id >> 4, r8 = (id & 15) * 8;
      long pos = tile_position_start + r8;
      long bi = tile_batch_start;
      while (pos >= NN) {
        pos -= NN;
        ++bi;
      }
      if (row0 + r8 + 7 < rows && pos + 7 < NN && (NN & 7) == 0) {
        cp_async16(shared_input_tile + h * position_stride + r8, x + (bi * H + h) * NN + pos);
      } else {
        bf16 tmp[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          long p = pos + t;
          long b2 = bi;
          while (p >= NN) {
            p -= NN;
            ++b2;
          }
          tmp[t] = (row0 + r8 + t < rows) ? x[(b2 * H + h) * NN + p] : __float2bfloat16(0.f);
        }
        *reinterpret_cast<uint4*>(shared_input_tile + h * position_stride + r8) =
            *reinterpret_cast<const uint4*>(tmp);
      }
    }
    cp_async_commit();
  };

  // B. Start the first tile, then alternate the two shared input buffers.
  int current_buffer = 0;
  if (blockIdx.x * 128 < rows)
    stage_input_tile(blockIdx.x, current_buffer);
  for (long tile = blockIdx.x; tile * 128 < rows; tile += gridDim.x) {
    const long row0 = tile * 128;
    bf16* shared_input_tile = shared_input_buffers[current_buffer];
    cp_async_wait0();
    // Every warp sees the completed loads. On later iterations this also
    // finishes all reads of the previous tile before that buffer is reused.
    __syncthreads();

    // C. Normalize in place. Each warp owns 16 positions; each lane pair owns
    // one position, with even/odd lanes summing even/odd channels in FP32.
    {
      int p = warp_id * 16 + (lane_id >> 1);
      int hpar = lane_id & 1;
      long row = row0 + p;
      float sum = 0.f, sq = 0.f;
      for (int h = hpar; h < H; h += 2) {
        float v = __bfloat162float(shared_input_tile[h * position_stride + p]);
        sum += v;
        sq += v * v;
      }
      sum += __shfl_xor_sync(FULL, sum, 1);
      sq += __shfl_xor_sync(FULL, sq, 1);
      float mean = sum * inverse_hidden_channels;
      float rstd = rsqrtf(sq * inverse_hidden_channels - mean * mean + 1e-5f);
      if (training && hpar == 0 && row < rows) {
        save_mean[row] = mean;
        save_rstd[row] = rstd;
      }
      bool live = row < rows;
      for (int h = hpar; h < H; h += 2) {
        float v = __bfloat162float(shared_input_tile[h * position_stride + p]);
        shared_input_tile[h * position_stride + p] = __float2bfloat16(
            live ? (v - mean) * rstd * shared_ln_parameters[h] + shared_ln_parameters[H + h] : 0.f);
      }
    }
    __syncthreads();  // Publish the normalized BF16 values to every warp.

    long next_tile = tile + gridDim.x;
    // Load the next tile into the other buffer while this tile is consumed.
    if (next_tile * 128 < rows)
      stage_input_tile(next_tile, current_buffer ^ 1);

    // D. Training saves the normalized [H, rows] transpose for backward.
    if (training) {
      for (int id = thread_id; id < H * 16; id += 256) {
        int h = id >> 4, r8 = (id & 15) * 8;
        long row = row0 + r8;
        if (row + 7 < rows && (rows & 7) == 0) {
          *reinterpret_cast<uint4*>(xlnT + (long)h * rows + row) =
              *reinterpret_cast<const uint4*>(shared_input_tile + h * position_stride + r8);
        } else {
          for (int t = 0; t < 8 && row + t < rows; ++t)
            xlnT[(long)h * rows + row + t] = shared_input_tile[h * position_stride + r8 + t];
        }
      }
    }

    // E. Compute the ungated output projection. A warp owns 32 rows by C/2
    // channels. Each thread keeps four FP32 values per 16x8 MMA fragment.
    const int warp_row_group = warp_id & 3, warp_column_half = warp_id >> 2;
    const int warp_row_start = warp_row_group * 32;
    const int columns_per_warp_group = C >> 1;
    const int warp_column_start = warp_column_half * columns_per_warp_group;
    float projection_accumulators[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        projection_accumulators[i][j][0] = projection_accumulators[i][j][1] =
            projection_accumulators[i][j][2] = projection_accumulators[i][j][3] = 0.f;

    // Transposed ldmatrix reads [channel, position] input as the MMA A operand.
    // Weight rows are output channels, with input channels contiguous.
    const bf16* xa0 = shared_input_tile + ((lane_id & 7) + ((lane_id >> 4) << 3)) * position_stride
                      + warp_row_start + (((lane_id >> 3) & 1) << 3);
    const bf16* xa1 = xa0 + 16;
    const bf16* wb0 = shared_weights
                      + (warp_column_start + (lane_id >> 4) * 8 + (lane_id & 7)) * weight_row_stride
                      + 8 * ((lane_id >> 3) & 1);
    auto gemm_all = [&](int ksteps, int ntp) {
      for (int ks = 0; ks < ksteps; ++ks) {
        int k0 = ks * 16;
        unsigned input_fragments[2][4];
        ldmatrix_x4_trans(
            input_fragments[0][0], input_fragments[0][1], input_fragments[0][2],
            input_fragments[0][3], xa0 + k0 * position_stride);
        ldmatrix_x4_trans(
            input_fragments[1][0], input_fragments[1][1], input_fragments[1][2],
            input_fragments[1][3], xa1 + k0 * position_stride);
        for (int jp = 0; jp < ntp; ++jp) {
          unsigned b00, b01, b10, b11;
          ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * weight_row_stride + k0);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma_16816(
                projection_accumulators[i][jp * 2][0], projection_accumulators[i][jp * 2][1],
                projection_accumulators[i][jp * 2][2], projection_accumulators[i][jp * 2][3],
                input_fragments[i][0], input_fragments[i][1], input_fragments[i][2],
                input_fragments[i][3], b00, b01);
            mma_16816(
                projection_accumulators[i][jp * 2 + 1][0],
                projection_accumulators[i][jp * 2 + 1][1],
                projection_accumulators[i][jp * 2 + 1][2],
                projection_accumulators[i][jp * 2 + 1][3], input_fragments[i][0],
                input_fragments[i][1], input_fragments[i][2], input_fragments[i][3], b10, b11);
          }
        }
      }
    };
    if (C == 128 && H == 128)
      gemm_all(8, 4);
    else
      gemm_all(H >> 4, C >> 5);

    // F. Apply sigmoid(gate) to FP32 projection values, then round each pair
    // to BF16 for the final row-major output. No shared output tile is needed.
    const int n0 = (lane_id & 3) * 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int p_lo = warp_row_start + i * 16 + (lane_id >> 2);
#pragma unroll 2
      for (int half = 0; half < 2; ++half) {  // p_lo and p_lo + 8
        long row = row0 + p_lo + half * 8;
        if (row >= rows)
          continue;
        const int fragment_pair_start = half * 2;  // Two adjacent output channels.
        for (int j = 0; j < 8; ++j) {
          int c = warp_column_start + j * 8 + n0;
          if (c >= warp_column_start + columns_per_warp_group)
            break;
          float r0 = projection_accumulators[i][j][fragment_pair_start],
                r1 = projection_accumulators[i][j][fragment_pair_start + 1];
          unsigned packed_gate_pair = *reinterpret_cast<const unsigned*>(gate + row * (long)C + c);
          const bf16* gate_pair = reinterpret_cast<const bf16*>(&packed_gate_pair);
          float g0 = __bfloat162float(gate_pair[0]), g1 = __bfloat162float(gate_pair[1]);
          float s0 = __fdividef(r0, 1.f + __expf(-g0));
          float s1 = __fdividef(r1, 1.f + __expf(-g1));
          __nv_bfloat162 po = __floats2bfloat162_rn(s0, s1);
          *reinterpret_cast<unsigned*>(out + row * (long)C + c) = *reinterpret_cast<unsigned*>(&po);
        }
      }
    }
    current_buffer ^= 1;  // The next iteration consumes the prefetched tile.
  }
}

std::vector<torch::Tensor> trimul_out_fwd(
    torch::Tensor x,
    torch::Tensor wz,
    torch::Tensor gate,
    torch::Tensor gamma,
    torch::Tensor beta,
    long B,
    long NN,
    int C,
    int H,
    bool training) {
  constexpr const char* op = "trimul_out_fwd";
  check_channels(C, H, op);
  int64_t rows = check_pair_extent(B, NN, H, op);
  check_cuda_contiguous(x, at::kBFloat16, op, "x");
  check_shape(x, {checked_mul(B, H, op, "B*H"), NN}, op, "x");
  check_alignment(x, 16, op, "x");
  check_cuda_contiguous(wz, at::kFloat, op, "wz");
  check_same_device(wz, x, op, "wz");
  check_shape(wz, {C, H}, op, "wz");
  check_cuda_contiguous(gate, at::kBFloat16, op, "gate");
  check_same_device(gate, x, op, "gate");
  check_shape(gate, {rows, C}, op, "gate");
  check_alignment(gate, 4, op, "gate");
  for (auto named_parameter :
       {std::pair<const char*, const torch::Tensor*>("gamma", &gamma),
        std::pair<const char*, const torch::Tensor*>("beta", &beta)}) {
    check_cuda_contiguous(*named_parameter.second, at::kFloat, op, named_parameter.first);
    check_same_device(*named_parameter.second, x, op, named_parameter.first);
    check_shape(*named_parameter.second, {H}, op, named_parameter.first);
  }
  const c10::cuda::CUDAGuard device_guard(x.device());
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());
  auto o = torch::empty({rows, (long)C}, opts);
  auto xln = torch::empty({training ? (long)H : 0, rows}, opts);
  auto fopts = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
  auto mean = torch::empty({training ? rows : 0}, fopts);
  auto rstd = torch::empty({training ? rows : 0}, fopts);
  size_t shared_bytes =
      ((size_t)2 * H * 136 + (size_t)C * (H + 8)) * sizeof(bf16) + (size_t)2 * H * sizeof(float);
  int device = x.get_device();
  check_dynamic_shared_memory_limit(kTrimulOutMaxDynamicSharedBytes, device, op);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      trimul_out_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kTrimulOutMaxDynamicSharedBytes));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(sm_count > 0, op, ": device reports no streaming multiprocessors");
  long g = std::max(1L, std::min(tiles, (long)(2 * sm_count)));
  trimul_out_fwd_kernel<<<dim3((unsigned)g), dim3(256), shared_bytes, stream>>>(
      (const bf16*)x.data_ptr(), wz.data_ptr<float>(), (const bf16*)gate.data_ptr(),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), rows, NN, C, H, (bf16*)o.data_ptr(),
      training ? (bf16*)xln.data_ptr() : nullptr, training ? mean.data_ptr<float>() : nullptr,
      training ? rstd.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, xln, mean, rstd};
}

// ---------------------------------------------------------------------------
// Input-gate backward: recompute the A/B projections and differentiate each gate.
//   zln: [rows, C] BF16 normalized input saved by trimul_in_fwd.
//   w_ap/w_ag/w_bp/w_bg: [H, C] FP32 weights, staged as BF16 as in forward.
//   d_a, d_b: [B*H, NN] BF16 channel-major contraction gradients.
//   d_proj5: [rows, 5*H] BF16; this kernel writes only the first four H slots.
// Each 256-thread block owns 32 output channels and visits 128-row tiles.
// The two epilogue waves share one buffer: d_ap/d_ag first, then d_bp/d_bg.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) trimul_gate_split_bwd_kernel(
    const bf16* __restrict__ zln,
    const float* __restrict__ w_ap,
    const float* __restrict__ w_ag,
    const float* __restrict__ w_bp,
    const float* __restrict__ w_bg,
    const float* __restrict__ mask,
    const bf16* __restrict__ d_a,
    const bf16* __restrict__ d_b,
    long rows,
    long NN,
    int C,
    int H,
    bf16* __restrict__ d_proj5) {
  extern __shared__ bf16 shared_storage[];
  const int input_row_stride = C + 8;
  // Four projections x 32 channels of weights, then one normalized input tile.
  // Each epilogue wave needs two 32-channel gradient blocks, padded to 72.
  bf16* shared_weights = shared_storage;  // [128, input_row_stride]
  bf16* shared_normalized_input =
      shared_weights + 128 * input_row_stride;  // [128, input_row_stride]
  bf16* shared_a_gradient = shared_normalized_input + 128 * input_row_stride;  // [32 ch][136 rows]
  bf16* shared_b_gradient = shared_a_gradient + 32 * 136;                      // [32 ch][136 rows]
  bf16* shared_projection_gradients =
      shared_b_gradient + 32 * 136;  // [128 rows][72] two-block wave

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id >> 5, lane_id = thread_id & 31;
  const int channel_slice = blockIdx.x;
  const long projection_row_stride = 5 * (long)H;

  // A. Stage four projections once for this block's 32-channel slice.
  // Weight rows are grouped as [channel half, projection, channel within half]:
  // half*64 + projection*16 + channel, with projections ordered ap/ag/bp/bg.
  {
    const float* wsrc[4] = {w_ap, w_ag, w_bp, w_bg};
    int chunks = C >> 3;
    for (int i = thread_id; i < 128 * chunks; i += 256) {
      int col = i / chunks, q = i % chunks;
      int wcg = col / 64, rem = col % 64;
      const float* src =
          wsrc[rem >> 4] + (long)(channel_slice * 32 + wcg * 16 + (rem & 15)) * C + q * 8;
      union {
        bf16 b[8];
        uint4 u;
      } packed_weights;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        packed_weights.b[t] = __float2bfloat16(src[t]);
      *reinterpret_cast<uint4*>(shared_weights + col * input_row_stride + q * 8) = packed_weights.u;
    }
  }

  // One shared normalized-input buffer is reused across tiles. The first load
  // zero-fills invalid rows; later invalid rows retain prior buffer values but
  // never produce a global output.
  auto stage_normalized_input_tile = [&](long tile, bool first) {
    const long row0 = tile * 128;
    int chunks = C >> 3;
    for (int id = thread_id; id < 128 * chunks; id += 256) {
      int r = id / chunks, q = id % chunks;
      long row = row0 + r;
      if (row < rows) {
        cp_async16(
            shared_normalized_input + r * input_row_stride + q * 8, zln + row * (long)C + q * 8);
      } else if (first) {  // zero-fill dead rows once
        *reinterpret_cast<uint4*>(shared_normalized_input + r * input_row_stride + q * 8) =
            make_uint4(0, 0, 0, 0);
      }
    }
    cp_async_commit();
  };

  long tile0 = blockIdx.y;
  if (tile0 * 128 < rows)
    stage_normalized_input_tile(tile0, true);

  for (long tile = tile0; tile * 128 < rows; tile += gridDim.y) {
    const long row0 = tile * 128;
    const long tile_batch_start = row0 / NN;
    const long tile_position_start = row0 - tile_batch_start * NN;

    // B. Stage channel-major A/B gradients, handling tails and batch boundaries.
    for (int id = thread_id; id < 32 * 16; id += 256) {
      int channel_in_slice = id >> 4, r = (id & 15) * 8;
      int h = channel_slice * 32 + channel_in_slice;
      long pos = tile_position_start + r;
      long bi = tile_batch_start;
      while (pos >= NN) {
        pos -= NN;
        ++bi;
      }
      long addr = (bi * H + h) * NN + pos;
      if (row0 + r + 7 < rows && pos + 7 < NN && (addr & 7) == 0) {
        *reinterpret_cast<uint4*>(shared_a_gradient + channel_in_slice * 136 + r) =
            *reinterpret_cast<const uint4*>(d_a + addr);
        *reinterpret_cast<uint4*>(shared_b_gradient + channel_in_slice * 136 + r) =
            *reinterpret_cast<const uint4*>(d_b + addr);
      } else {
        for (int t = 0; t < 8; ++t) {
          long row = row0 + r + t;
          bf16 va = __float2bfloat16(0.f), vb = va;
          if (row < rows) {
            long p = pos + t;
            long b2 = bi;
            while (p >= NN) {
              p -= NN;
              ++b2;
            }
            long ad = (b2 * H + h) * NN + p;
            va = d_a[ad];
            vb = d_b[ad];
          }
          shared_a_gradient[channel_in_slice * 136 + r + t] = va;
          shared_b_gradient[channel_in_slice * 136 + r + t] = vb;
        }
      }
    }
    cp_async_wait0();
    __syncthreads();  // All normalized inputs, weights, and A/B gradients are ready.

    // C. Recompute ap/ag/bp/bg using the forward BF16 inputs and FP32 MMA order.
    // Each warp owns 32 rows and 64 projection columns. These accumulators
    // remain in registers while the two epilogue waves reuse shared memory.
    const int warp_row_group = warp_id & 3, warp_column_group = warp_id >> 2;
    const int warp_row_start = warp_row_group * 32;
    float projection_accumulators[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
        projection_accumulators[i][j][0] = projection_accumulators[i][j][1] =
            projection_accumulators[i][j][2] = projection_accumulators[i][j][3] = 0.f;

    const bf16* za0 = shared_normalized_input + (warp_row_start + (lane_id & 15)) * input_row_stride
                      + 8 * (lane_id >> 4);
    const bf16* za1 = za0 + 16 * input_row_stride;
    const bf16* wb0 =
        shared_weights
        + (warp_column_group * 64 + (lane_id >> 4) * 8 + (lane_id & 7)) * input_row_stride
        + 8 * ((lane_id >> 3) & 1);
    auto gemm_all = [&](int ksteps) {
      for (int ks = 0; ks < ksteps; ++ks) {
        int k0 = ks * 16;
        unsigned input_fragments[2][4];
        ldmatrix_x4(
            input_fragments[0][0], input_fragments[0][1], input_fragments[0][2],
            input_fragments[0][3], za0 + k0);
        ldmatrix_x4(
            input_fragments[1][0], input_fragments[1][1], input_fragments[1][2],
            input_fragments[1][3], za1 + k0);
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          unsigned b00, b01, b10, b11;
          ldmatrix_x4(b00, b01, b10, b11, wb0 + jp * 16 * input_row_stride + k0);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma_16816(
                projection_accumulators[i][jp * 2][0], projection_accumulators[i][jp * 2][1],
                projection_accumulators[i][jp * 2][2], projection_accumulators[i][jp * 2][3],
                input_fragments[i][0], input_fragments[i][1], input_fragments[i][2],
                input_fragments[i][3], b00, b01);
            mma_16816(
                projection_accumulators[i][jp * 2 + 1][0],
                projection_accumulators[i][jp * 2 + 1][1],
                projection_accumulators[i][jp * 2 + 1][2],
                projection_accumulators[i][jp * 2 + 1][3], input_fragments[i][0],
                input_fragments[i][1], input_fragments[i][2], input_fragments[i][3], b10, b11);
          }
        }
      }
    };
    if (C == 128)
      gemm_all(8);
    else
      gemm_all(C >> 4);

    // D. Differentiate A's gate, then B's gate. Each wave writes two H slots.
    const int n0 = (lane_id & 3) * 2;
#pragma unroll
    for (int wave = 0; wave < 2; ++wave) {
      // Wave 0 waits for all GEMM reads. Wave 1 also waits for readers of
      // the previous shared gradient wave before overwriting that buffer.
      __syncthreads();
      if (wave == 1) {
        // No warp needs the current normalized input now: the projection
        // values are in registers. Refill its buffer for the next tile.
        long next_tile = tile + gridDim.y;
        if (next_tile * 128 < rows)
          stage_normalized_input_tile(next_tile, false);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int r_lo = warp_row_start + i * 16 + (lane_id >> 2);
        long row_lo = row0 + r_lo, row_hi = row_lo + 8;
        float m_lo = row_lo < rows ? mask[row_lo] : 0.f;
        float m_hi = row_hi < rows ? mask[row_hi] : 0.f;
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            int channel_in_slice = warp_column_group * 16 + jp * 8 + n0 + q;
            // Fragment columns: ap/ag start at 0/2, bp/bg at 4/6.
            float pv0 = projection_accumulators[i][wave * 4 + jp][q];
            float gv0 = projection_accumulators[i][wave * 4 + 2 + jp][q];
            float pv1 = projection_accumulators[i][wave * 4 + jp][2 + q];
            float gv1 = projection_accumulators[i][wave * 4 + 2 + jp][2 + q];
            const bf16* dsrc =
                (wave == 0 ? shared_a_gradient : shared_b_gradient) + channel_in_slice * 136;
            float d0 = __bfloat162float(dsrc[r_lo]);
            float d1 = __bfloat162float(dsrc[r_lo + 8]);
            float s0 = __fdividef(1.f, 1.f + __expf(-gv0));
            float s1 = __fdividef(1.f, 1.f + __expf(-gv1));
            // d_p = dy * sig(g) * m ; d_g = dy * p * sig'(g) * m
            shared_projection_gradients[r_lo * 72 + channel_in_slice] =
                __float2bfloat16(d0 * s0 * m_lo);
            shared_projection_gradients[r_lo * 72 + 32 + channel_in_slice] =
                __float2bfloat16(d0 * pv0 * s0 * (1.f - s0) * m_lo);
            shared_projection_gradients[(r_lo + 8) * 72 + channel_in_slice] =
                __float2bfloat16(d1 * s1 * m_hi);
            shared_projection_gradients[(r_lo + 8) * 72 + 32 + channel_in_slice] =
                __float2bfloat16(d1 * pv1 * s1 * (1.f - s1) * m_hi);
          }
        }
      }
      __syncthreads();  // Publish this wave's BF16 projection gradients.
      // Cooperatively store this 32-channel slice into two row-major H slots.
      for (int id = thread_id; id < 128 * 2 * 4; id += 256) {
        int r = id >> 3, blk = (id >> 2) & 1, q = id & 3;
        long row = row0 + r;
        if (row >= rows)
          continue;
        long col = (long)(2 * wave + blk) * H + channel_slice * 32 + q * 8;
        *reinterpret_cast<uint4*>(d_proj5 + row * projection_row_stride + col) =
            *reinterpret_cast<const uint4*>(
                shared_projection_gradients + r * 72 + blk * 32 + q * 8);
      }
    }
    __syncthreads();  // Finish shared gradient reads before the next tile.
  }
}

// ---------------------------------------------------------------------------
// sig_mul backward from out: d_raw = dy*sig(g); d_gate = dy*out*(1-sig(g)).
// d_gate written into a strided destination (the [4H:5H) slot of d_proj5).
// `out` is the saved, gated BF16 forward output. Both gradients are evaluated
// in FP32 from these saved values and rounded to BF16 at their stores.
// ---------------------------------------------------------------------------
__global__ void sig_mul_bwd_out_kernel(
    const bf16* __restrict__ out,
    const bf16* __restrict__ gate,
    const bf16* __restrict__ grad_out,
    long total,
    int H,
    long dgate_stride,
    bf16* __restrict__ draw,
    bf16* __restrict__ dgate) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  long row = idx / H;
  int h = idx % H;
  float gv = __bfloat162float(gate[row * (long)H + h]);
  float g = __fdividef(1.f, 1.f + __expf(-gv));
  float dy = __bfloat162float(grad_out[idx]);
  float o = __bfloat162float(out[idx]);
  draw[idx] = __float2bfloat16(dy * g);
  dgate[row * dgate_stride + h] = __float2bfloat16(dy * o * (1.f - g));
}

torch::Tensor trimul_gate_split_bwd(
    torch::Tensor zln,
    torch::Tensor w_ap,
    torch::Tensor w_ag,
    torch::Tensor w_bp,
    torch::Tensor w_bg,
    torch::Tensor mask,
    torch::Tensor d_a,
    torch::Tensor d_b,
    torch::Tensor d_proj5,
    long B,
    long NN,
    int C,
    int H) {
  constexpr const char* op = "trimul_gate_split_bwd";
  check_channels(C, H, op);
  int64_t rows = check_pair_extent(B, NN, H, op);
  check_cuda_contiguous(zln, at::kBFloat16, op, "zln");
  check_shape(zln, {rows, C}, op, "zln");
  check_alignment(zln, 16, op, "zln");
  for (auto named_weight :
       {std::pair<const char*, const torch::Tensor*>("w_ap", &w_ap),
        std::pair<const char*, const torch::Tensor*>("w_ag", &w_ag),
        std::pair<const char*, const torch::Tensor*>("w_bp", &w_bp),
        std::pair<const char*, const torch::Tensor*>("w_bg", &w_bg)}) {
    check_cuda_contiguous(*named_weight.second, at::kFloat, op, named_weight.first);
    check_same_device(*named_weight.second, zln, op, named_weight.first);
    check_shape(*named_weight.second, {H, C}, op, named_weight.first);
  }
  check_cuda_contiguous(mask, at::kFloat, op, "mask");
  check_same_device(mask, zln, op, "mask");
  check_shape(mask, {rows}, op, "mask");
  TORCH_CHECK(!mask.requires_grad(), op, ": mask must not require gradients");
  for (auto named_gradient :
       {std::pair<const char*, const torch::Tensor*>("d_a", &d_a),
        std::pair<const char*, const torch::Tensor*>("d_b", &d_b)}) {
    check_cuda_contiguous(*named_gradient.second, at::kBFloat16, op, named_gradient.first);
    check_same_device(*named_gradient.second, zln, op, named_gradient.first);
    check_shape(
        *named_gradient.second, {checked_mul(B, H, op, "B*H"), NN}, op, named_gradient.first);
    check_alignment(*named_gradient.second, 16, op, named_gradient.first);
  }
  check_cuda_contiguous(d_proj5, at::kBFloat16, op, "d_proj5");
  check_same_device(d_proj5, zln, op, "d_proj5");
  check_shape(d_proj5, {rows, 5LL * H}, op, "d_proj5");
  check_alignment(d_proj5, 16, op, "d_proj5");
  check_no_overlap(d_proj5, {&zln, &w_ap, &w_ag, &w_bp, &w_bg, &mask, &d_a, &d_b});
  const c10::cuda::CUDAGuard device_guard(zln.device());
  long tiles = (rows + 127) / 128;
  auto stream = at::cuda::getCurrentCUDAStream();
  int input_row_stride = C + 8;
  size_t shared_bytes =
      ((size_t)128 * input_row_stride + 128 * input_row_stride + 2 * 32 * 136 + 128 * 72)
      * sizeof(bf16);
  int device = zln.get_device();
  check_dynamic_shared_memory_limit(kTrimulGateBwdMaxDynamicSharedBytes, device, op);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      trimul_gate_split_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kTrimulGateBwdMaxDynamicSharedBytes));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  TORCH_CHECK(sm_count > 0, op, ": device reports no streaming multiprocessors");
  int S = H / 32;
  long y = std::max(1L, std::min(tiles, (long)(2 * sm_count) / S));
  TORCH_CHECK(y <= kMaxGridYZ, "trimul_gate_split_bwd: grid.y overflow");
  trimul_gate_split_bwd_kernel<<<dim3((unsigned)S, (unsigned)y), dim3(256), shared_bytes, stream>>>(
      (const bf16*)zln.data_ptr(), w_ap.data_ptr<float>(), w_ag.data_ptr<float>(),
      w_bp.data_ptr<float>(), w_bg.data_ptr<float>(), mask.data_ptr<float>(),
      (const bf16*)d_a.data_ptr(), (const bf16*)d_b.data_ptr(), rows, NN, C, H,
      (bf16*)d_proj5.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d_proj5;
}

torch::Tensor sig_mul_bwd_out(
    torch::Tensor out,
    torch::Tensor gate,
    torch::Tensor grad_out,
    torch::Tensor dgate_dst,
    int H,
    long dgate_stride) {
  constexpr const char* op = "sig_mul_bwd_out";
  check_channel(H, op, "H");
  check_cuda_contiguous(out, at::kBFloat16, op, "out");
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) > 0 && out.size(1) == H, op,
      ": out must have shape [rows, H] with rows > 0; got ", out.sizes());
  int64_t rows = out.size(0);
  for (auto named_input :
       {std::pair<const char*, const torch::Tensor*>("gate", &gate),
        std::pair<const char*, const torch::Tensor*>("grad_out", &grad_out)}) {
    check_cuda_contiguous(*named_input.second, at::kBFloat16, op, named_input.first);
    check_same_device(*named_input.second, out, op, named_input.first);
    check_shape(*named_input.second, {rows, H}, op, named_input.first);
  }
  check_cuda_tensor(dgate_dst, at::kBFloat16, op, "dgate_dst");
  check_same_device(dgate_dst, out, op, "dgate_dst");
  check_shape(dgate_dst, {rows, H}, op, "dgate_dst");
  checked_positive(dgate_stride, op, "dgate_stride");
  TORCH_CHECK(dgate_stride >= H, op, ": dgate_stride must be at least H");
  TORCH_CHECK(
      dgate_dst.stride(0) == dgate_stride && dgate_dst.stride(1) == 1, op,
      ": dgate_dst strides must equal [dgate_stride, 1], got ", dgate_dst.strides());
  check_no_overlap(dgate_dst, {&out, &gate, &grad_out});
  int64_t total = checked_mul(rows, H, op, "rows*H");
  TORCH_CHECK(
      ceil_div_positive(total, 256) <= kMaxGridX, op,
      ": launch grid-x exceeds the production limit");
  const c10::cuda::CUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto draw = torch::empty_like(out);
  sig_mul_bwd_out_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
      (const bf16*)out.data_ptr(), (const bf16*)gate.data_ptr(), (const bf16*)grad_out.data_ptr(),
      total, H, dgate_stride, (bf16*)draw.data_ptr(), (bf16*)dgate_dst.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return draw;
}

static inline int cdiv(long a, int b) {
  return (int)((a + b - 1) / b);
}

torch::Tensor ln_bwd_dx(
    torch::Tensor input,
    torch::Tensor grad_out,
    long rows,
    long row_stride,
    long col_stride,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd) {
  constexpr const char* op = "ln_bwd_dx";
  checked_positive(rows, op, "rows");
  check_cuda_contiguous(input, at::kBFloat16, op, "input");
  check_cuda_contiguous(gamma, at::kFloat, op, "gamma");
  check_same_device(gamma, input, op, "gamma");
  TORCH_CHECK(gamma.dim() == 1, op, ": gamma must be rank one");
  TORCH_CHECK(gamma.size(0) <= std::numeric_limits<int>::max(), op, ": gamma length exceeds int32");
  int C = static_cast<int>(gamma.size(0));
  check_channel(C, op, "C");
  TORCH_CHECK(
      row_stride == C && col_stride == 1, op,
      ": production layout requires row_stride=C and col_stride=1");
  check_shape(input, {rows, C}, op, "input");
  TORCH_CHECK(
      input.stride(0) == row_stride && input.stride(1) == col_stride, op,
      ": input strides do not match the supplied strides");
  check_cuda_contiguous(grad_out, at::kBFloat16, op, "grad_out");
  check_same_device(grad_out, input, op, "grad_out");
  check_shape(grad_out, {rows, C}, op, "grad_out");
  for (auto named_stat :
       {std::pair<const char*, const torch::Tensor*>("mean", &mean),
        std::pair<const char*, const torch::Tensor*>("rstd", &rstd)}) {
    check_cuda_contiguous(*named_stat.second, at::kFloat, op, named_stat.first);
    check_same_device(*named_stat.second, input, op, named_stat.first);
    check_shape(*named_stat.second, {rows}, op, named_stat.first);
  }
  TORCH_CHECK(
      ceil_div_positive(rows, 8) <= kMaxGridX, op, ": launch grid-x exceeds the production limit");
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto grad_in = torch::empty_like(input);
  int warps_per_block = 8;
  dim3 block(WARP * warps_per_block);
  dim3 grid(cdiv(rows, warps_per_block));
  ln_bwd_dx_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)input.data_ptr(), (const bf16*)grad_out.data_ptr(), rows, C, row_stride,
      col_stride, gamma.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
      (bf16*)grad_in.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_in;
}

std::vector<torch::Tensor> ln_bwd_dgb(
    torch::Tensor input,
    torch::Tensor grad_out,
    long rows,
    long row_stride,
    long col_stride,
    int C,
    torch::Tensor mean,
    torch::Tensor rstd) {
  constexpr const char* op = "ln_bwd_dgb";
  checked_positive(rows, op, "rows");
  check_channel(C, op, "C");
  TORCH_CHECK(
      row_stride == C && col_stride == 1, op,
      ": production layout requires row_stride=C and col_stride=1");
  check_cuda_contiguous(input, at::kBFloat16, op, "input");
  check_shape(input, {rows, C}, op, "input");
  TORCH_CHECK(
      input.stride(0) == row_stride && input.stride(1) == col_stride, op,
      ": input strides do not match the supplied strides");
  check_cuda_contiguous(grad_out, at::kBFloat16, op, "grad_out");
  check_same_device(grad_out, input, op, "grad_out");
  check_shape(grad_out, {rows, C}, op, "grad_out");
  for (auto named_stat :
       {std::pair<const char*, const torch::Tensor*>("mean", &mean),
        std::pair<const char*, const torch::Tensor*>("rstd", &rstd)}) {
    check_cuda_contiguous(*named_stat.second, at::kFloat, op, named_stat.first);
    check_same_device(*named_stat.second, input, op, named_stat.first);
    check_shape(*named_stat.second, {rows}, op, named_stat.first);
  }
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dgamma =
      torch::empty({(long)C}, torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
  auto dbeta = torch::empty_like(dgamma);
  // Choose row chunks from the current GPU's SM count and the number of rows.
  // Round to 64-row units and clamp to [64, 2048]; each chunk still accumulates
  // rows in ascending order, then the final kernel folds chunks in order.
  int sm_count = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, input.get_device()));
  TORCH_CHECK(sm_count > 0, op, ": device reports no streaming multiprocessors");
  constexpr int target_blocks_per_sm = 16;
  long target_chunks = (long)sm_count * target_blocks_per_sm;
  long chunk_rows = (rows + target_chunks - 1) / target_chunks;
  chunk_rows = ((chunk_rows + 63) / 64) * 64;
  chunk_rows = std::max(64L, std::min(2048L, chunk_rows));
  int64_t chunks64 = ceil_div_positive(rows, chunk_rows);
  TORCH_CHECK(chunks64 <= kMaxGridYZ, op, ": reduction chunks exceed the production grid-y limit");
  int chunks = static_cast<int>(chunks64);
  auto partial = torch::empty(
      {(long)chunks * 2, (long)C},
      torch::TensorOptions().dtype(torch::kFloat).device(input.device()));
  dim3 block(128);
  dim3 grid(cdiv(C, 128), chunks);
  ln_bwd_dgb_partial_kernel<<<grid, block, 0, stream>>>(
      (const bf16*)input.data_ptr(), (const bf16*)grad_out.data_ptr(), rows, C, row_stride,
      col_stride, chunk_rows, mean.data_ptr<float>(), rstd.data_ptr<float>(),
      partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ln_bwd_dgb_fold_kernel<<<cdiv(C, 128), 128, 0, stream>>>(
      partial.data_ptr<float>(), chunks, C, dgamma.data_ptr<float>(), dbeta.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dgamma, dbeta};
}

torch::Tensor trimul_wgrad_fixed_fold(
    torch::Tensor partials,
    torch::Tensor accumulator,
    bool initialize) {
  constexpr const char* op = "trimul_wgrad_fixed_fold";
  check_cuda_contiguous(partials, at::kFloat, op, "partials");
  TORCH_CHECK(
      partials.dim() == 3, op, ": partials must have shape [chunks, M, C]; got ", partials.sizes());
  int64_t chunks64 = checked_positive(partials.size(0), op, "chunks");
  int64_t matrix_rows = checked_positive(partials.size(1), op, "M");
  int64_t matrix_cols = checked_positive(partials.size(2), op, "C");
  TORCH_CHECK(chunks64 <= std::numeric_limits<int>::max(), op, ": chunks exceed int32");

  check_cuda_contiguous(accumulator, at::kFloat, op, "accumulator");
  check_same_device(accumulator, partials, op, "accumulator");
  check_shape(accumulator, {matrix_rows, matrix_cols}, op, "accumulator");
  check_no_overlap(accumulator, {&partials});

  int64_t elements = checked_mul(matrix_rows, matrix_cols, op, "M*C");
  int64_t partial_elements = checked_mul(chunks64, elements, op, "chunks*M*C");
  TORCH_CHECK(
      partial_elements == partials.numel(), op, ": chunks*M*C does not match partials.numel()");
  TORCH_CHECK(
      ceil_div_positive(elements, 256) <= kMaxGridX, op,
      ": launch grid-x exceeds the production limit");
  const c10::cuda::CUDAGuard device_guard(partials.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned blocks = (unsigned)ceil_div_positive(elements, 256);
  if (initialize) {
    trimul_wgrad_fixed_fold_kernel<true><<<blocks, 256, 0, stream>>>(
        partials.data_ptr<float>(), static_cast<int>(chunks64), elements,
        accumulator.data_ptr<float>());
  } else {
    trimul_wgrad_fixed_fold_kernel<false><<<blocks, 256, 0, stream>>>(
        partials.data_ptr<float>(), static_cast<int>(chunks64), elements,
        accumulator.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return accumulator;
}

std::vector<torch::Tensor> transpose_ln_bwd(
    torch::Tensor x,
    torch::Tensor d_xln,
    int N,
    int H,
    int B,
    torch::Tensor gamma,
    torch::Tensor mean,
    torch::Tensor rstd) {
  constexpr const char* op = "transpose_ln_bwd";
  check_channel(H, op, "H");
  checked_positive(N, op, "N");
  checked_positive(B, op, "B");
  TORCH_CHECK(B <= kMaxGridYZ, op, ": B exceeds the production grid-z limit");
  int64_t NN = checked_mul(N, N, op, "N*N");
  int64_t rows = checked_mul(B, NN, op, "B*N*N");
  int64_t tiles64 = ceil_div_positive(NN, 32);
  TORCH_CHECK(tiles64 <= kMaxGridX, op, ": transpose grid-x exceeds the production limit");
  int64_t chunks64 = checked_mul(tiles64, B, op, "ceil(N*N/32)*B");
  TORCH_CHECK(chunks64 <= kMaxGridX, op, ": transpose reduction chunks exceed signed int32");
  check_cuda_contiguous(x, at::kBFloat16, op, "x");
  check_shape(x, {checked_mul(B, H, op, "B*H"), NN}, op, "x");
  check_cuda_contiguous(d_xln, at::kBFloat16, op, "d_xln");
  check_same_device(d_xln, x, op, "d_xln");
  check_shape(d_xln, {rows, H}, op, "d_xln");
  check_cuda_contiguous(gamma, at::kFloat, op, "gamma");
  check_same_device(gamma, x, op, "gamma");
  check_shape(gamma, {H}, op, "gamma");
  for (auto named_stat :
       {std::pair<const char*, const torch::Tensor*>("mean", &mean),
        std::pair<const char*, const torch::Tensor*>("rstd", &rstd)}) {
    check_cuda_contiguous(*named_stat.second, at::kFloat, op, named_stat.first);
    check_same_device(*named_stat.second, x, op, named_stat.first);
    check_shape(*named_stat.second, {rows}, op, named_stat.first);
  }
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto d_x = torch::empty(
      {(long)B * H, (long)N, (long)N},
      torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));
  dim3 block(256);
  unsigned tiles = static_cast<unsigned>(tiles64);
  dim3 grid(tiles, 1, (unsigned)B);
  long chunks = static_cast<long>(chunks64);
  auto partial = torch::empty(
      {chunks * 2, (long)H}, torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  size_t shared_bytes = (size_t)H * 33 * sizeof(float) * 2;
  transpose_ln_bwd_kernel<<<grid, block, shared_bytes, stream>>>(
      (const bf16*)x.data_ptr(), (const bf16*)d_xln.data_ptr(), NN, H, gamma.data_ptr<float>(),
      mean.data_ptr<float>(), rstd.data_ptr<float>(), (bf16*)d_x.data_ptr(),
      partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto dgamma =
      torch::empty({(long)H}, torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  auto dbeta = torch::empty_like(dgamma);
  // Reduce [chunks, 2, H] partials through 64 groups, then fold those groups.
  // Keep both levels and their ascending accumulation order unchanged.
  int groups = 64;
  int group_len = (int)((chunks + groups - 1) / groups);
  auto stage = torch::empty(
      {(long)groups * 2, (long)H}, torch::TensorOptions().dtype(torch::kFloat).device(x.device()));
  dim3 fold_grid(cdiv(H, 128), groups);
  ln_bwd_dgb_group_fold_kernel<<<fold_grid, 128, 0, stream>>>(
      partial.data_ptr<float>(), (int)chunks, H, group_len, stage.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ln_bwd_dgb_fold_kernel<<<cdiv(H, 128), 128, 0, stream>>>(
      stage.data_ptr<float>(), groups, H, dgamma.data_ptr<float>(), dbeta.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_x, dgamma, dbeta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("trimul_in_fwd", &trimul_in_fwd);
  module.def("trimul_out_fwd", &trimul_out_fwd);
  module.def("trimul_gate_split_bwd", &trimul_gate_split_bwd);
  module.def("sig_mul_bwd_out", &sig_mul_bwd_out);
  module.def("ln_bwd_dx", &ln_bwd_dx);
  module.def("ln_bwd_dgb", &ln_bwd_dgb);
  module.def("transpose_ln_bwd", &transpose_ln_bwd);
  module.def("trimul_wgrad_fixed_fold", &trimul_wgrad_fixed_fold);
}
