// SPDX-License-Identifier: Apache-2.0
// FP32-storage inference triangle multiplication. In TF32 mode input rows are
// normalized once into bounded FP32 scratch, then projection/gate/layout work
// is fused. Output normalization/projection/gating stays fused in shared memory
// and registers. MMA operands are rounded to TF32 (nearest-even); all tensor
// storage and accumulators are FP32. The Python facade keeps the full-N
// triangle contraction in cuBLAS and selects the strict-FP32 vendor composition
// when TF32 is disabled. No global precision setting is changed here.
//
// The TF32 projection kernels follow the BF16 forward schedule: persistent
// blocks keep their weight slice resident in shared memory, loop over row
// tiles, and prefetch the next tile while the current one is multiplied.
// - Input projections: TMA loads the normalized tiles and wgmma multiplies them
//   (both operands from shared memory). A chain of m64nNk8 wgmma steps gives the
//   same bits as the m16n8k8 mma.sync chain it replaced (same ascending K order,
//   zero-initialized accumulators).
// - Output projection: cp.async prefetches the tiles and ldmatrix loads the
//   mma.sync fragments: one m8n8.b16 matrix row is 16 bytes, i.e. four FP32/TF32
//   words, so thread i receives word i%4 of row i/4, which is exactly the
//   m16n8k8 TF32 fragment layout.
// Each output value accumulates its K products in ascending order with the same
// epilogue expressions, so results do not depend on the block/warp tiling.
//
// gate_pack: [R,4H+C] projections -> masked/gated A/B [H,N,N] and gate [N*N,C].
// A 32x32 shared-memory tile makes both projection reads and channel-major
// writes coalesced. The extra shared column avoids transpose bank conflicts.
// gate_output: apply the stored sigmoid output gate directly into final output.

#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <cmath>

namespace {

constexpr int kTile = 32;
constexpr int kThreadRows = 8;
constexpr int64_t kMaxGridX = 2147483647;

constexpr int kFusedThreads = 256;
constexpr int kInputRows = 128;
constexpr int kInputChannels = 16;
constexpr int kNormalizationRows = kFusedThreads / 32;

// Persistent TF32 input projections: two 8-warp blocks per SM, so one block's
// gating epilogue and stores overlap the other block's MMAs. A block keeps 16
// output channels of all five projections resident (80 weight columns) and
// visits 64-row tiles of normalized input. Its shared memory (112.5 KiB at
// C=128) is sized so that two blocks fit in one SM.
constexpr int kInputPersistentThreads = 256;
constexpr int kPersistentInputRows = 64;
constexpr int kPersistentInputChannels = 16;
constexpr int kPersistentInputColumns = 5 * kPersistentInputChannels;
// Each warp owns 8 channels of all five projections: 5 * 8 = 40 columns.
constexpr int kPersistentInputGroupChannels = 8;
constexpr int kPersistentInputGroupColumns = 5 * kPersistentInputGroupChannels;
// Channel-major A/B staging: [16 channels][64 rows + 4]. The padding maps
// the MMA accumulator ownership (row=lane/4, channel=2*(lane%4)) onto 32
// distinct banks for each store instruction.
constexpr int kPersistentInputStagingStride = kPersistentInputRows + 4;
// Persistent output projection: one 16-warp block per SM (its resident
// weights and buffers allow only one). A block owns all C output channels
// and visits 64-row tiles of the channel-major contraction result.
constexpr int kOutputPersistentThreads = 512;
constexpr int kOutputPersistentWarps = kOutputPersistentThreads / 32;
constexpr int kPersistentOutputRows = 64;
// cp.async target [H][64 rows + 4]. Rows are 16-byte aligned, and the
// transpose reads 8 channels x 4 rows per instruction from 32 distinct banks.
constexpr int kPersistentOutputStagingStride = kPersistentOutputRows + 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ uint32_t round_tf32(float value) {
  uint32_t rounded;
  // Do not substitute cvt.rna or truncation: the frozen numerical contract
  // specifies ties-to-even, including the halfway cases of FP32 operands.
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(rounded) : "f"(value));
  return rounded;
}

__device__ __forceinline__ float sigmoid_fp32(float value) {
  // The fused TF32 path uses FP32 approximate exp/reciprocal instructions.
  // This removes the libdevice exponential and correctly-rounded division
  // instruction chains; the unchanged qualification gates bound their error.
  // NaN inputs remain NaN. This does not alter the strict-FP32 vendor path.
  // The exponential is computed as __expf(-value) computes it,
  // ex2.approx(value * -log2(e)), in its .ftz form: that form skips __expf's
  // rescaling of arguments below -126, whose results lie below 2^-126, and
  // 1 + such a value rounds to exactly 1 either way, so the result has the
  // same bits as __fdividef(1.0f, 1.0f + __expf(-value)).
  const float exponent = __fmul_rn(value, -1.44269502f);
  float exponential;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
  return __fdividef(1.0f, __fadd_rn(1.0f, exponential));
}

__device__ __forceinline__ void mma_tf32(
    float (&accumulator)[4], uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]),
        "+f"(accumulator[2]), "+f"(accumulator[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four 8x4-word matrices from shared memory. Lanes 8j..8j+7 supply the row
// addresses of matrix j; fragment[j] receives word lane%4 of row lane/4.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&fragment)[4], const float* row) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
               : "r"(address));
}

// Two 8x4-word matrices; lanes 0..15 supply the row addresses.
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&fragment)[2], const float* row) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(fragment[0]), "=r"(fragment[1])
               : "r"(address));
}

// 16-byte global-to-shared copy that bypasses L1. Both addresses must be
// 16-byte aligned. Completion is observed with cp_async_wait_all().
__device__ __forceinline__ void cp_async16(float* destination, const float* source) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address), "l"(source));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

// Waits for this thread's copies only; a block barrier must follow before
// other threads read the destination.
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// ---------------------------------------------------------------------------
// WGMMA (sm_90a) for the persistent input projections.
// A warpgroup (four warps) computes a 64-row x N-column TF32 product with
// wgmma.mma_async; both operands are read from shared memory through matrix
// descriptors. On H100 a chain of m64nNk8 wgmma steps gives the same bits as
// the chain of m16n8k8 mma.sync steps it replaces, when K advances in the same
// ascending order from the same zero-initialized accumulators (checked with
// probe/wgmma_tf32_bits.cu of the 2026-11-01 work: 0 differences in 2.76M outputs
// over normal, wide-exponent and cancelling inputs).
//
// Operand layout (128-byte swizzle, K-major): a [rows][K] tile is stored as K
// blocks of 32 TF32 words (128 bytes per row). Block kb holds rows 0..rows-1
// contiguously, and the 16-byte chunk c of row r sits at chunk c ^ (r % 8).
// Every block must start 1024-byte aligned; rows is a multiple of 8.
__device__ __forceinline__ int sw128_k_major_offset(int row, int column, int rows) {
  const int k_block = column / 32, column_in_block = column % 32;
  return k_block * rows * 32 + row * 32 + (((column_in_block / 4) ^ (row % 8)) * 4)
         + column_in_block % 4;
}

// Descriptor of the K-major operand rows [first_row, first_row + extent) at
// reduction index k (a multiple of 8) of a tile stored with sw128_k_major_offset.
// Fields are in 16-byte units: start address (bits 0-13); leading byte offset
// (bits 16-29, unused by swizzled K-major layouts, set to 1); stride byte offset
// between 8-row groups = 8 * 128 bytes (bits 32-45); swizzle mode 1 = 128 bytes
// (bits 62-63). An 8-word K step stays inside its 128-byte row segment.
__device__ __forceinline__ uint64_t sw128_k_major_descriptor(
    const float* tile, int rows, int first_row, int k) {
  const unsigned start = static_cast<unsigned>(__cvta_generic_to_shared(tile))
      + static_cast<unsigned>(((k / 32) * rows * 32 + first_row * 32 + k % 32) * sizeof(float));
  return (static_cast<uint64_t>(start >> 4) & 0x3FFF) | (uint64_t{1} << 16)
         | (uint64_t{(8 * 128) >> 4} << 32) | (uint64_t{1} << 62);
}

// Shared-memory writes by ordinary stores and cp.async become visible to the
// asynchronous proxy that wgmma reads through only after this fence.
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_all() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

// One-shot shared-memory barriers for the TMA tile loads. A phase completes when
// its single arrival (the thread that issues the loads, with the expected byte
// count) has arrived and all expected TMA bytes have landed; waiters test the
// phase parity.
__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, unsigned arrival_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                   static_cast<unsigned>(__cvta_generic_to_shared(barrier))),
               "r"(arrival_count)
               : "memory");
}

// Makes barrier initialization visible to the TMA unit (the asynchronous proxy).
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_bytes(uint64_t* barrier, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   static_cast<unsigned>(__cvta_generic_to_shared(barrier))),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* barrier, unsigned parity) {
  asm volatile(
      "{\n"
      ".reg .pred phase_complete;\n"
      "wait_for_phase:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 phase_complete, [%0], %1;\n"
      "@!phase_complete bra wait_for_phase;\n"
      "}\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(barrier))),
      "r"(parity)
      : "memory");
}

// TMA: copy the box at (column, row) of a 2-D tensor map into shared memory and
// count its bytes on `barrier`. The tensor map's 128-byte swizzle writes the box
// in the sw128_k_major_offset layout of one K block. Rows past the tensor are
// filled with zeros.
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* map, float* destination, uint64_t* barrier, int column, int64_t row) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(destination))),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(static_cast<int>(row)),
      "r"(static_cast<unsigned>(__cvta_generic_to_shared(barrier)))
      : "memory");
}

// accumulator += A[64 x 8] * B[40 x 8]^T. accumulator[j][e] is element e of
// the 8-column block j in the m16n8 fragment layout of this thread's warp:
// rows 16*(warp%4) + lane/4 (+8 for e >= 2), columns 8j + 2*(lane%4) + e%2.
__device__ __forceinline__ void wgmma_m64n40k8_tf32(
    float (&accumulator)[5][4], uint64_t descriptor_a, uint64_t descriptor_b) {
  // scale-d is always true (D += A*B), as for the mma.sync chain that starts
  // from zeros; scale-a and scale-b are +1.
  asm volatile(
      "{\n"
      ".reg .pred accumulate;\n"
      "setp.ne.b32 accumulate, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k8.f32.tf32.tf32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19}, "
      "%20, %21, accumulate, 1, 1;\n"
      "}\n"
      : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
        "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
        "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
        "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
        "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3])
      : "l"(descriptor_a), "l"(descriptor_b));
}

// One warp normalizes one pair row at a time. Centering around the first
// channel before both reductions avoids cancellation on near-constant inputs
// such as 100 + 1e-3 * noise. No E[x*x] - E[x]^2 variance is used.
template <int Channels>
__device__ __forceinline__ void normalize_shared_row(
    float* row, const float* gamma, const float* beta, float epsilon, int lane) {
  float values[Channels / 32];
  const float origin = row[0];
  float shifted_sum = 0.0f;
#pragma unroll
  for (int part = 0; part < Channels / 32; ++part) {
    values[part] = row[part * 32 + lane] - origin;
    shifted_sum += values[part];
  }
  const float mean_shift = warp_sum(shifted_sum) / Channels;
  float squared_sum = 0.0f;
#pragma unroll
  for (int part = 0; part < Channels / 32; ++part) {
    values[part] -= mean_shift;
    squared_sum = __fadd_rn(squared_sum, __fmul_rn(values[part], values[part]));
  }
  const float reciprocal_std = rsqrtf(warp_sum(squared_sum) / Channels + epsilon);
  // Shuffle reductions exchange registers but do not order shared memory.
  // All lanes must finish reading the original row, especially row[0],
  // before any lane overwrites it. Both callers execute a complete warp.
  __syncwarp(0xffffffffu);
#pragma unroll
  for (int part = 0; part < Channels / 32; ++part) {
    const int channel = part * 32 + lane;
    const float normalized = __fmul_rn(values[part], reciprocal_std);
    const float affine = __fadd_rn(
        __fmul_rn(normalized, gamma == nullptr ? 1.0f : gamma[channel]),
        beta == nullptr ? 0.0f : beta[channel]);
    row[channel] = __uint_as_float(round_tf32(affine));
  }
}

// Normalize each pair row once before projection tiles share it. This calls
// the same normalization helper as the original fused path, including its
// shifted reductions, affine operation order and nearest-even TF32 rounding.
template <int Channels>
__global__ void normalize_input_kernel(
    const float* __restrict__ input, const float* __restrict__ gamma,
    const float* __restrict__ beta, float epsilon,
    float* __restrict__ output, int64_t rows) {
  __shared__ float normalized[kNormalizationRows][Channels + 4];
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int64_t first_row = static_cast<int64_t>(blockIdx.x) * kNormalizationRows;
  for (int index = thread; index < kNormalizationRows * Channels;
       index += kFusedThreads) {
    const int row = index / Channels;
    const int channel = index % Channels;
    normalized[row][channel] = first_row + row < rows
        ? input[(first_row + row) * Channels + channel] : 0.0f;
  }
  __syncthreads();
  normalize_shared_row<Channels>(normalized[warp], gamma, beta, epsilon, lane);
  // Each lane reads only the elements it just wrote. Other lanes' shared
  // writes are not consumed here, and the row storage is never reused.
  if (first_row + warp < rows) {
#pragma unroll
    for (int part = 0; part < Channels / 32; ++part) {
      const int channel = part * 32 + lane;
      output[(first_row + warp) * Channels + channel] = normalized[warp][channel];
    }
  }
}

template <int Channels, bool InputIsNormalized>
__global__ __launch_bounds__(kFusedThreads, 2) void input_fused_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ gamma,
    const float* __restrict__ beta, float epsilon, const float* __restrict__ mask,
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ output_gate,
    int64_t rows, int64_t total_rows, int64_t first_row, int hidden,
    int channel_tiles, bool mask_after_projection) {
  constexpr int stride = Channels + 4;
  extern __shared__ float4 input_shared_vectors[];
  float* normalized = reinterpret_cast<float*>(input_shared_vectors);
  float* weights = normalized + kInputRows * stride;
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int channel_base = (blockIdx.x % channel_tiles) * kInputChannels;
  const int64_t row_base = (blockIdx.x / channel_tiles) * kInputRows;

  // Stage a complete C-wide input row and the selected 16 output channels of
  // all five projections. Biases stay FP32 and are applied after each GEMM.
  // Four FP32 values per load/store reduce address and memory-issue work.
  // A contiguous tensor can still begin at an unaligned storage offset, so
  // each operand has its own uniform alignment branch with a scalar path.
  if ((reinterpret_cast<uintptr_t>(input) & 15u) == 0) {
    constexpr int vectors_per_row = Channels / 4;
    for (int index = thread; index < kInputRows * vectors_per_row; index += kFusedThreads) {
      const int row = index / vectors_per_row;
      const int channel = (index % vectors_per_row) * 4;
      const float4 values = row_base + row < rows
          ? *reinterpret_cast<const float4*>(input + (row_base + row) * Channels + channel)
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(normalized + row * stride + channel) = values;
    }
  } else {
    for (int index = thread; index < kInputRows * Channels; index += kFusedThreads) {
      const int row = index / Channels;
      const int channel = index % Channels;
      normalized[row * stride + channel] = row_base + row < rows
          ? input[(row_base + row) * Channels + channel] : 0.0f;
    }
  }
  if ((reinterpret_cast<uintptr_t>(weight) & 15u) == 0) {
    constexpr int vectors_per_row = Channels / 4;
    for (int index = thread; index < 5 * kInputChannels * vectors_per_row; index += kFusedThreads) {
      const int projection = index / (kInputChannels * vectors_per_row);
      const int channel = (index / vectors_per_row) % kInputChannels;
      const int reduction_channel = (index % vectors_per_row) * 4;
      const int output_channel = channel_base + channel;
      const bool live = output_channel < (projection == 4 ? Channels : hidden);
      const int weight_row = projection == 4 ? 4 * hidden + output_channel
                                             : projection * hidden + output_channel;
      float4 values = live
          ? *reinterpret_cast<const float4*>(weight + weight_row * Channels + reduction_channel)
          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      values.x = __uint_as_float(round_tf32(values.x));
      values.y = __uint_as_float(round_tf32(values.y));
      values.z = __uint_as_float(round_tf32(values.z));
      values.w = __uint_as_float(round_tf32(values.w));
      *reinterpret_cast<float4*>(weights + (projection * kInputChannels + channel) * stride + reduction_channel) = values;
    }
  } else {
    for (int index = thread; index < 5 * kInputChannels * Channels; index += kFusedThreads) {
      const int projection = index / (kInputChannels * Channels);
      const int channel = (index / Channels) % kInputChannels;
      const int reduction_channel = index % Channels;
      const int output_channel = channel_base + channel;
      const bool live = output_channel < (projection == 4 ? Channels : hidden);
      const int weight_row = projection == 4 ? 4 * hidden + output_channel
                                             : projection * hidden + output_channel;
      weights[(projection * kInputChannels + channel) * stride + reduction_channel] =
          __uint_as_float(round_tf32(live ? weight[weight_row * Channels + reduction_channel] : 0.0f));
    }
  }
  if constexpr (!InputIsNormalized) {
    __syncthreads();
    for (int row = warp; row < kInputRows; row += 8)
      normalize_shared_row<Channels>(normalized + row * stride, gamma, beta, epsilon, lane);
  }
  // With pre-normalized input this orders both input and weight staging.
  // Otherwise it also finishes normalization before all warps consume it.
  __syncthreads();

  // Eight warps cover 128 rows x 16 channels. Each warp owns 16 x 16 and
  // carries the five projection accumulators together until their epilogues.
  const int warp_row = warp * 16;
  constexpr int warp_channel = 0;
  const int group = lane / 4;
  const int member = lane % 4;
  float accumulators[5][2][4] = {};
#pragma unroll
  for (int reduction = 0; reduction < Channels; reduction += 8) {
    const uint32_t a0 = __float_as_uint(normalized[(warp_row + group) * stride + reduction + member]);
    const uint32_t a1 = __float_as_uint(normalized[(warp_row + group + 8) * stride + reduction + member]);
    const uint32_t a2 = __float_as_uint(normalized[(warp_row + group) * stride + reduction + member + 4]);
    const uint32_t a3 = __float_as_uint(normalized[(warp_row + group + 8) * stride + reduction + member + 4]);
#pragma unroll
    for (int projection = 0; projection < 5; ++projection) {
#pragma unroll
      for (int column_half = 0; column_half < 2; ++column_half) {
        const int weight_row = projection * kInputChannels + warp_channel + column_half * 8 + group;
        const uint32_t b0 = __float_as_uint(weights[weight_row * stride + reduction + member]);
        const uint32_t b1 = __float_as_uint(weights[weight_row * stride + reduction + member + 4]);
        mma_tf32(accumulators[projection][column_half], a0, a1, a2, a3, b0, b1);
      }
    }
  }
  // Weight storage is dead after the final MMA and becomes the transpose
  // staging area. This reuse keeps the C=128 shared allocation below 120 KiB.
  __syncthreads();
  // MMA accumulator lanes own row=lane/4 and channel=2*(lane%4).
  // Four padding columns map these 32 stores to 32 distinct shared banks;
  // a single padding column creates four-way conflicts for this ownership.
  constexpr int transpose_stride = kInputRows + 4;
  float* staged_a = weights;
  float* staged_b = staged_a + kInputChannels * transpose_stride;
#pragma unroll
  for (int column_half = 0; column_half < 2; ++column_half) {
#pragma unroll
    for (int element = 0; element < 4; ++element) {
      const int local_row = warp_row + group + (element / 2) * 8;
      const int local_channel = warp_channel + column_half * 8 + member * 2 + element % 2;
      const int channel = channel_base + local_channel;
      const int64_t global_row = first_row + row_base + local_row;
      if (row_base + local_row < rows) {
        float projected[5];
#pragma unroll
        for (int projection = 0; projection < 5; ++projection) {
          const bool live = channel < (projection == 4 ? Channels : hidden);
          const int bias_channel = projection == 4 ? 4 * hidden + channel : projection * hidden + channel;
          projected[projection] = accumulators[projection][column_half][element]
              + (bias != nullptr && live ? bias[bias_channel] : 0.0f);
        }
        if (channel < hidden) {
          const float pair_mask = mask == nullptr ? 1.0f : mask[global_row];
          const float gate_a = sigmoid_fp32(projected[1]);
          const float gate_b = sigmoid_fp32(projected[3]);
          staged_a[local_channel * transpose_stride + local_row] = mask_after_projection
              ? __fmul_rn(__fmul_rn(projected[0], gate_a), pair_mask)
              : __fmul_rn(__fmul_rn(pair_mask, gate_a), projected[0]);
          staged_b[local_channel * transpose_stride + local_row] = mask_after_projection
              ? __fmul_rn(__fmul_rn(projected[2], gate_b), pair_mask)
              : __fmul_rn(__fmul_rn(pair_mask, gate_b), projected[2]);
        }
        if (channel < Channels)
          output_gate[global_row * Channels + channel] = sigmoid_fp32(projected[4]);
      }
    }
  }
  __syncthreads();
  for (int index = thread; index < kInputChannels * kInputRows; index += kFusedThreads) {
    const int local_channel = index / kInputRows;
    const int local_row = index % kInputRows;
    const int channel = channel_base + local_channel;
    if (channel < hidden && row_base + local_row < rows) {
      const int64_t offset = channel * total_rows + first_row + row_base + local_row;
      a[offset] = staged_a[local_channel * transpose_stride + local_row];
      b[offset] = staged_b[local_channel * transpose_stride + local_row];
    }
  }
}

// Persistent input projections for pre-normalized TF32 rows.
// Grid: x = 16-channel slice, y = persistent row-tile walkers. A block stages
// its slice of the five packed weights once, then repeatedly
//   1. waits for its prefetched 64-row tile and starts the TMA load of the next one,
//   2. multiplies the tile by the resident weights with wgmma: warpgroup g
//      (warps 4g..4g+3) computes all 64 rows x the 40 columns of channel group g,
//   3. forms gated/masked A/B and the sigmoid output gate with the same
//      arithmetic as input_fused_kernel, and
//   4. writes A/B channel-major through shared staging.
// Resident weight column = group * 40 + projection * 8 + channel_in_group, so
// channel group g owns channels slice*16 + g*8 + [0,8) of every projection.
// HasBias/HasMask/MaskAfterProjection are compile-time copies of the call's
// uniform options; they remove per-element branches, not arithmetic. Without
// a mask the products still multiply by 1.0f, as input_fused_kernel does.
// input_map describes the normalized chunk [rows][C] with 32-column x 64-row
// boxes and the 128-byte swizzle. The host guarantees an 8-byte-aligned gate.
template <int Channels, bool HasBias, bool HasMask, bool MaskAfterProjection>
__global__ __launch_bounds__(kInputPersistentThreads, 2) void input_persistent_kernel(
    const __grid_constant__ CUtensorMap input_map, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ mask,
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ output_gate,
    int64_t rows, int64_t total_rows, int64_t first_row, int hidden,
    bool vector_stores) {
  // Weights [80][C] and input tiles [64][C] use the wgmma operand layout
  // (sw128_k_major_offset); every K block starts 1024-byte aligned because
  // 80*128 and 64*128 bytes are multiples of 1024.
  extern __shared__ __align__(1024) float4 persistent_input_vectors[];
  float* weights = reinterpret_cast<float*>(persistent_input_vectors);  // [80][C]
  // Two input buffers; buffer i starts at input_tiles + i * tile size.
  // (Pointer arithmetic, not a pointer array, keeps the selection in registers.)
  constexpr int input_tile_values = kPersistentInputRows * Channels;
  float* input_tiles = weights + kPersistentInputColumns * Channels;
  float* staged_a = input_tiles + 2 * input_tile_values;                   // [16][68]
  float* staged_b = staged_a + kPersistentInputChannels * kPersistentInputStagingStride;
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int slice_channel_base = blockIdx.x * kPersistentInputChannels;
  const int64_t row_tiles = (rows + kPersistentInputRows - 1) / kPersistentInputRows;
  // tile_loaded[i] completes when input buffer i holds its next tile. The
  // barriers follow the staging buffers in dynamic shared memory: a static
  // __shared__ variable would push the 1024-aligned dynamic base up and leave
  // room for only one block per SM.
  uint64_t* tile_loaded = reinterpret_cast<uint64_t*>(
      staged_b + kPersistentInputChannels * kPersistentInputStagingStride);
  if (thread == 0) {
    mbarrier_init(&tile_loaded[0], 1);
    mbarrier_init(&tile_loaded[1], 1);
    fence_mbarrier_init();
  }

  // Stage this slice of all five projections once, rounded to TF32 exactly as
  // input_fused_kernel rounds its per-block copy. Dead channels are zero. Each
  // step copies four consecutive reduction channels: one 16-byte chunk of the
  // operand layout (the host guarantees a 16-byte-aligned weight).
  constexpr int weight_chunks_per_column = Channels / 4;
  for (int index = thread; index < kPersistentInputColumns * weight_chunks_per_column;
       index += kInputPersistentThreads) {
    const int column = index / weight_chunks_per_column;
    const int reduction_channel = (index % weight_chunks_per_column) * 4;
    const int channel_group = column / kPersistentInputGroupColumns;
    const int projection = (column % kPersistentInputGroupColumns) / kPersistentInputGroupChannels;
    const int output_channel = slice_channel_base + channel_group * kPersistentInputGroupChannels
                               + column % kPersistentInputGroupChannels;
    const bool live = output_channel < (projection == 4 ? Channels : hidden);
    const int weight_row = projection == 4 ? 4 * hidden + output_channel
                                           : projection * hidden + output_channel;
    const float4 values = live
        ? *reinterpret_cast<const float4*>(weight + weight_row * Channels + reduction_channel)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *reinterpret_cast<float4*>(
        weights + sw128_k_major_offset(column, reduction_channel, kPersistentInputColumns)) =
        make_float4(__uint_as_float(round_tf32(values.x)), __uint_as_float(round_tf32(values.y)),
                    __uint_as_float(round_tf32(values.z)), __uint_as_float(round_tf32(values.w)));
  }

  // Thread 0 starts the TMA load of one 64-row tile of normalized input: one
  // 32-column box per K block. Rows past the chunk arrive as zeros; their
  // projections are computed but never stored.
  auto load_input_tile = [&](int64_t tile, int tile_buffer_index) {
    float* tile_buffer = input_tiles + tile_buffer_index * input_tile_values;
    mbarrier_arrive_expect_bytes(&tile_loaded[tile_buffer_index],
                                 input_tile_values * static_cast<unsigned>(sizeof(float)));
#pragma unroll
    for (int k_block = 0; k_block < Channels / 32; ++k_block)
      tma_load_2d(&input_map, tile_buffer + k_block * kPersistentInputRows * 32,
                  &tile_loaded[tile_buffer_index], k_block * 32, tile * kPersistentInputRows);
  };
  // The weight stores above become visible to wgmma's operand reads after this
  // fence and the barrier below; the barrier also publishes the initialized
  // tile barriers before any thread waits on them.
  fence_proxy_async_shared();
  __syncthreads();

  // Warpgroup g = warp/4 computes channel group g (40 columns); inside it, warp
  // w owns rows (w%4)*16 + [0,16) of the accumulator, as an m16n8 fragment per
  // 8-column block (one block per projection).
  const int warp_row = (warp % 4) * 16;
  const int channel_group = warp / 4;
  const int group = lane / 4;
  const int member = lane % 4;
  // This thread's first output channel; the second fragment element of a
  // pair is the next channel.
  const int lane_local_channel = channel_group * kPersistentInputGroupChannels + member * 2;
  // Operand descriptors at reduction index 0 (input buffer 0). The other input
  // buffer and every K step add constant start-address offsets in 16-byte units
  // to the low word; shared addresses stay below 2^18 bytes, so the 14-bit
  // address field never carries. The high word (stride and swizzle) is the
  // same for both operands.
  const uint64_t input_descriptor = sw128_k_major_descriptor(input_tiles, kPersistentInputRows, 0, 0);
  const uint32_t input_descriptor_low = static_cast<uint32_t>(input_descriptor);
  const uint32_t descriptor_high = static_cast<uint32_t>(input_descriptor >> 32);
  constexpr uint32_t kInputBufferDescriptorStep = input_tile_values * sizeof(float) / 16;
  const uint32_t weight_descriptor_low = static_cast<uint32_t>(sw128_k_major_descriptor(
      weights, kPersistentInputColumns, channel_group * kPersistentInputGroupColumns, 0));
  auto descriptor = [&](uint32_t low) {
    return (static_cast<uint64_t>(descriptor_high) << 32) | low;
  };

  int buffer = 0;
  unsigned tile_iteration = 0;  // buffer i is reused every other iteration
  if (thread == 0 && blockIdx.y < row_tiles) load_input_tile(blockIdx.y, 0);
  for (int64_t tile = blockIdx.y; tile < row_tiles; tile += gridDim.y, ++tile_iteration) {
    const int64_t row_base = tile * kPersistentInputRows;
    // The mask values of this thread's two epilogue rows (rows group and
    // group + 8 of its warp's 16), loaded now so the loads complete during the MMA.
    float row_masks[2] = {1.0f, 1.0f};
    if constexpr (HasMask) {
#pragma unroll
      for (int row_half = 0; row_half < 2; ++row_half) {
        const int local_row = warp_row + group + row_half * 8;
        if (row_base + local_row < rows)
          row_masks[row_half] = mask[first_row + row_base + local_row];
      }
    }
    mbarrier_wait_parity(&tile_loaded[buffer], (tile_iteration / 2) & 1);
    // All warps have finished the previous tile: its MMA reads of the other
    // input buffer and its reads of the A/B staging.
    __syncthreads();
    if (thread == 0 && tile + gridDim.y < row_tiles) load_input_tile(tile + gridDim.y, buffer ^ 1);

    // [projection][fragment]. Every value accumulates its K products in the
    // same ascending order as input_fused_kernel.
    float accumulators[5][4] = {};
    wgmma_fence();  // the zeroed accumulators are ready for the asynchronous MMAs
    const uint32_t tile_descriptor_low = input_descriptor_low + buffer * kInputBufferDescriptorStep;
#pragma unroll
    for (int reduction = 0; reduction < Channels; reduction += 8) {
      // Byte offset of reduction index k: (k / 32) whole K blocks of rows x 128 bytes,
      // plus (k % 32) words inside the row segment.
      constexpr uint32_t kWordBytes = sizeof(float);
      const uint32_t in_block = (reduction % 32) * kWordBytes;
      wgmma_m64n40k8_tf32(
          accumulators,
          descriptor(tile_descriptor_low + ((reduction / 32) * kPersistentInputRows * 128 + in_block) / 16),
          descriptor(weight_descriptor_low
                     + ((reduction / 32) * kPersistentInputColumns * 128 + in_block) / 16));
    }
    wgmma_commit();
    wgmma_wait_all();  // the epilogue reads the accumulators

    // Epilogue. Fragment elements {0,1} belong to row group, {2,3} to row
    // group + 8; elements 2k and 2k+1 are adjacent channels. The per-value
    // arithmetic matches input_fused_kernel exactly.
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
      const int local_row = warp_row + group + row_half * 8;
      if (row_base + local_row >= rows) continue;
      const int64_t global_row = first_row + row_base + local_row;
      const float pair_mask = row_masks[row_half];
      const int channel = slice_channel_base + lane_local_channel;
      float gate_pair[2];
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int element = row_half * 2 + pair;
        const int pair_channel = channel + pair;
        if (pair_channel < hidden) {
          float projected[4];
#pragma unroll
          for (int projection = 0; projection < 4; ++projection)
            projected[projection] = accumulators[projection][element]
                + (HasBias ? bias[projection * hidden + pair_channel] : 0.0f);
          const float gate_a = sigmoid_fp32(projected[1]);
          const float gate_b = sigmoid_fp32(projected[3]);
          const int staged = (lane_local_channel + pair) * kPersistentInputStagingStride + local_row;
          staged_a[staged] = MaskAfterProjection
              ? __fmul_rn(__fmul_rn(projected[0], gate_a), pair_mask)
              : __fmul_rn(__fmul_rn(pair_mask, gate_a), projected[0]);
          staged_b[staged] = MaskAfterProjection
              ? __fmul_rn(__fmul_rn(projected[2], gate_b), pair_mask)
              : __fmul_rn(__fmul_rn(pair_mask, gate_b), projected[2]);
        }
        if (pair_channel < Channels)
          gate_pair[pair] = sigmoid_fp32(accumulators[4][element]
              + (HasBias ? bias[4 * hidden + pair_channel] : 0.0f));
      }
      // Channels is even and channel is even: the pair is live or dead together.
      if (channel < Channels)
        *reinterpret_cast<float2*>(output_gate + global_row * Channels + channel) =
            make_float2(gate_pair[0], gate_pair[1]);
    }
    __syncthreads();  // A/B staging is complete.
    if (vector_stores) {
      // rows, first_row and total_rows are multiples of four and a/b are
      // 16-byte aligned, so each group of four pair rows is one aligned float4
      // that is entirely live or entirely past the chunk.
      constexpr int row_vectors = kPersistentInputRows / 4;
      for (int index = thread; index < kPersistentInputChannels * row_vectors;
           index += kInputPersistentThreads) {
        const int local_channel = index / row_vectors;
        const int local_row = (index % row_vectors) * 4;
        const int channel = slice_channel_base + local_channel;
        if (channel < hidden && row_base + local_row < rows) {
          const int64_t offset = channel * total_rows + first_row + row_base + local_row;
          const int staged = local_channel * kPersistentInputStagingStride + local_row;
          *reinterpret_cast<float4*>(a + offset) = *reinterpret_cast<const float4*>(staged_a + staged);
          *reinterpret_cast<float4*>(b + offset) = *reinterpret_cast<const float4*>(staged_b + staged);
        }
      }
    } else {
      // Adjacent threads write adjacent pair rows of one channel.
      for (int index = thread; index < kPersistentInputChannels * kPersistentInputRows;
           index += kInputPersistentThreads) {
        const int local_channel = index / kPersistentInputRows;
        const int local_row = index % kPersistentInputRows;
        const int channel = slice_channel_base + local_channel;
        if (channel < hidden && row_base + local_row < rows) {
          const int64_t offset = channel * total_rows + first_row + row_base + local_row;
          const int staged = local_channel * kPersistentInputStagingStride + local_row;
          a[offset] = staged_a[staged];
          b[offset] = staged_b[staged];
        }
      }
    }
    // The next iteration's barrier orders these staging reads before the next
    // epilogue overwrites the staging buffers.
    buffer ^= 1;
  }
}

// Persistent output stage: LayerNorm over H, TF32 projection to C, and the
// stored sigmoid gate. A block owns all C output channels, so each pair row is
// normalized once (the earlier 32-channel tiling normalized it C/32 times).
// Per 64-row tile:
//   1. wait for the prefetched channel-major tile, start copying the next one,
//   2. transpose [H][rows] to row-major [rows][H] in shared memory,
//   3. normalize each row with normalize_shared_row (one warp per row),
//   4. multiply by the resident TF32 weights and apply bias and gate.
// VectorLoads requires rows % 4 == 0 and a 16-byte-aligned product; otherwise
// the tile is copied with scalar loads. The gate and output must be 8-byte
// aligned (checked by the host): each thread reads and writes channel pairs.
template <int Hidden, int Channels, bool VectorLoads, bool HasBias>
__global__ __launch_bounds__(kOutputPersistentThreads, 1) void output_persistent_kernel(
    const float* __restrict__ product, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ gamma,
    const float* __restrict__ beta, float epsilon,
    const float* __restrict__ gate, float* __restrict__ output, int64_t rows) {
  constexpr int stride = Hidden + 4;
  constexpr int vectors_per_channel = kPersistentOutputRows / 4;
  // Warp w owns rows (w%4)*16 + [0,16) and output channels (w/4)*C/4 + [0,C/4):
  // one to four 8-column MMA tiles for C = 32..128.
  constexpr int warp_columns = Channels / 4;
  constexpr int warp_column_tiles = warp_columns / 8;
  extern __shared__ float4 persistent_output_vectors[];
  float* weights = reinterpret_cast<float*>(persistent_output_vectors);  // [C][stride]
  // Two [H][68] cp.async buffers; buffer i starts at staging + i * tile size.
  constexpr int staging_tile_values = Hidden * kPersistentOutputStagingStride;
  float* staging = weights + Channels * stride;
  float* normalized = staging + 2 * staging_tile_values;  // [64][stride]
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int64_t row_tiles = (rows + kPersistentOutputRows - 1) / kPersistentOutputRows;

  // Stage all output weights once, rounded to TF32 as before.
  for (int index = thread; index < Channels * Hidden; index += kOutputPersistentThreads) {
    const int channel = index / Hidden;
    const int reduction_channel = index % Hidden;
    weights[channel * stride + reduction_channel] =
        __uint_as_float(round_tf32(weight[channel * Hidden + reduction_channel]));
  }

  // Copy one tile of the cuBLAS result in its native [H][rows] layout.
  auto stage_product_tile = [&](int64_t tile, float* tile_buffer) {
    const int64_t row_base = tile * kPersistentOutputRows;
    if constexpr (VectorLoads) {
      // rows % 4 == 0, so each four-row group is entirely live or dead.
      for (int index = thread; index < Hidden * vectors_per_channel; index += kOutputPersistentThreads) {
        const int channel = index / vectors_per_channel;
        const int row = (index % vectors_per_channel) * 4;
        float* destination = tile_buffer + channel * kPersistentOutputStagingStride + row;
        if (row_base + row < rows)
          cp_async16(destination, product + channel * rows + row_base + row);
        else
          *reinterpret_cast<float4*>(destination) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      cp_async_commit();
    } else {
      for (int index = thread; index < Hidden * kPersistentOutputRows; index += kOutputPersistentThreads) {
        const int channel = index / kPersistentOutputRows;
        const int row = index % kPersistentOutputRows;
        tile_buffer[channel * kPersistentOutputStagingStride + row] =
            row_base + row < rows ? product[channel * rows + row_base + row] : 0.0f;
      }
    }
  };

  const int warp_row = (warp % 4) * 16;
  const int warp_column = (warp / 4) * warp_columns;
  const int group = lane / 4;
  const int member = lane % 4;
  // Same ldmatrix fragment addressing as input_persistent_kernel.
  const int a_row = warp_row + ((lane / 8) % 2) * 8 + lane % 8;
  const int a_column = (lane / 16) * 4;
  const int b_row = warp_column + (lane / 16) * 8 + lane % 8;
  const int b_column = ((lane / 8) % 2) * 4;
  // Transpose ownership: one instruction moves channels c0+lane%8 of rows
  // r0+lane/8. Reads (stride 68 = 4 mod 32) and writes (stride H+4 = 4 mod 32)
  // both touch 32 distinct banks.
  const int transpose_channel = lane % 8;
  const int transpose_row = lane / 8;
  constexpr int transpose_blocks = (Hidden / 8) * (kPersistentOutputRows / 4);

  int buffer = 0;
  if (blockIdx.x < row_tiles) stage_product_tile(blockIdx.x, staging);
  for (int64_t tile = blockIdx.x; tile < row_tiles; tile += gridDim.x) {
    const int64_t row_base = tile * kPersistentOutputRows;
    if constexpr (VectorLoads) cp_async_wait_all();
    // The current tile (and, first time, the weights) is visible to every
    // warp, and all warps have finished the previous tile's MMA reads of the
    // normalized rows and transpose reads of the other staging buffer.
    __syncthreads();
    if (tile + gridDim.x < row_tiles)
      stage_product_tile(tile + gridDim.x, staging + (buffer ^ 1) * staging_tile_values);

    const float* tile_buffer = staging + buffer * staging_tile_values;
    for (int block = warp; block < transpose_blocks; block += kOutputPersistentWarps) {
      const int channel = (block % (Hidden / 8)) * 8 + transpose_channel;
      const int row = (block / (Hidden / 8)) * 4 + transpose_row;
      normalized[row * stride + channel] = tile_buffer[channel * kPersistentOutputStagingStride + row];
    }
    __syncthreads();  // Row-major tile is complete.
    for (int row = warp; row < kPersistentOutputRows; row += kOutputPersistentWarps)
      normalize_shared_row<Hidden>(normalized + row * stride, gamma, beta, epsilon, lane);
    __syncthreads();  // Normalized rows are visible to every warp.

    // Fragment elements {0,1} belong to row group and {2,3} to row group + 8;
    // each pair covers adjacent channels. Load this thread's gate pairs
    // before the MMA so their latency overlaps it.
    const int64_t thread_rows[2] = {row_base + warp_row + group, row_base + warp_row + group + 8};
    const int lane_channel = warp_column + member * 2;
    float2 gate_values[2][warp_column_tiles];
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
      const float* gate_row = gate + thread_rows[row_half] * Channels + lane_channel;
#pragma unroll
      for (int column_tile = 0; column_tile < warp_column_tiles; ++column_tile)
        gate_values[row_half][column_tile] = thread_rows[row_half] < rows
            ? *reinterpret_cast<const float2*>(gate_row + column_tile * 8) : make_float2(0.0f, 0.0f);
    }

    const float* a_address = normalized + a_row * stride + a_column;
    const float* b_address = weights + b_row * stride + b_column;
    float accumulators[warp_column_tiles][4] = {};
#pragma unroll
    for (int reduction = 0; reduction < Hidden; reduction += 8) {
      uint32_t input_fragment[4];
      ldmatrix_x4(input_fragment, a_address + reduction);
#pragma unroll
      for (int column_pair = 0; column_pair < warp_column_tiles / 2; ++column_pair) {
        uint32_t weight_fragment[4];
        ldmatrix_x4(weight_fragment, b_address + column_pair * 16 * stride + reduction);
        mma_tf32(accumulators[2 * column_pair], input_fragment[0], input_fragment[1],
                 input_fragment[2], input_fragment[3], weight_fragment[0], weight_fragment[1]);
        mma_tf32(accumulators[2 * column_pair + 1], input_fragment[0], input_fragment[1],
                 input_fragment[2], input_fragment[3], weight_fragment[2], weight_fragment[3]);
      }
      if constexpr (warp_column_tiles % 2 == 1) {
        constexpr int last = warp_column_tiles - 1;
        uint32_t weight_fragment[2];
        ldmatrix_x2(weight_fragment, b_address + last * 8 * stride + reduction);
        mma_tf32(accumulators[last], input_fragment[0], input_fragment[1],
                 input_fragment[2], input_fragment[3], weight_fragment[0], weight_fragment[1]);
      }
    }
    // Same per-value arithmetic as before: (projection + bias) * gate.
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
      if (thread_rows[row_half] >= rows) continue;
      float* output_row = output + thread_rows[row_half] * Channels + lane_channel;
#pragma unroll
      for (int column_tile = 0; column_tile < warp_column_tiles; ++column_tile) {
        const int channel = lane_channel + column_tile * 8;
        const float projected0 = accumulators[column_tile][row_half * 2]
            + (HasBias ? bias[channel] : 0.0f);
        const float projected1 = accumulators[column_tile][row_half * 2 + 1]
            + (HasBias ? bias[channel + 1] : 0.0f);
        *reinterpret_cast<float2*>(output_row + column_tile * 8) = make_float2(
            projected0 * gate_values[row_half][column_tile].x,
            projected1 * gate_values[row_half][column_tile].y);
      }
    }
    buffer ^= 1;
  }
}

__global__ void gate_pack_kernel(
    const float* __restrict__ projected,
    const float* __restrict__ mask,
    float* __restrict__ a,
    float* __restrict__ b,
    float* __restrict__ output_gate,
    int64_t chunk_rows,
    int64_t total_rows,
    int64_t first_row,
    int64_t hidden,
    int64_t channels,
    int64_t channel_tiles) {
  __shared__ float tile_a[kTile][kTile + 1];
  __shared__ float tile_b[kTile][kTile + 1];
  const int64_t row_tile = blockIdx.x / channel_tiles;
  const int64_t channel_tile = blockIdx.x % channel_tiles;
  const int64_t channel = channel_tile * kTile + threadIdx.x;
  const int64_t projection_width = 4 * hidden + channels;
  for (int tile_row = threadIdx.y; tile_row < kTile; tile_row += kThreadRows) {
    const int64_t row = row_tile * kTile + tile_row;
    float value_a = 0.0f;
    float value_b = 0.0f;
    if (row < chunk_rows) {
      const int64_t global_row = first_row + row;
      const int64_t offset = row * projection_width;
      if (channel < hidden) {
        const float pair_mask = mask == nullptr ? 1.0f : mask[global_row];
        const float gate_a = 1.0f / (1.0f + expf(-projected[offset + hidden + channel]));
        const float gate_b = 1.0f / (1.0f + expf(-projected[offset + 3 * hidden + channel]));
        // Preserve the eager equation's mask -> sigmoid gate -> projection order.
        value_a = (pair_mask * gate_a) * projected[offset + channel];
        value_b = (pair_mask * gate_b) * projected[offset + 2 * hidden + channel];
      }
      if (channel < channels) {
        output_gate[global_row * channels + channel] =
            1.0f / (1.0f + expf(-projected[offset + 4 * hidden + channel]));
      }
    }
    tile_a[tile_row][threadIdx.x] = value_a;
    tile_b[tile_row][threadIdx.x] = value_b;
  }
  __syncthreads();
  // Transpose tile ownership: adjacent lanes now write adjacent pair rows.
  const int64_t row = row_tile * kTile + threadIdx.x;
  for (int tile_channel = threadIdx.y; tile_channel < kTile; tile_channel += kThreadRows) {
    const int64_t output_channel = channel_tile * kTile + tile_channel;
    if (row < chunk_rows && output_channel < hidden) {
      const int64_t output_offset = output_channel * total_rows + first_row + row;
      a[output_offset] = tile_a[threadIdx.x][tile_channel];
      b[output_offset] = tile_b[threadIdx.x][tile_channel];
    }
  }
}

__global__ void gate_output_kernel(
    const float* __restrict__ projected,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = projected[index] * gate[index];
  }
}

void check_fp32_contiguous(const torch::Tensor& tensor, const char* name,
                           const torch::Device& device) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == device,
              name, " must be on the same CUDA device");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 && tensor.is_contiguous(),
              name, " must be contiguous FP32");
}

void gate_pack_out(
    const torch::Tensor& projected,
    const std::optional<torch::Tensor>& mask,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& output_gate,
    int64_t first_row) {
  check_fp32_contiguous(projected, "projected", projected.device());
  check_fp32_contiguous(a, "a", projected.device());
  check_fp32_contiguous(b, "b", projected.device());
  check_fp32_contiguous(output_gate, "output_gate", projected.device());
  TORCH_CHECK(projected.dim() == 2 && projected.size(0) > 0,
              "projected must have nonempty shape [R,4H+C]");
  TORCH_CHECK(a.dim() == 3 && a.size(0) > 0 && a.size(1) > 0
                  && a.size(1) == a.size(2) && b.sizes() == a.sizes(),
              "a and b must have matching shape [H,N,N]");
  const int64_t hidden = a.size(0);
  const int64_t total_rows = a.size(1) * a.size(2);
  TORCH_CHECK(output_gate.dim() == 2 && output_gate.size(0) == total_rows
                  && output_gate.size(1) > 0,
              "output_gate must have shape [N*N,C]");
  const int64_t channels = output_gate.size(1);
  const int64_t rows = projected.size(0);
  TORCH_CHECK(projected.size(1) == 4 * hidden + channels,
              "projection layout must be [ap,ag,bp,bg,output_gate]");
  TORCH_CHECK(first_row >= 0 && first_row <= total_rows && rows <= total_rows - first_row,
              "projection row chunk exceeds output storage");
  if (mask.has_value()) {
    check_fp32_contiguous(*mask, "mask", projected.device());
    TORCH_CHECK(mask->dim() == 1 && mask->numel() == total_rows,
                "mask must cover the complete [N*N] pair rows");
  }
  // Out tensors must not overlap each other or any read operand.
  at::assert_no_overlap(a, b);
  at::assert_no_overlap(a, output_gate);
  at::assert_no_overlap(b, output_gate);
  for (const auto& output : {a, b, output_gate}) {
    at::assert_no_overlap(output, projected);
    if (mask.has_value()) at::assert_no_overlap(output, *mask);
  }
  const int64_t channel_tiles = (std::max(hidden, channels) + kTile - 1) / kTile;
  const int64_t row_tiles = (rows + kTile - 1) / kTile;
  TORCH_CHECK(channel_tiles <= kMaxGridX && row_tiles <= kMaxGridX / channel_tiles,
              "gate/pack launch exceeds CUDA grid-x range");
  const c10::cuda::CUDAGuard device_guard(projected.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_pack_kernel<<<static_cast<unsigned int>(row_tiles * channel_tiles),
                     dim3(kTile, kThreadRows), 0, stream>>>(
      projected.data_ptr<float>(), mask.has_value() ? mask->data_ptr<float>() : nullptr,
      a.data_ptr<float>(), b.data_ptr<float>(), output_gate.data_ptr<float>(),
      rows, total_rows, first_row, hidden, channels, channel_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_output_out(const torch::Tensor& projected, const torch::Tensor& gate,
                     torch::Tensor& output) {
  check_fp32_contiguous(projected, "projected", projected.device());
  check_fp32_contiguous(gate, "gate", projected.device());
  check_fp32_contiguous(output, "output", projected.device());
  TORCH_CHECK(projected.dim() == 2 && projected.numel() > 0
                  && projected.sizes() == gate.sizes() && output.sizes() == projected.sizes(),
              "projected, gate and output must have matching nonempty [R,C] shapes");
  at::assert_no_overlap(output, projected);
  at::assert_no_overlap(output, gate);
  constexpr int threads = 256;
  const int64_t blocks = (projected.numel() + threads - 1) / threads;
  TORCH_CHECK(blocks <= kMaxGridX, "output gate launch exceeds CUDA grid-x range");
  const c10::cuda::CUDAGuard device_guard(projected.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_output_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      projected.data_ptr<float>(), gate.data_ptr<float>(), output.data_ptr<float>(),
      projected.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool fused_channel_count(int64_t channels) {
  return channels == 32 || channels == 64 || channels == 96 || channels == 128;
}

const float* optional_parameter(
    const std::optional<torch::Tensor>& tensor, int64_t elements,
    const torch::Device& device, const char* name) {
  if (!tensor.has_value()) return nullptr;
  check_fp32_contiguous(*tensor, name, device);
  TORCH_CHECK(tensor->dim() == 1 && tensor->numel() == elements,
              name, " has an incorrect parameter shape");
  return tensor->data_ptr<float>();
}

void normalize_input_out(
    const torch::Tensor& input, const std::optional<torch::Tensor>& gamma,
    const std::optional<torch::Tensor>& beta, double epsilon,
    torch::Tensor& output) {
  const auto device = input.device();
  check_fp32_contiguous(input, "input", device);
  check_fp32_contiguous(output, "output", device);
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0
                  && fused_channel_count(input.size(1))
                  && output.sizes() == input.sizes(),
              "normalized input/output must match [R,C], C in {32,64,96,128}");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0, "epsilon must be finite and positive");
  const int64_t channels = input.size(1);
  const float* gamma_pointer = optional_parameter(gamma, channels, device, "gamma");
  const float* beta_pointer = optional_parameter(beta, channels, device, "beta");
  at::assert_no_overlap(output, input);
  for (const auto* parameter : {&gamma, &beta})
    if (parameter->has_value()) at::assert_no_overlap(output, **parameter);
  const int64_t blocks = (input.size(0) + kNormalizationRows - 1) / kNormalizationRows;
  TORCH_CHECK(blocks <= kMaxGridX, "input normalization grid exceeds CUDA grid-x range");
  const c10::cuda::CUDAGuard device_guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_NORMALIZATION(C) normalize_input_kernel<C><<< \
    static_cast<unsigned int>(blocks), kFusedThreads, 0, stream>>>( \
        input.data_ptr<float>(), gamma_pointer, beta_pointer, \
        static_cast<float>(epsilon), output.data_ptr<float>(), input.size(0))
  switch (channels) {
    case 32: LAUNCH_NORMALIZATION(32); break;
    case 64: LAUNCH_NORMALIZATION(64); break;
    case 96: LAUNCH_NORMALIZATION(96); break;
    case 128: LAUNCH_NORMALIZATION(128); break;
  }
#undef LAUNCH_NORMALIZATION
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Channels, bool InputIsNormalized>
void launch_input_fused(
    const torch::Tensor& input, const torch::Tensor& weight, const float* bias,
    const float* gamma, const float* beta, float epsilon, const float* mask,
    torch::Tensor& a, torch::Tensor& b, torch::Tensor& gate, int64_t first_row,
    bool mask_after_projection, int64_t blocks, int channel_tiles) {
  constexpr int weight_values = 5 * kInputChannels * (Channels + 4);
  constexpr int transpose_values = 2 * kInputChannels * (kInputRows + 4);
  constexpr int reused_values = weight_values > transpose_values ? weight_values : transpose_values;
  constexpr int shared_bytes = (kInputRows * (Channels + 4) + reused_values) * sizeof(float);
  // Attributes belong to a device's loaded kernel. Cache per host thread and
  // device, while always taking the launch stream from the caller's context.
  static thread_local int configured_device = -1;
  if (configured_device != input.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(input_fused_kernel<Channels, InputIsNormalized>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    configured_device = input.get_device();
  }
  input_fused_kernel<Channels, InputIsNormalized><<<static_cast<unsigned int>(blocks), kFusedThreads,
      shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(), bias, gamma, beta, epsilon,
      mask, a.data_ptr<float>(), b.data_ptr<float>(), gate.data_ptr<float>(),
      input.size(0), gate.size(0), first_row, static_cast<int>(a.size(0)),
      channel_tiles, mask_after_projection);
}

// Multiprocessor count of the current device, cached per host thread.
int multiprocessor_count(int device) {
  static thread_local int cached_device = -1;
  static thread_local int cached_count = 0;
  if (cached_device != device) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&cached_count, cudaDevAttrMultiProcessorCount, device));
    TORCH_CHECK(cached_count > 0, "device reports no streaming multiprocessors");
    cached_device = device;
  }
  return cached_count;
}

bool aligned_to(const void* pointer, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(pointer) & (bytes - 1)) == 0;
}

// Resolve cuTensorMapEncodeTiled through the runtime instead of linking
// directly against libcuda.
using TensorMapEncodeFunction = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

TensorMapEncodeFunction tensor_map_encode_function() {
  static const TensorMapEncodeFunction encode = [] {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    const cudaError_t status =
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &symbol, cudaEnableDefault, &query);
    TORCH_CHECK(status == cudaSuccess && query == cudaDriverEntryPointSuccess && symbol != nullptr,
                "cuTensorMapEncodeTiled is unavailable in this driver");
    return reinterpret_cast<TensorMapEncodeFunction>(symbol);
  }();
  return encode;
}

// Normalized FP32 rows [rows][channels] as TMA boxes of 32 channels x box_rows
// rows with the 128-byte swizzle (one box = one K block of the wgmma operand
// layout). Out-of-range rows are filled with zeros.
CUtensorMap make_normalized_input_tensor_map(const torch::Tensor& input, int channels, int box_rows) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(channels),
                                    static_cast<cuuint64_t>(input.size(0))};
  const cuuint64_t global_stride[1] = {static_cast<cuuint64_t>(channels) * sizeof(float)};
  const cuuint32_t box_dim[2] = {32, static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element_stride[2] = {1, 1};
  const CUresult result = tensor_map_encode_function()(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, input.data_ptr(), global_dim, global_stride,
      box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return tensor_map;
}

template <int Channels, bool HasBias, bool HasMask, bool MaskAfterProjection>
void launch_input_persistent(
    const torch::Tensor& input, const torch::Tensor& weight, const float* bias,
    const float* mask, torch::Tensor& a, torch::Tensor& b, torch::Tensor& gate,
    int64_t first_row) {
  constexpr int shared_bytes =
      ((kPersistentInputColumns + 2 * kPersistentInputRows) * Channels
       + 2 * kPersistentInputChannels * kPersistentInputStagingStride) * sizeof(float)
      + 2 * sizeof(uint64_t);  // the two tile_loaded barriers
  const auto kernel = input_persistent_kernel<Channels, HasBias, HasMask, MaskAfterProjection>;
  static thread_local int configured_device = -1;
  if (configured_device != input.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    // Two blocks per SM need the largest shared-memory carveout.
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    configured_device = input.get_device();
  }
  const int64_t rows = input.size(0);
  const int64_t total_rows = gate.size(0);
  const int hidden = static_cast<int>(a.size(0));
  const int slices = (std::max<int>(Channels, hidden) + kPersistentInputChannels - 1)
                     / kPersistentInputChannels;
  const int64_t row_tiles = (rows + kPersistentInputRows - 1) / kPersistentInputRows;
  // Two resident blocks per SM, split evenly across the channel slices so
  // each block keeps one weight slice for its whole row-tile walk.
  const int64_t walkers = std::max<int64_t>(
      1, std::min<int64_t>(row_tiles, 2 * multiprocessor_count(input.get_device()) / slices));
  const bool vector_stores = rows % 4 == 0 && first_row % 4 == 0 && total_rows % 4 == 0
      && aligned_to(a.data_ptr<float>(), 16) && aligned_to(b.data_ptr<float>(), 16);
  const CUtensorMap input_map = make_normalized_input_tensor_map(input, Channels, kPersistentInputRows);
  kernel<<<dim3(slices, static_cast<unsigned int>(walkers)), kInputPersistentThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      input_map, weight.data_ptr<float>(), bias, mask,
      a.data_ptr<float>(), b.data_ptr<float>(), gate.data_ptr<float>(),
      rows, total_rows, first_row, hidden, vector_stores);
}

// Select the compile-time copy of the call's bias/mask options. Without a
// mask both multiplication orders reduce to the same product times 1.0f.
template <int Channels>
void dispatch_input_persistent(
    const torch::Tensor& input, const torch::Tensor& weight, const float* bias,
    const float* mask, torch::Tensor& a, torch::Tensor& b, torch::Tensor& gate,
    int64_t first_row, bool mask_after_projection) {
#define LAUNCH(HAS_BIAS, HAS_MASK, MASK_AFTER) launch_input_persistent<Channels, HAS_BIAS, HAS_MASK, MASK_AFTER>( \
    input, weight, bias, mask, a, b, gate, first_row)
  if (bias != nullptr) {
    if (mask == nullptr) LAUNCH(true, false, false);
    else if (mask_after_projection) LAUNCH(true, true, true);
    else LAUNCH(true, true, false);
  } else {
    if (mask == nullptr) LAUNCH(false, false, false);
    else if (mask_after_projection) LAUNCH(false, true, true);
    else LAUNCH(false, true, false);
  }
#undef LAUNCH
}

void input_fused_out(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& gamma,
    const std::optional<torch::Tensor>& beta, double epsilon,
    const std::optional<torch::Tensor>& mask,
    torch::Tensor& a, torch::Tensor& b, torch::Tensor& gate,
    int64_t first_row, bool mask_after_projection, bool input_is_normalized) {
  const auto device = input.device();
  check_fp32_contiguous(input, "input", device);
  check_fp32_contiguous(weight, "weight", device);
  check_fp32_contiguous(a, "a", device);
  check_fp32_contiguous(b, "b", device);
  check_fp32_contiguous(gate, "gate", device);
  TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 && fused_channel_count(input.size(1)),
              "fused input must have shape [R,C], C in {32,64,96,128}");
  TORCH_CHECK(a.dim() == 3 && fused_channel_count(a.size(0)) && a.size(1) > 0
                  && a.size(1) == a.size(2) && b.sizes() == a.sizes(),
              "fused a/b must have shape [H,N,N], H in {32,64,96,128}");
  const int64_t channels = input.size(1);
  const int64_t hidden = a.size(0);
  const int64_t total_rows = a.size(1) * a.size(2);
  TORCH_CHECK(gate.dim() == 2 && gate.size(0) == total_rows && gate.size(1) == channels,
              "fused gate must have shape [N*N,C]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 4 * hidden + channels
                  && weight.size(1) == channels,
              "fused input weights must have layout [ap,ag,bp,bg,output_gate]");
  TORCH_CHECK(first_row >= 0 && first_row <= total_rows && input.size(0) <= total_rows - first_row,
              "fused input row chunk exceeds output storage");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0, "epsilon must be finite and positive");
  const float* bias_pointer = optional_parameter(bias, 4 * hidden + channels, device, "bias");
  const float* gamma_pointer = optional_parameter(gamma, channels, device, "gamma");
  const float* beta_pointer = optional_parameter(beta, channels, device, "beta");
  const float* mask_pointer = optional_parameter(mask, total_rows, device, "mask");
  at::assert_no_overlap(a, b);
  at::assert_no_overlap(a, gate);
  at::assert_no_overlap(b, gate);
  for (const auto& output : {a, b, gate}) {
    at::assert_no_overlap(output, input);
    at::assert_no_overlap(output, weight);
    for (const auto* parameter : {&bias, &gamma, &beta, &mask})
      if (parameter->has_value()) at::assert_no_overlap(output, **parameter);
  }
  const c10::cuda::CUDAGuard device_guard(device);
  // Pre-normalized rows take the persistent schedule when TMA can read them
  // (16-byte-aligned base; every row is aligned because C % 4 == 0), the
  // weights can be read as float4, and gate pairs can be stored as float2.
  if (input_is_normalized && aligned_to(input.data_ptr<float>(), 16)
      && aligned_to(weight.data_ptr<float>(), 16) && aligned_to(gate.data_ptr<float>(), 8)) {
#define LAUNCH_PERSISTENT_INPUT(C) dispatch_input_persistent<C>( \
    input, weight, bias_pointer, mask_pointer, a, b, gate, first_row, mask_after_projection)
    switch (channels) {
      case 32: LAUNCH_PERSISTENT_INPUT(32); break;
      case 64: LAUNCH_PERSISTENT_INPUT(64); break;
      case 96: LAUNCH_PERSISTENT_INPUT(96); break;
      case 128: LAUNCH_PERSISTENT_INPUT(128); break;
    }
#undef LAUNCH_PERSISTENT_INPUT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int channel_tiles = static_cast<int>((std::max(channels, hidden) + kInputChannels - 1) / kInputChannels);
  const int64_t row_tiles = (input.size(0) + kInputRows - 1) / kInputRows;
  TORCH_CHECK(row_tiles <= kMaxGridX / channel_tiles, "fused input grid exceeds CUDA grid-x range");
  const int64_t blocks = row_tiles * channel_tiles;
#define LAUNCH_INPUT_MODE(C, NORMALIZED) launch_input_fused<C, NORMALIZED>( \
    input, weight, bias_pointer, gamma_pointer, beta_pointer, \
    static_cast<float>(epsilon), mask_pointer, a, b, gate, first_row, \
    mask_after_projection, blocks, channel_tiles)
#define LAUNCH_INPUT(C) do { \
    if (input_is_normalized) { LAUNCH_INPUT_MODE(C, true); } \
    else { LAUNCH_INPUT_MODE(C, false); } \
  } while (false)
  switch (channels) {
    case 32: LAUNCH_INPUT(32); break;
    case 64: LAUNCH_INPUT(64); break;
    case 96: LAUNCH_INPUT(96); break;
    case 128: LAUNCH_INPUT(128); break;
  }
#undef LAUNCH_INPUT
#undef LAUNCH_INPUT_MODE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Hidden, int Channels, bool VectorLoads, bool HasBias>
void launch_output_persistent(
    const torch::Tensor& product, const torch::Tensor& weight, const float* bias,
    const float* gamma, const float* beta, float epsilon,
    const torch::Tensor& gate, torch::Tensor& output) {
  constexpr int stride = Hidden + 4;
  constexpr int shared_bytes =
      ((Channels + kPersistentOutputRows) * stride
       + 2 * Hidden * kPersistentOutputStagingStride) * sizeof(float);
  const auto kernel = output_persistent_kernel<Hidden, Channels, VectorLoads, HasBias>;
  static thread_local int configured_device = -1;
  if (configured_device != product.get_device()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    configured_device = product.get_device();
  }
  const int64_t rows = output.size(0);
  const int64_t row_tiles = (rows + kPersistentOutputRows - 1) / kPersistentOutputRows;
  const int64_t blocks = std::max<int64_t>(
      1, std::min<int64_t>(row_tiles, multiprocessor_count(product.get_device())));
  kernel<<<static_cast<unsigned int>(blocks), kOutputPersistentThreads, shared_bytes,
           at::cuda::getCurrentCUDAStream()>>>(
      product.data_ptr<float>(), weight.data_ptr<float>(), bias, gamma, beta,
      epsilon, gate.data_ptr<float>(), output.data_ptr<float>(), rows);
}

template <int Hidden, int Channels>
void dispatch_output_persistent(
    const torch::Tensor& product, const torch::Tensor& weight, const float* bias,
    const float* gamma, const float* beta, float epsilon,
    const torch::Tensor& gate, torch::Tensor& output) {
  const bool vector_loads = output.size(0) % 4 == 0 && aligned_to(product.data_ptr<float>(), 16);
#define LAUNCH(VECTOR_LOADS, HAS_BIAS) launch_output_persistent<Hidden, Channels, VECTOR_LOADS, HAS_BIAS>( \
    product, weight, bias, gamma, beta, epsilon, gate, output)
  if (vector_loads) {
    if (bias != nullptr) LAUNCH(true, true); else LAUNCH(true, false);
  } else {
    if (bias != nullptr) LAUNCH(false, true); else LAUNCH(false, false);
  }
#undef LAUNCH
}

template <int Hidden>
void dispatch_output_channels(
    const torch::Tensor& product, const torch::Tensor& weight, const float* bias,
    const float* gamma, const float* beta, float epsilon,
    const torch::Tensor& gate, torch::Tensor& output) {
  switch (output.size(1)) {
    case 32: dispatch_output_persistent<Hidden, 32>(product, weight, bias, gamma, beta, epsilon, gate, output); break;
    case 64: dispatch_output_persistent<Hidden, 64>(product, weight, bias, gamma, beta, epsilon, gate, output); break;
    case 96: dispatch_output_persistent<Hidden, 96>(product, weight, bias, gamma, beta, epsilon, gate, output); break;
    case 128: dispatch_output_persistent<Hidden, 128>(product, weight, bias, gamma, beta, epsilon, gate, output); break;
  }
}

void output_fused_out(
    const torch::Tensor& product, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& gamma,
    const std::optional<torch::Tensor>& beta, double epsilon,
    const torch::Tensor& gate, torch::Tensor& output) {
  const auto device = product.device();
  check_fp32_contiguous(product, "product", device);
  check_fp32_contiguous(weight, "weight", device);
  check_fp32_contiguous(gate, "gate", device);
  check_fp32_contiguous(output, "output", device);
  TORCH_CHECK(product.dim() == 3 && fused_channel_count(product.size(0))
                  && product.size(1) > 0 && product.size(2) > 0,
              "fused product must have nonempty channel-major shape [H,R,N]");
  const int64_t hidden = product.size(0);
  const int64_t rows = product.size(1) * product.size(2);
  TORCH_CHECK(output.dim() == 2 && output.size(0) == rows && fused_channel_count(output.size(1))
                  && gate.sizes() == output.sizes(),
              "fused output/gate must have shape [R*N,C], C in {32,64,96,128}");
  const int64_t channels = output.size(1);
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == hidden,
              "fused output weight must have shape [C,H]");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0, "epsilon must be finite and positive");
  const float* bias_pointer = optional_parameter(bias, channels, device, "bias");
  const float* gamma_pointer = optional_parameter(gamma, hidden, device, "gamma");
  const float* beta_pointer = optional_parameter(beta, hidden, device, "beta");
  for (const auto& input : {product, weight, gate}) at::assert_no_overlap(output, input);
  for (const auto* parameter : {&bias, &gamma, &beta})
    if (parameter->has_value()) at::assert_no_overlap(output, **parameter);
  TORCH_CHECK(aligned_to(gate.data_ptr<float>(), 8) && aligned_to(output.data_ptr<float>(), 8),
              "fused output/gate must be 8-byte aligned (channel pairs are accessed as float2)");
  const c10::cuda::CUDAGuard device_guard(device);
#define LAUNCH_OUTPUT(H) dispatch_output_channels<H>(product, weight, bias_pointer, gamma_pointer, \
    beta_pointer, static_cast<float>(epsilon), gate, output)
  switch (hidden) {
    case 32: LAUNCH_OUTPUT(32); break;
    case 64: LAUNCH_OUTPUT(64); break;
    case 96: LAUNCH_OUTPUT(96); break;
    case 128: LAUNCH_OUTPUT(128); break;
  }
#undef LAUNCH_OUTPUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("trimul_fp32_gate_pack_out", &gate_pack_out,
             "FP32 projection gates/mask and channel-major A/B packing");
  module.def("trimul_fp32_gate_output_out", &gate_output_out,
             "FP32 output projection times stored sigmoid gate");
  module.def("trimul_fp32_normalize_input_out", &normalize_input_out,
             "Normalize FP32 input rows once with nearest-even TF32 output rounding");
  module.def("trimul_fp32_input_fused_out", &input_fused_out,
             "FP32 input projections, gate/mask and A/B packing; optionally normalize input",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
             pybind11::arg("gamma"), pybind11::arg("beta"), pybind11::arg("epsilon"),
             pybind11::arg("mask"), pybind11::arg("a"), pybind11::arg("b"),
             pybind11::arg("gate"), pybind11::arg("first_row"),
             pybind11::arg("mask_after_projection"),
             pybind11::arg("input_is_normalized") = false);
  module.def("trimul_fp32_output_fused_out", &output_fused_out,
             "FP32 storage fused channel-major LN, TF32 output projection and gate");
}
