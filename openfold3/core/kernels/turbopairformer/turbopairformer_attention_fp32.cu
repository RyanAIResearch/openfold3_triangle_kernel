// SPDX-License-Identifier: Apache-2.0
// Inference-only FP32 Triangle Attention. All stored operands remain FP32.
//
// TF32 uses a fused, tiled QK / online-softmax / PV kernel. Its tensor storage,
// score arithmetic, running softmax state, and output accumulators are FP32;
// only Tensor Core multiplicands are explicitly rounded to TF32. K/V tiles
// are double-buffered in shared memory, with no global score/probability tile.
// Strict FP32 retains the bounded cuBLAS PEDANTIC implementation below.
// Neither path calls cuEquivariance, PyTorch attention, or the BF16 kernel.

#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kQueryTile = 128;
constexpr int64_t kMaxGridX = 2147483647LL;

void check_blas(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
                " failed with cuBLAS status ", static_cast<int>(status));
}

// PyTorch owns a per-thread handle. Restore both settings even if a later
// GEMM or CUDA launch fails; no numerical policy leaks into the host model.
class ScopedBlasSettings {
public:
    explicit ScopedBlasSettings(cublasHandle_t handle) : handle_(handle) {
        check_blas(cublasGetMathMode(handle_, &math_mode_), "cublasGetMathMode");
        check_blas(cublasGetPointerMode(handle_, &pointer_mode_), "cublasGetPointerMode");
    }

    void configure(bool allow_tf32) {
        check_blas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
                   "cublasSetPointerMode");
        check_blas(cublasSetMathMode(handle_, allow_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                                       : CUBLAS_PEDANTIC_MATH),
                   "cublasSetMathMode");
    }

    void restore() {
        // Attempt both restorations before reporting either failure.
        const auto math_status = cublasSetMathMode(handle_, math_mode_);
        const auto pointer_status = cublasSetPointerMode(handle_, pointer_mode_);
        check_blas(math_status, "restore cuBLAS math mode");
        check_blas(pointer_status, "restore cuBLAS pointer mode");
        restored_ = true;
    }

    ~ScopedBlasSettings() {
        if (!restored_) {
            // Destructors must not throw while unwinding the original failure.
            (void)cublasSetMathMode(handle_, math_mode_);
            (void)cublasSetPointerMode(handle_, pointer_mode_);
        }
    }

    ScopedBlasSettings(const ScopedBlasSettings&) = delete;
    ScopedBlasSettings& operator=(const ScopedBlasSettings&) = delete;

private:
    cublasHandle_t handle_;
    cublasMath_t math_mode_;
    cublasPointerMode_t pointer_mode_;
    bool restored_ = false;
};

// max(a, b), or the canonical NaN when either input is NaN. One max.NaN
// instruction has exactly these semantics: the same comparison as fmaxf for
// non-NaN inputs, and the canonical NaN (CUDART_NAN_F) otherwise.
__device__ __forceinline__ float maximum_propagating_nan(float a, float b) {
    float maximum;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(maximum) : "f"(a), "f"(b));
    return maximum;
}

// -----------------------------------------------------------------------------
// Fused TF32 forward
// -----------------------------------------------------------------------------

constexpr int kFusedWarps = 4;
constexpr int kFusedThreads = kFusedWarps * 32;
constexpr int kFusedQueryRows = kFusedWarps * 16;
constexpr int kFusedKeyRows = 64;

// The four accumulators of m16n8 are (row,2*lane_in_group+[0,1])
// followed by (row+8,2*lane_in_group+[0,1]), where row = lane / 4.
struct Fp32Accumulator {
    float values[4];
};

__device__ __forceinline__ uint32_t round_tf32(float value) {
    uint32_t rounded;
    // RN, rather than RZ, is part of the frozen TF32 numerical contract.
    asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(rounded) : "f"(value));
    return rounded;
}

__device__ __forceinline__ void mma_tf32(
    Fp32Accumulator& accumulator,
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(accumulator.values[0]), "+f"(accumulator.values[1]),
          "+f"(accumulator.values[2]), "+f"(accumulator.values[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void copy_fp32_async(float* destination, const float* source) {
    const uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(shared_address), "l"(source));
}

__device__ __forceinline__ float quad_maximum(float value) {
    value = maximum_propagating_nan(value, __shfl_xor_sync(0xffffffffU, value, 1));
    return maximum_propagating_nan(value, __shfl_xor_sync(0xffffffffU, value, 2));
}

__device__ __forceinline__ float quad_sum(float value) {
    value += __shfl_xor_sync(0xffffffffU, value, 1);
    return value + __shfl_xor_sync(0xffffffffU, value, 2);
}

// Convert the m16n8 accumulator layout into the m16k8 A layout without
// storing probabilities. The latter holds columns lane%4 and lane%4+4,
// rather than the accumulator's adjacent pair 2*(lane%4)+[0,1].
__device__ __forceinline__ void weights_to_tf32_a(
    const Fp32Accumulator& weights, int lane, uint32_t (&fragment)[4]) {
    const int within_row = lane & 3;
    const int first_source = (lane & ~3) + (within_row >> 1);
#pragma unroll
    for (int column_half = 0; column_half < 2; ++column_half) {
        const int source = first_source + 2 * column_half;
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
            // Both shuffles must execute in all lanes before the selection.
            const float even = __shfl_sync(0xffffffffU, weights.values[2 * row_half], source);
            const float odd = __shfl_sync(0xffffffffU, weights.values[2 * row_half + 1], source);
            fragment[2 * column_half + row_half] = round_tf32((within_row & 1) ? odd : even);
        }
    }
}

// A CTA owns 64 query rows from one [batch,outer,head] matrix. Four warps
// independently own 16 rows each, while all threads stage the next K/V tile.
// HeadDim pads only local storage/MMA lanes; input and output remain logical D.
// Gated multiplies each output value by the element of `gate` at the same
// offset (gate has the output's [B,O,N,H,D] layout), as the module's
// `output * gate` does after attention: one more rounded FP32 multiply. gate
// is the last parameter, so the ungated kernel keeps its previous code.
template <int HeadDim, bool Gated>
// Four blocks per SM: the register budget is capped at 128 per thread, the
// occupancy the 16-warp-per-SM schedule was tuned for.
__global__ __launch_bounds__(kFusedThreads, 4) void fused_tf32_attention(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t outer,
    int64_t heads,
    int sequence,
    int io_dim,
    int query_tiles,
    int64_t batch_stride,
    int64_t outer_stride,
    int64_t head_stride,
    int64_t sequence_stride,
    bool vector_copy_aligned,
    float scale,
    const float* __restrict__ gate) {
    // Four extra FP32 elements keep 16-byte cp.async alignment and reduce
    // shared-memory bank conflicts for the transposed K/V fragment reads.
    constexpr int kSharedStride = HeadDim + 4;
    constexpr int kTileElements = kFusedKeyRows * kSharedStride;
    extern __shared__ __align__(16) unsigned char shared_storage[];
    float* shared_kv = reinterpret_cast<float*>(shared_storage);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_row = lane >> 2;
    const int lane_column = lane & 3;
    const int64_t matrix = static_cast<int64_t>(blockIdx.x) / query_tiles;
    const int first_query = static_cast<int>(blockIdx.x % query_tiles) * kFusedQueryRows;
    const int warp_query = first_query + warp * 16;
    const int64_t batch_outer = matrix / heads;
    const int64_t head = matrix % heads;
    const int64_t batch = batch_outer / outer;
    const int64_t outer_index = batch_outer - batch * outer;
    // The model's projections arrive as [B,O,N,H,D] views transposed into
    // [B,O,H,N,D]. Read that layout directly instead of packing three tensors.
    // Q/K/V share these strides; arbitrary layouts are packed by the facade.
    const int64_t qkv_base = batch * batch_stride + outer_index * outer_stride
        + head * head_stride;
    const int64_t mask_base = batch_outer * sequence;
    const int64_t bias_base = (batch * heads + head) * sequence * sequence;
    const int64_t output_base = batch_outer * sequence * heads * io_dim + head * io_dim;

    // Q is loaded once and split into two TF32 terms. QK uses the three
    // products Q_hi*K_hi + Q_hi*K_lo + Q_lo*K_hi, omitting only the product
    // of the two small residuals. This avoids letting QK rounding errors grow
    // through softmax while still using TF32 Tensor Core instructions.
    // TF32 m16n8k8 A registers are
    // (r,c), (r+8,c), (r,c+4), (r+8,c+4), with r=lane/4,c=lane%4.
    uint32_t query_fragments[HeadDim / 8][4];
    uint32_t query_residuals[HeadDim / 8][4];
#pragma unroll
    for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
#pragma unroll
        for (int column_half = 0; column_half < 2; ++column_half) {
            const int channel = channel_tile * 8 + lane_column + column_half * 4;
#pragma unroll
            for (int row_half = 0; row_half < 2; ++row_half) {
                const int row = warp_query + lane_row + row_half * 8;
                const float value = row < sequence && channel < io_dim
                    ? q[qkv_base + static_cast<int64_t>(row) * sequence_stride + channel] : 0.0F;
                const int element = 2 * column_half + row_half;
                query_fragments[channel_tile][element] = round_tf32(value);
                query_residuals[channel_tile][element] = round_tf32(
                    value - __uint_as_float(query_fragments[channel_tile][element]));
            }
        }
    }

    // This thread's two query rows: validity, and their mask/bias rows (an
    // invalid row's pointer is clamped into the tensor and never dereferenced).
    bool query_valid[2];
    const float* bias_rows[2];
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
        const int query = warp_query + lane_row + row_half * 8;
        query_valid[row_half] = query < sequence;
        bias_rows[row_half] = bias + bias_base + static_cast<int64_t>(min(query, sequence - 1)) * sequence;
    }
    const float* mask_row = mask + mask_base;
    // Mask and bias pairs (two adjacent keys) can be read as one float2 when
    // every row starts 8-byte aligned.
    const bool pair_loads = sequence % 2 == 0
        && reinterpret_cast<uintptr_t>(mask_row) % 8 == 0
        && reinterpret_cast<uintptr_t>(bias + bias_base) % 8 == 0;

    float row_maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float weight_sum[2] = {0.0F, 0.0F};
    Fp32Accumulator weighted_output[HeadDim / 8];
#pragma unroll
    for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
        weighted_output[channel_tile] = Fp32Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    }

    // Vector K/V copies: a thread always copies the same 16-byte channel group
    // of rows copy_row, copy_row + kCopyRowsPerPass, ... of each 64-key tile.
    constexpr int kCopyVectorsPerRow = HeadDim / 4;
    constexpr int kCopyRowsPerPass = kFusedThreads / kCopyVectorsPerRow;
    const int copy_channel = (static_cast<int>(threadIdx.x) % kCopyVectorsPerRow) * 4;
    const int copy_row = static_cast<int>(threadIdx.x) / kCopyVectorsPerRow;
    const bool copy_channel_live = copy_channel < io_dim;
    const int64_t copy_pass_stride = static_cast<int64_t>(kCopyRowsPerPass) * sequence_stride;

    const auto prefetch = [&](int key_start, int slot) {
        float* destination_k = shared_kv + slot * 2 * kTileElements;
        float* destination_v = destination_k + kTileElements;
        if (vector_copy_aligned) {
            // A full 16-byte vector never crosses a logical input row.
            const int64_t first_source = qkv_base
                + static_cast<int64_t>(key_start + copy_row) * sequence_stride + copy_channel;
#pragma unroll
            for (int pass = 0; pass < kFusedKeyRows / kCopyRowsPerPass; ++pass) {
                const int row = copy_row + pass * kCopyRowsPerPass;
                const int destination = row * kSharedStride + copy_channel;
                if (key_start + row < sequence && copy_channel_live) {
                    const int64_t source = first_source + pass * copy_pass_stride;
                    copy_fp32_async(destination_k + destination, k + source);
                    copy_fp32_async(destination_v + destination, v + source);
                } else {
                    *reinterpret_cast<float4*>(destination_k + destination) =
                        make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                    *reinterpret_cast<float4*>(destination_v + destination) =
                        make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                }
            }
        } else {
            // Odd widths and unaligned strided views use guarded scalar
            // copies. Neither row padding nor pointer alignment is assumed.
            for (int element = threadIdx.x; element < kFusedKeyRows * HeadDim;
                 element += kFusedThreads) {
                const int row = element / HeadDim;
                const int channel = element % HeadDim;
                const bool valid = key_start + row < sequence && channel < io_dim;
                const int64_t source = qkv_base + static_cast<int64_t>(key_start + row)
                    * sequence_stride + channel;
                destination_k[row * kSharedStride + channel] = valid ? k[source] : 0.0F;
                destination_v[row * kSharedStride + channel] = valid ? v[source] : 0.0F;
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    prefetch(0, 0);

    const int key_tiles = (sequence - 1) / kFusedKeyRows + 1;
#pragma unroll 1
    for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
        const int key_start = key_tile * kFusedKeyRows;
        const int slot = key_tile & 1;
        const float* shared_k = shared_kv + slot * 2 * kTileElements;
        const float* shared_v = shared_k + kTileElements;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        // Also completes reads of the previous tile before that slot is reused.
        __syncthreads();
        if (key_tile + 1 < key_tiles) {
            prefetch(key_start + kFusedKeyRows, slot ^ 1);
        }

        // Stage 1: QK. Each MMA produces 16 query rows by eight key columns.
        Fp32Accumulator scores[kFusedKeyRows / 8];
#pragma unroll
        for (int key_column_tile = 0; key_column_tile < kFusedKeyRows / 8; ++key_column_tile) {
            scores[key_column_tile] = Fp32Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
            for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
                const int key_row = key_column_tile * 8 + lane_row;
                const int channel = channel_tile * 8 + lane_column;
                const float key_low_column = shared_k[key_row * kSharedStride + channel];
                const float key_high_column = shared_k[key_row * kSharedStride + channel + 4];
                uint32_t key_fragment[2] = {round_tf32(key_low_column), round_tf32(key_high_column)};
                uint32_t key_residual[2] = {
                    round_tf32(key_low_column - __uint_as_float(key_fragment[0])),
                    round_tf32(key_high_column - __uint_as_float(key_fragment[1]))};
                mma_tf32(scores[key_column_tile], query_residuals[channel_tile], key_fragment);
                mma_tf32(scores[key_column_tile], query_fragments[channel_tile], key_residual);
                mma_tf32(scores[key_column_tile], query_fragments[channel_tile], key_fragment);
            }
        }

        // Stage 2: preserve FP32 scale -> mask -> bias rounding order. FMA
        // contraction here would change finite all--1e9-mask semantics.
        float tile_maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
        // In a full key tile every key is in range, and pairs load as float2.
        const bool full_key_tile = key_start + kFusedKeyRows <= sequence;
        const bool pair_tile = full_key_tile && pair_loads;
#pragma unroll
        for (int key_column_tile = 0; key_column_tile < kFusedKeyRows / 8; ++key_column_tile) {
            const int key_pair = key_start + key_column_tile * 8 + lane_column * 2;
            float mask_values[2] = {0.0F, 0.0F};
            if (pair_tile) {
                const float2 pair = *reinterpret_cast<const float2*>(mask_row + key_pair);
                mask_values[0] = pair.x;
                mask_values[1] = pair.y;
            } else {
#pragma unroll
                for (int adjacent = 0; adjacent < 2; ++adjacent)
                    if (key_pair + adjacent < sequence) mask_values[adjacent] = mask_row[key_pair + adjacent];
            }
#pragma unroll
            for (int row_half = 0; row_half < 2; ++row_half) {
                float bias_values[2] = {0.0F, 0.0F};
                if (pair_tile) {
                    const float2 pair = *reinterpret_cast<const float2*>(bias_rows[row_half] + key_pair);
                    bias_values[0] = pair.x;
                    bias_values[1] = pair.y;
                } else {
#pragma unroll
                    for (int adjacent = 0; adjacent < 2; ++adjacent)
                        if (key_pair + adjacent < sequence)
                            bias_values[adjacent] = bias_rows[row_half][key_pair + adjacent];
                }
#pragma unroll
                for (int adjacent = 0; adjacent < 2; ++adjacent) {
                    float value = -CUDART_INF_F;
                    if (query_valid[row_half] && (full_key_tile || key_pair + adjacent < sequence)) {
                        value = __fmul_rn(scores[key_column_tile].values[row_half * 2 + adjacent], scale);
                        value = __fadd_rn(value, mask_values[adjacent]);
                        value = __fadd_rn(value, bias_values[adjacent]);
                    }
                    scores[key_column_tile].values[row_half * 2 + adjacent] = value;
                    tile_maximum[row_half] = maximum_propagating_nan(tile_maximum[row_half], value);
                }
            }
        }
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
            const float updated_maximum = maximum_propagating_nan(
                row_maximum[row_half], quad_maximum(tile_maximum[row_half]));
            // expf is evaluated unconditionally and then selected, so the
            // compiler emits no branch around it; the selected values are the same.
            const float history_exponential = expf(row_maximum[row_half] - updated_maximum);
            const float history_scale = row_maximum[row_half] == -CUDART_INF_F
                ? 0.0F : history_exponential;
            row_maximum[row_half] = updated_maximum;
            // The running sum is rescaled and then extended with this tile's sum as
            // two separately rounded operations (never one fused multiply-add).
            weight_sum[row_half] = __fmul_rn(weight_sum[row_half], history_scale);
#pragma unroll
            for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
                weighted_output[channel_tile].values[row_half * 2] *= history_scale;
                weighted_output[channel_tile].values[row_half * 2 + 1] *= history_scale;
            }
        }

        // Stage 3: online-softmax weights stay in registers. Their FP32 sum
        // normalizes the output after all key tiles have been visited.
        float tile_weight_sum[2] = {0.0F, 0.0F};
#pragma unroll
        for (int key_column_tile = 0; key_column_tile < kFusedKeyRows / 8; ++key_column_tile) {
#pragma unroll
            for (int row_half = 0; row_half < 2; ++row_half) {
#pragma unroll
                for (int adjacent = 0; adjacent < 2; ++adjacent) {
                    float& weight = scores[key_column_tile].values[row_half * 2 + adjacent];
                    // Evaluated unconditionally and then selected (no branch per element).
                    const float exponential = expf(weight - row_maximum[row_half]);
                    weight = weight == -CUDART_INF_F ? 0.0F : exponential;
                    tile_weight_sum[row_half] += weight;
                }
            }
        }
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
            weight_sum[row_half] = __fadd_rn(weight_sum[row_half], quad_sum(tile_weight_sum[row_half]));
        }

        // Stage 4: PV with ordinary TF32 rounding of both multiplicands and
        // FP32 accumulators. QK's three-term decomposition above protects the
        // softmax logits; PV uses one MMA per fragment to avoid unnecessary
        // residual products. Online weights are not yet normalized, so this
        // arithmetic is qualified against the fixed numerical gate rather
        // than claimed bitwise equal to a normalized-probability GEMM.
        // MMA B holds V[key=lane%4+[0,4], channel=lane/4] in each 8-key tile.
#pragma unroll
        for (int key_chunk = 0; key_chunk < kFusedKeyRows / 8; ++key_chunk) {
            uint32_t probability_fragment[4];
            weights_to_tf32_a(scores[key_chunk], lane, probability_fragment);
#pragma unroll
            for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
                const int key = key_chunk * 8 + lane_column;
                const int channel = channel_tile * 8 + lane_row;
                uint32_t value_fragment[2] = {
                    round_tf32(shared_v[key * kSharedStride + channel]),
                    round_tf32(shared_v[(key + 4) * kSharedStride + channel])};
                mma_tf32(weighted_output[channel_tile], probability_fragment, value_fragment);
            }
        }
    }

    // Direct public-layout stores replace the separate output-scatter launch.
#pragma unroll
    for (int row_half = 0; row_half < 2; ++row_half) {
        const int query = warp_query + lane_row + row_half * 8;
        const float inverse_sum = weight_sum[row_half] == 0.0F ? 0.0F : 1.0F / weight_sum[row_half];
#pragma unroll
        for (int channel_tile = 0; channel_tile < HeadDim / 8; ++channel_tile) {
#pragma unroll
            for (int adjacent = 0; adjacent < 2; ++adjacent) {
                const int channel = channel_tile * 8 + lane_column * 2 + adjacent;
                if (query < sequence && channel < io_dim) {
                    if constexpr (Gated) {
                        const int64_t offset =
                            output_base + static_cast<int64_t>(query) * heads * io_dim + channel;
                        const float attention = __fmul_rn(
                            weighted_output[channel_tile].values[row_half * 2 + adjacent], inverse_sum);
                        output[offset] = __fmul_rn(attention, gate[offset]);
                    } else {
                        // Written as before the gate option existed: the ungated
                        // kernels compile to the same SASS.
                        output[output_base + static_cast<int64_t>(query) * heads * io_dim + channel]
                            = weighted_output[channel_tile].values[row_half * 2 + adjacent] * inverse_sum;
                    }
                }
            }
        }
    }
}

// gate is null unless Gated.
template <int HeadDim, bool Gated>
void launch_fused_tf32(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& mask, const torch::Tensor& bias, const float* gate, torch::Tensor& output,
    int64_t matrices, cudaStream_t stream) {
    const int sequence = static_cast<int>(q.size(3));
    const int query_tiles = (sequence - 1) / kFusedQueryRows + 1;
    bool vector_copy_aligned = (q.size(4) % 4 == 0);
    for (int axis = 0; axis < 4; ++axis) {
        vector_copy_aligned = vector_copy_aligned && (q.stride(axis) % 4 == 0);
    }
    vector_copy_aligned = vector_copy_aligned
        && (reinterpret_cast<uintptr_t>(q.data_ptr<float>()) % 16 == 0)
        && (reinterpret_cast<uintptr_t>(k.data_ptr<float>()) % 16 == 0)
        && (reinterpret_cast<uintptr_t>(v.data_ptr<float>()) % 16 == 0);
    constexpr int shared_bytes = 4 * kFusedKeyRows * (HeadDim + 4) * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(fused_tf32_attention<HeadDim, Gated>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    fused_tf32_attention<HeadDim, Gated><<<static_cast<unsigned int>(matrices * query_tiles),
        kFusedThreads, shared_bytes, stream>>>(q.data_ptr<float>(), k.data_ptr<float>(),
        v.data_ptr<float>(), mask.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        q.size(1), q.size(2), sequence, static_cast<int>(q.size(4)), query_tiles,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), vector_copy_aligned,
        1.0F / std::sqrt(static_cast<float>(q.size(4))), gate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// One CTA owns one query row. The two fixed-order shared reductions do not
// use atomics. Bias is shared across O; mask is shared across H and queries.
__global__ void mask_bias_softmax(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    const float* __restrict__ bias,
    int64_t first_matrix,
    int64_t outer,
    int64_t heads,
    int sequence,
    int first_query,
    int query_count,
    int query_capacity) {
    __shared__ float reduction[kThreads];
    const int64_t local_matrix = static_cast<int64_t>(blockIdx.x) / query_count;
    const int query = static_cast<int>(blockIdx.x % query_count);
    const int64_t matrix = first_matrix + local_matrix;
    const int64_t batch_outer = matrix / heads;
    const int64_t batch = batch_outer / outer;
    const int64_t head = matrix % heads;
    const int64_t score_base = (local_matrix * query_capacity + query) * sequence;
    const int64_t mask_base = batch_outer * sequence;
    const int64_t bias_base = ((batch * heads + head) * sequence + first_query + query)
                              * sequence;

    float row_maximum = -CUDART_INF_F;
    for (int64_t key = threadIdx.x; key < sequence; key += blockDim.x) {
        // Match the model's mask-then-pair-bias additive order. In particular,
        // finite -1e9 masks are not interpreted as boolean masked positions.
        float value = scores[score_base + key] + mask[mask_base + key];
        value += bias[bias_base + key];
        scores[score_base + key] = value;
        row_maximum = maximum_propagating_nan(row_maximum, value);
    }
    reduction[threadIdx.x] = row_maximum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] = maximum_propagating_nan(
                reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    row_maximum = reduction[0];
    // All true -inf logits have a zero output, as in the documented inference
    // contract. NaN maxima take the ordinary arithmetic path and propagate.
    const bool all_negative_infinity = row_maximum == -CUDART_INF_F;
    float row_sum = 0.0F;
    for (int64_t key = threadIdx.x; key < sequence; key += blockDim.x) {
        const float probability = all_negative_infinity
                                      ? 0.0F
                                      : expf(scores[score_base + key] - row_maximum);
        scores[score_base + key] = probability;
        row_sum += probability;
    }
    __syncthreads();
    reduction[threadIdx.x] = row_sum;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float denominator = all_negative_infinity ? 1.0F : reduction[0];
    for (int64_t key = threadIdx.x; key < sequence; key += blockDim.x) {
        scores[score_base + key] /= denominator;
    }
}

__global__ void scatter_output(
    const float* __restrict__ tile,
    float* __restrict__ output,
    int64_t first_matrix,
    int64_t heads,
    int sequence,
    int head_dim,
    int first_query,
    int query_count,
    int query_capacity,
    int64_t elements) {
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += step) {
        const int channel = static_cast<int>(index % head_dim);
        const int64_t row = index / head_dim;
        const int query = static_cast<int>(row % query_count);
        const int64_t local_matrix = row / query_count;
        const int64_t matrix = first_matrix + local_matrix;
        const int64_t batch_outer = matrix / heads;
        const int64_t head = matrix % heads;
        const int64_t source = (local_matrix * query_capacity + query) * head_dim + channel;
        const int64_t destination =
            ((batch_outer * sequence + first_query + query) * heads + head) * head_dim
            + channel;
        output[destination] = tile[source];
    }
}

void check_input(const torch::Tensor& tensor, const torch::Tensor& q, const char* name,
                 bool require_contiguous = true) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device()
                    && tensor.scalar_type() == at::kFloat
                    && (!require_contiguous || tensor.is_contiguous()),
                name, " must be FP32 on q's CUDA device",
                require_contiguous ? " and contiguous" : "");
}

// -----------------------------------------------------------------------------
// TriangleAttention module front end (TF32 inference)
// -----------------------------------------------------------------------------
// For every pair row x of C = 128 channels the module computes
//   x_ln = LayerNorm(x),
//   q, k, v = x_ln W_q^T, x_ln W_k^T, x_ln W_v^T          (128 values each),
//   gate = sigmoid(x_ln W_g^T),  pair_bias[h] = x_ln . W_z[h].
// PyTorch runs this as one LayerNorm, five cuBLAS GEMMs and a sigmoid, which
// writes x_ln once and reads it five times. Two kernels produce the same bits:
//   1. layer_norm_rows_tf32 repeats PyTorch's vectorized LayerNorm arithmetic
//      with one warp per row and stores x_ln already rounded to TF32.
//   2. attention_projections_tf32 multiplies 64-row tiles of x_ln by resident
//      weights with wgmma and writes q, k, v, the gate and the pair bias.
// The bit equality rests on three facts measured on H100 (2026-11-01 probe):
//   - PyTorch 2.10's vectorized_layer_norm_kernel<float> evaluates exactly the
//     roundings and fused multiply-adds written below (read from its SASS;
//     0 differences in 65M outputs, including near-constant and wide rows);
//   - cuBLAS's TF32 GEMMs round both operands to TF32 with ties-to-even and
//     accumulate one ascending-K chain from zero (0 differences at the
//     model's shapes, N = 128 and N = 4);
//   - torch.sigmoid equals 1 / (1 + expf(-x)) for all 2^32 FP32 inputs.

// Welford state as PyTorch's LayerNorm keeps it: running mean, sum of squared
// deviations from the mean, and the element count (a float, as in PyTorch).
struct WelfordState {
    float mean;
    float squared_deviations;
    float count;
};

// One element of cuWelfordOnlineSum. PyTorch multiplies by the reciprocal of
// the new count instead of dividing, and its build fuses both updates:
//   mean' = fma(delta, 1/count', mean),  m2' = fma(delta, value - mean', m2).
// reciprocal_count is the correctly rounded reciprocal of the new count.
__device__ __forceinline__ WelfordState welford_add(
    WelfordState state, float value, float reciprocal_count) {
    const float delta = __fsub_rn(value, state.mean);
    const float mean = __fmaf_rn(delta, reciprocal_count, state.mean);
    const float squared_deviations =
        __fmaf_rn(delta, __fsub_rn(value, mean), state.squared_deviations);
    return {mean, squared_deviations, __fadd_rn(state.count, 1.0F)};
}

// cuWelfordCombine(own, other) as PyTorch compiles it. An empty pair keeps zero
// mean and deviations. Otherwise, with shares n = count / (own + other count):
//   mean = fma(other.mean, n_other, own.mean * n_own)
//   m2   = fma(n_own, other.count * delta^2, other.m2 + own.m2).
__device__ __forceinline__ WelfordState welford_combine(WelfordState own, WelfordState other) {
    const float count = __fadd_rn(own.count, other.count);
    if (!(own.count > -other.count)) return {0.0F, 0.0F, count};
    const float delta = __fsub_rn(own.mean, other.mean);
    const float reciprocal = __frcp_rn(count);
    const float own_share = __fmul_rn(own.count, reciprocal);
    const float other_share = __fmul_rn(other.count, reciprocal);
    const float mean = __fmaf_rn(other.mean, other_share, __fmul_rn(own.mean, own_share));
    const float spread = __fmul_rn(other.count, __fmul_rn(delta, delta));
    const float squared_deviations = __fmaf_rn(
        own_share, spread, __fadd_rn(other.squared_deviations, own.squared_deviations));
    return {mean, squared_deviations, count};
}

constexpr int kLayerNormWarps = 8;

// One warp per pair row of x [B, N, N, Channels]: any batch/row/column strides,
// unit channel stride, 16-byte-aligned rows (checked by the host). Lane l holds
// channels 4l..4l+3, like thread l of the first warp in PyTorch's 128-thread
// LayerNorm block; lanes past Channels/4 and PyTorch's other three warps hold
// empty states. The normalized row is stored contiguously, rounded to TF32
// (ties to even), which is the value every projection GEMM consumes.
template <int Channels>
__global__ __launch_bounds__(kLayerNormWarps * 32) void layer_norm_rows_tf32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int64_t rows,
    int64_t sequence,
    int64_t batch_stride,
    int64_t row_stride,
    int64_t column_stride,
    float epsilon) {
    static_assert(Channels % 4 == 0 && Channels <= 128, "one warp holds at most 128 channels");
    constexpr int kLiveLanes = Channels / 4;
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kLayerNormWarps + threadIdx.x / 32;
    if (row >= rows) return;
    const int64_t plane = sequence * sequence;
    const int64_t batch = row / plane;
    const int64_t within_plane = row - batch * plane;
    const int64_t first_index = within_plane / sequence;
    const int64_t second_index = within_plane - first_index * sequence;
    const float* source = input + batch * batch_stride + first_index * row_stride
        + second_index * column_stride;

    const bool live = lane < kLiveLanes;
    float4 values = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (live) values = *reinterpret_cast<const float4*>(source + 4 * lane);

    // The first four Welford steps of a thread; 1/3 rounds to 0x3EAAAAAB.
    WelfordState state{0.0F, 0.0F, 0.0F};
    if (live) {
        state = welford_add(state, values.x, 1.0F);
        state = welford_add(state, values.y, 0.5F);
        state = welford_add(state, values.z, __uint_as_float(0x3EAAAAABU));
        state = welford_add(state, values.w, 0.25F);
    }
    // PyTorch's shuffle-down tree; lane 0 ends with the row's statistics.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const WelfordState other{__shfl_down_sync(0xffffffffU, state.mean, offset),
                                 __shfl_down_sync(0xffffffffU, state.squared_deviations, offset),
                                 __shfl_down_sync(0xffffffffU, state.count, offset)};
        state = welford_combine(state, other);
    }
    // PyTorch then merges warp 0 with its block's three empty warps (two merge
    // steps). They change nothing for finite statistics but are kept so that
    // overflowing rows give PyTorch's result too.
    state = welford_combine(state, WelfordState{0.0F, 0.0F, 0.0F});
    state = welford_combine(state, WelfordState{0.0F, 0.0F, 0.0F});
    const float mean = __shfl_sync(0xffffffffU, state.mean, 0);
    const float variance = __fdiv_rn(__shfl_sync(0xffffffffU, state.squared_deviations, 0),
                                     static_cast<float>(Channels));
    const float inverse_deviation = rsqrtf(__fadd_rn(variance, epsilon));
    if (!live) return;

    // y = fma(gamma, rstd * (x - mean), beta), as PyTorch's output loop.
    const float4 scale = reinterpret_cast<const float4*>(gamma)[lane];
    const float4 shift = reinterpret_cast<const float4*>(beta)[lane];
    const auto normalize = [&](float value, float gamma_value, float beta_value) {
        const float centered = __fmul_rn(inverse_deviation, __fsub_rn(value, mean));
        return __uint_as_float(round_tf32(__fmaf_rn(gamma_value, centered, beta_value)));
    };
    reinterpret_cast<float4*>(output + row * Channels)[lane] = make_float4(
        normalize(values.x, scale.x, shift.x), normalize(values.y, scale.y, shift.y),
        normalize(values.z, scale.z, shift.z), normalize(values.w, scale.w, shift.w));
}

// torch.sigmoid for FP32: 1 / (1 + exp(-x)) with the accurate expf and an
// IEEE division (equal to PyTorch's kernel for all 2^32 inputs).
__device__ __forceinline__ float sigmoid_like_pytorch(float value) {
    return __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-value)));
}

// --- wgmma operands and helpers (sm_90a) -------------------------------------
// The same layouts as the TF32 TriMul projections. A warpgroup (four warps)
// computes a 64-row x N-column product with wgmma.mma_async from shared-memory
// operands. A chain of m64nNk8 steps gives the same bits as the chain of
// m16n8k8 mma.sync steps when K advances in the same ascending order from
// zero-initialized accumulators (probe/wgmma_tf32_bits.cu of 2026-11-01).
//
// Operand layout (128-byte swizzle, K-major): a [rows][K] tile is stored as K
// blocks of 32 TF32 words (128 bytes per row). Block kb holds rows 0..rows-1
// contiguously, and the 16-byte chunk c of row r sits at chunk c ^ (r % 8).
// Every block starts 1024-byte aligned; rows is a multiple of 8.
__device__ __forceinline__ int sw128_k_major_offset(int row, int column, int rows) {
    const int k_block = column / 32, column_in_block = column % 32;
    return k_block * rows * 32 + row * 32 + (((column_in_block / 4) ^ (row % 8)) * 4)
        + column_in_block % 4;
}

// Descriptor of operand rows [first_row, first_row + extent) at reduction index
// 0 of a tile stored with sw128_k_major_offset. Fields are in 16-byte units:
// start address (bits 0-13); leading byte offset (bits 16-29, unused by
// swizzled K-major layouts, set to 1); stride between 8-row groups = 8 * 128
// bytes (bits 32-45); swizzle mode 1 = 128 bytes (bits 62-63).
__device__ __forceinline__ uint64_t sw128_k_major_descriptor(const float* tile, int first_row) {
    const unsigned start = static_cast<unsigned>(__cvta_generic_to_shared(tile))
        + static_cast<unsigned>(first_row * 32 * sizeof(float));
    return (static_cast<uint64_t>(start >> 4) & 0x3FFF) | (uint64_t{1} << 16)
        | (uint64_t{(8 * 128) >> 4} << 32) | (uint64_t{1} << 62);
}

// Ordinary shared-memory stores become visible to wgmma's asynchronous-proxy
// reads only after this fence.
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

// Synchronizes the 128 threads of one warpgroup (named barrier 1 + warpgroup;
// barrier 0 is __syncthreads).
__device__ __forceinline__ void warpgroup_barrier(int warpgroup) {
    asm volatile("bar.sync %0, 128;" ::"r"(1 + warpgroup) : "memory");
}

// Shared-memory barriers for the TMA tile loads. A phase completes when its
// single arrival (the thread that issues the loads, announcing the byte count)
// has arrived and all expected bytes have landed; waiters test the parity.
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
// count its bytes on `barrier`. The map's 128-byte swizzle writes the box in
// the sw128_k_major_offset layout of one K block. Rows past the tensor arrive
// as zeros.
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* map, float* destination, uint64_t* barrier, int column, int64_t row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(destination))),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(column), "r"(static_cast<int>(row)),
        "r"(static_cast<unsigned>(__cvta_generic_to_shared(barrier)))
        : "memory");
}

// accumulator += A[64 x 8] * B[64 x 8]^T. accumulator[j][e] is element e of the
// 8-column block j in the m16n8 fragment layout of this thread's warp:
// row 16*(warp%4) + lane/4 (+8 for e >= 2), column 8j + 2*(lane%4) + e%2.
// scale-d is always true (D += A*B), as for an mma.sync chain from zeros.
__device__ __forceinline__ void wgmma_m64n64k8_tf32(
    float (&accumulator)[8][4], uint64_t descriptor_a, uint64_t descriptor_b) {
    asm volatile(
        "{\n"
        ".reg .pred accumulate;\n"
        "setp.ne.b32 accumulate, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, accumulate, 1, 1;\n"
        "}\n"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
          "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
          "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]),
          "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]),
          "+f"(accumulator[6][0]), "+f"(accumulator[6][1]), "+f"(accumulator[6][2]), "+f"(accumulator[6][3]),
          "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3])
        : "l"(descriptor_a), "l"(descriptor_b));
}

// The same for one 8-column block (the pair-bias heads).
__device__ __forceinline__ void wgmma_m64n8k8_tf32(
    float (&accumulator)[4], uint64_t descriptor_a, uint64_t descriptor_b) {
    asm volatile(
        "{\n"
        ".reg .pred accumulate;\n"
        "setp.ne.b32 accumulate, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
        "{%0,%1,%2,%3}, %4, %5, accumulate, 1, 1;\n"
        "}\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "l"(descriptor_a), "l"(descriptor_b));
}

constexpr int kProjectionChannels = 128;   // C, the reduction length
constexpr int kProjectionWidth = 128;      // H * D columns of q, k, v and gate
constexpr int kProjectionBiasColumns = 8;  // pair-bias heads, padded to one n8 block
constexpr int kProjectionWeightRows = kProjectionWidth + kProjectionBiasColumns;
constexpr int kProjectionTileRows = 64;    // wgmma M: rows per tile
constexpr int kProjectionWarpgroups = 2;
constexpr int kProjectionThreads = kProjectionWarpgroups * 128;
constexpr int kProjectionStages = 3;       // TMA input buffers in flight
constexpr int kProjectionTileValues = kProjectionTileRows * kProjectionChannels;
constexpr int kProjectionSharedBytes =
    (kProjectionWeightRows * kProjectionChannels             // resident weights
     + kProjectionStages * kProjectionTileValues)            // input buffers
        * static_cast<int>(sizeof(float))
    + kProjectionStages * static_cast<int>(sizeof(uint64_t));  // tile_loaded barriers

// A block owns a 32-column slice of each of q, k, v and gate, and warpgroup w
// multiplies by weight rows [64w, 64w + 64): eight 8-column blocks. Blocks 0-3
// are q (w = 0) or v (w = 1); blocks 4-5 are gate columns [16w, 16w + 16) and
// blocks 6-7 are k columns [16w, 16w + 16) of the slice, so both warpgroups
// apply the sigmoid to the same number of values and reach the per-tile
// barrier together. 0 = q, 1 = k, 2 = v, 3 = gate.
__device__ __forceinline__ int projection_of_block(int warpgroup, int block) {
    return block < 4 ? (warpgroup == 0 ? 0 : 2) : (block < 6 ? 3 : 1);
}

// First column of the block inside the slice's 32 columns of its projection.
__device__ __forceinline__ int slice_column_of_block(int warpgroup, int block) {
    return block < 4 ? 8 * block : 16 * warpgroup + 8 * ((block - 4) % 2);
}

// Which outputs one launch of attention_projections_tf32 writes. The chunked
// module computes the pair bias of every row first (kPairBias), because each
// chunk's attention reads all of it, and then q, k, v and the gate one outer-row
// chunk at a time (kProjections); the unchunked module does both in one pass.
// Each mode is a separate compile-time copy, so no branch separates the wgmma
// steps (see the kernel comment).
enum class ProjectionOutputs { kAll, kProjections, kPairBias };

// Projection kernel over the rows [first_row, first_row + rows) of x_ln.
// blockIdx.x = slice s owns columns [32s, 32s + 32) of each of q, k, v and
// gate, so the four slices share the sigmoid work evenly (kPairBias launches
// one slice). The blockIdx.y blocks walk the 64-row tiles, and the slices with
// the same blockIdx.y read the same tiles at about the same time, so x_ln is
// fetched from DRAM about once.
//
// Each block keeps its 128 + 8 TF32 weight rows in shared memory for the whole
// walk. Both warpgroups work on the same tile: warpgroup w multiplies it by
// weight rows [64w, 64w + 64) (projection_of_block), and both multiply it by
// the 8 bias rows; the bias stores are shared out over slices and warpgroups.
// Every thread runs the same loop, with no branch on the thread index around
// the wgmma sequence: ptxas serializes wgmma steps whose warpgroup it cannot
// prove convergent. Thread 0 keeps kProjectionStages tiles in flight with TMA.
//
// Values: q/k/v = the FP32 accumulator; gate = sigmoid_like_pytorch(accumulator),
// stored at row r - first_row; pair_bias[b, h, i, j] = accumulator of x_ln row
// (b, i, j), or of row (b, j, i) when transpose_bias. Rows past the range are
// computed (from the next rows or TMA's zero fill) and never stored. The host
// guarantees 8-byte-aligned outputs and sequence^2 rows per batch item.
template <ProjectionOutputs Outputs>
__global__ __launch_bounds__(kProjectionThreads, 1) void attention_projections_tf32(
    const __grid_constant__ CUtensorMap normalized_map,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ v_weight,
    const float* __restrict__ gate_weight,
    const float* __restrict__ bias_weight,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    float* __restrict__ gate,
    float* __restrict__ pair_bias,
    int64_t first_row,
    int64_t rows,
    int64_t sequence,
    int heads,
    bool transpose_bias) {
    constexpr bool kWritesProjections = Outputs != ProjectionOutputs::kPairBias;
    constexpr bool kWritesPairBias = Outputs != ProjectionOutputs::kProjections;
    // Weights [136][C] and the input tiles [64][C] use the wgmma operand layout;
    // each K block starts 1024-byte aligned because 136 * 128 and 64 * 128 bytes
    // are multiples of 1024. Weight rows [64w, 64w + 64) belong to warpgroup w
    // (projection_of_block); rows 128..135 are the bias heads.
    extern __shared__ __align__(1024) float4 projection_shared[];
    float* weights = reinterpret_cast<float*>(projection_shared);
    float* input_tiles = weights + kProjectionWeightRows * kProjectionChannels;
    // tile_loaded[stage]; placed after the buffers, because a static
    // __shared__ variable would move the 1024-aligned dynamic base.
    uint64_t* tile_loaded = reinterpret_cast<uint64_t*>(input_tiles + kProjectionStages * kProjectionTileValues);

    constexpr int kSliceColumns = kProjectionWidth / 4;
    const int slice = blockIdx.x;
    const int thread = threadIdx.x;
    const int warpgroup = thread / 128;
    const int warp_in_group = (thread % 128) / 32;
    const int lane = thread % 32;

    if (thread == 0) {
        for (int stage = 0; stage < kProjectionStages; ++stage) mbarrier_init(&tile_loaded[stage], 1);
        fence_mbarrier_init();
    }
    // Stage the weights this mode multiplies by, once, rounded to TF32 with
    // ties to even as cuBLAS rounds them. Bias rows past `heads` are zero. Each
    // step copies four reduction channels: one 16-byte chunk of the operand layout.
    constexpr int kChunksPerRow = kProjectionChannels / 4;
    constexpr int kFirstWeightRow = kWritesProjections ? 0 : kProjectionWidth;
    constexpr int kWeightRowsEnd = kWritesPairBias ? kProjectionWeightRows : kProjectionWidth;
    for (int index = thread + kFirstWeightRow * kChunksPerRow; index < kWeightRowsEnd * kChunksPerRow;
         index += kProjectionThreads) {
        const int row = index / kChunksPerRow;
        const int channel = (index % kChunksPerRow) * 4;
        float4 values = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        if (row < kProjectionWidth) {
            const int row_warpgroup = row / 64, block = (row % 64) / 8;
            const int projection = projection_of_block(row_warpgroup, block);
            const float* weight = projection == 0 ? q_weight
                : projection == 1 ? k_weight : projection == 2 ? v_weight : gate_weight;
            const int source_row = slice * kSliceColumns + slice_column_of_block(row_warpgroup, block) + row % 8;
            values = *reinterpret_cast<const float4*>(weight + source_row * kProjectionChannels + channel);
        } else if (row - kProjectionWidth < heads) {
            values = *reinterpret_cast<const float4*>(
                bias_weight + (row - kProjectionWidth) * kProjectionChannels + channel);
        }
        *reinterpret_cast<float4*>(weights + sw128_k_major_offset(row, channel, kProjectionWeightRows)) =
            make_float4(__uint_as_float(round_tf32(values.x)), __uint_as_float(round_tf32(values.y)),
                        __uint_as_float(round_tf32(values.z)), __uint_as_float(round_tf32(values.w)));
    }
    // The weight stores reach wgmma's operand reads after this fence and the
    // barrier, which also publishes the initialized tile barriers.
    fence_proxy_async_shared();
    __syncthreads();

    const int64_t row_tiles = (rows + kProjectionTileRows - 1) / kProjectionTileRows;
    const int64_t first_tile = blockIdx.y;
    const int64_t tile_step = gridDim.y;
    // One 32-channel box per K block, all counted on the stage's barrier.
    const auto load_tile = [&](int64_t tile, int stage) {
        float* destination = input_tiles + stage * kProjectionTileValues;
        mbarrier_arrive_expect_bytes(&tile_loaded[stage], kProjectionTileValues * sizeof(float));
#pragma unroll
        for (int k_block = 0; k_block < kProjectionChannels / 32; ++k_block)
            tma_load_2d(&normalized_map, destination + k_block * kProjectionTileRows * 32,
                        &tile_loaded[stage], k_block * 32, first_row + tile * kProjectionTileRows);
    };
    if (thread == 0) {
        for (int stage = 0; stage < kProjectionStages; ++stage)
            if (first_tile + stage * tile_step < row_tiles) load_tile(first_tile + stage * tile_step, stage);
    }

    // Descriptors at reduction index 0. A K step or a stage adds its byte offset
    // (in 16-byte units) to the low word: shared addresses stay below 2^18
    // bytes, so the 14-bit start field never carries. Both operands share the
    // high word. This warpgroup's weight rows start at 64 * warpgroup.
    const uint64_t tile_descriptor = sw128_k_major_descriptor(input_tiles, 0);
    const uint32_t descriptor_high = static_cast<uint32_t>(tile_descriptor >> 32);
    const uint32_t tile_low = static_cast<uint32_t>(tile_descriptor);
    const uint32_t weight_low = static_cast<uint32_t>(sw128_k_major_descriptor(weights, 64 * warpgroup));
    const uint32_t bias_weight_low = static_cast<uint32_t>(sw128_k_major_descriptor(weights, kProjectionWidth));
    const auto descriptor = [&](uint32_t low) {
        return (static_cast<uint64_t>(descriptor_high) << 32) | low;
    };
    constexpr uint32_t kStageStep = kProjectionTileValues * sizeof(float) / 16;
    constexpr uint32_t kTileKBlockStep = kProjectionTileRows * 128 / 16;
    constexpr uint32_t kWeightKBlockStep = kProjectionWeightRows * 128 / 16;

    // This thread's fragment rows (group and group + 8 of its warp's 16) and
    // its first column pair inside each 8-column block.
    const int fragment_row = warp_in_group * 16 + lane / 4;
    const int fragment_column = 2 * (lane % 4);
    const int64_t plane = sequence * sequence;

    unsigned iteration = 0;
    for (int64_t tile = first_tile; tile < row_tiles; tile += tile_step, ++iteration) {
        const int stage = static_cast<int>(iteration % kProjectionStages);
        mbarrier_wait_parity(&tile_loaded[stage], (iteration / kProjectionStages) & 1);

        // accumulators[b] holds block b (projection_of_block). Every value
        // accumulates its 128 products in ascending K order from zero.
        float accumulators[8][4] = {};
        float bias_accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        const uint32_t stage_low = tile_low + stage * kStageStep;
        wgmma_fence();
#pragma unroll
        for (int reduction = 0; reduction < kProjectionChannels; reduction += 8) {
            // (k / 32) whole K blocks, plus (k % 32) words inside the 128-byte segment.
            const uint32_t in_block = (reduction % 32) * sizeof(float) / 16;
            const uint32_t a_low = stage_low + (reduction / 32) * kTileKBlockStep + in_block;
            const uint32_t b_offset = (reduction / 32) * kWeightKBlockStep + in_block;
            if constexpr (kWritesProjections)
                wgmma_m64n64k8_tf32(accumulators, descriptor(a_low), descriptor(weight_low + b_offset));
            if constexpr (kWritesPairBias)
                wgmma_m64n8k8_tf32(bias_accumulator, descriptor(a_low), descriptor(bias_weight_low + b_offset));
        }
        wgmma_commit();
        wgmma_wait_all();
        // Both warpgroups have finished reading this stage; refill it with the
        // tile kProjectionStages steps ahead while the epilogue runs.
        __syncthreads();
        if (thread == 0 && tile + kProjectionStages * tile_step < row_tiles)
            load_tile(tile + kProjectionStages * tile_step, stage);

        const int64_t tile_row = tile * kProjectionTileRows;
#pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half) {
            const int64_t row = tile_row + fragment_row + row_half * 8;  // inside the range
            if (row >= rows) continue;
            const int64_t row_offset = row * kProjectionWidth + slice * kSliceColumns + fragment_column;
#pragma unroll
            for (int block = 0; block < (kWritesProjections ? 8 : 0); ++block) {
                float2 pair = make_float2(accumulators[block][row_half * 2],
                                          accumulators[block][row_half * 2 + 1]);
                if (block == 4 || block == 5) {  // the gate blocks of both warpgroups
                    pair.x = sigmoid_like_pytorch(pair.x);
                    pair.y = sigmoid_like_pytorch(pair.y);
                }
                const int projection = projection_of_block(warpgroup, block);
                float* output = projection == 0 ? q : projection == 1 ? k : projection == 2 ? v : gate;
                *reinterpret_cast<float2*>(output + row_offset + slice_column_of_block(warpgroup, block)) = pair;
            }
            // Every warpgroup holds the bias columns; the stores are shared out
            // (they are scattered when transpose_bias): slice s stores heads
            // h = s (mod 4) (kPairBias's single slice stores all of them), and
            // warpgroup w the rows of fragment half w.
            if (kWritesPairBias && row_half == warpgroup) {
                const int64_t pair_row = first_row + row;
                const int64_t batch = pair_row / plane;
                const int64_t within_plane = pair_row - batch * plane;
                const int64_t first_index = within_plane / sequence;
                const int64_t second_index = within_plane - first_index * sequence;
                const int64_t bias_position = transpose_bias
                    ? second_index * sequence + first_index : within_plane;
#pragma unroll
                for (int adjacent = 0; adjacent < 2; ++adjacent) {
                    const int head = fragment_column + adjacent;
                    const bool slice_owns_head = Outputs == ProjectionOutputs::kPairBias || head % 4 == slice;
                    if (head < heads && slice_owns_head)
                        pair_bias[(batch * heads + head) * plane + bias_position] =
                            bias_accumulator[row_half * 2 + adjacent];
                }
            }
        }
    }
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

// x_ln [rows][128] as TMA boxes of 32 channels x 64 rows with the 128-byte
// swizzle (one box = one K block of the wgmma operand layout).
CUtensorMap make_normalized_tensor_map(const torch::Tensor& normalized) {
    CUtensorMap tensor_map;
    const cuuint64_t global_dim[2] = {static_cast<cuuint64_t>(kProjectionChannels),
                                      static_cast<cuuint64_t>(normalized.size(0))};
    const cuuint64_t global_stride[1] = {kProjectionChannels * sizeof(float)};
    const cuuint32_t box_dim[2] = {32, kProjectionTileRows};
    const cuuint32_t element_stride[2] = {1, 1};
    const CUresult result = tensor_map_encode_function()(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, normalized.data_ptr(), global_dim,
        global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
    return tensor_map;
}

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

}  // namespace

torch::Tensor turbopairformer_attention_fp32_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    bool allow_tf32,
    int64_t workspace_bytes) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "FP32 Triangle Attention is inference-only");
    TORCH_CHECK(q.dim() == 5, "q must be [B,O,H,N,D]");
    check_input(q, q, "q", !allow_tf32);
    check_input(k, q, "k", !allow_tf32);
    check_input(v, q, "v", !allow_tf32);
    check_input(mask, q, "mask");
    check_input(bias, q, "bias");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q/k/v shape mismatch");
    bool common_qkv_strides = q.size(4) == 1 || q.stride(4) == 1;
    for (int axis = 0; axis < 5; ++axis) {
        if (q.size(axis) > 1) {
            common_qkv_strides = common_qkv_strides && q.stride(axis) == k.stride(axis)
                && q.stride(axis) == v.stride(axis);
        }
    }
    TORCH_CHECK(common_qkv_strides,
                "Q/K/V must share strides and have contiguous head channels");
    const int64_t batch = q.size(0), outer = q.size(1), heads = q.size(2);
    const int64_t sequence_long = q.size(3), head_dim_long = q.size(4);
    TORCH_CHECK(batch > 0 && outer > 0 && heads > 0 && sequence_long > 0
                    && sequence_long <= std::numeric_limits<int>::max()
                    && head_dim_long > 0 && head_dim_long <= 64,
                "invalid FP32 Triangle Attention dimensions");
    TORCH_CHECK(mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == outer
                    && mask.size(2) == sequence_long,
                "mask must be [B,O,N]");
    TORCH_CHECK(bias.dim() == 4 && bias.size(0) == batch && bias.size(1) == heads
                    && bias.size(2) == sequence_long && bias.size(3) == sequence_long,
                "bias must be [B,H,N,N]");
    const int sequence = static_cast<int>(sequence_long);
    const int head_dim = static_cast<int>(head_dim_long);
    // Tensor numel validation bounds the flattened products; use int64_t for
    // every address/stride even though cuBLAS's individual dimensions are int.
    const int64_t matrix_stride = sequence_long * head_dim;
    const int64_t matrices = q.numel() / matrix_stride;
    const int64_t row_elements = sequence_long + head_dim;
    TORCH_CHECK(workspace_bytes >= row_elements * static_cast<int64_t>(sizeof(float)),
                "workspace_bytes must hold at least one score/output row");
    const int64_t workspace_elements = workspace_bytes / sizeof(float);
    const int query_capacity = static_cast<int>(std::min<int64_t>(
        std::min(sequence, kQueryTile), workspace_elements / row_elements));
    const int64_t matrix_capacity = std::min(
        std::min(matrices, workspace_elements / (query_capacity * row_elements)),
        kMaxGridX / query_capacity);

    const c10::cuda::CUDAGuard device_guard(q.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "FP32 Triangle Attention currently requires Hopper CC 9.0");
    auto output = torch::empty({batch, outer, sequence_long, heads, head_dim_long}, q.options());
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    const int64_t fused_query_tiles = (sequence_long + kFusedQueryRows - 1) / kFusedQueryRows;
    if (allow_tf32 && matrices <= kMaxGridX / fused_query_tiles) {
        // This route needs no global scratch. workspace_bytes remains a bound
        // on temporary global score/output tensors, including for tiny limits.
        if (head_dim <= 8) {
            launch_fused_tf32<8, false>(q, k, v, mask, bias, nullptr, output, matrices, stream);
        } else if (head_dim <= 16) {
            launch_fused_tf32<16, false>(q, k, v, mask, bias, nullptr, output, matrices, stream);
        } else if (head_dim <= 32) {
            launch_fused_tf32<32, false>(q, k, v, mask, bias, nullptr, output, matrices, stream);
        } else {
            launch_fused_tf32<64, false>(q, k, v, mask, bias, nullptr, output, matrices, stream);
        }
        return output;
    }
    // Strict FP32 must not execute a rounded TF32 MMA. The bounded vendor
    // route also handles the exceptional grid extent beyond one fused launch.
    const auto q_for_gemm = q.contiguous();
    const auto k_for_gemm = k.contiguous();
    const auto v_for_gemm = v.contiguous();
    auto scores = torch::empty({matrix_capacity, query_capacity, sequence_long}, q.options());
    auto output_tile = torch::empty({matrix_capacity, query_capacity, head_dim_long}, q.options());
    const auto handle = at::cuda::getCurrentCUDABlasHandle();
    ScopedBlasSettings settings(handle);
    settings.configure(allow_tf32);
    const auto compute_type = allow_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                                         : CUBLAS_COMPUTE_32F_PEDANTIC;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const float one = 1.0F, zero = 0.0F;
    const int64_t score_stride = static_cast<int64_t>(query_capacity) * sequence;
    const int64_t output_stride = static_cast<int64_t>(query_capacity) * head_dim;

    for (int64_t first_matrix = 0; first_matrix < matrices; first_matrix += matrix_capacity) {
        const int matrix_count = static_cast<int>(std::min(matrix_capacity, matrices - first_matrix));
        const int64_t matrix_offset = first_matrix * matrix_stride;
        for (int64_t query_offset = 0; query_offset < sequence; query_offset += query_capacity) {
            const int first_query = static_cast<int>(query_offset);
            const int query_count = std::min(query_capacity, sequence - first_query);
            // Row-major QK^T is the column-major product K * Q^T with dimensions
            // [N,query_count]. K and Q themselves are stored as [D,N] columns.
            check_blas(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, sequence, query_count, head_dim,
                &scale, k_for_gemm.data_ptr<float>() + matrix_offset, CUDA_R_32F, head_dim,
                matrix_stride, q_for_gemm.data_ptr<float>() + matrix_offset
                                   + static_cast<int64_t>(first_query) * head_dim,
                CUDA_R_32F, head_dim, matrix_stride, &zero, scores.data_ptr<float>(),
                CUDA_R_32F, sequence, score_stride, matrix_count, compute_type,
                CUBLAS_GEMM_DEFAULT), "QK cuBLAS GEMM");

            const int64_t rows = static_cast<int64_t>(matrix_count) * query_count;
            mask_bias_softmax<<<static_cast<unsigned int>(rows), kThreads, 0, stream>>>(
                scores.data_ptr<float>(), mask.data_ptr<float>(), bias.data_ptr<float>(),
                first_matrix, outer, heads, sequence, first_query, query_count, query_capacity);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // [query_count,N] probabilities times [N,D] V is represented as
            // the column-major product [D,N] * [N,query_count]. No cast occurs.
            check_blas(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, query_count, sequence,
                &one, v_for_gemm.data_ptr<float>() + matrix_offset, CUDA_R_32F, head_dim,
                matrix_stride, scores.data_ptr<float>(), CUDA_R_32F, sequence,
                score_stride, &zero, output_tile.data_ptr<float>(), CUDA_R_32F,
                head_dim, output_stride, matrix_count, compute_type,
                CUBLAS_GEMM_DEFAULT), "PV cuBLAS GEMM");

            const int64_t elements = rows * head_dim;
            const int64_t blocks = std::min(kMaxGridX, (elements + kThreads - 1) / kThreads);
            scatter_output<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
                output_tile.data_ptr<float>(), output.data_ptr<float>(), first_matrix,
                heads, sequence, head_dim, first_query, query_count, query_capacity, elements);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    settings.restore();
    return output;
}

namespace {

void check_module_parameter(const torch::Tensor& tensor, const torch::Tensor& like, const char* name,
                            std::initializer_list<int64_t> shape) {
    TORCH_CHECK(tensor.device() == like.device() && tensor.scalar_type() == at::kFloat
                    && tensor.is_contiguous() && tensor.sizes() == c10::IntArrayRef(shape)
                    && aligned_to(tensor.data_ptr<float>(), 16),
                name, " must be a contiguous, 16-byte-aligned FP32 tensor of the module's shape");
}

int64_t check_bias_weight(const torch::Tensor& bias_weight, const torch::Tensor& like) {
    TORCH_CHECK(bias_weight.dim() == 2 && bias_weight.size(0) >= 1
                    && bias_weight.size(0) <= kProjectionBiasColumns,
                "bias_weight must be [heads <= 8, 128]");
    check_module_parameter(bias_weight, like, "bias_weight", {bias_weight.size(0), kProjectionChannels});
    return bias_weight.size(0);
}

void check_hopper_inference(const torch::Tensor& like) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "FP32 Triangle Attention is inference-only");
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "FP32 Triangle Attention currently requires Hopper CC 9.0");
    TORCH_CHECK(like.is_cuda(), "inputs must be CUDA tensors");
}

// x [B, N, N, 128] with any batch/row/column strides and unit channel stride
// (the ending node passes a transposed view) -> x_ln [B * N * N, 128] rounded to TF32.
torch::Tensor layer_norm_pair_rows(const torch::Tensor& x, const torch::Tensor& weight,
                                   const torch::Tensor& bias, double epsilon) {
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.dim() == 4, "x must be FP32 CUDA [B, N, N, C]");
    TORCH_CHECK(x.size(1) == x.size(2) && x.size(0) > 0 && x.size(1) > 0,
                "x must be a square pair representation");
    TORCH_CHECK(x.size(3) == kProjectionChannels && x.stride(3) == 1,
                "x must have 128 channels with unit channel stride");
    for (int axis = 0; axis < 3; ++axis)
        TORCH_CHECK(x.size(axis) == 1 || x.stride(axis) % 4 == 0, "x rows must be 16-byte aligned");
    TORCH_CHECK(aligned_to(x.data_ptr<float>(), 16), "x must be 16-byte aligned");
    check_module_parameter(weight, x, "layer_norm_weight", {kProjectionChannels});
    check_module_parameter(bias, x, "layer_norm_bias", {kProjectionChannels});
    const int64_t sequence = x.size(1);
    const int64_t rows = x.size(0) * sequence * sequence;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "too many pair rows for one TMA map");
    auto normalized = torch::empty({rows, kProjectionChannels}, x.options());
    const int64_t blocks = (rows + kLayerNormWarps - 1) / kLayerNormWarps;
    TORCH_CHECK(blocks <= kMaxGridX, "too many pair rows");
    // Resolve the kernel handle explicitly instead of using the host stub's
    // implicit lookup that produced an INVALID_HANDLE API report at the first
    // fused LayerNorm launch with the CUDA 12.9 runtime. Keep the
    // handle local to this call, without caching across device/context lifetimes.
    // The launch stays asynchronous on PyTorch's current stream.
    cudaKernel_t kernel_handle = nullptr;
    C10_CUDA_CHECK(cudaGetKernel(&kernel_handle, layer_norm_rows_tf32<kProjectionChannels>));
    const float* input_values = x.data_ptr<float>();
    const float* weight_values = weight.data_ptr<float>();
    const float* bias_values = bias.data_ptr<float>();
    float* output_values = normalized.data_ptr<float>();
    int64_t row_count = rows, sequence_length = sequence;
    int64_t batch_stride = x.stride(0), row_stride = x.stride(1), column_stride = x.stride(2);
    float epsilon_value = static_cast<float>(epsilon);
    void* arguments[] = {&input_values, &weight_values, &bias_values, &output_values,
                        &row_count, &sequence_length, &batch_stride, &row_stride,
                        &column_stride, &epsilon_value};
    C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel_handle),
        dim3(static_cast<unsigned int>(blocks)), dim3(kLayerNormWarps * 32), arguments,
        0, at::cuda::getCurrentCUDAStream(x.get_device())));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return normalized;
}

// One launch of attention_projections_tf32<Outputs> over x_ln rows
// [first_row, first_row + rows). Pointers a mode does not use may be null.
template <ProjectionOutputs Outputs>
void launch_projections(const torch::Tensor& normalized, int64_t first_row, int64_t rows,
                        const float* q_weight, const float* k_weight, const float* v_weight,
                        const float* gate_weight, const float* bias_weight, float* q, float* k,
                        float* v, float* gate, float* pair_bias, int64_t sequence, int64_t heads,
                        bool transpose_bias) {
    const auto kernel = attention_projections_tf32<Outputs>;
    const int device = normalized.get_device();
    static thread_local int configured_device = -1;
    if (configured_device != device) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            kProjectionSharedBytes));
        configured_device = device;
    }
    // One block per SM: four column slices (one for the pair bias alone) times
    // tile walkers.
    constexpr int kSlices = Outputs == ProjectionOutputs::kPairBias ? 1 : 4;
    const int64_t row_tiles = (rows + kProjectionTileRows - 1) / kProjectionTileRows;
    const int64_t walkers = std::max<int64_t>(1, std::min<int64_t>(
        row_tiles, multiprocessor_count(device) / kSlices));
    const CUtensorMap normalized_map = make_normalized_tensor_map(normalized);
    kernel<<<dim3(kSlices, static_cast<unsigned int>(walkers)), kProjectionThreads,
             kProjectionSharedBytes, at::cuda::getCurrentCUDAStream(device)>>>(
        normalized_map, q_weight, k_weight, v_weight, gate_weight, bias_weight, q, k, v, gate,
        pair_bias, first_row, rows, sequence, static_cast<int>(heads), transpose_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// TriangleAttention module front end for TF32 inference: LayerNorm, the q, k,
// v and gate projections, the sigmoid gate, and the pair bias, bit-identical
// to PyTorch's LayerNorm + nn.functional.linear (cuBLAS TF32) + torch.sigmoid.
// x is the module input [B, N, N, 128] (any batch/row/column strides, unit
// channel stride, as the transposed view of the ending-node call); weights are
// [128, 128] and [heads, 128]. Returns q, k, v, gate as contiguous
// [B, N, N, 128] and pair_bias as contiguous [B, heads, N, N].
std::vector<torch::Tensor> turbopairformer_attention_fp32_module_projections(
    const torch::Tensor& x,
    const torch::Tensor& layer_norm_weight,
    const torch::Tensor& layer_norm_bias,
    double layer_norm_epsilon,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& gate_weight,
    const torch::Tensor& bias_weight,
    bool transpose_bias) {
    check_hopper_inference(x);
    const c10::cuda::CUDAGuard device_guard(x.device());
    for (const auto* weight : {&q_weight, &k_weight, &v_weight, &gate_weight})
        check_module_parameter(*weight, x, "projection weight", {kProjectionWidth, kProjectionChannels});
    const int64_t heads = check_bias_weight(bias_weight, x);
    const auto normalized = layer_norm_pair_rows(x, layer_norm_weight, layer_norm_bias, layer_norm_epsilon);
    const int64_t batch = x.size(0), sequence = x.size(1);
    auto q = torch::empty({batch, sequence, sequence, kProjectionWidth}, x.options());
    auto k = torch::empty_like(q);
    auto v = torch::empty_like(q);
    auto gate = torch::empty_like(q);
    auto pair_bias = torch::empty({batch, heads, sequence, sequence}, x.options());
    launch_projections<ProjectionOutputs::kAll>(
        normalized, 0, normalized.size(0), q_weight.data_ptr<float>(), k_weight.data_ptr<float>(),
        v_weight.data_ptr<float>(), gate_weight.data_ptr<float>(), bias_weight.data_ptr<float>(),
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), gate.data_ptr<float>(),
        pair_bias.data_ptr<float>(), sequence, heads, transpose_bias);
    return {q, k, v, gate, pair_bias};
}

// The chunked module, step 1: x_ln [B * N * N, 128], rounded to TF32 as every
// projection consumes it. The caller also uses it as the output buffer, one
// chunk at a time, once a chunk's projections have read its rows.
torch::Tensor turbopairformer_attention_fp32_module_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& layer_norm_weight,
    const torch::Tensor& layer_norm_bias,
    double layer_norm_epsilon) {
    check_hopper_inference(x);
    const c10::cuda::CUDAGuard device_guard(x.device());
    return layer_norm_pair_rows(x, layer_norm_weight, layer_norm_bias, layer_norm_epsilon);
}

// The chunked module, step 2: the pair bias [B, heads, N, N] of every x_ln
// row, which each chunk's attention reads in full.
torch::Tensor turbopairformer_attention_fp32_module_pair_bias(
    const torch::Tensor& normalized,
    const torch::Tensor& bias_weight,
    int64_t batch,
    int64_t sequence,
    bool transpose_bias) {
    check_hopper_inference(normalized);
    const c10::cuda::CUDAGuard device_guard(normalized.device());
    TORCH_CHECK(normalized.scalar_type() == at::kFloat && normalized.is_contiguous()
                    && normalized.dim() == 2 && normalized.size(1) == kProjectionChannels
                    && batch > 0 && sequence > 0 && normalized.size(0) == batch * sequence * sequence,
                "normalized must be contiguous FP32 [batch * sequence^2, 128]");
    const int64_t heads = check_bias_weight(bias_weight, normalized);
    auto pair_bias = torch::empty({batch, heads, sequence, sequence}, normalized.options());
    launch_projections<ProjectionOutputs::kPairBias>(
        normalized, 0, normalized.size(0), nullptr, nullptr, nullptr, nullptr,
        bias_weight.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
        pair_bias.data_ptr<float>(), sequence, heads, transpose_bias);
    return pair_bias;
}

// The chunked module, step 3: q, k, v and the gate of the x_ln rows
// [first_row, first_row + rows), each contiguous [rows, 128].
std::vector<torch::Tensor> turbopairformer_attention_fp32_module_chunk_projections(
    const torch::Tensor& normalized,
    int64_t first_row,
    int64_t rows,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& gate_weight) {
    check_hopper_inference(normalized);
    const c10::cuda::CUDAGuard device_guard(normalized.device());
    TORCH_CHECK(normalized.scalar_type() == at::kFloat && normalized.is_contiguous()
                    && normalized.dim() == 2 && normalized.size(1) == kProjectionChannels,
                "normalized must be contiguous FP32 [rows, 128]");
    TORCH_CHECK(first_row >= 0 && rows > 0 && first_row + rows <= normalized.size(0),
                "row range is outside normalized");
    for (const auto* weight : {&q_weight, &k_weight, &v_weight, &gate_weight})
        check_module_parameter(*weight, normalized, "projection weight",
                               {kProjectionWidth, kProjectionChannels});
    auto q = torch::empty({rows, kProjectionWidth}, normalized.options());
    auto k = torch::empty_like(q);
    auto v = torch::empty_like(q);
    auto gate = torch::empty_like(q);
    launch_projections<ProjectionOutputs::kProjections>(
        normalized, first_row, rows, q_weight.data_ptr<float>(), k_weight.data_ptr<float>(),
        v_weight.data_ptr<float>(), gate_weight.data_ptr<float>(), nullptr, q.data_ptr<float>(),
        k.data_ptr<float>(), v.data_ptr<float>(), gate.data_ptr<float>(), nullptr, 1, 1, false);
    return {q, k, v, gate};
}

// The fused TF32 attention of turbopairformer_attention_fp32_forward followed by
// the module's `output * gate`, in the same kernel. gate is contiguous with the
// output's [B, O, N, H*D] layout. TF32 only; the caller checks the policy.
torch::Tensor turbopairformer_attention_fp32_gated_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    const torch::Tensor& gate) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "FP32 Triangle Attention is inference-only");
    TORCH_CHECK(q.dim() == 5, "q must be [B,O,H,N,D]");
    check_input(q, q, "q", false);
    check_input(k, q, "k", false);
    check_input(v, q, "v", false);
    check_input(mask, q, "mask");
    check_input(bias, q, "bias");
    check_input(gate, q, "gate");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q/k/v shape mismatch");
    bool common_qkv_strides = q.size(4) == 1 || q.stride(4) == 1;
    for (int axis = 0; axis < 5; ++axis) {
        if (q.size(axis) > 1) {
            common_qkv_strides = common_qkv_strides && q.stride(axis) == k.stride(axis)
                && q.stride(axis) == v.stride(axis);
        }
    }
    TORCH_CHECK(common_qkv_strides,
                "Q/K/V must share strides and have contiguous head channels");
    const int64_t batch = q.size(0), outer = q.size(1), heads = q.size(2);
    const int64_t sequence = q.size(3), head_dim = q.size(4);
    TORCH_CHECK(sequence > 0 && sequence <= std::numeric_limits<int>::max()
                    && head_dim > 0 && head_dim <= 64,
                "invalid FP32 Triangle Attention dimensions");
    TORCH_CHECK(mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == outer
                    && mask.size(2) == sequence,
                "mask must be [B,O,N]");
    TORCH_CHECK(bias.dim() == 4 && bias.size(0) == batch && bias.size(1) == heads
                    && bias.size(2) == sequence && bias.size(3) == sequence,
                "bias must be [B,H,N,N]");
    TORCH_CHECK(gate.dim() == 4 && gate.size(0) == batch && gate.size(1) == outer
                    && gate.size(2) == sequence && gate.size(3) == heads * head_dim,
                "gate must be [B,O,N,H*D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "FP32 Triangle Attention currently requires Hopper CC 9.0");
    const int64_t matrices = batch * outer * heads;
    const int64_t query_tiles = (sequence + kFusedQueryRows - 1) / kFusedQueryRows;
    TORCH_CHECK(matrices <= kMaxGridX / query_tiles, "too many attention rows for one launch");
    auto output = torch::empty({batch, outer, sequence, heads, head_dim}, q.options());
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    const float* gate_values = gate.data_ptr<float>();
    if (head_dim <= 8) {
        launch_fused_tf32<8, true>(q, k, v, mask, bias, gate_values, output, matrices, stream);
    } else if (head_dim <= 16) {
        launch_fused_tf32<16, true>(q, k, v, mask, bias, gate_values, output, matrices, stream);
    } else if (head_dim <= 32) {
        launch_fused_tf32<32, true>(q, k, v, mask, bias, gate_values, output, matrices, stream);
    } else {
        launch_fused_tf32<64, true>(q, k, v, mask, bias, gate_values, output, matrices, stream);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("turbopairformer_attention_fp32_forward",
               &turbopairformer_attention_fp32_forward,
               "Inference-only Triangle Attention (fused TF32 / strict FP32 cuBLAS)");
    module.def("turbopairformer_attention_fp32_module_projections",
               &turbopairformer_attention_fp32_module_projections,
               "TriangleAttention LayerNorm and q/k/v/gate/pair-bias projections (TF32)");
    module.def("turbopairformer_attention_fp32_module_layer_norm",
               &turbopairformer_attention_fp32_module_layer_norm,
               "TriangleAttention LayerNorm of every pair row, rounded to TF32 (chunked module)");
    module.def("turbopairformer_attention_fp32_module_pair_bias",
               &turbopairformer_attention_fp32_module_pair_bias,
               "TriangleAttention pair bias of every pair row (chunked module)");
    module.def("turbopairformer_attention_fp32_module_chunk_projections",
               &turbopairformer_attention_fp32_module_chunk_projections,
               "TriangleAttention q/k/v/gate projections of one outer-row chunk (chunked module)");
    module.def("turbopairformer_attention_fp32_gated_forward",
               &turbopairformer_attention_fp32_gated_forward,
               "Fused TF32 Triangle Attention with the output gate applied");
}
