// SPDX-License-Identifier: Apache-2.0
// TurboPairFormer Triangle Attention forward with online softmax.
//
// Each block owns 64 query rows and visits the keys in 64-row tiles. Each
// warp handles 16 query rows. Online softmax keeps scores on chip and
// accumulates the output in a fixed key order.
// Each output row has one owning block and a fixed reduction order. Replay
// determinism means bitwise agreement for repeated runs with the same inputs,
// binary, and device. Accuracy is validated by error ratios against FP64;
// bitwise agreement with PyTorch is not the numerical contract.
//
// There are two forward kernels:
//   - cp.async + mma.sync supports all head dimensions accepted by this file.
//   - TMA + WGMMA handles D = 16, 32, or 64 when the plane count fits int32.
//     It uses WGMMA for QK^T and mma.sync for PV. Bias also uses TMA when its
//     address and row stride satisfy the tensor-map alignment requirements.
// Both kernels can save a BF16 approximation of the output rounding residual
// for the backward pass. The Python facade enables this by default.
//
// Tensor layouts at the C++ entry point (the Python facade has its own limits):
//   q, k, v : [B, O, H, N, D] BF16, contiguous
//   mask    : [B, O, 1, 1, N] FP32 additive mask
//   bias    : [B, 1, H, N, N] BF16 additive
//   out     : [B, O, N, H, D] BF16; optional residual has the same layout
//   lse     : [B, O, H, N] FP32 (m + log l, natural log)
// B = batch, O = independent outer planes, H = heads, N = query/key length,
// D = head channels. O need not equal N. Bias is shared across O; the mask
// is shared across heads and query rows within each (B, O) plane.
// Typical mask values are 0 and -inf or a large negative finite value.
// The zero-weight special case tests scores for true -inf; finite negative
// masks follow ordinary softmax arithmetic, including fully masked rows.
//
// Reading guide:
//   - Start at turbopairformer_triangle_forward_impl for validation/dispatch.
//   - launch_forward / launch_forward_tma select the kernel specialization.
//   - Both kernels follow the same six stages, labeled below. Their shared
//     softmax recurrence and precision rules precede the cp.async kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kRowsPerWarp = 16;
constexpr int kBlockM = kWarpsPerBlock * kRowsPerWarp;  // 64 query rows per block.
constexpr int kBlockN = 64;                             // Keys processed in one iteration.

// Keep launch-grid bounds separate from the TMA coordinate bound.
constexpr int kMaxGridYZ = 65535;
constexpr long long kMaxGridX = 2147483647LL;
constexpr long long kMaxTmaPlanes = 2147483647LL;

// -----------------------------------------------------------------------------
// QKV layout conversion
// -----------------------------------------------------------------------------

// Reorder Q, K, and V from [batch_outer, N, H, D] to [batch_outer, H, N, D]
// in one launch. The caller passes flat views in the source memory order.
// CopyT sets the copy width; the launcher chooses it from row size and
// pointer alignment so that each copy stays within one row.
template <typename CopyT>
__global__ void turbopairformer_pack_qkv_kernel(
    const CopyT* __restrict__ q,
    const CopyT* __restrict__ k,
    const CopyT* __restrict__ v,
    CopyT* __restrict__ out_q,
    CopyT* __restrict__ out_k,
    CopyT* __restrict__ out_v,
    int64_t total_vectors,
    int heads,
    int sequence,
    int64_t vectors_per_row) {
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total_vectors;
         linear += grid_stride) {
        const int64_t row = linear / vectors_per_row;
        const int64_t column = linear % vectors_per_row;
        const int64_t rows_per_outer = static_cast<int64_t>(heads) * sequence;
        const int64_t batch_outer = row / rows_per_outer;
        const int64_t remainder = row % rows_per_outer;
        const int head = static_cast<int>(remainder / sequence);
        const int query = static_cast<int>(remainder % sequence);
        const int64_t source_row = (batch_outer * sequence + query) * heads + head;
        const int64_t source = source_row * vectors_per_row + column;
        // Copy the BF16 bits without converting their values.
        out_q[linear] = q[source];
        out_k[linear] = k[source];
        out_v[linear] = v[source];
    }
}

// -----------------------------------------------------------------------------
// Matrix fragments and asynchronous copies
// -----------------------------------------------------------------------------

// Round two FP32 values to BF16 and pack them into one register.
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
    const __nv_bfloat16 l = __float2bfloat16_rn(low);
    const __nv_bfloat16 h = __float2bfloat16_rn(high);
    return static_cast<uint32_t>(__bfloat16_as_ushort(l))
           | (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

// Four FP32 accumulator values held by one thread for an m16n8 result.
// Relative to this warp's 16-row tile and the current 8-column tile:
//   For lane index l: r = l / 4, c = 2 * (l % 4).
//   values[0], values[1] -> (r,     c), (r,     c + 1)
//   values[2], values[3] -> (r + 8, c), (r + 8, c + 1)
// Columns represent keys in score fragments and channels in output fragments.
struct Accumulator {
    float values[4];
};

__device__ __forceinline__ void mma_m16n8k16(
    Accumulator& d,
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d.values[0]), "+f"(d.values[1]), "+f"(d.values[2]), "+f"(d.values[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(__nv_bfloat16* dst, const __nv_bfloat16* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all_committed() {
    asm volatile("cp.async.wait_group 0;\n");
}

// Load a 16x16 A fragment from a row-major shared-memory tile.
__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile,
    int ld,
    int dim_start,
    int lane,
    uint32_t (&frag)[4]) {
    const int row = lane & 15;
    const int col = dim_start + ((lane >> 4) << 3);
    const uint32_t addr = shared_u32(tile + row * ld + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                 : "r"(addr));
}

// Load the K fragment for QK^T: 8 keys and 16 head channels.
// Adjacent BF16 values come from adjacent channels in the [keys, D] tile.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile,
    int ld,
    int key_start,
    int dim_start,
    int lane,
    uint32_t (&frag)[2]) {
    const int key = key_start + (lane & 7);
    const int col = dim_start + (((lane >> 3) & 1) << 3);
    const uint32_t addr = shared_u32(tile + key * ld + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(frag[0]), "=r"(frag[1])
                 : "r"(addr));
}

// Load the V fragment for PV: 16 keys and 8 output channels.
// The transposed load arranges these values for the MMA B operand.
__device__ __forceinline__ void ldmatrix_b_keytrans(
    const __nv_bfloat16* tile,
    int ld,
    int key_start,
    int dim_start,
    int lane,
    uint32_t (&frag)[2]) {
    const int key = key_start + (lane & 15);
    const uint32_t addr = shared_u32(tile + key * ld + dim_start);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(frag[0]), "=r"(frag[1])
                 : "r"(addr));
}

// -----------------------------------------------------------------------------
// TMA and WGMMA helpers
// -----------------------------------------------------------------------------
// TMA fills swizzled shared-memory tiles. WGMMA reads K through a descriptor;
// the PV step reads V through ldmatrix and uses mma.sync.

__device__ __forceinline__ void proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(shared_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(shared_u32(bar)),
        "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        "  .reg .pred done;\n"
        "  wait:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "  @!done bra wait;\n"
        "}\n" ::"r"(shared_u32(bar)),
        "r"(parity));
}

__device__ __forceinline__ void tma_load_3d_slice(
    const CUtensorMap* map,
    __nv_bfloat16* dst,
    uint64_t* bar,
    int coord_key,
    int coord_plane) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)),
        "r"(0),
        "r"(coord_key),
        "r"(coord_plane),
        "r"(shared_u32(bar)));
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_zero() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// All four warps cooperate on one 64x64 QK^T tile over 16 head channels.
// Each warp's score fragments still represent only its own 16 query rows.
// Q comes from registers; the descriptor identifies K in shared memory.
// Keep the per-thread outputs ordered as
// score[0].values[0..3], ..., score[7].values[0..3] for the following softmax step.
__device__ __forceinline__ void wgmma_m64n64k16(
    Accumulator (&score)[8],
    const uint32_t (&a)[4],
    uint64_t descriptor_b,
    int accumulate) {
    asm volatile("{\n"
                 "  .reg .pred use_acc;\n"
                 "  setp.ne.b32 use_acc, %37, 0;\n"
                 "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                 "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "   %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
                 "  {%32,%33,%34,%35}, %36, use_acc, 1, 1, 0;\n"
                 "}\n"
                 : "+f"(score[0].values[0]),
                   "+f"(score[0].values[1]),
                   "+f"(score[0].values[2]),
                   "+f"(score[0].values[3]),
                   "+f"(score[1].values[0]),
                   "+f"(score[1].values[1]),
                   "+f"(score[1].values[2]),
                   "+f"(score[1].values[3]),
                   "+f"(score[2].values[0]),
                   "+f"(score[2].values[1]),
                   "+f"(score[2].values[2]),
                   "+f"(score[2].values[3]),
                   "+f"(score[3].values[0]),
                   "+f"(score[3].values[1]),
                   "+f"(score[3].values[2]),
                   "+f"(score[3].values[3]),
                   "+f"(score[4].values[0]),
                   "+f"(score[4].values[1]),
                   "+f"(score[4].values[2]),
                   "+f"(score[4].values[3]),
                   "+f"(score[5].values[0]),
                   "+f"(score[5].values[1]),
                   "+f"(score[5].values[2]),
                   "+f"(score[5].values[3]),
                   "+f"(score[6].values[0]),
                   "+f"(score[6].values[1]),
                   "+f"(score[6].values[2]),
                   "+f"(score[6].values[3]),
                   "+f"(score[7].values[0]),
                   "+f"(score[7].values[1]),
                   "+f"(score[7].values[2]),
                   "+f"(score[7].values[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(descriptor_b), "r"(accumulate));
}

// Load a V fragment from the swizzled TMA layout. Each row has HeadDim/8
// chunks of 16 bytes. The XOR selects the stored chunk for this row and
// channel; its row term advances every 128 bytes.
template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_keytrans_sw(
    const __nv_bfloat16* tile,
    int key_start,
    int dim_start,
    int lane,
    uint32_t (&frag)[2]) {
    constexpr int kAtoms = HeadDim / 8;
    const int key = key_start + (lane & 15);
    const int atom = (dim_start >> 3) ^ (((key * HeadDim) >> 6) & (kAtoms - 1));
    const uint32_t addr = shared_u32(tile + key * HeadDim + atom * 8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(frag[0]), "=r"(frag[1])
                 : "r"(addr));
}

// Load a 3-D TMA box. Coordinates are ordered from innermost to outermost.
__device__ __forceinline__ void tma_load_3d_box(
    const CUtensorMap* map,
    __nv_bfloat16* dst,
    uint64_t* bar,
    int coord0,
    int coord1,
    int coord2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)),
        "r"(coord0),
        "r"(coord1),
        "r"(coord2),
        "r"(shared_u32(bar)));
}

// Load a 16x16 bias tile into the same register layout as the scores.
// The shared tile uses a 128-byte swizzle: chunk c is stored at c ^ (row & 7).
// frag[0] and frag[1] cover rows r and r+8 for the lower 8 keys;
// frag[2] and frag[3] cover the upper 8 keys. Each register holds two BF16
// values.
__device__ __forceinline__ void ldmatrix_bias_x4(
    const __nv_bfloat16* tile,
    int row_base,
    int key_pair,
    int lane,
    uint32_t (&frag)[4]) {
    static_assert(
        kBlockN * sizeof(__nv_bfloat16) == 128,
        "bias box rows must span exactly the 128-byte swizzle");
    const int row = row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int chunk = key_pair * 2 + ((lane >> 4) & 1);
    const uint32_t addr = shared_u32(tile + row * kBlockN + ((chunk ^ (row & 7)) << 3));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                 : "r"(addr));
}

// Reduce across the four lanes that hold values from the same query row.
__device__ __forceinline__ float quad_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 1, kWarpSize));
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 2, kWarpSize));
    return value;
}

__device__ __forceinline__ float quad_sum(float value) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, 1, kWarpSize);
    value += __shfl_xor_sync(0xFFFFFFFFU, value, 2, kWarpSize);
    return value;
}

// -----------------------------------------------------------------------------
// Forward computation shared by both kernels
// -----------------------------------------------------------------------------
// For each query row, visit key tiles in order. The compact symbols below
// denote the per-row values used by both kernels:
//   m = running_row_max; l = running_weight_sum.
//   o = output_accumulators, with the row's output split across lanes.
//   m_next = updated_row_max; alpha = history_scale.
// With S = QK^T * attention_scale + mask + bias, update the state:
//   m_next = max(m, max(S_tile))
//   alpha  = exp(m - m_next)          (zero when the old m is -inf)
//   p      = exp(S_tile - m_next)     (zero for each -inf score)
//   l      = alpha * l + sum(p)
//   o      = alpha * o + BF16(p) @ V_tile
//   m      = m_next
// The state and reductions are FP32. Before PV, p is rounded to BF16;
// the denominator sums the FP32 weights. The epilogue writes
// BF16(o / l), LSE = m + log(l), and optionally the output rounding
// residual. With finite V, rows whose scores are all -inf produce zero output
// and LSE = -inf. No score/probability matrix is materialized in global memory.
//
// Kernel template flags (used by both paths):
//   HeadDim        Compiled storage width; io_dim is the cp.async input width.
//   TileMajorGrid  Query tiles use grid.x; otherwise they use grid.z.
//   NoPartialTiles N fills complete query/key tiles; no tail guards are needed.
//   OuterEqSeq     O == N, so sequence also supplies the outer extent.
// The TMA kernel additionally uses BiasTma to select staged vs direct bias.
//
// -----------------------------------------------------------------------------
// Forward path 1: cp.async + mma.sync
// -----------------------------------------------------------------------------
// The Python facade pads N to full tiles, allowing NoPartialTiles to remove
// tail checks. Direct callers can use the guarded version for partial tiles.
// Global tensors have io_dim channels. Shared-memory tiles have HeadDim
// channels. Extra Q/K channels are both zero, so their dot-product terms
// are exactly zero; scaling still uses the logical io_dim. This storage
// padding is separate from any BF16 query rescaling in the Python facade.
template <int HeadDim, bool TileMajorGrid, bool NoPartialTiles, bool OuterEqSeq>
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_forward_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    __nv_bfloat16* __restrict__ shadow_residual,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_dim,
    float attention_scale) {
    constexpr int kKVStride = HeadDim + 8;
    static_assert(HeadDim % 16 == 0 && (kKVStride * 2) % 16 == 0);
    const int lane_id = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    // Map this block to one batch item, outer index, head, and query tile.
    // Query tiles use grid.x unless batch * outer is too large for grid.z.
    const int outer_extent = OuterEqSeq ? sequence : outer_count;
    const int batch_outer_index = TileMajorGrid ? blockIdx.z : blockIdx.x;
    const int batch_index = batch_outer_index / outer_extent;
    const int outer_index = batch_outer_index - batch_index * outer_extent;
    const int block_query_start = (TileMajorGrid ? blockIdx.x : blockIdx.z) * kBlockM;
    const int head_index = blockIdx.y;
    // Warps beyond the last query row still help load K/V and reach every
    // block barrier. Their scores are masked and their output stores skipped.
    const int warp_query_start = block_query_start + warp_id * kRowsPerWarp;

    extern __shared__ __align__(16) unsigned char cp_async_shared_storage[];
    // Shared-memory layout: Q | K0 | V0 | K1 | V1.
    // All rows use kKVStride; each warp owns 16 rows of Q.
    __nv_bfloat16* shared_warp_q = reinterpret_cast<__nv_bfloat16*>(cp_async_shared_storage)
                                   + warp_id * kRowsPerWarp * kKVStride;
    __nv_bfloat16* shared_kv_buffers =
        reinterpret_cast<__nv_bfloat16*>(cp_async_shared_storage) + kBlockM * kKVStride;

    // Reuse one flattened batch/outer index for related tensor base addresses.
    // Keeping these expressions together helps the compiler share the 64-bit
    // address products instead of recomputing them in each access path.
    const int64_t batch_outer_plane =
        static_cast<int64_t>(batch_index) * outer_extent + outer_index;
    const int64_t head_row_offset =
        (batch_outer_plane * heads + head_index) * static_cast<int64_t>(sequence);
    const int64_t qkv_head_offset = head_row_offset * io_dim;
    const int64_t bias_head_offset = (static_cast<int64_t>(batch_index) * heads + head_index)
                                     * static_cast<int64_t>(sequence) * sequence;
    const int64_t mask_plane_offset = batch_outer_plane * sequence;
    // Write directly to the public output layout: [B, O, N, H, io_dim].
    const int64_t output_head_offset =
        batch_outer_plane * static_cast<int64_t>(sequence) * heads * io_dim
        + static_cast<int64_t>(head_index) * io_dim;

    // Load this warp's Q rows once; every key tile reuses them.
    // Invalid query rows load row 0, but their outputs are never stored.
    for (int element_index = lane_id; element_index < kRowsPerWarp * HeadDim;
         element_index += kWarpSize) {
        const int row = element_index / HeadDim;
        const int channel = element_index - row * HeadDim;
        const int query_row =
            (NoPartialTiles || warp_query_start + row < sequence) ? warp_query_start + row : 0;
        shared_warp_q[row * kKVStride + channel] =
            channel < io_dim
                ? q[qkv_head_offset + static_cast<int64_t>(query_row) * io_dim + channel]
                : __float2bfloat16_rn(0.0F);
    }
    __syncwarp();

    uint32_t q_fragments[HeadDim / 16][4];
#pragma unroll
    for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        ldmatrix_a16x16(
            shared_warp_q, kKVStride, channel_chunk * 16, lane_id, q_fragments[channel_chunk]);
    }

    // Online softmax state for the two rows held by each lane:
    // Keep the running row maximum, weight sum, and weighted output in FP32.
    // The epilogue divides output_accumulators by running_weight_sum.
    float running_row_max[2] = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    float running_weight_sum[2] = {0.0F, 0.0F};
    Accumulator output_accumulators[HeadDim / 8];
#pragma unroll
    for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
        output_accumulators[output_column_tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    }

    const int key_tile_count = (sequence + kBlockN - 1) / kBlockN;
    // Copy tile t+1 while computing tile t, alternating between two buffers.
    // Each copy moves 16 bytes (8 BF16 values). Invalid key rows load key 0;
    // the score mask below removes their contribution.
    constexpr int kChunksPerRow = HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    constexpr int kChunkElements = 16 / static_cast<int>(sizeof(__nv_bfloat16));
    const auto prefetch_kv_tile = [&](int key_tile_index, int buffer_slot) {
        __nv_bfloat16* destination_k_tile =
            shared_kv_buffers + buffer_slot * 2 * kBlockN * kKVStride;
        __nv_bfloat16* destination_v_tile = destination_k_tile + kBlockN * kKVStride;
        const int key_tile_start = key_tile_index * kBlockN;
        for (int chunk = threadIdx.x; chunk < kBlockN * kChunksPerRow; chunk += kThreads) {
            const int row = chunk / kChunksPerRow;
            const int channel = (chunk - row * kChunksPerRow) * kChunkElements;
            const int key_row =
                (NoPartialTiles || key_tile_start + row < sequence) ? key_tile_start + row : 0;
            if (channel < io_dim) {
                const int64_t source_offset =
                    qkv_head_offset + static_cast<int64_t>(key_row) * io_dim + channel;
                cp_async_16(destination_k_tile + row * kKVStride + channel, k + source_offset);
                cp_async_16(destination_v_tile + row * kKVStride + channel, v + source_offset);
            } else {
                *reinterpret_cast<uint4*>(destination_k_tile + row * kKVStride + channel) =
                    uint4{0U, 0U, 0U, 0U};
                *reinterpret_cast<uint4*>(destination_v_tile + row * kKVStride + channel) =
                    uint4{0U, 0U, 0U, 0U};
            }
        }
        cp_async_commit();
    };

    // Start the first load before entering the key loop.
    prefetch_kv_tile(0, 0);

#pragma unroll 1
    for (int key_tile_index = 0; key_tile_index < key_tile_count; ++key_tile_index) {
        const int key_tile_start = key_tile_index * kBlockN;
        const int buffer_slot = key_tile_index & 1;
        __nv_bfloat16* shared_k_tile = shared_kv_buffers + buffer_slot * 2 * kBlockN * kKVStride;
        __nv_bfloat16* shared_v_tile = shared_k_tile + kBlockN * kKVStride;
        // Stage 1: acquire this tile and prefetch the next tile.
        // The block barrier also finishes the previous tile's reads before
        // prefetch_kv_tile reuses that previous tile's buffer.
        cp_async_wait_all_committed();
        __syncthreads();
        if (key_tile_index + 1 < key_tile_count) {
            prefetch_kv_tile(key_tile_index + 1, buffer_slot ^ 1);
        }

        // Stage 2: compute QK^T for 16 query rows and the current 64 keys.
        // Each MMA covers 8 keys and sums over 16 head channels.
        Accumulator score_fragments[kBlockN / 8];
#pragma unroll
        for (int score_column_tile = 0; score_column_tile < kBlockN / 8; ++score_column_tile) {
            score_fragments[score_column_tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
            for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
                uint32_t k_fragment[2];
                ldmatrix_b_dimpair(
                    shared_k_tile, kKVStride, score_column_tile * 8, channel_chunk * 16, lane_id,
                    k_fragment);
                mma_m16n8k16(
                    score_fragments[score_column_tile], q_fragments[channel_chunk], k_fragment);
            }
        }

        // Stage 3: apply scale/mask/bias and find each row's tile maximum.
        // Invalid query/key positions receive -inf.
        float tile_row_max[2] = {
            -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        const int lane_row_group = lane_id >> 2;
        const int lane_in_row_group = lane_id & 3;
#pragma unroll
        for (int score_column_tile = 0; score_column_tile < kBlockN / 8; ++score_column_tile) {
#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int row = warp_query_start + lane_row_group + row_slot * 8;
#pragma unroll
                for (int pair_element = 0; pair_element < 2; ++pair_element) {
                    const int key_row = key_tile_start + score_column_tile * 8
                                        + lane_in_row_group * 2 + pair_element;
                    float score_value =
                        score_fragments[score_column_tile].values[row_slot * 2 + pair_element];
                    if (NoPartialTiles || (key_row < sequence && row < sequence)) {
                        score_value = score_value * attention_scale
                                      + mask[mask_plane_offset + key_row]
                                      + __bfloat162float(
                                          bias
                                              [bias_head_offset
                                               + static_cast<int64_t>(row) * sequence + key_row]);
                    } else {
                        score_value = -std::numeric_limits<float>::infinity();
                    }
                    score_fragments[score_column_tile].values[row_slot * 2 + pair_element] =
                        score_value;
                    tile_row_max[row_slot] = fmaxf(tile_row_max[row_slot], score_value);
                }
            }
        }
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            tile_row_max[row_slot] = quad_max(tile_row_max[row_slot]);
        }

        // Stage 4: update the maximum and rescale the accumulated history.
        // Previous weight/output sums must use the new softmax scale.
        float history_scale[2];
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            const float updated_row_max = fmaxf(running_row_max[row_slot], tile_row_max[row_slot]);
            history_scale[row_slot] =
                running_row_max[row_slot] == -std::numeric_limits<float>::infinity()
                    ? 0.0F
                    : __expf(running_row_max[row_slot] - updated_row_max);
            running_row_max[row_slot] = updated_row_max;
            running_weight_sum[row_slot] *= history_scale[row_slot];
        }
#pragma unroll
        for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                output_accumulators[output_column_tile].values[row_slot * 2] *=
                    history_scale[row_slot];
                output_accumulators[output_column_tile].values[row_slot * 2 + 1] *=
                    history_scale[row_slot];
            }
        }

        // Stage 5: accumulate the unnormalized weights and their PV product.
        // First compute exp(S - m) and update the FP32 row sums.
        // Keep the weights in score_fragments for the following PV step.
        float lane_tile_weight_sum[2] = {0.0F, 0.0F};
#pragma unroll
        for (int score_column_tile = 0; score_column_tile < kBlockN / 8; ++score_column_tile) {
#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                float weight0 = score_fragments[score_column_tile].values[row_slot * 2];
                float weight1 = score_fragments[score_column_tile].values[row_slot * 2 + 1];
                weight0 = weight0 == -std::numeric_limits<float>::infinity()
                              ? 0.0F
                              : __expf(weight0 - running_row_max[row_slot]);
                weight1 = weight1 == -std::numeric_limits<float>::infinity()
                              ? 0.0F
                              : __expf(weight1 - running_row_max[row_slot]);
                lane_tile_weight_sum[row_slot] += weight0 + weight1;
                score_fragments[score_column_tile].values[row_slot * 2] = weight0;
                score_fragments[score_column_tile].values[row_slot * 2 + 1] = weight1;
            }
        }
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            running_weight_sum[row_slot] += quad_sum(lane_tile_weight_sum[row_slot]);
        }

        // Then round weights into BF16 register fragments and accumulate P @ V.
        // Visit 16-key chunks in ascending order; normalize after the key loop.
#pragma unroll
        for (int key_chunk = 0; key_chunk < kBlockN / 16; ++key_chunk) {
            uint32_t weight_fragment[4];
            weight_fragment[0] = pack_bf16_pair(
                score_fragments[key_chunk * 2].values[0], score_fragments[key_chunk * 2].values[1]);
            weight_fragment[1] = pack_bf16_pair(
                score_fragments[key_chunk * 2].values[2], score_fragments[key_chunk * 2].values[3]);
            weight_fragment[2] = pack_bf16_pair(
                score_fragments[key_chunk * 2 + 1].values[0],
                score_fragments[key_chunk * 2 + 1].values[1]);
            weight_fragment[3] = pack_bf16_pair(
                score_fragments[key_chunk * 2 + 1].values[2],
                score_fragments[key_chunk * 2 + 1].values[3]);
#pragma unroll
            for (int output_column_tile = 0; output_column_tile < HeadDim / 8;
                 ++output_column_tile) {
                uint32_t v_fragment[2];
                ldmatrix_b_keytrans(
                    shared_v_tile, kKVStride, key_chunk * 16, output_column_tile * 8, lane_id,
                    v_fragment);
                mma_m16n8k16(output_accumulators[output_column_tile], weight_fragment, v_fragment);
            }
        }
    }

    // Stage 6: normalize and store output, LSE, and optional shadow residual.
    // shadow = BF16(o_fp32 - float(o_bf16)) captures final output rounding;
    // it does not recover the earlier BF16 rounding of weights used by PV.
    const int lane_row_group = lane_id >> 2;
    const int lane_in_row_group = lane_id & 3;
#pragma unroll
    for (int row_slot = 0; row_slot < 2; ++row_slot) {
        const int row = warp_query_start + lane_row_group + row_slot * 8;
        if (!NoPartialTiles && row >= sequence) {
            continue;
        }
        const float inverse_weight_sum =
            running_weight_sum[row_slot] == 0.0F ? 0.0F : 1.0F / running_weight_sum[row_slot];
#pragma unroll
        for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
#pragma unroll
            for (int pair_element = 0; pair_element < 2; ++pair_element) {
                const int output_channel =
                    output_column_tile * 8 + lane_in_row_group * 2 + pair_element;
                if (output_channel < io_dim) {
                    const float output_value =
                        output_accumulators[output_column_tile].values[row_slot * 2 + pair_element]
                        * inverse_weight_sum;
                    const __nv_bfloat16 output_bf16 = __float2bfloat16_rn(output_value);
                    const int64_t output_index = output_head_offset
                                                 + static_cast<int64_t>(row) * heads * io_dim
                                                 + output_channel;
                    out[output_index] = output_bf16;
                    if (shadow_residual != nullptr) {
                        shadow_residual[output_index] =
                            __float2bfloat16_rn(output_value - __bfloat162float(output_bf16));
                    }
                }
            }
        }
        if (lane_in_row_group == 0) {
            lse[head_row_offset + row] =
                running_row_max[row_slot] + __logf(running_weight_sum[row_slot]);
        }
    }
}
// -----------------------------------------------------------------------------
// Forward path 2: TMA + WGMMA (HeadDim = 16, 32, or 64)
// -----------------------------------------------------------------------------
// TMA loads K/V into alternating shared buffers. With BiasTma, it also loads
// bias; otherwise bias is read directly from global memory. The mask is
// prefetched one tile ahead. QK^T uses WGMMA and PV uses mma.sync.
// Bias fragments match the score layout so the score expression keeps the
// same operation order as the cp.async kernel.
template <int HeadDim, bool TileMajorGrid, bool NoPartialTiles, bool OuterEqSeq, bool BiasTma>
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_forward_tma_kernel(
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap v_map,
    const __grid_constant__ CUtensorMap bias_map,
    const __nv_bfloat16* __restrict__ q,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    __nv_bfloat16* __restrict__ shadow_residual,
    int sequence,
    int outer_count,
    int heads,
    float attention_scale) {
    // The Python facade and AOT/JIT loaders restrict execution to Hopper CC 9.0;
    // the extension is built for sm_90a. This guard is for device compilation,
    // not a fallback implementation for earlier architectures.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // A. Prepare this block's query rows, shared buffers, and running state.
    // Q stays fixed; the main loop will visit every K/V tile for these rows.
    constexpr int kQueryRowStride = HeadDim + 8;
    constexpr int kKVTileElements = kBlockN * HeadDim;    // K/V TMA box, swizzled
    constexpr int kBiasTileElements = kBlockM * kBlockN;  // bias box, 128B swizzle
    // Keep each TMA box aligned to 1024 bytes in shared memory.
    static_assert(
        ((kBlockM * kQueryRowStride) * static_cast<int>(sizeof(__nv_bfloat16))) % 1024 == 0
            && (kKVTileElements * static_cast<int>(sizeof(__nv_bfloat16))) % 1024 == 0
            && (kBiasTileElements * static_cast<int>(sizeof(__nv_bfloat16))) % 1024 == 0,
        "TMA boxes must be 1024-byte aligned in shared memory");
    const int lane_id = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    // Map this block to one batch item, outer index, head, and query tile.
    const int outer_extent = OuterEqSeq ? sequence : outer_count;
    const int batch_outer_index = TileMajorGrid ? blockIdx.z : blockIdx.x;
    const int batch_index = batch_outer_index / outer_extent;
    const int outer_index = batch_outer_index - batch_index * outer_extent;
    const int block_query_start = (TileMajorGrid ? blockIdx.x : blockIdx.z) * kBlockM;
    const int head_index = blockIdx.y;
    const int warp_query_start = block_query_start + warp_id * kRowsPerWarp;

    // Shared-memory layout: Q | K0 | K1 | V0 | V1 | [B0 | B1] | M0 | M1.
    // Bias buffers exist only with BiasTma. Q rows use kQueryRowStride; K/V and
    // bias use swizzled TMA boxes. M0/M1 hold the FP32 masks for alternating
    // tiles. shared_warp_q points only to this warp's 16 rows within the Q
    // region.
    extern __shared__ __align__(1024) unsigned char shared_storage[];
    __nv_bfloat16* shared_warp_q =
        reinterpret_cast<__nv_bfloat16*>(shared_storage) + warp_id * kRowsPerWarp * kQueryRowStride;
    __nv_bfloat16* shared_k_buffers =
        reinterpret_cast<__nv_bfloat16*>(shared_storage) + kBlockM * kQueryRowStride;
    __nv_bfloat16* shared_v_buffers = shared_k_buffers + 2 * kKVTileElements;
    __nv_bfloat16* shared_bias_buffers = shared_v_buffers + 2 * kKVTileElements;
    float* shared_mask_buffers =
        reinterpret_cast<float*>(shared_bias_buffers + (BiasTma ? 2 * kBiasTileElements : 0));
    __shared__ __align__(8) uint64_t k_load_barriers[2];
    __shared__ __align__(8) uint64_t v_load_barriers[2];
    __shared__ __align__(8) uint64_t bias_load_barriers[2];

    // Global-memory offsets below count elements, not bytes.
    const int64_t batch_outer_plane =
        static_cast<int64_t>(batch_index) * outer_extent + outer_index;
    const int64_t head_row_offset =
        (batch_outer_plane * heads + head_index) * static_cast<int64_t>(sequence);
    const int64_t q_head_offset = head_row_offset * HeadDim;
    const int64_t mask_plane_offset = batch_outer_plane * sequence;
    const int64_t bias_head_offset = (static_cast<int64_t>(batch_index) * heads + head_index)
                                     * static_cast<int64_t>(sequence) * sequence;
    // Write directly to [B, O, N, H, D], so the facade can return this layout.
    const int64_t output_head_offset =
        batch_outer_plane * static_cast<int64_t>(sequence) * heads * HeadDim
        + static_cast<int64_t>(head_index) * HeadDim;
    // The launcher checks that batch * outer * heads fits int32.
    const int kv_tma_plane = static_cast<int>(batch_outer_plane * heads + head_index);
    // Each (batch, head) has one bias matrix shared by all outer indices.
    const int bias_tma_plane = batch_index * heads + head_index;

    // One thread initializes the barriers used by the TMA transfers.
    if (threadIdx.x == 0) {
        mbarrier_init(&k_load_barriers[0], 1);
        mbarrier_init(&k_load_barriers[1], 1);
        mbarrier_init(&v_load_barriers[0], 1);
        mbarrier_init(&v_load_barriers[1], 1);
        if constexpr (BiasTma) {
            mbarrier_init(&bias_load_barriers[0], 1);
            mbarrier_init(&bias_load_barriers[1], 1);
        }
    }

    // Each warp loads its 16 Q rows into shared memory, then into registers.
    // These register fragments are reused for every key tile.
    // Invalid query rows borrow row 0; their outputs are never stored.
    for (int element_index = lane_id; element_index < kRowsPerWarp * HeadDim;
         element_index += kWarpSize) {
        const int row = element_index / HeadDim;
        const int channel = element_index - row * HeadDim;
        const int query_row =
            (NoPartialTiles || warp_query_start + row < sequence) ? warp_query_start + row : 0;
        shared_warp_q[row * kQueryRowStride + channel] =
            q[q_head_offset + static_cast<int64_t>(query_row) * HeadDim + channel];
    }
    // Load the first 64 mask values. All query rows in this block share them;
    // later iterations prefetch the next mask tile.
    if (threadIdx.x < kBlockN) {
        const int key_row = threadIdx.x;
        shared_mask_buffers[key_row] =
            (NoPartialTiles || key_row < sequence) ? mask[mask_plane_offset + key_row] : 0.0F;
    }
    __syncthreads();  // Make the Q loads, barriers, and first mask visible to the
                      // block.

    uint32_t q_fragments[HeadDim / 16][4];
#pragma unroll
    for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        ldmatrix_a16x16(
            shared_warp_q,
            kQueryRowStride,
            channel_chunk * 16,
            lane_id,
            q_fragments[channel_chunk]);
    }

    // Four adjacent lanes cooperate on two query rows, eight rows apart:
    //   query row = warp_query_start + lane_id / 4 + 8 * row_slot.
    // row_slot selects the row; the four lanes divide its columns among them.
    // Keep FP32 state across all key tiles: the row maximum (m), weight sum (l),
    // and this lane's unnormalized output fragments. Normalize only at the end.
    float running_row_max[2] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    float running_weight_sum[2] = {0.0F, 0.0F};
    Accumulator output_accumulators[HeadDim / 8];
#pragma unroll
    for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
        output_accumulators[output_column_tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    }

    // These helpers submit asynchronous copies into the requested buffer slot.
    // Thread 0 submits each request; TMA performs the actual transfer.
    // The byte count lets the matching barrier track transfer completion.
    const auto prefetch_k_tile = [&](int key_tile_index, int buffer_slot) {
        if (threadIdx.x == 0) {
            mbarrier_expect_tx(
                &k_load_barriers[buffer_slot],
                kKVTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
            tma_load_3d_slice(
                &k_map,
                shared_k_buffers + buffer_slot * kKVTileElements,
                &k_load_barriers[buffer_slot],
                key_tile_index * kBlockN,
                kv_tma_plane);
        }
    };
    const auto prefetch_v_tile = [&](int key_tile_index, int buffer_slot) {
        if (threadIdx.x == 0) {
            mbarrier_expect_tx(
                &v_load_barriers[buffer_slot],
                kKVTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
            tma_load_3d_slice(
                &v_map,
                shared_v_buffers + buffer_slot * kKVTileElements,
                &v_load_barriers[buffer_slot],
                key_tile_index * kBlockN,
                kv_tma_plane);
        }
    };
    const auto prefetch_bias_tile = [&](int key_tile_index, int buffer_slot) {
        if constexpr (BiasTma) {
            if (threadIdx.x == 0) {
                mbarrier_expect_tx(
                    &bias_load_barriers[buffer_slot],
                    kBiasTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
                tma_load_3d_box(
                    &bias_map,
                    shared_bias_buffers + buffer_slot * kBiasTileElements,
                    &bias_load_barriers[buffer_slot],
                    key_tile_index * kBlockN,
                    block_query_start,
                    bias_tma_plane);
            }
        }
    };

    // Start the first transfers before entering the key loop.
    const int key_tile_count = (sequence + kBlockN - 1) / kBlockN;
    prefetch_k_tile(0, 0);
    prefetch_v_tile(0, 0);
    prefetch_bias_tile(0, 0);

    // B. Visit all key tiles, accumulating the same query rows (Stages 1-5).
    // The buffer slot selects storage. Its phase bit distinguishes consecutive
    // transfers into that slot:
    //   key tile:  0  1  2  3  4  5
    //   slot:      0  1  0  1  0  1
    //   phase:     0  0  1  1  0  0
    // Wait K -> launch QK -> prefetch t+1 -> wait QK/bias -> softmax -> PV.
    // Separate K/V barriers let V finish later, just before PV needs it.
    // Each iteration ends by publishing the next mask and synchronizing readers.

#pragma unroll 1
    for (int key_tile_index = 0; key_tile_index < key_tile_count; ++key_tile_index) {
        const int key_tile_start = key_tile_index * kBlockN;
        const int buffer_slot = key_tile_index & 1;
        // Each buffer is reused every other tile; its barrier phase flips on reuse.
        const uint32_t barrier_phase = (key_tile_index >> 1) & 1;
        __nv_bfloat16* shared_k_tile = shared_k_buffers + buffer_slot * kKVTileElements;
        __nv_bfloat16* shared_v_tile = shared_v_buffers + buffer_slot * kKVTileElements;
        const __nv_bfloat16* shared_bias_tile =
            shared_bias_buffers + (BiasTma ? buffer_slot * kBiasTileElements : 0);
        const float* shared_mask_tile = shared_mask_buffers + buffer_slot * kBlockN;
        const bool has_next_key_tile = key_tile_index + 1 < key_tile_count;

        // Prefetch the next mask value into a register. Store it in the other
        // shared buffer at the end of this iteration, before the block barrier.
        // That slot is initially unused, or its old readers finished before
        // the previous iteration's trailing barrier, so it can be overwritten.
        float next_mask_value = 0.0F;
        if (has_next_key_tile && threadIdx.x < kBlockN) {
            const int key_row = key_tile_start + kBlockN + threadIdx.x;
            next_mask_value =
                (NoPartialTiles || key_row < sequence) ? mask[mask_plane_offset + key_row] : 0.0F;
        }

        // Stage 1: wait for this tile's K transfer before QK reads shared memory.
        // V can keep loading until the later PV step needs it.
        mbarrier_wait_parity(&k_load_barriers[buffer_slot], barrier_phase);
        proxy_fence();

        // Stage 2: compute QK^T over 16 head channels per WGMMA instruction.
        // Descriptor address/LBO/SBO fields are encoded in 16-byte units:
        //   SBO = (8 key rows * HeadDim channels * 2 bytes) / 16 = HeadDim.
        //   Mode 1/2/3 selects the 128B/64B/32B shared-memory swizzle.
        // PTX does not use LBO for this swizzled K-major layout; the encoded
        // value below is retained. Each 16-channel chunk advances by 32 bytes.
        constexpr uint64_t kKSwizzleMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
        constexpr uint64_t kDescriptorLbo = 64;
        constexpr uint64_t kDescriptorSbo = HeadDim;
        Accumulator score_fragments[kBlockN / 8];
#pragma unroll
        for (int score_column_tile = 0; score_column_tile < kBlockN / 8; ++score_column_tile) {
            score_fragments[score_column_tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        }
        wgmma_fence();
#pragma unroll
        for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
            const uint64_t k_tile_descriptor =
                ((static_cast<uint64_t>(shared_u32(shared_k_tile)) + channel_chunk * 32) >> 4)
                | (kDescriptorLbo << 16) | (kDescriptorSbo << 32) | (kKSwizzleMode << 62);
            wgmma_m64n64k16(
                score_fragments,
                q_fragments[channel_chunk],
                k_tile_descriptor,
                channel_chunk != 0);
        }
        wgmma_commit();

        // Load the next K/V tiles and optional bias tile while QK is in flight.
        if (has_next_key_tile) {
            prefetch_k_tile(key_tile_index + 1, buffer_slot ^ 1);
            prefetch_v_tile(key_tile_index + 1, buffer_slot ^ 1);
            prefetch_bias_tile(key_tile_index + 1, buffer_slot ^ 1);
        }
        wgmma_wait_zero();  // QK must finish before the scores are used below.

        // Wait for this tile's bias before reading it with ldmatrix.
        if constexpr (BiasTma) {
            mbarrier_wait_parity(&bias_load_barriers[buffer_slot], barrier_phase);
            proxy_fence();
        }

        // Stage 3: apply scale/mask/bias and find each row's tile maximum.
        float tile_row_max[2] = {
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        // Each group of four lanes shares two rows and divides their columns.
        const int lane_row_group = lane_id >> 2;
        const int lane_in_row_group = lane_id & 3;
#pragma unroll
        for (int bias_key_chunk = 0; bias_key_chunk < kBlockN / 16; ++bias_key_chunk) {
            // 16 rows x 16 keys of bias in the accumulator fragment layout.
            uint32_t bias_fragment[4];
            if constexpr (BiasTma) {
                ldmatrix_bias_x4(
                    shared_bias_tile,
                    warp_id * kRowsPerWarp,
                    bias_key_chunk,
                    lane_id,
                    bias_fragment);
            }
#pragma unroll
            for (int key_subtile = 0; key_subtile < 2; ++key_subtile) {
                const int score_column_tile = bias_key_chunk * 2 + key_subtile;
                const float2 mask_pair = *reinterpret_cast<const float2*>(
                    shared_mask_tile + score_column_tile * 8 + lane_in_row_group * 2);
#pragma unroll
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    const int row = warp_query_start + lane_row_group + row_slot * 8;
                    const int key_pair_start =
                        key_tile_start + score_column_tile * 8 + lane_in_row_group * 2;
#pragma unroll
                    for (int pair_element = 0; pair_element < 2; ++pair_element) {
                        const int key_row = key_pair_start + pair_element;
                        float score_value =
                            score_fragments[score_column_tile].values[row_slot * 2 + pair_element];
                        if (NoPartialTiles || (key_row < sequence && row < sequence)) {
                            float bias_value;
                            if constexpr (BiasTma) {
                                const __nv_bfloat16* bias_pair_values =
                                    reinterpret_cast<const __nv_bfloat16*>(
                                        &bias_fragment[key_subtile * 2 + row_slot]);
                                bias_value = __bfloat162float(bias_pair_values[pair_element]);
                            } else {
                                // Load bias directly when its layout cannot use TMA.
                                bias_value = __bfloat162float(
                                    bias
                                        [bias_head_offset + static_cast<int64_t>(row) * sequence
                                         + key_row]);
                            }
                            score_value = score_value * attention_scale
                                          + (pair_element == 0 ? mask_pair.x : mask_pair.y)
                                          + bias_value;
                        } else {
                            score_value = -std::numeric_limits<float>::infinity();
                        }
                        score_fragments[score_column_tile].values[row_slot * 2 + pair_element] =
                            score_value;
                        tile_row_max[row_slot] = fmaxf(tile_row_max[row_slot], score_value);
                    }
                }
            }
        }
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            tile_row_max[row_slot] = quad_max(tile_row_max[row_slot]);
        }

        // Stage 4: update the maximum and rescale the accumulated history.
        // The old sums use the old row maximum. Express them relative to the
        // updated maximum before adding this tile's weights and weighted V.
        float history_scale[2];
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            const float updated_row_max = fmaxf(running_row_max[row_slot], tile_row_max[row_slot]);
            history_scale[row_slot] =
                running_row_max[row_slot] == -std::numeric_limits<float>::infinity()
                    ? 0.0F
                    : __expf(running_row_max[row_slot] - updated_row_max);
            running_row_max[row_slot] = updated_row_max;
            running_weight_sum[row_slot] *= history_scale[row_slot];
        }
#pragma unroll
        for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                output_accumulators[output_column_tile].values[row_slot * 2] *=
                    history_scale[row_slot];
                output_accumulators[output_column_tile].values[row_slot * 2 + 1] *=
                    history_scale[row_slot];
            }
        }

        // Stage 5: accumulate the unnormalized weights and their PV product.
        // Here, weights are computed just before each 16-key PV chunk, rather
        // than for all 64 keys before PV as in the cp.async path. Keep a separate
        // -inf check for each score: another score can be NaN while fmaxf leaves
        // the row maximum at -inf. Preserve the row-sum and MMA addition orders.
        // Reuse score_fragments for unnormalized weights, computed in FP32.
        // Sum the FP32 weights for the denominator; pack BF16 copies for PV.
        // Each lane first sums its columns; quad_sum below combines four lanes.
        float lane_tile_weight_sum[2] = {0.0F, 0.0F};

        // Wait for V before loading its fragments for PV.
        mbarrier_wait_parity(&v_load_barriers[buffer_slot], barrier_phase);
        proxy_fence();
#pragma unroll
        for (int key_chunk = 0; key_chunk < kBlockN / 16; ++key_chunk) {
#pragma unroll
            for (int key_subtile = 0; key_subtile < 2; ++key_subtile) {
                const int score_column_tile = key_chunk * 2 + key_subtile;
#pragma unroll
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    float weight0 = score_fragments[score_column_tile].values[row_slot * 2];
                    float weight1 = score_fragments[score_column_tile].values[row_slot * 2 + 1];
                    weight0 = weight0 == -std::numeric_limits<float>::infinity()
                                  ? 0.0F
                                  : __expf(weight0 - running_row_max[row_slot]);
                    weight1 = weight1 == -std::numeric_limits<float>::infinity()
                                  ? 0.0F
                                  : __expf(weight1 - running_row_max[row_slot]);
                    lane_tile_weight_sum[row_slot] += weight0 + weight1;
                    score_fragments[score_column_tile].values[row_slot * 2] = weight0;
                    score_fragments[score_column_tile].values[row_slot * 2 + 1] = weight1;
                }
            }
            uint32_t weight_fragment[4];
            weight_fragment[0] = pack_bf16_pair(
                score_fragments[key_chunk * 2].values[0],
                score_fragments[key_chunk * 2].values[1]);
            weight_fragment[1] = pack_bf16_pair(
                score_fragments[key_chunk * 2].values[2],
                score_fragments[key_chunk * 2].values[3]);
            weight_fragment[2] = pack_bf16_pair(
                score_fragments[key_chunk * 2 + 1].values[0],
                score_fragments[key_chunk * 2 + 1].values[1]);
            weight_fragment[3] = pack_bf16_pair(
                score_fragments[key_chunk * 2 + 1].values[2],
                score_fragments[key_chunk * 2 + 1].values[3]);
#pragma unroll
            for (int output_column_tile = 0; output_column_tile < HeadDim / 8;
                 ++output_column_tile) {
                uint32_t v_fragment[2];
                ldmatrix_b_keytrans_sw<HeadDim>(
                    shared_v_tile,
                    key_chunk * 16,
                    output_column_tile * 8,
                    lane_id,
                    v_fragment);
                mma_m16n8k16(output_accumulators[output_column_tile], weight_fragment, v_fragment);
            }
        }
#pragma unroll
        for (int row_slot = 0; row_slot < 2; ++row_slot) {
            running_weight_sum[row_slot] += quad_sum(lane_tile_weight_sum[row_slot]);
        }

        // Publish the prefetched mask before the next iteration reads it.
        if (has_next_key_tile && threadIdx.x < kBlockN) {
            shared_mask_buffers[(buffer_slot ^ 1) * kBlockN + threadIdx.x] = next_mask_value;
        }
        __syncthreads();  // Finish all reads before reusing this tile's buffers.
    }

    // C. Finish after every key tile has contributed to the running state.
    // Stage 6: normalize and store output, LSE, and optional shadow residual.
    // Use the same final-rounding residual as the cp.async epilogue.
    const int lane_row_group = lane_id >> 2;
    const int lane_in_row_group = lane_id & 3;
#pragma unroll
    for (int row_slot = 0; row_slot < 2; ++row_slot) {
        const int row = warp_query_start + lane_row_group + row_slot * 8;
        if (!NoPartialTiles && row >= sequence) {
            continue;
        }
        const float inverse_weight_sum =
            running_weight_sum[row_slot] == 0.0F ? 0.0F : 1.0F / running_weight_sum[row_slot];
#pragma unroll
        for (int output_column_tile = 0; output_column_tile < HeadDim / 8; ++output_column_tile) {
#pragma unroll
            for (int pair_element = 0; pair_element < 2; ++pair_element) {
                const int output_channel =
                    output_column_tile * 8 + lane_in_row_group * 2 + pair_element;
                const float output_value =
                    output_accumulators[output_column_tile].values[row_slot * 2 + pair_element]
                    * inverse_weight_sum;
                const __nv_bfloat16 output_bf16 = __float2bfloat16_rn(output_value);
                const int64_t output_index = output_head_offset
                                             + static_cast<int64_t>(row) * heads * HeadDim
                                             + output_channel;
                out[output_index] = output_bf16;
                if (shadow_residual != nullptr) {
                    // Save a BF16 approximation of the rounding residual. Backward
                    // uses it to compute delta without another probability traversal.
                    shadow_residual[output_index] =
                        __float2bfloat16_rn(output_value - __bfloat162float(output_bf16));
                }
            }
        }
        if (lane_in_row_group == 0) {
            lse[head_row_offset + row] =
                running_row_max[row_slot] + __logf(running_weight_sum[row_slot]);
        }
    }
#endif
}

// -----------------------------------------------------------------------------
// Shared-memory sizes and TMA tensor maps
// -----------------------------------------------------------------------------

template <int HeadDim> constexpr int shared_bytes() {
    return (kBlockM * (HeadDim + 8) + 4 * kBlockN * (HeadDim + 8))
           * static_cast<int>(sizeof(__nv_bfloat16));
}

template <int HeadDim, bool BiasTma> constexpr int shared_bytes_tma() {
    // Per-warp Q, double-buffered K/V, optional double-buffered bias, and
    // a double-buffered FP32 mask row. Each TMA box is 1024-byte aligned.
    return (kBlockM * (HeadDim + 8) + 4 * kBlockN * HeadDim + (BiasTma ? 2 * kBlockM * kBlockN : 0))
               * static_cast<int>(sizeof(__nv_bfloat16))
           + 2 * kBlockN * static_cast<int>(sizeof(float));
}

// Resolve cuTensorMapEncodeTiled through the runtime instead of linking
// directly against libcuda.
using TensorMapEncodeFn = CUresult (*)(
    CUtensorMap*,
    CUtensorMapDataType,
    cuuint32_t,
    void*,
    const cuuint64_t*,
    const cuuint64_t*,
    const cuuint32_t*,
    const cuuint32_t*,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

TensorMapEncodeFn tensor_map_encode_fn() {
    static const TensorMapEncodeFn encode = [] {
        void* symbol = nullptr;
        cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
        const cudaError_t status =
            cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &symbol, cudaEnableDefault, &query);
        TORCH_CHECK(
            status == cudaSuccess && query == cudaDriverEntryPointSuccess && symbol != nullptr,
            "cuTensorMapEncodeTiled is unavailable in this driver");
        return reinterpret_cast<TensorMapEncodeFn>(symbol);
    }();
    return encode;
}

// Describe K or V as [planes, sequence, HeadDim], where planes = B * O * H.
// Each TMA load covers one key tile, with the swizzle matched to its row width.
CUtensorMap make_kv_tensor_map(
    const torch::Tensor& source,
    int head_dim,
    int sequence,
    int64_t planes) {
    CUtensorMap tensor_map;
    const cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(head_dim), static_cast<cuuint64_t>(sequence),
        static_cast<cuuint64_t>(planes)};
    const cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(head_dim) * sizeof(__nv_bfloat16),
        static_cast<cuuint64_t>(head_dim) * sequence * sizeof(__nv_bfloat16)};
    const cuuint32_t box_dim[3] = {static_cast<cuuint32_t>(head_dim), kBlockN, 1};
    const cuuint32_t element_stride[3] = {1, 1, 1};
    const CUtensorMapSwizzle swizzle =
        head_dim == 16 ? CU_TENSOR_MAP_SWIZZLE_32B
                       : (head_dim == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B);
    const CUresult result = tensor_map_encode_fn()(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, source.data_ptr(), global_dim,
        global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(
        result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
    return tensor_map;
}

// Describe bias as [B * H, query, key]. Each load covers one query/key tile
// and uses the 128-byte swizzle expected by ldmatrix_bias_x4.
// The caller checks the required 16-byte address and row-stride alignment.
CUtensorMap make_bias_tensor_map(const torch::Tensor& bias, int sequence, int64_t planes) {
    CUtensorMap tensor_map;
    const cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(sequence), static_cast<cuuint64_t>(sequence),
        static_cast<cuuint64_t>(planes)};
    const cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(sequence) * sizeof(__nv_bfloat16),
        static_cast<cuuint64_t>(sequence) * sequence * sizeof(__nv_bfloat16)};
    const cuuint32_t box_dim[3] = {kBlockN, kBlockM, 1};
    const cuuint32_t element_stride[3] = {1, 1, 1};
    const CUresult result = tensor_map_encode_fn()(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, bias.data_ptr(), global_dim,
        global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(
        result == CUDA_SUCCESS, "cuTensorMapEncodeTiled (bias) failed: ", static_cast<int>(result));
    return tensor_map;
}

// -----------------------------------------------------------------------------
// Host launch helpers: grid, shared-memory size, and kernel specialization
// -----------------------------------------------------------------------------

template <int HeadDim, bool TileMajorGrid, bool NoPartialTiles, bool OuterEqSeq, bool BiasTma>
void launch_forward_tma_kernel(
    const CUtensorMap& k_map,
    const CUtensorMap& v_map,
    const CUtensorMap& bias_map,
    const torch::Tensor& q,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    __nv_bfloat16* shadow_residual,
    int sequence,
    int outer_count,
    int heads,
    float attention_scale,
    dim3 grid,
    cudaStream_t stream) {
    const int shared_memory_bytes = shared_bytes_tma<HeadDim, BiasTma>();

    // Allow this specialization to use the required dynamic shared memory.
    const cudaError_t status = cudaFuncSetAttribute(
        turbopairformer_triangle_forward_tma_kernel<
            HeadDim, TileMajorGrid, NoPartialTiles, OuterEqSeq, BiasTma>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    TORCH_CHECK(
        status == cudaSuccess,
        "failed to opt in TurboPairFormer TMA forward shared memory: ", cudaGetErrorString(status));

    // Launch the selected kernel on the caller's stream.
    turbopairformer_triangle_forward_tma_kernel<
        HeadDim, TileMajorGrid, NoPartialTiles, OuterEqSeq, BiasTma>
        <<<grid, kThreads, shared_memory_bytes, stream>>>(
            k_map, v_map, bias_map, reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            mask.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
            shadow_residual, sequence, outer_count, heads, attention_scale);
}

template <int HeadDim, bool BiasTma>
auto select_forward_tma_launcher(
    bool query_tiles_on_grid_x,
    bool no_partial_tiles,
    bool outer_matches_sequence) {
    // Each choice names a compiled specialization of the CPU launch helper.
    // Taking its address does not launch the GPU; the caller invokes it later.
    if (query_tiles_on_grid_x && no_partial_tiles) {
        // Query tiles on grid.x; every query/key tile is full.
        return outer_matches_sequence ? &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/true, BiasTma>
                                      : &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/false, BiasTma>;
    }

    if (query_tiles_on_grid_x) {
        // Query tiles on grid.x; the last query/key tile needs bounds checks.
        return outer_matches_sequence ? &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/false,
                                            /*OuterEqSeq=*/true, BiasTma>
                                      : &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/false,
                                            /*OuterEqSeq=*/false, BiasTma>;
    }

    if (no_partial_tiles) {
        // Outer tasks on grid.x; every query/key tile is full.
        return outer_matches_sequence ? &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/false,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/true, BiasTma>
                                      : &launch_forward_tma_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/false,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/false, BiasTma>;
    }

    // Outer tasks on grid.x; the last query/key tile needs bounds checks.
    return outer_matches_sequence ? &launch_forward_tma_kernel<
                                        HeadDim,
                                        /*TileMajorGrid=*/false,
                                        /*NoPartialTiles=*/false,
                                        /*OuterEqSeq=*/true, BiasTma>
                                  : &launch_forward_tma_kernel<
                                        HeadDim,
                                        /*TileMajorGrid=*/false,
                                        /*NoPartialTiles=*/false,
                                        /*OuterEqSeq=*/false, BiasTma>;
}

template <int HeadDim, bool BiasTma>
void launch_forward_tma_specialized(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    __nv_bfloat16* shadow_residual,
    cudaStream_t stream) {
    // 1. Describe the K/V tensors, and bias when it also uses TMA.
    const int64_t kv_plane_count = static_cast<int64_t>(batch) * outer_count * heads;
    const CUtensorMap k_map = make_kv_tensor_map(k, HeadDim, sequence, kv_plane_count);
    const CUtensorMap v_map = make_kv_tensor_map(v, HeadDim, sequence, kv_plane_count);
    CUtensorMap bias_map{};
    if constexpr (BiasTma) {
        bias_map = make_bias_tensor_map(bias, sequence, static_cast<int64_t>(batch) * heads);
    }

    // 2. Set the attention scale and assign blocks to
    //    (batch, outer, head, query tile) tasks.
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
    const int query_tile_count = (sequence + kBlockM - 1) / kBlockM;
    const int64_t batch_outer_count = static_cast<int64_t>(batch) * outer_count;
    const bool query_tiles_on_grid_x = batch_outer_count <= kMaxGridYZ;
    dim3 grid;
    if (query_tiles_on_grid_x) {
        // grid = (query tiles, heads, batch * outer).
        grid = dim3(
            static_cast<unsigned int>(query_tile_count), static_cast<unsigned int>(heads),
            static_cast<unsigned int>(batch_outer_count));
    } else {
        // batch * outer exceeds grid.z's limit, so exchange the x/z roles.
        grid = dim3(
            static_cast<unsigned int>(batch_outer_count), static_cast<unsigned int>(heads),
            static_cast<unsigned int>(query_tile_count));
    }

    // 3. Choose the compiled version that matches this input shape.
    const bool no_partial_tiles = sequence % kBlockM == 0 && sequence % kBlockN == 0;
    const bool outer_matches_sequence = outer_count == sequence;

    const auto selected_launcher = select_forward_tma_launcher<HeadDim, BiasTma>(
        query_tiles_on_grid_x, no_partial_tiles, outer_matches_sequence);

    // 4. Configure shared memory and launch once with the common arguments.
    selected_launcher(
        k_map, v_map, bias_map, q, mask, bias, out, lse, shadow_residual, sequence, outer_count,
        heads, attention_scale, grid, stream);
}

template <int HeadDim>
void launch_forward_tma(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    __nv_bfloat16* shadow_residual,
    cudaStream_t stream) {
    // Bias TMA needs a 16-byte-aligned address and row stride.
    // Otherwise, keep K/V on TMA and load bias directly.
    const bool bias_tma_ok =
        sequence % 8 == 0 && reinterpret_cast<std::uintptr_t>(bias.data_ptr()) % 16 == 0;
    if (bias_tma_ok) {
        launch_forward_tma_specialized<HeadDim, true>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, shadow_residual,
            stream);
    } else {
        launch_forward_tma_specialized<HeadDim, false>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, shadow_residual,
            stream);
    }
}

template <int HeadDim, bool TileMajorGrid, bool NoPartialTiles, bool OuterEqSeq>
void launch_forward_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    __nv_bfloat16* shadow_residual,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_dim,
    float attention_scale,
    dim3 grid,
    cudaStream_t stream) {
    const int shared_memory_bytes = shared_bytes<HeadDim>();

    // Allow this specialization to use the required dynamic shared memory.
    const cudaError_t status = cudaFuncSetAttribute(
        turbopairformer_triangle_forward_kernel<HeadDim, TileMajorGrid, NoPartialTiles, OuterEqSeq>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    TORCH_CHECK(
        status == cudaSuccess,
        "failed to opt in TurboPairFormer forward shared memory: ", cudaGetErrorString(status));

    // Launch the selected kernel on the caller's stream.
    turbopairformer_triangle_forward_kernel<HeadDim, TileMajorGrid, NoPartialTiles, OuterEqSeq>
        <<<grid, kThreads, shared_memory_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), mask.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
            shadow_residual, sequence, outer_count, heads, batch, io_dim, attention_scale);
}

template <int HeadDim>
auto select_forward_launcher(
    bool query_tiles_on_grid_x,
    bool no_partial_tiles,
    bool outer_matches_sequence) {
    // Each choice names a compiled specialization of the CPU launch helper.
    // Taking its address does not launch the GPU; the caller invokes it later.
    if (query_tiles_on_grid_x && no_partial_tiles) {
        // Query tiles on grid.x; every query/key tile is full.
        return outer_matches_sequence ? &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/true>
                                      : &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/false>;
    }

    if (query_tiles_on_grid_x) {
        // Query tiles on grid.x; the last query/key tile needs bounds checks.
        return outer_matches_sequence ? &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/false,
                                            /*OuterEqSeq=*/true>
                                      : &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/true,
                                            /*NoPartialTiles=*/false,
                                            /*OuterEqSeq=*/false>;
    }

    if (no_partial_tiles) {
        // Outer tasks on grid.x; every query/key tile is full.
        return outer_matches_sequence ? &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/false,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/true>
                                      : &launch_forward_kernel<
                                            HeadDim,
                                            /*TileMajorGrid=*/false,
                                            /*NoPartialTiles=*/true,
                                            /*OuterEqSeq=*/false>;
    }

    // Outer tasks on grid.x; the last query/key tile needs bounds checks.
    return outer_matches_sequence ? &launch_forward_kernel<
                                        HeadDim,
                                        /*TileMajorGrid=*/false,
                                        /*NoPartialTiles=*/false,
                                        /*OuterEqSeq=*/true>
                                  : &launch_forward_kernel<
                                        HeadDim,
                                        /*TileMajorGrid=*/false,
                                        /*NoPartialTiles=*/false,
                                        /*OuterEqSeq=*/false>;
}

template <int HeadDim>
void launch_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_dim,
    __nv_bfloat16* shadow_residual,
    cudaStream_t stream) {
    // 1. Set the attention scale and assign blocks to
    //    (batch, outer, head, query tile) tasks.
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
    const int query_tile_count = (sequence + kBlockM - 1) / kBlockM;
    const int64_t batch_outer_count = static_cast<int64_t>(batch) * outer_count;
    const bool query_tiles_on_grid_x = batch_outer_count <= kMaxGridYZ;
    dim3 grid;
    if (query_tiles_on_grid_x) {
        // grid = (query tiles, heads, batch * outer).
        grid = dim3(
            static_cast<unsigned int>(query_tile_count), static_cast<unsigned int>(heads),
            static_cast<unsigned int>(batch_outer_count));
    } else {
        // batch * outer exceeds grid.z's limit, so exchange the x/z roles.
        grid = dim3(
            static_cast<unsigned int>(batch_outer_count), static_cast<unsigned int>(heads),
            static_cast<unsigned int>(query_tile_count));
    }

    // 2. Choose the compiled version that matches this input shape.
    const bool no_partial_tiles = sequence % kBlockM == 0 && sequence % kBlockN == 0;
    const bool outer_matches_sequence = outer_count == sequence;

    const auto selected_launcher = select_forward_launcher<HeadDim>(
        query_tiles_on_grid_x, no_partial_tiles, outer_matches_sequence);

    // 3. Configure shared memory and launch once with the common arguments.
    selected_launcher(
        q, k, v, mask, bias, out, lse, shadow_residual, sequence, outer_count, heads, batch, io_dim,
        attention_scale, grid, stream);
}

// -----------------------------------------------------------------------------
// PyTorch entry points: layout conversion, validation, and forward dispatch
// -----------------------------------------------------------------------------

std::vector<torch::Tensor> turbopairformer_pack_qkv(
    const torch::Tensor& q_source,
    const torch::Tensor& k_source,
    const torch::Tensor& v_source,
    int64_t batch_outer,
    int64_t heads64,
    int64_t sequence64,
    int64_t head_dim64) {
    TORCH_CHECK(
        batch_outer > 0 && heads64 > 0 && heads64 <= kMaxGridYZ && sequence64 > 0
            && sequence64 <= std::numeric_limits<int>::max() && head_dim64 > 0 && head_dim64 <= 64
            && head_dim64 % 2 == 0,
        "TurboPairFormer pack requires positive batch_outer/N, H<=65535, and an "
        "even D<=64");
    TORCH_CHECK(
        batch_outer <= std::numeric_limits<int64_t>::max() / heads64 / sequence64 / head_dim64,
        "TurboPairFormer pack element count overflows int64");
    const int64_t elements = batch_outer * heads64 * sequence64 * head_dim64;
    const auto check_source = [&](const torch::Tensor& tensor) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == q_source.device()
                && tensor.dtype() == torch::kBFloat16 && tensor.is_contiguous()
                && tensor.numel() == elements,
            "TurboPairFormer pack sources must be contiguous BF16 tensors with the "
            "declared element count on one CUDA device");
    };
    check_source(q_source);
    check_source(k_source);
    check_source(v_source);

    const int64_t rows = batch_outer * heads64 * sequence64;
    const int64_t blocks = (rows + 7) / 8;
    TORCH_CHECK(blocks <= kMaxGridX, "TurboPairFormer pack exceeds the CUDA grid-x contract");
    const int heads = static_cast<int>(heads64);
    const int sequence = static_cast<int>(sequence64);
    const int head_dim = static_cast<int>(head_dim64);
    const c10::cuda::CUDAGuard device_guard(q_source.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto options = q_source.options();
    auto out_q = torch::empty({batch_outer, heads64, sequence64, head_dim64}, options);
    auto out_k = torch::empty_like(out_q);
    auto out_v = torch::empty_like(out_q);
    const auto aligned = [](const torch::Tensor& tensor, int bytes) {
        return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr())
                & static_cast<std::uintptr_t>(bytes - 1))
               == 0;
    };
    const int row_bytes = head_dim * static_cast<int>(sizeof(__nv_bfloat16));
    const auto safe_width = [&](int bytes) {
        return row_bytes % bytes == 0 && aligned(q_source, bytes) && aligned(k_source, bytes)
               && aligned(v_source, bytes) && aligned(out_q, bytes) && aligned(out_k, bytes)
               && aligned(out_v, bytes);
    };
    // Choose the widest aligned copy that stays within a row.
    // A single BF16 value (2 bytes) is the fallback.
    const int copy_bytes = safe_width(16) ? 16 : (safe_width(8) ? 8 : (safe_width(4) ? 4 : 2));
    const int64_t vectors_per_row = row_bytes / copy_bytes;
    const int64_t total_vectors = rows * vectors_per_row;
    constexpr int kCopyThreads = 128;
    const int64_t copy_blocks =
        std::min<int64_t>((total_vectors + kCopyThreads - 1) / kCopyThreads, kMaxGridX);
    const dim3 block(kCopyThreads);
    const dim3 grid(static_cast<unsigned int>(copy_blocks));

#define LAUNCH_TURBOPAIRFORMER_PACK_COPY(CopyType)                                                 \
    turbopairformer_pack_qkv_kernel<CopyType><<<grid, block, 0, stream>>>(                         \
        reinterpret_cast<const CopyType*>(q_source.data_ptr()),                                    \
        reinterpret_cast<const CopyType*>(k_source.data_ptr()),                                    \
        reinterpret_cast<const CopyType*>(v_source.data_ptr()),                                    \
        reinterpret_cast<CopyType*>(out_q.data_ptr()),                                             \
        reinterpret_cast<CopyType*>(out_k.data_ptr()),                                             \
        reinterpret_cast<CopyType*>(out_v.data_ptr()), total_vectors, heads, sequence,             \
        vectors_per_row)
    switch (copy_bytes) {
    case 16:
        LAUNCH_TURBOPAIRFORMER_PACK_COPY(uint4);
        break;
    case 8:
        LAUNCH_TURBOPAIRFORMER_PACK_COPY(uint2);
        break;
    case 4:
        LAUNCH_TURBOPAIRFORMER_PACK_COPY(std::uint32_t);
        break;
    default:
        LAUNCH_TURBOPAIRFORMER_PACK_COPY(std::uint16_t);
        break;
    }

#undef LAUNCH_TURBOPAIRFORMER_PACK_COPY
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer pack QKV launch failed");
    return {out_q, out_k, out_v};
}

void turbopairformer_triangle_forward_impl(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    __nv_bfloat16* shadow_residual) {
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);

    // Read [B, O, H, N, D] and check the launch dimensions before narrowing.
    const int64_t batch64 = q.size(0);
    const int64_t outer64 = q.size(1);
    const int64_t heads64 = q.size(2);
    const int64_t sequence64 = q.size(3);
    const int64_t head_dim64 = q.size(4);
    TORCH_CHECK(
        batch64 > 0 && batch64 <= std::numeric_limits<int>::max() && outer64 > 0
            && outer64 <= std::numeric_limits<int>::max() && sequence64 > 0
            && sequence64 <= std::numeric_limits<int>::max() && heads64 > 0 && heads64 <= kMaxGridYZ
            && head_dim64 > 0 && head_dim64 <= std::numeric_limits<int>::max()
            && batch64 * outer64 <= kMaxGridX && (sequence64 + kBlockM - 1) / kBlockM <= kMaxGridYZ,
        "batch/outer/sequence/head count exceeds the CUDA grid contract");
    const int batch = static_cast<int>(batch64);
    const int outer_count = static_cast<int>(outer64);
    const int sequence = static_cast<int>(sequence64);
    const int heads = static_cast<int>(heads64);
    const int head_dim = static_cast<int>(head_dim64);

    // Check the tensors used by both forward kernels.
    const auto check_q_like = [&](const torch::Tensor& tensor) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kBFloat16
                && tensor.sizes() == q.sizes() && tensor.is_contiguous(),
            "k/v/out must be contiguous BF16 tensors matching q on one device");
    };
    check_q_like(k);
    check_q_like(v);
    TORCH_CHECK(
        out.is_cuda() && out.device() == q.device() && out.dtype() == torch::kBFloat16
            && out.is_contiguous()
            && out.sizes() == torch::IntArrayRef({batch, outer_count, sequence, heads, head_dim}),
        "out must be contiguous BF16 [B,O,N,H,D] on q.device");
    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(
        mask.is_cuda() && mask.device() == q.device() && mask.dtype() == torch::kFloat32
            && mask.is_contiguous()
            && mask.sizes() == torch::IntArrayRef({batch, outer_count, 1, 1, sequence}),
        "mask must be contiguous FP32 [B,O,1,1,N] on q.device");
    TORCH_CHECK(
        bias.is_cuda() && bias.device() == q.device() && bias.dtype() == torch::kBFloat16
            && bias.is_contiguous()
            && bias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
        "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
    TORCH_CHECK(
        lse.is_cuda() && lse.device() == q.device() && lse.dtype() == torch::kFloat32
            && lse.is_contiguous()
            && lse.sizes() == torch::IntArrayRef({batch, outer_count, heads, sequence}),
        "lse must be contiguous FP32 [B,O,H,N] on q.device");

    // Both TMA and the 16-byte cp.async copies require aligned K/V addresses.
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(k.data_ptr()) % 16 == 0
            && reinterpret_cast<std::uintptr_t>(v.data_ptr()) % 16 == 0,
        "TurboPairFormer forward K/V storage must be 16-byte aligned");

    const c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Native dispatch accepts multiples of 8 through 128 (the public Python
    // facade limits D to 64):
    //   D = 16/32/64 and B*O*H fits int32 -> TMA + WGMMA.
    //   All other accepted cases        -> cp.async + mma.sync.
    // storage is the compiled width, not a change to global tensor strides:
    // the cp.async path zero-fills channels from head_dim through storage-1.
    // Bias alignment is handled independently inside launch_forward_tma.
    const bool tma_planes_ok = batch64 * outer64 * heads64 <= kMaxTmaPlanes;
    TORCH_CHECK(
        head_dim % 8 == 0 && head_dim <= 128, "head_dim must be a multiple of 8 and <= 128, got ",
        head_dim);
    const int storage = head_dim <= 16 ? 16 : (head_dim <= 32 ? 32 : (head_dim <= 64 ? 64 : 128));
    switch (storage) {
    case 16:
        if (tma_planes_ok && head_dim == 16) {
            launch_forward_tma<16>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, shadow_residual,
                stream);
        } else {
            launch_forward<16>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, head_dim,
                shadow_residual, stream);
        }
        break;
    case 32:
        if (tma_planes_ok && head_dim == 32) {
            launch_forward_tma<32>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, shadow_residual,
                stream);
        } else {
            launch_forward<32>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, head_dim,
                shadow_residual, stream);
        }
        break;
    case 64:
        if (tma_planes_ok && head_dim == 64) {
            launch_forward_tma<64>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, shadow_residual,
                stream);
        } else {
            launch_forward<64>(
                q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, head_dim,
                shadow_residual, stream);
        }
        break;
    default:
        launch_forward<128>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch, head_dim,
            shadow_residual, stream);
        break;
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

void turbopairformer_triangle_forward_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse) {
    turbopairformer_triangle_forward_impl(q, k, v, mask, bias, out, lse, nullptr);
}

// Also save a BF16 approximation of the output rounding residual in
// [B, O, N, H, D] layout. Backward uses it to improve the row correction delta.
void turbopairformer_triangle_forward_shadow_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out,
    torch::Tensor& lse,
    torch::Tensor& shadow_residual) {
    TORCH_CHECK(
        shadow_residual.is_cuda() && shadow_residual.device() == out.device()
            && shadow_residual.dtype() == torch::kBFloat16 && shadow_residual.is_contiguous()
            && shadow_residual.sizes() == out.sizes(),
        "shadow_residual must be contiguous BF16 matching out");
    turbopairformer_triangle_forward_impl(
        q, k, v, mask, bias, out, lse,
        reinterpret_cast<__nv_bfloat16*>(shadow_residual.data_ptr()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "turbopairformer_pack_qkv", &turbopairformer_pack_qkv,
        "TurboPairFormer fused physical [N,H,D] to logical [H,N,D] QKV pack");
    module.def(
        "turbopairformer_triangle_forward_out", &turbopairformer_triangle_forward_out,
        "TurboPairFormer Triangle Attention forward (online softmax)");
    module.def(
        "turbopairformer_triangle_forward_shadow_out", &turbopairformer_triangle_forward_shadow_out,
        "Forward + BF16 rounding-residual shadow for the SR delta");
}
