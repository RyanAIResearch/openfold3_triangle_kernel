// SPDX-License-Identifier: Apache-2.0
// TurboPairFormer single-reconstruction backward and split dQ/dBias kernels.
// The public facade uses single reconstruction (SR): reconstruct each
// probability tile once and reuse it for dQ, dK, dV, and dBias. Direct split
// calls compute dQ/dBias here and dK/dV in the sibling dK/dV source.
// Both routes use bounded dBias workspace and fixed reduction orders.
//
// All three leaf kernels reconstruct the same quantities:
//   P  = exp(attention_scale * QK^T + mask + bias - logsumexp)
//   dP = dO V^T;  dS = P * (dP - delta)
//   dQ = attention_scale * dS K;  dBias = sum_outer(dS)
// SR also computes dK = attention_scale * dS^T Q and dV = P^T dO.
// Matrix products accumulate in FP32; their P/dS operands are rounded to
// BF16. dS contributes to the FP32 dBias tree before this operand rounding.
// The supplied logsumexp and delta are FP32 row statistics. A logsumexp of
// true -inf gives zero probability; finite values follow the usual formula.
//
// An outer leaf groups four consecutive outer indices. Each dBias leaf
// plane has one owning block per tile and uses (o0 + o1) + (o2 + o3).
// The single-outer SR case directly uses its one active contribution.
// Each leaf plane is rounded to BF16 in workspace; the finalizer folds these
// planes into an FP32 root in ascending leaf order. Workspace is reused in
// leaf waves, query slabs, and batch chunks. Wave boundaries only store and
// reload the FP32 root; they do not introduce another arithmetic reduction.
// Replay determinism means bitwise agreement with the same inputs, binary,
// and device; bitwise agreement with PyTorch is not the numerical contract.
//
// Internal CUDA ABI layouts:
// q/k/v/out_dq/out_dk/out_dv: [B,O,H,N,D] BF16; dO: [B,O,N,H,D] BF16.
// mask: [B,O,1,1,N] FP32 additive; bias: [B,1,H,N,N] BF16 additive.
// lse/delta: [B,O,H,N] FP32; out_dbias: [B,1,H,N,N] FP32.
// B = batch, O = independent outer planes, H = heads, N = sequence length,
// D = tensor head channels. io_head_dim is D; HeadDim is the compile-time
// storage width and may be larger. Staging fills extra channels with zero.
//
// Reading guide:
//   - turbopairformer_triangle_sr_backward_out validates the fused SR route;
//     launch_sr schedules turbopairformer_sr_leaf_kernel and its reducers.
//   - turbopairformer_triangle_dq_dbias_out validates the split route;
//     dispatch_sequence chooses the per-warp or warpgroup leaf kernel.
//   - Each leaf kernel labels preparation, reconstruction, accumulation,
//     and writeback stages. Shared-memory and register lifetimes are local
//     to that kernel; SR keeps K/V fixed while the split kernels keep Q/dO.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <pybind11/stl.h>
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
constexpr int kQueryTile = 16;  // Query rows per warp's MMA tile in the split kernels.
constexpr int kQuerySubTiles = 2;
constexpr int kQueryBlock = kQueryTile * kQuerySubTiles;  // Per-warp split leaf: 32 rows.
constexpr int kKeyTile = 32;   // Keys per iteration in both split kernels.
constexpr int kOuterLeaf = 4;  // Outer planes combined by one dBias leaf.
constexpr int kPasses = kOuterLeaf / kWarpsPerBlock;
constexpr int kMaxPrefixRank = 16;

constexpr int kMaxGridYZ = 65535;
constexpr long long kMaxGridX = 2147483647LL;
// Retain the grid-stride reducers' launch cap; this is not the grid.x hardware limit.
constexpr int kReductionGridBlocks = 65535;

// Map a flattened target batch back to the original, possibly broadcast bias.
struct PrefixBroadcastPlan {
  int rank;
  int target_dims[kMaxPrefixRank];
  int bias_dims[kMaxPrefixRank];
  int64_t bias_strides[kMaxPrefixRank];
};

// -----------------------------------------------------------------------------
// Matrix fragments and shared-memory addressing
// -----------------------------------------------------------------------------

// Four FP32 values owned by one thread in an m16n8 accumulator tile.
// r = lane_id / 4 and c = 2 * (lane_id % 4):
//   values[0], values[1] -> (r,     c), (r,     c + 1)
//   values[2], values[3] -> (r + 8, c), (r + 8, c + 1)
// A warp collectively owns 16 rows by 8 columns. Split score rows are queries;
// SR score rows are keys because SR reconstructs the transposed score tile.
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

// Convert an existing shared-memory pointer to the address used by PTX.
__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Load a warp's 16x16 A tile into four packed BF16 registers per thread.
// ld counts BF16 elements between row starts, including any padding.
__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile,
    int ld,
    int column_start,
    int lane_id,
    uint32_t (&frag)[4]) {
  const int row = lane_id & 15;
  const int column = column_start + ((lane_id >> 4) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + column);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
               : "r"(addr));
}

// B for QK^T and dO@V^T: contraction over the contiguous dim axis.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile,
    int ld,
    int key_start,
    int dim_start,
    int lane_id,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane_id & 7);
  const int col = dim_start + (((lane_id >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(tile + key * ld + col);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1])
               : "r"(addr));
}

// B for dS@K: contraction over 16 keys, n = 8 dims.
__device__ __forceinline__ void ldmatrix_b_keytrans(
    const __nv_bfloat16* tile,
    int ld,
    int key_start,
    int dim_start,
    int lane_id,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane_id & 15);
  const uint32_t addr = shared_u32(tile + key * ld + dim_start);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1])
               : "r"(addr));
}

// Round two FP32 values to BF16 and pack their bits into one register.
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l))
         | (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

// The +8 columns in each warp's Q tile are intentionally outside every
// ldmatrix operand.  Across four warps they form an exact 32x32 BF16 tile:
// warp w owns bias key columns [8*w, 8*w+8).  Staging the batch/head bias
// once per CTA removes four identical L1/TEX reads (one from each outer
// warp) without increasing dynamic shared memory or changing arithmetic.
template <int HeadDim>
__device__ __forceinline__ __nv_bfloat16* shared_bias_cell(
    __nv_bfloat16* shared_q,
    int query,
    int key) {
  constexpr int kQueryRowStride = HeadDim + 8;
  const int owner_warp = key >> 3;
  return shared_q + owner_warp * kQueryBlock * kQueryRowStride + query * kQueryRowStride + HeadDim
         + (key & 7);
}

// -----------------------------------------------------------------------------
// Split dQ/dBias: each warp owns one outer plane and 32 query rows
// -----------------------------------------------------------------------------

template <int HeadDim, int StaticSequence, bool FullQueryTiles, bool TaskInGridZ, bool OuterEqSeq>
__global__ __launch_bounds__(kThreads) void turbopairformer_dq_dbias_leaf_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ delta,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out_dq,
    __nv_bfloat16* __restrict__ workspace,
    int runtime_sequence,
    int outer_count,
    int heads,
    int batch_base,
    int active_batch,
    int batch_capacity,
    int query_base,
    int query_rows,
    int query_capacity,
    int leaf_base,
    int io_head_dim,
    float attention_scale) {
  // A. Map this block to one query tile, head, and four-outer leaf.
  // StaticSequence == -1 selects full tiles with runtime sequence length;
  // StaticSequence == 0 keeps key-tail guards for unpadded direct calls.
  // FullQueryTiles independently controls the active query-slab tail.
  const int sequence = StaticSequence > 0 ? StaticSequence : runtime_sequence;
  constexpr int outer_leaf = kOuterLeaf;
  constexpr int passes = kPasses;
  constexpr bool kSequenceExact = StaticSequence != 0;
  constexpr int kQueryRowStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
  const int local_batch = wave_task % active_batch;
  const int batch_index = batch_base + local_batch;
  const int wave_leaf = wave_task / active_batch;
  const int query_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
  const int query_local_start = query_block * kQueryBlock;
  const int query_start = query_base + query_local_start;
  const int query_end = query_base + query_rows;
  const int head = blockIdx.y;
  const int leaf = leaf_base + wave_leaf;
  const int lane_row = lane_id >> 2;
  const int lane_in_row_group = lane_id & 3;

  // Shared layout: Q | dO | K/V | per-warp FP32 dBias contributions.
  // Each warp has private Q/dO/K/V tiles. Q padding holds the common bias
  // tile; the final CTA barrier publishes the next K/V/bias tile together.
  extern __shared__ __align__(16) unsigned char shared_storage[];
  __nv_bfloat16* shared_q = reinterpret_cast<__nv_bfloat16*>(shared_storage);
  __nv_bfloat16* shared_d_output = shared_q + kWarpsPerBlock * kQueryBlock * kQueryRowStride;
  __nv_bfloat16* shared_kv = shared_d_output + kWarpsPerBlock * kQueryBlock * kQueryRowStride;
  float* shared_warp_bias_gradients =
      reinterpret_cast<float*>(shared_kv + kWarpsPerBlock * 2 * kKeyTile * kKVStride);

  __nv_bfloat16* shared_warp_q = shared_q + warp_id * kQueryBlock * kQueryRowStride;
  __nv_bfloat16* shared_warp_d_output = shared_d_output + warp_id * kQueryBlock * kQueryRowStride;
  __nv_bfloat16* shared_warp_k = shared_kv + warp_id * 2 * kKeyTile * kKVStride;
  __nv_bfloat16* shared_warp_v = shared_warp_k + kKeyTile * kKVStride;
  float* shared_warp_bias_gradient = shared_warp_bias_gradients + warp_id * kQueryBlock * kKeyTile;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) * heads + head)
           * static_cast<int64_t>(query_capacity)
       + query_local_start)
      * sequence;
  const int64_t bias_base =
      ((static_cast<int64_t>(batch_index) * heads + head) * sequence + query_start)
      * static_cast<int64_t>(sequence);

#pragma unroll 1
  for (int pass = 0; pass < passes; ++pass) {
    const int outer_extent = OuterEqSeq ? sequence : outer_count;
    const int outer = leaf * outer_leaf + pass * kWarpsPerBlock + warp_id;
    // Outer is a pure plane count (decoupled from sequence); its tail guard
    // must stay a runtime compare in every lane -- values are unchanged when
    // outer_count is a leaf multiple, and arbitrary outer stays in bounds.
    const bool outer_valid = outer < outer_extent;
    // Multiplier form keeps the hot-loop condition compile-foldable on the
    // exact lanes: valid outers multiply by 1.0f (bitwise identity), tail
    // outers zero their (finite, clamped-load) contributions exactly.
    const float outer_valid_multiplier = outer_valid ? 1.0F : 0.0F;
    const int outer_load = outer_valid ? outer : outer_extent - 1;
    const int64_t plane = static_cast<int64_t>(batch_index) * outer_extent + outer_load;
    const int64_t row_stats_base = (plane * heads + head) * static_cast<int64_t>(sequence);
    const int64_t tensor_base = row_stats_base * io_head_dim;
    // d_output arrives in the public [B, O, N, H, io_head_dim] layout.
    const int64_t dout_plane_base = plane * static_cast<int64_t>(sequence) * heads * io_head_dim
                                    + static_cast<int64_t>(head) * io_head_dim;
    const int64_t mask_base = plane * sequence;

    // B. Stage this outer's Q/dO and retain row statistics and dQ state.
    // Q/dO stay in shared memory across the key loop; dQ stays in registers.
    for (int element = lane_id; element < kQueryBlock * HeadDim; element += kWarpSize) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int global_query =
          FullQueryTiles || query_start + row < query_end ? query_start + row : sequence - 1;
      const int64_t src = tensor_base + static_cast<int64_t>(global_query) * io_head_dim + col;
      const int64_t dout_src =
          dout_plane_base + static_cast<int64_t>(global_query) * heads * io_head_dim + col;
      shared_warp_q[row * kQueryRowStride + col] =
          col < io_head_dim ? q[src] : __float2bfloat16_rn(0.0F);
      shared_warp_d_output[row * kQueryRowStride + col] =
          col < io_head_dim ? d_output[dout_src] : __float2bfloat16_rn(0.0F);
    }
    __syncwarp();

    float row_logsumexp_pairs[kQuerySubTiles][2];
    float row_delta_pairs[kQuerySubTiles][2];
#pragma unroll
    for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int query_row = query_start + sub * kQueryTile + lane_row + half * 8;
        const int64_t stats =
            row_stats_base + (FullQueryTiles || query_row < query_end ? query_row : sequence - 1);
        row_logsumexp_pairs[sub][half] = logsumexp[stats];
        row_delta_pairs[sub][half] = delta[stats];
      }
    }

    Accumulator dq_accumulators[kQuerySubTiles][HeadDim / 8];
#pragma unroll
    for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        dq_accumulators[sub][tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
    }

    // C. Stage the first K/V/bias tile; later calls reuse the same buffers.
    const auto stage_kv_and_bias = [&](int key_start) {
      // K/V are outer-dependent, so each warp retains its private tile.
      for (int element = lane_id; element < kKeyTile * HeadDim; element += kWarpSize) {
        const int row = element / HeadDim;
        const int col = element - row * HeadDim;
        const int global_key =
            kSequenceExact || key_start + row < sequence ? key_start + row : sequence - 1;
        const int64_t src = tensor_base + static_cast<int64_t>(global_key) * io_head_dim + col;
        shared_warp_k[row * kKVStride + col] =
            col < io_head_dim ? k[src] : __float2bfloat16_rn(0.0F);
        shared_warp_v[row * kKVStride + col] =
            col < io_head_dim ? v[src] : __float2bfloat16_rn(0.0F);
      }
      // Bias is shared by all four outer warps.  The full CTA cooperatively
      // stages it into otherwise-unused Q padding.
      for (int element = threadIdx.x; element < kQueryBlock * kKeyTile; element += kThreads) {
        const int row = element / kKeyTile;
        const int key = element - row * kKeyTile;
        const bool valid = (FullQueryTiles || query_start + row < query_end)
                           && (kSequenceExact || key_start + key < sequence);
        *shared_bias_cell<HeadDim>(shared_q, row, key) =
            valid ? bias[bias_base + static_cast<int64_t>(row) * sequence + key_start + key]
                  : __float2bfloat16_rn(0.0F);
      }
    };

    stage_kv_and_bias(0);
    __syncthreads();

#pragma unroll 1
    for (int key_start = 0; key_start < sequence; key_start += kKeyTile) {
#pragma unroll
      for (int sub = 0; sub < kQuerySubTiles; ++sub) {
        uint32_t q_fragments[HeadDim / 16][4];
        uint32_t d_output_fragments[HeadDim / 16][4];
#pragma unroll
        for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
          ldmatrix_a16x16(
              shared_warp_q + sub * kQueryTile * kQueryRowStride,
              kQueryRowStride,
              channel_chunk * 16,
              lane_id,
              q_fragments[channel_chunk]);
          ldmatrix_a16x16(
              shared_warp_d_output + sub * kQueryTile * kQueryRowStride,
              kQueryRowStride,
              channel_chunk * 16,
              lane_id,
              d_output_fragments[channel_chunk]);
        }

        // D. Compute QK^T and dP = dO V^T for these 16 query rows.
        Accumulator score_fragments[kKeyTile / 8];
        Accumulator dp_fragments[kKeyTile / 8];
#pragma unroll
        for (int column = 0; column < kKeyTile / 8; ++column) {
          score_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
          dp_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
          for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
            uint32_t b_fragment[2];
            ldmatrix_b_dimpair(
                shared_warp_k,
                kKVStride,
                column * 8,
                channel_chunk * 16,
                lane_id,
                b_fragment);
            mma_m16n8k16(score_fragments[column], q_fragments[channel_chunk], b_fragment);
            ldmatrix_b_dimpair(
                shared_warp_v,
                kKVStride,
                column * 8,
                channel_chunk * 16,
                lane_id,
                b_fragment);
            mma_m16n8k16(dp_fragments[column], d_output_fragments[channel_chunk], b_fragment);
          }
        }

#pragma unroll
        for (int column = 0; column < kKeyTile / 8; ++column) {
          // Reconstruct P from the saved row logsumexp, then form dS in FP32.
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const float row_logsumexp = row_logsumexp_pairs[sub][half];
            const float row_delta = row_delta_pairs[sub][half];
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              const int key = key_start + column * 8 + lane_in_row_group * 2 + pair;
              const int query_row = query_start + sub * kQueryTile + lane_row + half * 8;
              const int index = half * 2 + pair;
              float score_gradient = 0.0F;
              if ((kSequenceExact || key < sequence) && (FullQueryTiles || query_row < query_end)) {
                const float value = score_fragments[column].values[index] * attention_scale
                                    + mask[mask_base + key]
                                    + __bfloat162float(*shared_bias_cell<HeadDim>(
                                        shared_q,
                                        sub * kQueryTile + lane_row + half * 8,
                                        column * 8 + lane_in_row_group * 2 + pair));
                const float probability = row_logsumexp == -std::numeric_limits<float>::infinity()
                                              ? 0.0F
                                              : __expf(value - row_logsumexp);
                score_gradient = probability * (dp_fragments[column].values[index] - row_delta);
              }
              if constexpr (!(OuterEqSeq && kSequenceExact)) {
                // Exact-tiles + outer==sequence lane covers every outer:
                // the tail multiplier is compile-time dead there.
                score_gradient *= outer_valid_multiplier;
              }
              // Reuse the score registers for dS; save FP32 dS for dBias.
              score_fragments[column].values[index] = score_gradient;
              shared_warp_bias_gradient
                  [(sub * kQueryTile + lane_row + half * 8) * kKeyTile + column * 8
                   + lane_in_row_group * 2 + pair] = score_gradient;
            }
          }
        }

#pragma unroll
        for (int chunk = 0; chunk < kKeyTile / 16; ++chunk) {
          // E. Round dS to BF16 for dQ += dS K; keep the dQ sum in FP32.
          uint32_t score_gradient_fragment[4];
          score_gradient_fragment[0] = pack_bf16_pair(
              score_fragments[chunk * 2].values[0],
              score_fragments[chunk * 2].values[1]);
          score_gradient_fragment[1] = pack_bf16_pair(
              score_fragments[chunk * 2].values[2],
              score_fragments[chunk * 2].values[3]);
          score_gradient_fragment[2] = pack_bf16_pair(
              score_fragments[chunk * 2 + 1].values[0],
              score_fragments[chunk * 2 + 1].values[1]);
          score_gradient_fragment[3] = pack_bf16_pair(
              score_fragments[chunk * 2 + 1].values[2],
              score_fragments[chunk * 2 + 1].values[3]);
#pragma unroll
          for (int tile = 0; tile < HeadDim / 8; ++tile) {
            uint32_t key_fragment[2];
            ldmatrix_b_keytrans(
                shared_warp_k,
                kKVStride,
                chunk * 16,
                tile * 8,
                lane_id,
                key_fragment);
            mma_m16n8k16(dq_accumulators[sub][tile], score_gradient_fragment, key_fragment);
          }
        }
      }

      // F. Publish all four dBias tiles and combine (o0+o1)+(o2+o3).
      // Each workspace cell has one writer; its leaf sum is stored as BF16.
      __syncthreads();
      const int64_t plane_tile_base = plane_base + key_start;
      for (int index = threadIdx.x; index < kQueryBlock * kKeyTile; index += kThreads) {
        const int row = index / kKeyTile;
        const int key = index - row * kKeyTile;
        const float left = __fadd_rn(
            shared_warp_bias_gradients[row * kKeyTile + key],
            shared_warp_bias_gradients[kQueryBlock * kKeyTile + row * kKeyTile + key]);
        const float right = __fadd_rn(
            shared_warp_bias_gradients[2 * kQueryBlock * kKeyTile + row * kKeyTile + key],
            shared_warp_bias_gradients[3 * kQueryBlock * kKeyTile + row * kKeyTile + key]);
        const float tile_sum = __fadd_rn(left, right);
        if ((!FullQueryTiles && query_start + row >= query_end)
            || (!kSequenceExact && key_start + key >= sequence)) {
          continue;
        }
        __nv_bfloat16* destination =
            workspace + plane_tile_base + static_cast<int64_t>(row) * sequence + key;
        if (pass == 0) {
          *destination = __float2bfloat16_rn(tile_sum);
        } else {
          *destination = __float2bfloat16_rn(__fadd_rn(__bfloat162float(*destination), tile_sum));
        }
      }
      if (key_start + kKeyTile < sequence) {
        // The first CTA barrier above proves every warp has finished reading
        // the current K/V and bias tiles.  Prefetch the next operands while
        // the same threads finish their disjoint workspace writes; the
        // existing trailing barrier publishes both, so steady state gains no
        // additional CTA barrier.
        stage_kv_and_bias(key_start + kKeyTile);
      }
      __syncthreads();
    }

    // G. All keys are consumed. Scale this outer's dQ once and store BF16.
    const int64_t dq_row_base = tensor_base + static_cast<int64_t>(query_start) * io_head_dim;
    if (outer_valid) {
#pragma unroll
      for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = sub * kQueryTile + lane_row + half * 8;
            if (!FullQueryTiles && query_start + row >= query_end) {
              continue;
            }
            const int column = tile * 8 + lane_in_row_group * 2;
            if (column >= io_head_dim) {
              continue;
            }
            out_dq[dq_row_base + static_cast<int64_t>(row) * io_head_dim + column] =
                __float2bfloat16_rn(dq_accumulators[sub][tile].values[half * 2] * attention_scale);
            out_dq[dq_row_base + static_cast<int64_t>(row) * io_head_dim + column + 1] =
                __float2bfloat16_rn(
                    dq_accumulators[sub][tile].values[half * 2 + 1] * attention_scale);
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Split dQ/dBias: two warpgroups share a 64-query tile across four outer planes
// -----------------------------------------------------------------------------
// Four warps cooperate on each outer's QK^T and dO V^T using WGMMA.
// Warpgroup 0 processes outer slots {0,1}; warpgroup 1 processes {2,3}.
// Their FP32 dBias pair sums are combined as (o0+o1)+(o2+o3), matching the
// per-warp split leaf's reduction tree and BF16 workspace rounding point.
// This path requires full 64-row query tiles and HeadDim <= 64. Runtime
// outer tails are still handled independently from the sequence length.
constexpr int kQueryBlockWg = 64;
// kThreads is one four-warp group; this block contains two such groups.
constexpr int kThreadsWg = 2 * kThreads;
constexpr int kSlotsPerWarpgroup = 2;
// Padded shared strides: the FP32 tree at a raw kKeyTile stride wraps the
// 32 banks exactly (128B) so every row hits one bank; +1 float staggers
// rows across banks.  The BF16 bias slab gets +4 halves for the same
// reason (64B rows alias two bank halves).
constexpr int kTreeStride = kKeyTile + 1;
constexpr int kBiasStride = kKeyTile + 4;

// Map a logical BF16 column to its position in a compact swizzled tile.
// The 16-byte atom XOR advances once per 128 bytes of row-major tile offset.
// Staging stores, WGMMA descriptors, and ldmatrix reads must use this layout.
template <int HeadDim>
__device__ __forceinline__ int wg_swz_col(int row, int col) {
  constexpr int kAtoms = HeadDim / 8;
  const int atom = (col >> 3) ^ (((row * HeadDim) >> 6) & (kAtoms - 1));
  return atom * 8 + (col & 7);
}

// B for dS@K over a compact swizzled K tile: contraction over 16 keys,
// n = 8 dims (the legacy keytrans read, swizzle-aware).
template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_keytrans_sw(
    const __nv_bfloat16* tile,
    int key_start,
    int dim_start,
    int lane_id,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane_id & 15);
  const uint32_t addr = shared_u32(tile + key * HeadDim + wg_swz_col<HeadDim>(key, dim_start));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1])
               : "r"(addr));
}

// One warpgroup computes a 64x32 result over 16 head channels. Both operands
// are shared-memory descriptors. Each thread owns four m16n8 accumulator
// fragments; their row/column placement matches the mma.sync layout above.
__device__ __forceinline__ void wg_ss_wgmma_m64n32k16(
    Accumulator (&o)[4],
    uint64_t descriptor_a,
    uint64_t descriptor_b,
    int accumulate) {
  asm volatile("{\n"
               "  .reg .pred use_acc;\n"
               "  setp.ne.b32 use_acc, %18, 0;\n"
               "  wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
               "  %16, %17, use_acc, 1, 1, 0, 0;\n"
               "}\n"
               : "+f"(o[0].values[0]),
                 "+f"(o[0].values[1]),
                 "+f"(o[0].values[2]),
                 "+f"(o[0].values[3]),
                 "+f"(o[1].values[0]),
                 "+f"(o[1].values[1]),
                 "+f"(o[1].values[2]),
                 "+f"(o[1].values[3]),
                 "+f"(o[2].values[0]),
                 "+f"(o[2].values[1]),
                 "+f"(o[2].values[2]),
                 "+f"(o[2].values[3]),
                 "+f"(o[3].values[0]),
                 "+f"(o[3].values[1]),
                 "+f"(o[3].values[2]),
                 "+f"(o[3].values[3])
               : "l"(descriptor_a), "l"(descriptor_b), "r"(accumulate));
}

// Publish shared-memory stores to the async proxy before WGMMA reads them.
// This visibility fence and the CTA barrier serve different purposes.
__device__ __forceinline__ void wg_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wg_wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wg_wgmma_wait_zero() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// Historical tuning recorded <=128 registers with this two-block bound,
// versus 164 registers and one CTA/SM without it (main 98eff406). Those counts
// describe that measurement, not every build. Keep the bound with 256 threads.
template <int HeadDim, bool TaskInGridZ, bool OuterEqSeq>
__global__ __launch_bounds__(kThreadsWg, 2) void turbopairformer_dq_dbias_wg_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ delta,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out_dq,
    __nv_bfloat16* __restrict__ workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch_base,
    int active_batch,
    int batch_capacity,
    int query_base,
    int query_capacity,
    int leaf_base,
    int io_head_dim,
    float attention_scale) {
  // A. Assign the query tile and outer leaf, then identify this warpgroup's
  // two outer slots and this thread's two rows within the warp's 16-row tile.
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warpgroup_id = warp_id / kWarpsPerBlock;
  const int warp_in_group = warp_id & (kWarpsPerBlock - 1);
  const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
  const int local_batch = wave_task % active_batch;
  const int batch_index = batch_base + local_batch;
  const int wave_leaf = wave_task / active_batch;
  const int query_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
  const int query_local_start = query_block * kQueryBlockWg;
  const int query_start = query_base + query_local_start;
  const int head = blockIdx.y;
  const int leaf = leaf_base + wave_leaf;
  const int lane_row = lane_id >> 2;
  const int lane_in_row_group = lane_id & 3;

  // Shared layout: all slots' Q | dO | K/V | common bias | masks | pair sums.
  // Q/dO remain resident across the key loop; K/V/bias/masks are reused for
  // each key tile. Align the swizzled tiles for their WGMMA descriptors.
  extern __shared__ __align__(1024) unsigned char shared_storage[];
  __nv_bfloat16* shared_q = reinterpret_cast<__nv_bfloat16*>(shared_storage);
  __nv_bfloat16* shared_d_output = shared_q + kOuterLeaf * kQueryBlockWg * HeadDim;
  __nv_bfloat16* shared_kv = shared_d_output + kOuterLeaf * kQueryBlockWg * HeadDim;
  __nv_bfloat16* shared_bias = shared_kv + kOuterLeaf * 2 * kKeyTile * HeadDim;
  float* shared_mask_tiles = reinterpret_cast<float*>(shared_bias + kQueryBlockWg * kBiasStride);
  float* shared_bias_pair_sums = shared_mask_tiles + kOuterLeaf * kKeyTile;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) * heads + head)
           * static_cast<int64_t>(query_capacity)
       + query_local_start)
      * sequence;
  const int64_t bias_base =
      ((static_cast<int64_t>(batch_index) * heads + head) * sequence + query_start)
      * static_cast<int64_t>(sequence);

  // Recompute global bases at staging sites instead of retaining four
  // arrays of addresses in registers throughout the key loop.
  const int outer_extent = OuterEqSeq ? sequence : outer_count;
  const auto slot_plane = [&](int slot) -> int64_t {
    const int outer = leaf * kOuterLeaf + slot;
    const int outer_load = outer < outer_extent ? outer : outer_extent - 1;
    return static_cast<int64_t>(batch_index) * outer_extent + outer_load;
  };
  float outer_valid_multiplier[kOuterLeaf];
#pragma unroll
  for (int slot = 0; slot < kOuterLeaf; ++slot) {
    outer_valid_multiplier[slot] = leaf * kOuterLeaf + slot < outer_extent ? 1.0F : 0.0F;
  }

  // B. Stage Q/dO for all four outers once; keep dQ and row statistics live
  // across the key loop. All block threads cooperate on the shared stores.
#pragma unroll 1
  for (int slot = 0; slot < kOuterLeaf; ++slot) {
    __nv_bfloat16* dst_q = shared_q + slot * kQueryBlockWg * HeadDim;
    __nv_bfloat16* dst_do = shared_d_output + slot * kQueryBlockWg * HeadDim;
    const int64_t plane = slot_plane(slot);
    const int64_t tensor_base =
        (plane * heads + head) * static_cast<int64_t>(sequence) * io_head_dim;
    const int64_t dout_base = plane * static_cast<int64_t>(sequence) * heads * io_head_dim
                              + static_cast<int64_t>(head) * io_head_dim;
    for (int element = threadIdx.x; element < kQueryBlockWg * HeadDim; element += kThreadsWg) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int global_query = query_start + row;
      const int64_t src = tensor_base + static_cast<int64_t>(global_query) * io_head_dim + col;
      const int64_t dout_src =
          dout_base + static_cast<int64_t>(global_query) * heads * io_head_dim + col;
      const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
      dst_q[dst_index] = col < io_head_dim ? q[src] : __float2bfloat16_rn(0.0F);
      dst_do[dst_index] = col < io_head_dim ? d_output[dout_src] : __float2bfloat16_rn(0.0F);
    }
  }

  // Each warpgroup carries only its own slot pair's row statistics and dQ
  // accumulators; the local index stays a compile-time constant so the
  // state lives in registers.
  float slot_logsumexp[kSlotsPerWarpgroup][2];
  float slot_delta[kSlotsPerWarpgroup][2];
#pragma unroll
  for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
    const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
    const int64_t row_stats_base =
        (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int query_row = query_start + warp_in_group * kQueryTile + lane_row + half * 8;
      const int64_t stats = row_stats_base + query_row;
      slot_logsumexp[local_slot][half] = logsumexp[stats];
      slot_delta[local_slot][half] = delta[stats];
    }
  }

  Accumulator dq_accumulators[kSlotsPerWarpgroup][HeadDim / 8];
#pragma unroll
  for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
      dq_accumulators[local_slot][tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    }
  }

  // C. Each iteration replaces K/V, the common bias tile, and per-slot masks.
  const auto stage_kv_and_bias = [&](int key_start) {
#pragma unroll 1
    for (int slot = 0; slot < kOuterLeaf; ++slot) {
      __nv_bfloat16* dst_k = shared_kv + slot * 2 * kKeyTile * HeadDim;
      __nv_bfloat16* dst_v = dst_k + kKeyTile * HeadDim;
      const int64_t tensor_base =
          (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence) * io_head_dim;
      for (int element = threadIdx.x; element < kKeyTile * HeadDim; element += kThreadsWg) {
        const int row = element / HeadDim;
        const int col = element - row * HeadDim;
        const int global_key = key_start + row;
        const int64_t src = tensor_base + static_cast<int64_t>(global_key) * io_head_dim + col;
        const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
        dst_k[dst_index] = col < io_head_dim ? k[src] : __float2bfloat16_rn(0.0F);
        dst_v[dst_index] = col < io_head_dim ? v[src] : __float2bfloat16_rn(0.0F);
      }
    }
    for (int element = threadIdx.x; element < kQueryBlockWg * kKeyTile; element += kThreadsWg) {
      const int row = element / kKeyTile;
      const int key = element - row * kKeyTile;
      shared_bias[row * kBiasStride + key] =
          bias[bias_base + static_cast<int64_t>(row) * sequence + key_start + key];
    }
    // Additive mask depends only on (outer plane, key): one FP32 row per
    // slot replaces sixteen per-thread global reads per slot-tile.
    for (int element = threadIdx.x; element < kOuterLeaf * kKeyTile; element += kThreadsWg) {
      const int slot = element / kKeyTile;
      const int key = element - slot * kKeyTile;
      shared_mask_tiles[element] = mask[slot_plane(slot) * sequence + key_start + key];
    }
  };

  stage_kv_and_bias(0);

#pragma unroll 1
  for (int key_start = 0; key_start < sequence; key_start += kKeyTile) {
    // Publishes the tile staged behind the previous combine (or the
    // prologue stage) and proves every warp left the prior tile.  The
    // proxy fence orders this thread's generic-proxy staging writes
    // before the warpgroup's async-proxy descriptor reads.
    wg_proxy_fence();
    __syncthreads();
    // Unrolling keeps the per-slot register-array indices compile-time
    // constants. Each warpgroup visits its own two slots in order.
#pragma unroll
    for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
      const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
      const __nv_bfloat16* shared_q_tile = shared_q + slot * kQueryBlockWg * HeadDim;
      const __nv_bfloat16* shared_d_output_tile = shared_d_output + slot * kQueryBlockWg * HeadDim;
      const __nv_bfloat16* shared_k_tile = shared_kv + slot * 2 * kKeyTile * HeadDim;
      const __nv_bfloat16* shared_v_tile = shared_k_tile + kKeyTile * HeadDim;

      // D. Compute S = QK^T and dP = dO V^T over successive 16-channel chunks.
      // A has 64 query rows; B has 32 key rows. Each warp receives 16 query
      // rows in the same accumulator layout used by the per-warp split leaf.
      constexpr uint64_t kSsMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
      constexpr uint64_t kSsLbo = 64;
      constexpr uint64_t kSsSbo = HeadDim;
      Accumulator score_fragments[kKeyTile / 8];
      Accumulator dp_fragments[kKeyTile / 8];
#pragma unroll
      for (int column = 0; column < kKeyTile / 8; ++column) {
        score_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
      wg_wgmma_fence();
#pragma unroll
      for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        const uint64_t channel_byte_offset = static_cast<uint64_t>(channel_chunk) * 32;
        const uint64_t q_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_q_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t k_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_k_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(score_fragments, q_descriptor, k_descriptor, channel_chunk != 0);
      }
#pragma unroll
      for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        const uint64_t channel_byte_offset = static_cast<uint64_t>(channel_chunk) * 32;
        const uint64_t d_output_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_d_output_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t v_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_v_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(dp_fragments, d_output_descriptor, v_descriptor, channel_chunk != 0);
      }
      wg_wgmma_commit();
      wg_wgmma_wait_zero();

      // E. Reconstruct P and dS in FP32 after WGMMA has completed.
      // Reuse score_fragments for dS and fold this outer into its pair sum.
      float* shared_bias_pair_sum =
          shared_bias_pair_sums + warpgroup_id * (kQueryBlockWg * kTreeStride);
      const float slot_valid_multiplier = outer_valid_multiplier[slot];
      const float* shared_slot_mask = shared_mask_tiles + slot * kKeyTile;
#pragma unroll
      for (int column = 0; column < kKeyTile / 8; ++column) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float row_logsumexp = slot_logsumexp[local_slot][half];
          const float row_delta = slot_delta[local_slot][half];
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const int tile_key = column * 8 + lane_in_row_group * 2 + pair;
            const int block_row = warp_in_group * kQueryTile + lane_row + half * 8;
            const int index = half * 2 + pair;
            const float value = score_fragments[column].values[index] * attention_scale
                                + shared_slot_mask[tile_key]
                                + __bfloat162float(shared_bias[block_row * kBiasStride + tile_key]);
            const float probability = row_logsumexp == -std::numeric_limits<float>::infinity()
                                          ? 0.0F
                                          : __expf(value - row_logsumexp);
            float score_gradient = probability * (dp_fragments[column].values[index] - row_delta);
            if constexpr (!OuterEqSeq) {
              score_gradient *= slot_valid_multiplier;
            }
            score_fragments[column].values[index] = score_gradient;
            const int cell = block_row * kTreeStride + tile_key;
            if (local_slot == 0) {
              shared_bias_pair_sum[cell] = score_gradient;
            } else {
              shared_bias_pair_sum[cell] = __fadd_rn(shared_bias_pair_sum[cell], score_gradient);
            }
          }
        }
      }

#pragma unroll
      for (int chunk = 0; chunk < kKeyTile / 16; ++chunk) {
        // F. dQ += BF16(dS) K; dQ remains FP32 across all key tiles.
        uint32_t score_gradient_fragment[4];
        score_gradient_fragment[0] = pack_bf16_pair(
            score_fragments[chunk * 2].values[0],
            score_fragments[chunk * 2].values[1]);
        score_gradient_fragment[1] = pack_bf16_pair(
            score_fragments[chunk * 2].values[2],
            score_fragments[chunk * 2].values[3]);
        score_gradient_fragment[2] = pack_bf16_pair(
            score_fragments[chunk * 2 + 1].values[0],
            score_fragments[chunk * 2 + 1].values[1]);
        score_gradient_fragment[3] = pack_bf16_pair(
            score_fragments[chunk * 2 + 1].values[2],
            score_fragments[chunk * 2 + 1].values[3]);
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          uint32_t key_fragment[2];
          ldmatrix_b_keytrans_sw<HeadDim>(
              shared_k_tile,
              chunk * 16,
              tile * 8,
              lane_id,
              key_fragment);
          mma_m16n8k16(dq_accumulators[local_slot][tile], score_gradient_fragment, key_fragment);
        }
      }
    }

    // G. Publish the two pair sums, combine them, and write the BF16 leaf
    // plane. This barrier also finishes reads of K/V/bias before reuse.
    __syncthreads();
    const int64_t plane_tile_base = plane_base + key_start;
    for (int index = threadIdx.x; index < kQueryBlockWg * kKeyTile; index += kThreadsWg) {
      const int row = index / kKeyTile;
      const int key = index - row * kKeyTile;
      const int cell = row * kTreeStride + key;
      const float tile_sum = __fadd_rn(
          shared_bias_pair_sums[cell],
          shared_bias_pair_sums[kQueryBlockWg * kTreeStride + cell]);
      workspace[plane_tile_base + static_cast<int64_t>(row) * sequence + key] =
          __float2bfloat16_rn(tile_sum);
    }
    if (key_start + kKeyTile < sequence) {
      stage_kv_and_bias(key_start + kKeyTile);
    }
  }

  // H. All keys are consumed. Scale and store dQ for this warp's query rows
  // in each of its warpgroup's two outer slots.
#pragma unroll
  for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
    const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
    if (outer_valid_multiplier[slot] == 0.0F) {
      continue;
    }
    const int64_t dq_row_base =
        (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence) * io_head_dim
        + static_cast<int64_t>(query_start) * io_head_dim;
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp_in_group * kQueryTile + lane_row + half * 8;
        const int column = tile * 8 + lane_in_row_group * 2;
        if (column >= io_head_dim) {
          continue;
        }
        out_dq[dq_row_base + static_cast<int64_t>(row) * io_head_dim + column] =
            __float2bfloat16_rn(
                dq_accumulators[local_slot][tile].values[half * 2] * attention_scale);
        out_dq[dq_row_base + static_cast<int64_t>(row) * io_head_dim + column + 1] =
            __float2bfloat16_rn(
                dq_accumulators[local_slot][tile].values[half * 2 + 1] * attention_scale);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Single-reconstruction backward: keep K/V resident and visit query tiles
// -----------------------------------------------------------------------------
// One block owns 64 keys, one head, and four outer slots. Two warpgroups
// process slot pairs {0,1} and {2,3}, sweeping a query slab in 32-row tiles.
// Each reconstruction produces S^T, P^T, dP^T, and dS^T in [key, query] order:
//   - dK/dV accumulate in registers over queries. FP32 continuation buffers
//     preserve these sums between query slabs; only the final slab stores BF16.
//   - dBias uses FP32 outer-pair sums and the same BF16 leaf workspace as split.
//   - dQ needs [query, key] dS. Shared memory transposes the BF16 operand;
//     this block then writes one BF16 dQ partial for its 64 keys. A separate
//     reducer folds key-block partials in ascending order and scales dQ once.
constexpr int kSrKeyBlock = 64;
constexpr int kSrQueryTile = 32;
constexpr int kSrTreeStride = kSrKeyBlock + 1;
constexpr int kSrBiasStride = kSrKeyBlock + 8;
constexpr int kSrDsStride = kSrKeyBlock + 8;

__device__ __forceinline__ void sr_cp_async_16(__nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)), "l"(src));
}

__device__ __forceinline__ void sr_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void sr_cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Load the query-row contraction operand for dK = dS^T Q and dV = P^T dO.
template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_rowtrans_sw_sr(
    const __nv_bfloat16* tile,
    int row_start,
    int dim_start,
    int lane_id,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane_id & 15);
  const uint32_t addr = shared_u32(tile + row * HeadDim + wg_swz_col<HeadDim>(row, dim_start));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1])
               : "r"(addr));
}

template <int HeadDim, bool TaskInGridZ, bool OuterEqSeq>
__global__ __launch_bounds__(kThreadsWg, 2) void turbopairformer_sr_leaf_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ d_output,
    const float* __restrict__ logsumexp,
    const float* __restrict__ delta,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out_dk,
    __nv_bfloat16* __restrict__ out_dv,
    float* __restrict__ dk_continuation,
    float* __restrict__ dv_continuation,
    __nv_bfloat16* __restrict__ dq_partials,
    __nv_bfloat16* __restrict__ workspace,
    int sequence,
    int outer_count,
    int total_planes,
    int heads,
    int batch_base,
    int active_batch,
    int batch_capacity,
    int query_base,
    int query_rows,
    int query_capacity,
    int leaf_base,
    int io_head_dim,
    float attention_scale) {
  // A. Assign the key block, head, and outer leaf. Here each warp owns
  // 16 key rows, unlike the query-row ownership in the split kernels.
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warpgroup_id = warp_id / kWarpsPerBlock;
  const int warp_in_group = warp_id & (kWarpsPerBlock - 1);
  const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
  const int local_batch = wave_task % active_batch;
  const int batch_index = batch_base + local_batch;
  const int wave_leaf = wave_task / active_batch;
  const int key_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
  const int key_start = key_block * kSrKeyBlock;
  const int head = blockIdx.y;
  const int leaf = leaf_base + wave_leaf;
  const int lane_row = lane_id >> 2;
  const int lane_in_row_group = lane_id & 3;
  const int warp_key_start = warp_in_group * kQueryTile;

  // Shared layout: resident K | V | double-buffered Q | dO | dS | bias
  //                | resident masks | two FP32 dBias pair sums.
  // Q/dO/bias alternate buffers so the next query tile can load during math.
  // dS is written in [query, key] order for the later dQ multiplication.
  extern __shared__ __align__(1024) unsigned char shared_storage[];
  __nv_bfloat16* shared_k = reinterpret_cast<__nv_bfloat16*>(shared_storage);
  __nv_bfloat16* shared_v = shared_k + kOuterLeaf * kSrKeyBlock * HeadDim;
  __nv_bfloat16* shared_q = shared_v + kOuterLeaf * kSrKeyBlock * HeadDim;
  __nv_bfloat16* shared_d_output = shared_q + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
  __nv_bfloat16* shared_score_gradients = shared_d_output + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
  __nv_bfloat16* shared_bias = shared_score_gradients + kOuterLeaf * kSrQueryTile * kSrDsStride;
  float* shared_mask = reinterpret_cast<float*>(shared_bias + 2 * kSrQueryTile * kSrBiasStride);
  float* shared_bias_pair_sums = shared_mask + kOuterLeaf * kSrKeyBlock;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) * heads + head)
       * static_cast<int64_t>(query_capacity))
      * sequence;
  const int64_t bias_base = ((static_cast<int64_t>(batch_index) * heads + head) * sequence)
                            * static_cast<int64_t>(sequence);
  const bool single_outer = !OuterEqSeq && outer_count == 1;

  const int outer_extent = OuterEqSeq ? sequence : outer_count;
  const auto slot_plane = [&](int slot) -> int64_t {
    const int outer = leaf * kOuterLeaf + slot;
    const int outer_load = outer < outer_extent ? outer : outer_extent - 1;
    return static_cast<int64_t>(batch_index) * outer_extent + outer_load;
  };
  const auto slot_is_active = [&](int slot) {
    if constexpr (OuterEqSeq) {
      // The SR launch requires N % 64 == 0, so O == N has full leaves.
      return true;
    } else {
      return leaf * kOuterLeaf + slot < outer_extent;
    }
  };

  // B. Stage this block's K/V and masks once, retaining them across queries.
  // The single-outer case stages only slot 0; the other slots are not read.
#pragma unroll 1
  const int staged_slots = single_outer ? 1 : kOuterLeaf;
  for (int slot = 0; slot < staged_slots; ++slot) {
    __nv_bfloat16* dst_k = shared_k + slot * kSrKeyBlock * HeadDim;
    __nv_bfloat16* dst_v = shared_v + slot * kSrKeyBlock * HeadDim;
    const int64_t tensor_base =
        (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence) * io_head_dim;
    for (int element = threadIdx.x; element < kSrKeyBlock * HeadDim; element += kThreadsWg) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int64_t src = tensor_base + static_cast<int64_t>(key_start + row) * io_head_dim + col;
      const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
      dst_k[dst_index] = col < io_head_dim ? k[src] : __float2bfloat16_rn(0.0F);
      dst_v[dst_index] = col < io_head_dim ? v[src] : __float2bfloat16_rn(0.0F);
    }
  }
  for (int element = threadIdx.x; element < staged_slots * kSrKeyBlock; element += kThreadsWg) {
    const int slot = element / kSrKeyBlock;
    const int key = element - slot * kSrKeyBlock;
    shared_mask[element] = mask[slot_plane(slot) * sequence + key_start + key];
  }

  // C. Start dK/dV at zero, or resume the raw FP32 sums from an earlier slab.
  // Their register values persist across every query tile in this launch.
  Accumulator dk_accumulators[kSlotsPerWarpgroup][HeadDim / 8];
  Accumulator dv_accumulators[kSlotsPerWarpgroup][HeadDim / 8];
#pragma unroll
  for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
      dk_accumulators[local_slot][tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      dv_accumulators[local_slot][tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    }
  }
  if (query_base != 0) {
#pragma unroll
    for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
      const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
      if (!slot_is_active(slot)) {
        continue;
      }
      const int64_t out_base =
          (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence) * io_head_dim
          + static_cast<int64_t>(key_start + warp_key_start) * io_head_dim;
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = lane_row + half * 8;
          const int column = tile * 8 + lane_in_row_group * 2;
          if (column >= io_head_dim) {
            continue;
          }
          const int64_t index = out_base + static_cast<int64_t>(row) * io_head_dim + column;
          const float2 dv_pair = *reinterpret_cast<const float2*>(dv_continuation + index);
          const float2 dk_pair = *reinterpret_cast<const float2*>(dk_continuation + index);
          dv_accumulators[local_slot][tile].values[half * 2] = dv_pair.x;
          dv_accumulators[local_slot][tile].values[half * 2 + 1] = dv_pair.y;
          dk_accumulators[local_slot][tile].values[half * 2] = dk_pair.x;
          dk_accumulators[local_slot][tile].values[half * 2 + 1] = dk_pair.y;
        }
      }
    }
  }

  // D. Stage Q/dO and bias with 16-byte copies into the selected buffer.
  // Commit groups are per thread; the loop waits and then publishes to WGMMA.
  constexpr int kSrChunks = HeadDim / 8;
  const auto stage_q_d_output_and_bias = [&](int query_start, int parity) {
    __nv_bfloat16* staging_q_buffer = shared_q + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    __nv_bfloat16* staging_d_output_buffer =
        shared_d_output + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    __nv_bfloat16* staging_bias_buffer = shared_bias + parity * kSrQueryTile * kSrBiasStride;
#pragma unroll 1
    for (int slot = 0; slot < staged_slots; ++slot) {
      __nv_bfloat16* dst_q = staging_q_buffer + slot * kSrQueryTile * HeadDim;
      __nv_bfloat16* dst_do = staging_d_output_buffer + slot * kSrQueryTile * HeadDim;
      const int64_t plane = slot_plane(slot);
      const int64_t tensor_base =
          (plane * heads + head) * static_cast<int64_t>(sequence) * io_head_dim;
      const int64_t dout_base = plane * static_cast<int64_t>(sequence) * heads * io_head_dim
                                + static_cast<int64_t>(head) * io_head_dim;
      for (int chunk = threadIdx.x; chunk < kSrQueryTile * kSrChunks; chunk += kThreadsWg) {
        const int row = chunk / kSrChunks;
        const int col = (chunk - row * kSrChunks) * 8;
        const int global_query = query_start + row;
        const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
        if (col < io_head_dim) {
          sr_cp_async_16(
              dst_q + dst_index,
              q + tensor_base + static_cast<int64_t>(global_query) * io_head_dim + col);
          sr_cp_async_16(
              dst_do + dst_index,
              d_output + dout_base + static_cast<int64_t>(global_query) * heads * io_head_dim
                  + col);
        } else {
          *reinterpret_cast<uint4*>(dst_q + dst_index) = uint4{0U, 0U, 0U, 0U};
          *reinterpret_cast<uint4*>(dst_do + dst_index) = uint4{0U, 0U, 0U, 0U};
        }
      }
    }
    for (int chunk = threadIdx.x; chunk < kSrQueryTile * (kSrKeyBlock / 8); chunk += kThreadsWg) {
      const int row = chunk / (kSrKeyBlock / 8);
      const int key = (chunk - row * (kSrKeyBlock / 8)) * 8;
      sr_cp_async_16(
          staging_bias_buffer + row * kSrBiasStride + key,
          bias + bias_base + static_cast<int64_t>(query_start + row) * sequence + key_start + key);
    }
    sr_cp_async_commit();
  };

  stage_q_d_output_and_bias(query_base, 0);

  const int query_end = query_base + query_rows;
#pragma unroll 1
  for (int query_start = query_base; query_start < query_end; query_start += kSrQueryTile) {
    const int parity = ((query_start - query_base) / kSrQueryTile) & 1;
    // Complete the current buffer, then prefetch into the other buffer.
    sr_cp_async_wait_all();
    wg_proxy_fence();
    __syncthreads();
    if (query_start + kSrQueryTile < query_end) {
      stage_q_d_output_and_bias(query_start + kSrQueryTile, parity ^ 1);
    }
    const __nv_bfloat16* current_q_buffer = shared_q + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    const __nv_bfloat16* current_d_output_buffer =
        shared_d_output + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    const __nv_bfloat16* current_bias_buffer = shared_bias + parity * kSrQueryTile * kSrBiasStride;
#pragma unroll
    for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
      const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
      if (single_outer && slot != 0) {
        continue;
      }
      const __nv_bfloat16* shared_k_tile = shared_k + slot * kSrKeyBlock * HeadDim;
      const __nv_bfloat16* shared_v_tile = shared_v + slot * kSrKeyBlock * HeadDim;
      const __nv_bfloat16* shared_q_tile = current_q_buffer + slot * kSrQueryTile * HeadDim;
      const __nv_bfloat16* shared_d_output_tile =
          current_d_output_buffer + slot * kSrQueryTile * HeadDim;

      // E. Compute S^T = KQ^T and dP^T = V dO^T with WGMMA.
      // A has 64 key rows; B has 32 query rows. Each warp gets 16 key rows.
      constexpr uint64_t kSsMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
      constexpr uint64_t kSsLbo = 64;
      constexpr uint64_t kSsSbo = HeadDim;
      Accumulator score_fragments[kSrQueryTile / 8];
      Accumulator dp_fragments[kSrQueryTile / 8];
#pragma unroll
      for (int column = 0; column < kSrQueryTile / 8; ++column) {
        score_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
      wg_wgmma_fence();
#pragma unroll
      for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        const uint64_t channel_byte_offset = static_cast<uint64_t>(channel_chunk) * 32;
        const uint64_t k_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_k_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t q_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_q_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(score_fragments, k_descriptor, q_descriptor, channel_chunk != 0);
      }
#pragma unroll
      for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        const uint64_t channel_byte_offset = static_cast<uint64_t>(channel_chunk) * 32;
        const uint64_t v_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_v_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t d_output_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_d_output_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(dp_fragments, v_descriptor, d_output_descriptor, channel_chunk != 0);
      }
      wg_wgmma_commit();
      wg_wgmma_wait_zero();

      // F. Reconstruct in the [key, query] accumulator layout:
      // P^T = exp(S^T*scale + mask[key] + bias[q,k] - LSE[q]);
      // dS^T = P^T * (dP^T - delta[q]).  Column index = query.
      float* shared_bias_pair_sum =
          shared_bias_pair_sums + warpgroup_id * (kSrQueryTile * kSrTreeStride);
      __nv_bfloat16* shared_score_gradient_tile =
          shared_score_gradients + slot * kSrQueryTile * kSrDsStride;
      const float slot_valid_multiplier = slot_is_active(slot) ? 1.0F : 0.0F;
      const float* shared_slot_mask = shared_mask + slot * kSrKeyBlock;
      const int64_t stats_base = (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence);
      // score_fragments become dS^T; dp_fragments become P^T. Round these
      // operands to BF16 after forming FP32 dS and its dBias contribution.
      uint32_t packed_probabilities[kSrQueryTile / 8][2];
      uint32_t packed_score_gradients[kSrQueryTile / 8][2];
#pragma unroll
      for (int column = 0; column < kSrQueryTile / 8; ++column) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int key_row = warp_key_start + lane_row + half * 8;
          const float mask_value = shared_slot_mask[key_row];
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const int query_in_tile = column * 8 + lane_in_row_group * 2 + pair;
            const int query = query_start + query_in_tile;
            const int index = half * 2 + pair;
            const int64_t stats = stats_base + query;
            const float row_logsumexp = logsumexp[stats];
            const float value =
                score_fragments[column].values[index] * attention_scale + mask_value
                + __bfloat162float(current_bias_buffer[query_in_tile * kSrBiasStride + key_row]);
            const float probability = row_logsumexp == -std::numeric_limits<float>::infinity()
                                          ? 0.0F
                                          : __expf(value - row_logsumexp);
            float score_gradient =
                probability * (dp_fragments[column].values[index] - delta[stats]);
            if constexpr (!OuterEqSeq) {
              score_gradient *= slot_valid_multiplier;
            }
            float p_value = probability;
            if constexpr (!OuterEqSeq) {
              p_value *= slot_valid_multiplier;
            }
            dp_fragments[column].values[index] = p_value;
            score_fragments[column].values[index] = score_gradient;
            const int cell = query_in_tile * kSrTreeStride + key_row;
            if (local_slot == 0) {
              shared_bias_pair_sum[cell] = score_gradient;
            } else {
              shared_bias_pair_sum[cell] = __fadd_rn(shared_bias_pair_sum[cell], score_gradient);
            }
            shared_score_gradient_tile[query_in_tile * kSrDsStride + key_row] =
                __float2bfloat16_rn(score_gradient);
          }
          packed_probabilities[column][half] = pack_bf16_pair(
              dp_fragments[column].values[half * 2],
              dp_fragments[column].values[half * 2 + 1]);
          packed_score_gradients[column][half] = pack_bf16_pair(
              score_fragments[column].values[half * 2],
              score_fragments[column].values[half * 2 + 1]);
        }
      }

      // G. dV += BF16(P^T) dO and dK += BF16(dS^T) Q over the 32 queries.
      // Both sums remain FP32; dK's attention scale is applied at writeback.
#pragma unroll
      for (int chunk = 0; chunk < kSrQueryTile / 16; ++chunk) {
        uint32_t probability_fragment[4];
        uint32_t score_gradient_fragment[4];
        probability_fragment[0] = packed_probabilities[chunk * 2][0];
        probability_fragment[1] = packed_probabilities[chunk * 2][1];
        probability_fragment[2] = packed_probabilities[chunk * 2 + 1][0];
        probability_fragment[3] = packed_probabilities[chunk * 2 + 1][1];
        score_gradient_fragment[0] = packed_score_gradients[chunk * 2][0];
        score_gradient_fragment[1] = packed_score_gradients[chunk * 2][1];
        score_gradient_fragment[2] = packed_score_gradients[chunk * 2 + 1][0];
        score_gradient_fragment[3] = packed_score_gradients[chunk * 2 + 1][1];
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          uint32_t b_fragment[2];
          ldmatrix_b_rowtrans_sw_sr<HeadDim>(
              shared_d_output_tile,
              chunk * 16,
              tile * 8,
              lane_id,
              b_fragment);
          mma_m16n8k16(dv_accumulators[local_slot][tile], probability_fragment, b_fragment);
          ldmatrix_b_rowtrans_sw_sr<HeadDim>(
              shared_q_tile,
              chunk * 16,
              tile * 8,
              lane_id,
              b_fragment);
          mma_m16n8k16(dk_accumulators[local_slot][tile], score_gradient_fragment, b_fragment);
        }
      }
    }

    // Publish the completed dBias pair sums and transposed dS to all warps.
    __syncthreads();

    // H. Combine (o0+o1)+(o2+o3) and store this tile's BF16 dBias leaf plane.
    // With a single outer, only its contribution was initialized and is read.
    const int64_t plane_tile_base =
        plane_base + static_cast<int64_t>(query_start - query_base) * sequence + key_start;
    for (int element = threadIdx.x; element < kSrQueryTile * kSrKeyBlock; element += kThreadsWg) {
      const int row = element / kSrKeyBlock;
      const int key = element - row * kSrKeyBlock;
      const int cell = row * kSrTreeStride + key;
      const float tile_sum = single_outer
                                 ? shared_bias_pair_sums[cell]
                                 : __fadd_rn(
                                       shared_bias_pair_sums[cell],
                                       shared_bias_pair_sums[kSrQueryTile * kSrTreeStride + cell]);
      workspace[plane_tile_base + static_cast<int64_t>(row) * sequence + key] =
          __float2bfloat16_rn(tile_sum);
    }

    // I. Compute this key block's dQ partial from BF16(dS) K over 64 keys.
    // Ownership changes here: warps 0/1 in a group take the first slot's
    // two 16-query halves, and warps 2/3 take the second slot's halves.
    {
      const int local_slot = warp_in_group >> 1;
      const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
      const int row_half = warp_in_group & 1;
      if (slot_is_active(slot)) {
        const __nv_bfloat16* shared_score_gradient_tile = shared_score_gradients
                                                          + slot * kSrQueryTile * kSrDsStride
                                                          + row_half * kQueryTile * kSrDsStride;
        const __nv_bfloat16* shared_k_tile = shared_k + slot * kSrKeyBlock * HeadDim;
        Accumulator dq_accumulators[HeadDim / 8];
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          dq_accumulators[tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        }
#pragma unroll
        for (int chunk = 0; chunk < kSrKeyBlock / 16; ++chunk) {
          uint32_t a_fragment[4];
          ldmatrix_a16x16(shared_score_gradient_tile, kSrDsStride, chunk * 16, lane_id, a_fragment);
#pragma unroll
          for (int tile = 0; tile < HeadDim / 8; ++tile) {
            uint32_t key_fragment[2];
            ldmatrix_b_keytrans_sw<HeadDim>(
                shared_k_tile,
                chunk * 16,
                tile * 8,
                lane_id,
                key_fragment);
            mma_m16n8k16(dq_accumulators[tile], a_fragment, key_fragment);
          }
        }
        // Partial layout matches the contiguous [B,O,H,N,D] output per
        // key block: idx = ((kb * total_planes + plane) * heads + head)
        //                  * sequence + q) * io_head_dim + col
        const int64_t plane = slot_plane(slot);
        const int64_t idx_base =
            (((static_cast<int64_t>(key_block) * total_planes + plane) * heads + head)
                 * static_cast<int64_t>(sequence)
             + (query_start + row_half * kQueryTile))
            * io_head_dim;
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = lane_row + half * 8;
            const int column = tile * 8 + lane_in_row_group * 2;
            if (column >= io_head_dim) {
              continue;
            }
            const int64_t index = idx_base + static_cast<int64_t>(row) * io_head_dim + column;
            *reinterpret_cast<uint32_t*>(dq_partials + index) = pack_bf16_pair(
                dq_accumulators[tile].values[half * 2],
                dq_accumulators[tile].values[half * 2 + 1]);
          }
        }
      }
    }
  }

  // J. Write dK/dV for both slots' 16 key rows owned by this warp.
  // Intermediate query slabs preserve raw FP32 sums; the final slab scales
  // dK once and stores dK/dV as BF16. dQ partials are reduced after all slabs.
  const bool final_query_stripe = query_base + query_rows == sequence;
#pragma unroll
  for (int local_slot = 0; local_slot < kSlotsPerWarpgroup; ++local_slot) {
    const int slot = warpgroup_id * kSlotsPerWarpgroup + local_slot;
    if (!slot_is_active(slot)) {
      continue;
    }
    const int64_t out_base =
        (slot_plane(slot) * heads + head) * static_cast<int64_t>(sequence) * io_head_dim
        + static_cast<int64_t>(key_start + warp_key_start) * io_head_dim;
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = lane_row + half * 8;
        const int column = tile * 8 + lane_in_row_group * 2;
        if (column >= io_head_dim) {
          continue;
        }
        const int64_t index = out_base + static_cast<int64_t>(row) * io_head_dim + column;
        if (final_query_stripe) {
          *reinterpret_cast<uint32_t*>(out_dv + index) = pack_bf16_pair(
              dv_accumulators[local_slot][tile].values[half * 2],
              dv_accumulators[local_slot][tile].values[half * 2 + 1]);
          *reinterpret_cast<uint32_t*>(out_dk + index) = pack_bf16_pair(
              dk_accumulators[local_slot][tile].values[half * 2] * attention_scale,
              dk_accumulators[local_slot][tile].values[half * 2 + 1] * attention_scale);
        } else {
          // Preserve the audited scalar continuation contract exactly:
          //   dv_continuation[index] = {dv0, dv1}
          //   dk_continuation[index] = {dk0, dk1}
          // The aligned float2 stores only combine each adjacent pair into
          // one transaction; they do not change values or stripe ordering.
          *reinterpret_cast<float2*>(dv_continuation + index) = make_float2(
              dv_accumulators[local_slot][tile].values[half * 2],
              dv_accumulators[local_slot][tile].values[half * 2 + 1]);
          *reinterpret_cast<float2*>(dk_continuation + index) = make_float2(
              dk_accumulators[local_slot][tile].values[half * 2],
              dk_accumulators[local_slot][tile].values[half * 2 + 1]);
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Fixed-order reductions: SR dQ partials, dBias leaves, and broadcast batches
// -----------------------------------------------------------------------------

// Fixed ascending-key-block fold of the SR dQ partials into BF16 dQ. Each
// thread owns eight consecutive elements (one 16-byte vector) so the
// per-kb loads coalesce to full lines and the eight per-element fadd
// chains stay independent; per-element fold order (ascending kb) and
// therefore the output bits match the scalar form exactly.
__global__ void turbopairformer_sr_dq_reduce_kernel(
    const __nv_bfloat16* __restrict__ dq_partials,
    __nv_bfloat16* __restrict__ out_dq,
    int64_t output_elements,
    int64_t partial_stride,
    int key_blocks,
    float attention_scale) {
  const int64_t vectors = output_elements / 8;
  const int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vec = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vec < vectors;
       vec += grid_stride) {
    const int64_t linear = vec * 8;
    // Seed with the first partial; do not insert an extra addition to zero.
    float root[8];
    {
      const uint4 first = *reinterpret_cast<const uint4*>(dq_partials + linear);
      const __nv_bfloat16* half_values = reinterpret_cast<const __nv_bfloat16*>(&first);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        root[element] = __bfloat162float(half_values[element]);
      }
    }
#pragma unroll 1
    for (int kb = 1; kb < key_blocks; ++kb) {
      const uint4 chunk = *reinterpret_cast<const uint4*>(
          dq_partials + static_cast<int64_t>(kb) * partial_stride + linear);
      const __nv_bfloat16* half_values = reinterpret_cast<const __nv_bfloat16*>(&chunk);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        root[element] = __fadd_rn(root[element], __bfloat162float(half_values[element]));
      }
    }
    uint4 packed;
    __nv_bfloat16* out_values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      out_values[element] = __float2bfloat16_rn(root[element] * attention_scale);
    }
    *reinterpret_cast<uint4*>(out_dq + linear) = packed;
  }
}

// dbias[h,q,k] = ((plane0 + plane1) + plane2) + ... in leaf order.
// A thread owns up to eight adjacent keys, with independent FP32 folds.
// Vectors never cross a row: workspace capacity can exceed the active slab.
// Unaligned rows and row tails retain scalar memory accesses and the same fold.
__global__ void turbopairformer_dq_dbias_finalize_kernel(
    const __nv_bfloat16* __restrict__ workspace,
    float* __restrict__ out_dbias,
    int64_t active_elements,
    int sequence,
    int heads,
    int batch_base,
    int active_batch,
    int batch_capacity,
    int query_base,
    int query_rows,
    int query_capacity,
    int wave_leaves,
    bool continue_root) {
  constexpr int kElements = 8;
  const int64_t vectors_per_row = (static_cast<int64_t>(sequence) + kElements - 1) / kElements;
  const int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t active_rows = active_elements / sequence;
  if (vector >= active_rows * vectors_per_row) {
    return;
  }
  const int key = static_cast<int>(vector % vectors_per_row) * kElements;
  int64_t remaining = vector / vectors_per_row;
  const int query_local = static_cast<int>(remaining % query_rows);
  remaining /= query_rows;
  const int head = static_cast<int>(remaining % heads);
  const int local_batch = static_cast<int>(remaining / heads);
  if (local_batch >= active_batch) {
    return;
  }
  const int64_t workspace_plane_elements =
      static_cast<int64_t>(batch_capacity) * heads * query_capacity * sequence;
  const int64_t workspace_linear =
      (((static_cast<int64_t>(local_batch) * heads + head) * query_capacity + query_local)
           * sequence
       + key);
  const int64_t output_linear =
      (((static_cast<int64_t>(batch_base + local_batch) * heads + head) * sequence + query_base
        + query_local)
           * sequence
       + key);
  const int active_keys = sequence - key < kElements ? sequence - key : kElements;
  const bool full_vector = active_keys == kElements;
  const bool vector_workspace =
      full_vector && workspace_plane_elements % kElements == 0
      && (reinterpret_cast<uintptr_t>(workspace + workspace_linear) & 15U) == 0;
  const bool vector_output =
      full_vector && (reinterpret_cast<uintptr_t>(out_dbias + output_linear) & 15U) == 0;
  // The first wave seeds the root from leaf 0. Later waves resume the
  // previous FP32 root, so wave boundaries leave the addition order intact.
  float root[kElements];
  if (continue_root) {
    if (vector_output) {
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
        const float4 values =
            *reinterpret_cast<const float4*>(out_dbias + output_linear + chunk * 4);
        root[chunk * 4] = values.x;
        root[chunk * 4 + 1] = values.y;
        root[chunk * 4 + 2] = values.z;
        root[chunk * 4 + 3] = values.w;
      }
    } else {
#pragma unroll
      for (int element = 0; element < kElements; ++element) {
        if (element < active_keys) {
          root[element] = out_dbias[output_linear + element];
        }
      }
    }
  } else if (vector_workspace) {
    const uint4 first = *reinterpret_cast<const uint4*>(workspace + workspace_linear);
    const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&first);
#pragma unroll
    for (int element = 0; element < kElements; ++element) {
      root[element] = __bfloat162float(values[element]);
    }
  } else {
#pragma unroll
    for (int element = 0; element < kElements; ++element) {
      if (element < active_keys) {
        root[element] = __bfloat162float(workspace[workspace_linear + element]);
      }
    }
  }
  // Always visit active leaves in ascending order, including a short wave.
  const int first_leaf = continue_root ? 0 : 1;
#pragma unroll 1
  for (int leaf = first_leaf; leaf < wave_leaves; ++leaf) {
    const __nv_bfloat16* leaf_values =
        workspace + static_cast<int64_t>(leaf) * workspace_plane_elements + workspace_linear;
    if (vector_workspace) {
      const uint4 chunk = *reinterpret_cast<const uint4*>(leaf_values);
      const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&chunk);
#pragma unroll
      for (int element = 0; element < kElements; ++element) {
        root[element] = __fadd_rn(root[element], __bfloat162float(values[element]));
      }
    } else {
#pragma unroll
      for (int element = 0; element < kElements; ++element) {
        if (element < active_keys) {
          root[element] = __fadd_rn(root[element], __bfloat162float(leaf_values[element]));
        }
      }
    }
  }
  if (vector_output) {
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      *reinterpret_cast<float4*>(out_dbias + output_linear + chunk * 4) = make_float4(
          root[chunk * 4],
          root[chunk * 4 + 1],
          root[chunk * 4 + 2],
          root[chunk * 4 + 3]);
    }
  } else {
#pragma unroll
    for (int element = 0; element < kElements; ++element) {
      if (element < active_keys) {
        out_dbias[output_linear + element] = root[element];
      }
    }
  }
}

// Reduce the public per-batch BF16 gradient in flattened batch order. Every
// FP32 partial is first rounded to the public BF16 per-batch contract, then
// accumulated with explicit FP32 round-to-nearest additions. One thread owns
// one output element; no atomics or framework reduction tree participate.
__global__ void turbopairformer_reduce_broadcast_dbias_kernel(
    const float* __restrict__ per_batch_dbias,
    __nv_bfloat16* __restrict__ out,
    int64_t output_elements,
    int64_t plane_elements,
    int target_batch,
    PrefixBroadcastPlan plan) {
  const int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < output_elements;
       linear += grid_stride) {
    const int64_t bias_batch_index = linear / plane_elements;
    const int64_t element = linear - bias_batch_index * plane_elements;
    float root = 0.0F;
    bool initialized = false;
#pragma unroll 1
    for (int target_index = 0; target_index < target_batch; ++target_index) {
      int remaining = target_index;
      int64_t mapped_bias_index = 0;
#pragma unroll 1
      for (int axis = plan.rank - 1; axis >= 0; --axis) {
        const int coordinate = remaining % plan.target_dims[axis];
        remaining /= plan.target_dims[axis];
        if (plan.bias_dims[axis] != 1) {
          mapped_bias_index += static_cast<int64_t>(coordinate) * plan.bias_strides[axis];
        }
      }
      if (mapped_bias_index != bias_batch_index) {
        continue;
      }
      const float public_partial = __bfloat162float(__float2bfloat16_rn(
          per_batch_dbias[static_cast<int64_t>(target_index) * plane_elements + element]));
      root = initialized ? __fadd_rn(root, public_partial) : public_partial;
      initialized = true;
    }
    out[linear] = __float2bfloat16_rn(initialized ? root : 0.0F);
  }
}

// -----------------------------------------------------------------------------
// Host launch planning and workspace reuse
// -----------------------------------------------------------------------------
// All launchers iterate batch chunks -> query slabs -> outer-leaf waves.
// Each wave writes BF16 leaf planes, then the dBias finalizer consumes them
// before the same stream reuses workspace. It carries the FP32 root between
// waves. SR also carries FP32 dK/dV between query slabs and reduces dQ after
// all leaf launches. These partitions must preserve the stated fold orders.
// Typed host helpers keep each kernel's launch arguments in one place.
// Selectors choose its compile-time booleans without changing the wave schedule.

int64_t checked_positive_product(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left > 0 && right > 0, label, " dimensions must be positive");
  TORCH_CHECK(
      left <= std::numeric_limits<int64_t>::max() / right, label, " element count overflow");
  return left * right;
}

int64_t positive_ceil_div(int64_t numerator, int64_t denominator) {
  TORCH_CHECK(numerator > 0 && denominator > 0, "ceil-div inputs must be positive");
  return numerator / denominator + (numerator % denominator != 0);
}

// Each attribute helper preserves the paired OuterEq=true/false setup and check.
// Kernel helpers launch only; callers keep launch-error checks and reductions.

template <int HeadDim, int StaticSequence, bool FullTiles, bool TaskInZ>
void set_dq_dbias_leaf_shared_memory(int kSharedBytes) {
  const cudaError_t status = cudaFuncSetAttribute(
      turbopairformer_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, TaskInZ, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  const cudaError_t status_g = cudaFuncSetAttribute(
      turbopairformer_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, TaskInZ, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  TORCH_CHECK(
      status == cudaSuccess && status_g == cudaSuccess,
      "failed to opt in TurboPairFormer dQ/dBias shared memory");
}

template <int HeadDim, int StaticSequence, bool FullTiles, bool TaskInZ, bool OuterEq>
void launch_dq_dbias_leaf_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch_base,
    int active_batch,
    int batch_chunk,
    int query_base,
    int query_rows,
    int query_span,
    int leaf_base,
    int io_head_dim,
    float attention_scale,
    dim3 grid,
    int kSharedBytes,
    cudaStream_t stream) {
  turbopairformer_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, TaskInZ, OuterEq>
      <<<grid, kThreads, kSharedBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()), logsumexp.data_ptr<float>(),
          delta.data_ptr<float>(), mask.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()), sequence, outer_count, heads,
          batch_base, active_batch, batch_chunk, query_base, query_rows, query_span, leaf_base,
          io_head_dim, attention_scale);
}

template <int HeadDim, int StaticSequence, bool FullTiles>
auto select_dq_dbias_leaf_launcher(bool task_in_grid_z, bool outer_eq) {
  // Select a CPU launch helper; this does not launch the GPU.
  if (task_in_grid_z) {
    return outer_eq ? &launch_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, true, true>
                    : &launch_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, true, false>;
  }
  return outer_eq ? &launch_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, false, true>
                  : &launch_dq_dbias_leaf_kernel<HeadDim, StaticSequence, FullTiles, false, false>;
}

template <int HeadDim, bool TaskInZ>
void set_dq_dbias_wg_shared_memory(int kSharedBytesWg) {
  const cudaError_t status = cudaFuncSetAttribute(
      turbopairformer_dq_dbias_wg_kernel<HeadDim, TaskInZ, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytesWg);
  const cudaError_t status_g = cudaFuncSetAttribute(
      turbopairformer_dq_dbias_wg_kernel<HeadDim, TaskInZ, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytesWg);
  TORCH_CHECK(
      status == cudaSuccess && status_g == cudaSuccess,
      "failed to opt in TurboPairFormer dQ/dBias warpgroup shared memory");
}

template <int HeadDim, bool TaskInZ, bool OuterEq>
void launch_dq_dbias_wg_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch_base,
    int active_batch,
    int batch_chunk,
    int query_base,
    int query_span,
    int leaf_base,
    int io_head_dim,
    float attention_scale,
    dim3 grid,
    int kSharedBytesWg,
    cudaStream_t stream) {
  turbopairformer_dq_dbias_wg_kernel<HeadDim, TaskInZ, OuterEq>
      <<<grid, kThreadsWg, kSharedBytesWg, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()), logsumexp.data_ptr<float>(),
          delta.data_ptr<float>(), mask.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()), sequence, outer_count, heads,
          batch_base, active_batch, batch_chunk, query_base, query_span, leaf_base, io_head_dim,
          attention_scale);
}

template <int HeadDim>
auto select_dq_dbias_wg_launcher(bool task_in_grid_z, bool outer_eq) {
  // Select a CPU launch helper; this does not launch the GPU.
  if (task_in_grid_z) {
    return outer_eq ? &launch_dq_dbias_wg_kernel<HeadDim, true, true>
                    : &launch_dq_dbias_wg_kernel<HeadDim, true, false>;
  }
  return outer_eq ? &launch_dq_dbias_wg_kernel<HeadDim, false, true>
                  : &launch_dq_dbias_wg_kernel<HeadDim, false, false>;
}

template <int HeadDim, bool TaskInZ>
void set_sr_leaf_shared_memory(int kSharedBytesSr) {
  const cudaError_t status = cudaFuncSetAttribute(
      turbopairformer_sr_leaf_kernel<HeadDim, TaskInZ, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytesSr);
  const cudaError_t status_g = cudaFuncSetAttribute(
      turbopairformer_sr_leaf_kernel<HeadDim, TaskInZ, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytesSr);
  TORCH_CHECK(
      status == cudaSuccess && status_g == cudaSuccess,
      "failed to opt in TurboPairFormer SR shared memory");
}

template <int HeadDim, bool TaskInZ, bool OuterEq>
void launch_sr_leaf_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv,
    torch::Tensor& dk_continuation,
    torch::Tensor& dv_continuation,
    torch::Tensor& dq_partials,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int total_planes,
    int heads,
    int batch_base,
    int active_batch,
    int batch_chunk,
    int query_base,
    int query_rows,
    int query_span,
    int leaf_base,
    int io_head_dim,
    float attention_scale,
    dim3 grid,
    int kSharedBytesSr,
    cudaStream_t stream) {
  turbopairformer_sr_leaf_kernel<HeadDim, TaskInZ, OuterEq>
      <<<grid, kThreadsWg, kSharedBytesSr, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()), logsumexp.data_ptr<float>(),
          delta.data_ptr<float>(), mask.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()), dk_continuation.data_ptr<float>(),
          dv_continuation.data_ptr<float>(),
          reinterpret_cast<__nv_bfloat16*>(dq_partials.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()), sequence, outer_count,
          total_planes, heads, batch_base, active_batch, batch_chunk, query_base, query_rows,
          query_span, leaf_base, io_head_dim, attention_scale);
}

template <int HeadDim>
auto select_sr_leaf_launcher(bool task_in_grid_z, bool outer_eq) {
  // Select a CPU launch helper; this does not launch the GPU.
  if (task_in_grid_z) {
    return outer_eq ? &launch_sr_leaf_kernel<HeadDim, true, true>
                    : &launch_sr_leaf_kernel<HeadDim, true, false>;
  }
  return outer_eq ? &launch_sr_leaf_kernel<HeadDim, false, true>
                  : &launch_sr_leaf_kernel<HeadDim, false, false>;
}

// Per-warp split launch: full-tile specialization or guarded query/key tails.
template <int HeadDim, int StaticSequence>
void launch(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    int io_head_dim,
    cudaStream_t stream) {
  constexpr int kQueryRowStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  constexpr int kSharedBytes =
      (2 * kWarpsPerBlock * kQueryBlock * kQueryRowStride
       + kWarpsPerBlock * 2 * kKeyTile * kKVStride)
          * static_cast<int>(sizeof(__nv_bfloat16))
      + kWarpsPerBlock * kQueryBlock * kKeyTile * static_cast<int>(sizeof(float));
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(io_head_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  TORCH_CHECK(
      query_span > 0 && query_span <= std::numeric_limits<int>::max() && resident_leaves > 0
          && resident_leaves <= leaves && batch_chunk > 0 && batch_chunk <= batch,
      "invalid query-slab workspace plan");
  int64_t required_workspace = checked_positive_product(resident_leaves, batch_chunk, "workspace");
  required_workspace = checked_positive_product(required_workspace, heads, "workspace");
  required_workspace = checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace = checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace, "workspace is smaller than the query-slab plan");
  set_dq_dbias_leaf_shared_memory<HeadDim, StaticSequence, true, true>(kSharedBytes);
  set_dq_dbias_leaf_shared_memory<HeadDim, StaticSequence, true, false>(kSharedBytes);
  if constexpr (StaticSequence != -1) {
    set_dq_dbias_leaf_shared_memory<HeadDim, StaticSequence, false, true>(kSharedBytes);
    set_dq_dbias_leaf_shared_memory<HeadDim, StaticSequence, false, false>(kSharedBytes);
  }
  const int finalize_threads = 256;
  for (int batch_base = 0; batch_base < batch;) {
    const int active_batch = std::min(batch_chunk, batch - batch_base);
    for (int query_base = 0; query_base < sequence;) {
      const int query_rows = std::min(query_span, sequence - query_base);
      const int query_blocks = (query_rows + kQueryBlock - 1) / kQueryBlock;
      TORCH_CHECK(
          query_blocks > 0 && query_blocks <= kMaxGridYZ,
          "query slab exceeds the CUDA grid-z contract");
      int64_t active_elements = checked_positive_product(active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(active_elements, sequence, "active dBias slab");
      const int64_t finalize_vectors = checked_positive_product(
          active_elements / sequence, positive_ceil_div(sequence, 8), "dBias slab vectors");
      const int64_t finalize_blocks = positive_ceil_div(finalize_vectors, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= kMaxGridX,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves = std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks = static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= kMaxGridX,
            "dBias leaf wave exceeds the CUDA grid-x contract");
        // Preserve the old query-major launch geometry whenever the task
        // axis fits CUDA grid-z.  Large prefix batches use the outer-major
        // fallback.  Both layouts decode the same logical CTA coordinates.
        const bool task_in_grid_z = leaf_tasks <= kMaxGridYZ;
        const dim3 grid =
            task_in_grid_z
                ? dim3(
                      static_cast<unsigned int>(query_blocks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(leaf_tasks))
                : dim3(
                      static_cast<unsigned int>(leaf_tasks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(query_blocks));
        const auto launch_leaf = [&] {
          if constexpr (StaticSequence == -1) {
            // This specialization only accepts complete query tiles.
            TORCH_CHECK(
                query_rows % kQueryBlock == 0,
                "exact-tiles lane requires query rows in full blocks");
            return select_dq_dbias_leaf_launcher<HeadDim, StaticSequence, true>(
                task_in_grid_z, outer_eq);
          } else {
            if (query_rows % kQueryBlock == 0) {
              return select_dq_dbias_leaf_launcher<HeadDim, StaticSequence, true>(
                  task_in_grid_z, outer_eq);
            } else {
              return select_dq_dbias_leaf_launcher<HeadDim, StaticSequence, false>(
                  task_in_grid_z, outer_eq);
            }
          }
        }();
        launch_leaf(
            q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, workspace, sequence,
            outer_count, heads, batch_base, active_batch, batch_chunk, query_base, query_rows,
            query_span, leaf_base, io_head_dim, attention_scale, grid, kSharedBytes, stream);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess, "TurboPairFormer dQ/dBias slab leaf launch failed");
        turbopairformer_dq_dbias_finalize_kernel<<<
            static_cast<unsigned int>(finalize_blocks), finalize_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
            out_dbias.data_ptr<float>(), active_elements, sequence, heads, batch_base, active_batch,
            batch_chunk, query_base, query_rows, query_span, wave_leaves, leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "TurboPairFormer dQ/dBias slab finalizer launch failed");
        // Increment by the validated active extent, so even the theoretical
        // INT_MAX boundary cannot overflow a signed host loop counter.
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
}

// Warpgroup split launch: full 64-query tiles, using the same dBias finalizer.
template <int HeadDim>
void launch_wg(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    int io_head_dim,
    cudaStream_t stream) {
  constexpr int kSharedBytesWg =
      (2 * kOuterLeaf * kQueryBlockWg * HeadDim + kOuterLeaf * 2 * kKeyTile * HeadDim
       + kQueryBlockWg * kBiasStride)
          * static_cast<int>(sizeof(__nv_bfloat16))
      + (kOuterLeaf * kKeyTile + 2 * kQueryBlockWg * kTreeStride) * static_cast<int>(sizeof(float));
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(io_head_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  TORCH_CHECK(
      query_span > 0 && query_span % kQueryBlockWg == 0 && sequence % kQueryBlockWg == 0
          && resident_leaves > 0 && resident_leaves <= leaves && batch_chunk > 0
          && batch_chunk <= batch,
      "invalid warpgroup query-slab plan");
  int64_t required_workspace = checked_positive_product(resident_leaves, batch_chunk, "workspace");
  required_workspace = checked_positive_product(required_workspace, heads, "workspace");
  required_workspace = checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace = checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace, "workspace is smaller than the query-slab plan");
  set_dq_dbias_wg_shared_memory<HeadDim, true>(kSharedBytesWg);
  set_dq_dbias_wg_shared_memory<HeadDim, false>(kSharedBytesWg);
  const int finalize_threads = 256;
  for (int batch_base = 0; batch_base < batch;) {
    const int active_batch = std::min(batch_chunk, batch - batch_base);
    for (int query_base = 0; query_base < sequence;) {
      const int query_rows = std::min(query_span, sequence - query_base);
      TORCH_CHECK(
          query_rows % kQueryBlockWg == 0,
          "warpgroup lane requires query rows in full 64-row blocks");
      const int query_blocks = query_rows / kQueryBlockWg;
      TORCH_CHECK(
          query_blocks > 0 && query_blocks <= kMaxGridYZ,
          "query slab exceeds the CUDA grid-z contract");
      int64_t active_elements = checked_positive_product(active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(active_elements, sequence, "active dBias slab");
      const int64_t finalize_vectors = checked_positive_product(
          active_elements / sequence, positive_ceil_div(sequence, 8), "dBias slab vectors");
      const int64_t finalize_blocks = positive_ceil_div(finalize_vectors, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= kMaxGridX,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves = std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks = static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= kMaxGridX,
            "dBias leaf wave exceeds the CUDA grid-x contract");
        const bool task_in_grid_z = leaf_tasks <= kMaxGridYZ;
        const dim3 grid =
            task_in_grid_z
                ? dim3(
                      static_cast<unsigned int>(query_blocks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(leaf_tasks))
                : dim3(
                      static_cast<unsigned int>(leaf_tasks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(query_blocks));
        const auto launch_leaf = select_dq_dbias_wg_launcher<HeadDim>(task_in_grid_z, outer_eq);
        launch_leaf(
            q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, workspace, sequence,
            outer_count, heads, batch_base, active_batch, batch_chunk, query_base, query_span,
            leaf_base, io_head_dim, attention_scale, grid, kSharedBytesWg, stream);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "TurboPairFormer dQ/dBias warpgroup leaf launch failed");
        turbopairformer_dq_dbias_finalize_kernel<<<
            static_cast<unsigned int>(finalize_blocks), finalize_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
            out_dbias.data_ptr<float>(), active_elements, sequence, heads, batch_base, active_batch,
            batch_chunk, query_base, query_rows, query_span, wave_leaves, leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "TurboPairFormer dQ/dBias slab finalizer launch failed");
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
}

// Fused SR launch: full 64-key blocks and 32-query tiles. dQ partials are
// indexed by key block; dK/dV continuation is needed only for multiple slabs.
template <int HeadDim>
void launch_sr(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    torch::Tensor& dq_partials,
    torch::Tensor& dk_continuation,
    torch::Tensor& dv_continuation,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    int io_head_dim,
    cudaStream_t stream) {
  constexpr int kSharedBytesSr =
      (2 * kOuterLeaf * kSrKeyBlock * HeadDim + 4 * kOuterLeaf * kSrQueryTile * HeadDim
       + kOuterLeaf * kSrQueryTile * kSrDsStride + 2 * kSrQueryTile * kSrBiasStride)
          * static_cast<int>(sizeof(__nv_bfloat16))
      + (kOuterLeaf * kSrKeyBlock + 2 * kSrQueryTile * kSrTreeStride)
            * static_cast<int>(sizeof(float));
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(io_head_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  const int key_blocks = sequence / kSrKeyBlock;
  const int total_planes = batch * outer_count;
  TORCH_CHECK(
      sequence % kSrKeyBlock == 0 && key_blocks <= kMaxGridYZ && query_span > 0
          && query_span % kSrQueryTile == 0 && resident_leaves > 0 && resident_leaves <= leaves
          && batch_chunk > 0 && batch_chunk <= batch,
      "invalid SR backward plan");
  int64_t required_workspace = checked_positive_product(resident_leaves, batch_chunk, "workspace");
  required_workspace = checked_positive_product(required_workspace, heads, "workspace");
  required_workspace = checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace = checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace, "workspace is smaller than the query-slab plan");
  int64_t required_partials = checked_positive_product(key_blocks, total_planes, "dq partials");
  required_partials = checked_positive_product(required_partials, heads, "dq partials");
  required_partials = checked_positive_product(required_partials, sequence, "dq partials");
  required_partials = checked_positive_product(required_partials, io_head_dim, "dq partials");
  TORCH_CHECK(
      dq_partials.numel() >= required_partials, "dq_partials is smaller than the key-block plan");
  const int64_t required_continuation = query_span < sequence ? q.numel() : 0;
  TORCH_CHECK(
      dk_continuation.numel() >= required_continuation
          && dv_continuation.numel() >= required_continuation,
      "dK/dV continuation state is smaller than the query-stripe plan");
  set_sr_leaf_shared_memory<HeadDim, true>(kSharedBytesSr);
  set_sr_leaf_shared_memory<HeadDim, false>(kSharedBytesSr);
  const int finalize_threads = 256;
  for (int batch_base = 0; batch_base < batch;) {
    const int active_batch = std::min(batch_chunk, batch - batch_base);
    for (int query_base = 0; query_base < sequence;) {
      const int query_rows = std::min(query_span, sequence - query_base);
      TORCH_CHECK(query_rows % kSrQueryTile == 0, "SR lane requires query rows in 32-row tiles");
      int64_t active_elements = checked_positive_product(active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(active_elements, sequence, "active dBias slab");
      const int64_t finalize_vectors = checked_positive_product(
          active_elements / sequence, positive_ceil_div(sequence, 8), "dBias slab vectors");
      const int64_t finalize_blocks = positive_ceil_div(finalize_vectors, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= kMaxGridX,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves = std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks = static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= kMaxGridX,
            "SR leaf wave exceeds the CUDA grid contract");
        const bool task_in_grid_z = leaf_tasks <= kMaxGridYZ;
        const dim3 grid =
            task_in_grid_z
                ? dim3(
                      static_cast<unsigned int>(key_blocks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(leaf_tasks))
                : dim3(
                      static_cast<unsigned int>(leaf_tasks), static_cast<unsigned int>(heads),
                      static_cast<unsigned int>(key_blocks));
        const auto launch_leaf = select_sr_leaf_launcher<HeadDim>(task_in_grid_z, outer_eq);
        launch_leaf(
            q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, dk_continuation,
            dv_continuation, dq_partials, workspace, sequence, outer_count, total_planes, heads,
            batch_base, active_batch, batch_chunk, query_base, query_rows, query_span, leaf_base,
            io_head_dim, attention_scale, grid, kSharedBytesSr, stream);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer SR leaf launch failed");
        turbopairformer_dq_dbias_finalize_kernel<<<
            static_cast<unsigned int>(finalize_blocks), finalize_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
            out_dbias.data_ptr<float>(), active_elements, sequence, heads, batch_base, active_batch,
            batch_chunk, query_base, query_rows, query_span, wave_leaves, leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess, "TurboPairFormer SR finalizer launch failed");
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
  const int64_t output_elements = out_dq.numel();
  TORCH_CHECK(output_elements % 8 == 0, "SR dQ reduce requires an 8-element-aligned output");
  const int64_t partial_stride =
      static_cast<int64_t>(total_planes) * heads * sequence * io_head_dim;
  constexpr int reduce_threads = 256;
  const int64_t reduce_blocks_needed = positive_ceil_div(output_elements / 8, reduce_threads);
  const unsigned int reduce_blocks =
      static_cast<unsigned int>(std::min<int64_t>(reduce_blocks_needed, kReductionGridBlocks));
  turbopairformer_sr_dq_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(dq_partials.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()), output_elements, partial_stride,
      key_blocks, attention_scale);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer SR dQ reduce launch failed");
}

// -----------------------------------------------------------------------------
// C++ entry points: validate tensor layouts, then choose storage specialization
// -----------------------------------------------------------------------------

// Production SR entry point. The Python facade supplies padded contiguous
// tensors, a bounded workspace plan, forward logsumexp, and backward delta.
void turbopairformer_triangle_sr_backward_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    torch::Tensor& dq_partials,
    torch::Tensor& dk_continuation,
    torch::Tensor& dv_continuation,
    int64_t query_span64,
    int64_t resident_leaves64,
    int64_t batch_chunk64) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
  const int batch = static_cast<int>(q.size(0));
  const int outer_count = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int sequence = static_cast<int>(q.size(3));
  const int head_dim = static_cast<int>(q.size(4));
  TORCH_CHECK(
      sequence % kSrKeyBlock == 0, "SR backward requires the 64-padded exact-tiles contract");
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 64,
      "SR backward serves D<=64 (D=128 keeps the two-pass path)");
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kBFloat16
            && tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/out_dq/out_dk/out_dv must be contiguous BF16 matching q");
  };
  check_q_like(k);
  check_q_like(v);
  check_q_like(out_dq);
  check_q_like(out_dk);
  check_q_like(out_dv);
  TORCH_CHECK(
      d_output.is_cuda() && d_output.dtype() == torch::kBFloat16 && d_output.is_contiguous()
          && d_output.dim() == 5 && d_output.size(0) == q.size(0) && d_output.size(1) == q.size(1)
          && d_output.size(2) == q.size(3) && d_output.size(3) == q.size(2)
          && d_output.size(4) == q.size(4),
      "d_output must be contiguous BF16 [B,O,N,H,D]");
  TORCH_CHECK(
      logsumexp.is_cuda() && logsumexp.dtype() == torch::kFloat32 && logsumexp.is_contiguous()
      && logsumexp.numel() == q.numel() / head_dim);
  TORCH_CHECK(
      delta.is_cuda() && delta.dtype() == torch::kFloat32 && delta.is_contiguous()
      && delta.sizes() == logsumexp.sizes());
  TORCH_CHECK(
      mask.is_cuda() && mask.dtype() == torch::kFloat32 && mask.is_contiguous()
      && mask.sizes() == torch::IntArrayRef({batch, outer_count, 1, 1, sequence}));
  TORCH_CHECK(
      bias.is_cuda() && bias.dtype() == torch::kBFloat16 && bias.is_contiguous()
      && bias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      out_dbias.is_cuda() && out_dbias.dtype() == torch::kFloat32 && out_dbias.is_contiguous()
      && out_dbias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      workspace.is_cuda() && workspace.dtype() == torch::kBFloat16 && workspace.is_contiguous());
  TORCH_CHECK(
      dq_partials.is_cuda() && dq_partials.dtype() == torch::kBFloat16
      && dq_partials.is_contiguous());
  const auto check_continuation = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kFloat32
            && tensor.is_contiguous(),
        "dK/dV continuation state must be contiguous FP32 on q.device");
  };
  check_continuation(dk_continuation);
  check_continuation(dv_continuation);
  const int query_span = static_cast<int>(query_span64);
  const int resident_leaves = static_cast<int>(resident_leaves64);
  const int batch_chunk = static_cast<int>(batch_chunk64);

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int io_head_dim = head_dim;
  const int storage = head_dim <= 16 ? 16 : (head_dim <= 32 ? 32 : 64);
  switch (storage) {
  case 16:
    launch_sr<16>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dk, out_dv, out_dbias,
        workspace, dq_partials, dk_continuation, dv_continuation, sequence, outer_count, heads,
        batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  case 32:
    launch_sr<32>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dk, out_dv, out_dbias,
        workspace, dq_partials, dk_continuation, dv_continuation, sequence, outer_count, heads,
        batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  default:
    launch_sr<64>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dk, out_dv, out_dbias,
        workspace, dq_partials, dk_continuation, dv_continuation, sequence, outer_count, heads,
        batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer SR launch failed");
}

template <int HeadDim>
void dispatch_sequence(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    int io_head_dim,
    cudaStream_t stream) {
  // Split route only. Prefer WGMMA for supported storage widths and full
  // 64-row slabs, then per-warp MMA with full tiles, then guarded tails.
  // Sequence length remains a runtime value in all three choices.
  if (HeadDim <= 64 && sequence % kQueryBlockWg == 0 && query_span % kQueryBlockWg == 0) {
    // The WGMMA swizzle descriptors support storage widths 16, 32, and 64.
    // D=128 uses the per-warp split kernel below.
    launch_wg<HeadDim>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
  } else if (sequence % kQueryBlock == 0 && sequence % kOuterLeaf == 0) {
    launch<HeadDim, -1>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
  } else {
    launch<HeadDim, 0>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
  }
}

// Direct split dQ/dBias entry point; dK/dV are computed by the separate module.
void turbopairformer_triangle_dq_dbias_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dq,
    torch::Tensor& out_dbias,
    torch::Tensor& workspace,
    int64_t query_span64,
    int64_t resident_leaves64,
    int64_t batch_chunk64) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
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
          && batch64 * outer64 <= kMaxGridX
          && (sequence64 + kQueryBlock - 1) / kQueryBlock <= kMaxGridYZ,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const int64_t leaves64 = (outer64 + kOuterLeaf - 1) / kOuterLeaf;
  TORCH_CHECK(
      query_span64 > 0 && query_span64 <= std::numeric_limits<int>::max() && resident_leaves64 > 0
          && resident_leaves64 <= leaves64 && resident_leaves64 <= std::numeric_limits<int>::max()
          && batch_chunk64 > 0 && batch_chunk64 <= batch64
          && batch_chunk64 <= std::numeric_limits<int>::max(),
      "invalid query-slab plan dimensions");
  const int query_span = static_cast<int>(query_span64);
  const int resident_leaves = static_cast<int>(resident_leaves64);
  const int batch_chunk = static_cast<int>(batch_chunk64);
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kBFloat16
            && tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/d_output/out_dq must be contiguous BF16 tensors matching q "
        "on one device");
  };
  check_q_like(k);
  check_q_like(v);
  TORCH_CHECK(
      d_output.is_cuda() && d_output.device() == q.device() && d_output.dtype() == torch::kBFloat16
          && d_output.is_contiguous() && d_output.dim() == 5 && d_output.size(0) == q.size(0)
          && d_output.size(1) == q.size(1) && d_output.size(2) == q.size(3)
          && d_output.size(3) == q.size(2) && d_output.size(4) == q.size(4),
      "d_output must be contiguous BF16 [B,O,N,H,D] on q.device");
  check_q_like(out_dq);
  TORCH_CHECK(
      logsumexp.is_cuda() && logsumexp.device() == q.device()
      && logsumexp.dtype() == torch::kFloat32 && logsumexp.is_contiguous()
      && logsumexp.numel() == q.numel() / head_dim);
  TORCH_CHECK(
      delta.is_cuda() && delta.device() == q.device() && delta.dtype() == torch::kFloat32
      && delta.is_contiguous() && delta.sizes() == logsumexp.sizes());
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() && mask.dtype() == torch::kFloat32
      && mask.is_contiguous()
      && mask.sizes() == torch::IntArrayRef({batch, outer_count, 1, 1, sequence}));
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() && bias.dtype() == torch::kBFloat16
      && bias.is_contiguous()
      && bias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      out_dbias.is_cuda() && out_dbias.device() == q.device()
      && out_dbias.dtype() == torch::kFloat32 && out_dbias.is_contiguous()
      && out_dbias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  int64_t required_workspace =
      checked_positive_product(resident_leaves64, batch_chunk64, "workspace plan");
  required_workspace = checked_positive_product(required_workspace, heads64, "workspace plan");
  required_workspace = checked_positive_product(required_workspace, query_span64, "workspace plan");
  required_workspace = checked_positive_product(required_workspace, sequence64, "workspace plan");
  TORCH_CHECK(
      workspace.is_cuda() && workspace.device() == q.device()
          && workspace.dtype() == torch::kBFloat16 && workspace.is_contiguous()
          && workspace.numel() >= required_workspace,
      "workspace is smaller than the validated query-slab plan");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int io_head_dim = head_dim;
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 128, "head_dim must be a multiple of 8 and <= 128, got ",
      head_dim);
  const int storage = head_dim <= 16 ? 16 : (head_dim <= 32 ? 32 : (head_dim <= 64 ? 64 : 128));
  switch (storage) {
  case 16:
    dispatch_sequence<16>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  case 32:
    dispatch_sequence<32>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  case 64:
    dispatch_sequence<64>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  default:
    dispatch_sequence<128>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dq, out_dbias, workspace, sequence,
        outer_count, heads, batch, query_span, resident_leaves, batch_chunk, io_head_dim, stream);
    break;
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer dQ/dBias launch failed");
}

// Restore the original bias prefix by folding the per-batch dBias results.
void turbopairformer_reduce_broadcast_dbias_out(
    const torch::Tensor& per_batch_dbias,
    torch::Tensor& out,
    const std::vector<int64_t>& target_prefix,
    const std::vector<int64_t>& bias_prefix) {
  TORCH_CHECK(
      !target_prefix.empty() && target_prefix.size() <= kMaxPrefixRank,
      "target prefix rank must be in [1, ", kMaxPrefixRank, "]");
  TORCH_CHECK(
      bias_prefix.size() <= target_prefix.size(),
      "bias prefix rank cannot exceed target prefix rank");
  PrefixBroadcastPlan plan{};
  plan.rank = static_cast<int>(target_prefix.size());
  int64_t target_batch64 = 1;
  const int rank_delta = static_cast<int>(target_prefix.size() - bias_prefix.size());
  for (int axis = 0; axis < plan.rank; ++axis) {
    const int64_t target = target_prefix[axis];
    const int64_t bias = axis < rank_delta ? 1 : bias_prefix[axis - rank_delta];
    TORCH_CHECK(
        target > 0 && target <= std::numeric_limits<int>::max() && bias > 0
            && bias <= std::numeric_limits<int>::max() && (bias == 1 || bias == target),
        "bias prefix must broadcast exactly to the positive target prefix");
    TORCH_CHECK(
        target_batch64 <= std::numeric_limits<int>::max() / target,
        "flattened target prefix exceeds the CUDA int indexing contract");
    target_batch64 *= target;
    plan.target_dims[axis] = static_cast<int>(target);
    plan.bias_dims[axis] = static_cast<int>(bias);
  }
  int64_t bias_batch = 1;
  for (int axis = plan.rank - 1; axis >= 0; --axis) {
    plan.bias_strides[axis] = bias_batch;
    TORCH_CHECK(
        bias_batch <= std::numeric_limits<int64_t>::max() / plan.bias_dims[axis],
        "bias prefix product overflow");
    bias_batch *= plan.bias_dims[axis];
  }
  const int target_batch = static_cast<int>(target_batch64);
  TORCH_CHECK(
      per_batch_dbias.is_cuda() && per_batch_dbias.dtype() == torch::kFloat32
          && per_batch_dbias.is_contiguous() && per_batch_dbias.dim() == 5
          && per_batch_dbias.size(0) == target_batch && per_batch_dbias.size(1) == 1,
      "per_batch_dbias must be contiguous FP32 [B,1,H,N,N]");
  const int64_t heads = per_batch_dbias.size(2);
  const int64_t sequence = per_batch_dbias.size(3);
  TORCH_CHECK(
      heads > 0 && sequence > 0 && per_batch_dbias.size(4) == sequence,
      "per_batch_dbias must have positive square pair dimensions");
  TORCH_CHECK(
      out.is_cuda() && out.device() == per_batch_dbias.device() && out.dtype() == torch::kBFloat16
          && out.is_contiguous() && out.dim() == static_cast<int64_t>(bias_prefix.size()) + 4,
      "out must be contiguous BF16 with the source bias rank");
  for (int axis = 0; axis < static_cast<int>(bias_prefix.size()); ++axis) {
    TORCH_CHECK(out.size(axis) == bias_prefix[axis], "out bias prefix mismatch");
  }
  const int tail = static_cast<int>(bias_prefix.size());
  TORCH_CHECK(
      out.size(tail) == 1 && out.size(tail + 1) == heads && out.size(tail + 2) == sequence
          && out.size(tail + 3) == sequence,
      "out bias tail must be [1,H,N,N]");
  int64_t plane_elements = checked_positive_product(heads, sequence, "broadcast dBias plane");
  plane_elements = checked_positive_product(plane_elements, sequence, "broadcast dBias plane");
  const int64_t expected_output_elements =
      checked_positive_product(bias_batch, plane_elements, "broadcast dBias output");
  TORCH_CHECK(
      out.numel() == expected_output_elements, "out element count does not match the bias prefix");

  const c10::cuda::CUDAGuard device_guard(per_batch_dbias.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int64_t required_blocks = positive_ceil_div(out.numel(), threads);
  const unsigned int blocks =
      static_cast<unsigned int>(std::min<int64_t>(required_blocks, kReductionGridBlocks));
  turbopairformer_reduce_broadcast_dbias_kernel<<<blocks, threads, 0, stream>>>(
      per_batch_dbias.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.numel(), plane_elements, target_batch, plan);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fixed broadcast dBias reduction launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "turbopairformer_triangle_dq_dbias_out", &turbopairformer_triangle_dq_dbias_out,
      "TurboPairFormer fused dQ+dBias (query-slab deterministic waves)");
  module.def(
      "turbopairformer_reduce_broadcast_dbias_out", &turbopairformer_reduce_broadcast_dbias_out,
      "Fixed-order broadcast dBias reduction");
  module.def(
      "turbopairformer_triangle_sr_backward_out", &turbopairformer_triangle_sr_backward_out,
      "TurboPairFormer single-reconstruction backward (dQ/dK/dV/dBias)");
}
