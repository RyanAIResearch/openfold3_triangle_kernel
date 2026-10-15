// SPDX-License-Identifier: Apache-2.0
// TurboPairFormer split-backward dK/dV with runtime shape dispatch.
// These split kernels remain available to direct callers. The public facade
// computes all four gradients with the single-reconstruction (SR) kernels
// in the sibling single-reconstruction backward source.
//
// One CTA owns one (batch, outer, head, 64-key block) and its dK/dV accumulators
// outright: dK[k] and dV[k] use column k of dS and P, respectively, so key-block
// ownership requires no atomics or dK/dV accumulation workspace. The optional
// delta_true output uses a separate FP32 partial workspace. The kernel
// streams query tiles in fixed ascending order,
// recomputes S^T = K@Q^T directly in the [key, query] accumulator layout
// (P^T and dS^T then follow elementwise from LSE/delta), and accumulates
//   dV += P^T @ dO   and   dK += dS^T @ Q.
// Only dK is scaled by 1/sqrt(D) at the epilogue.
//
// Two staging paths share the same body and gradient recurrence:
//   - TMA loads compact swizzled Q/dO/K/V tiles. WGMMA computes KQ^T and VdO^T;
//     mma.sync computes dK/dV. This path serves D = 16, 32, 64 when TMA plane
//     coordinates fit int32.
//   - The fallback stages K/V with thread loads and Q/dO with cp.async, using
//     padded shared-memory rows and mma.sync for all matrix products.
// Both paths keep K/V fixed and alternate two Q/dO buffers over the query loop.
// P and dS are formed in FP32, then rounded to BF16 as gradient MMA operands;
// dK/dV accumulate in FP32 and are stored as BF16 after all queries finish.
//
// Optional delta_true is an additional output, not a replacement for the
// supplied delta used in dS. It folds FP32 P*dP over keys before P is rounded
// to BF16. No division by the recomputed sum(P) is performed. Its partials
// have flat physical layout [B, O, H, key_blocks, N]; the reducer emits
// [B, O, H, N] in a fixed ascending key-block order.
//
// Internal CUDA ABI layouts:
//   q/k/v/out_dk/out_dv: [B, O, H, N, D] BF16; dO: [B, O, N, H, D] BF16
//   mask: [B, O, 1, 1, N] FP32 additive; bias: [B, 1, H, N, N] BF16 additive
//   logsumexp/delta: [B, O, H, N] FP32
// B = batch, O = independent outer planes, H = heads, N = query/key length.
// io_head_dim is the tensor's D; HeadDim is the storage width 16/32/64/128.
// Extra storage channels are zero-filled. Partial sequence tiles retain load
// and contribution guards; the public facade applies its own shape limits.
//
// Reading guide:
//   - run_dkdv validates tensors and selects storage width and TMA eligibility.
//   - dispatch_sequence / launch select tail handling and launch geometry.
//   - turbopairformer_triangle_dkdv_body contains the labeled execution stages.
//   - The delta_true entry point adds a fixed-order reduction after that body.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kKeyRowsPerWarp = 16;
constexpr int kKeyBlock = kWarpsPerBlock * kKeyRowsPerWarp;  // 64 keys
constexpr int kQueryTile = 32;

// Keep launch-grid bounds separate from the TMA coordinate bound.
constexpr int kMaxGridYZ = 65535;
constexpr long long kMaxGridX = 2147483647LL;
constexpr long long kMaxTmaPlanes = 2147483647LL;

// -----------------------------------------------------------------------------
// Matrix fragments and shared-memory copies
// -----------------------------------------------------------------------------

// Four FP32 values owned by one thread in an m16n8 accumulator tile.
// r = lane_id / 4 and c = 2 * (lane_id % 4):
//   values[0], values[1] -> (r,     c), (r,     c + 1)
//   values[2], values[3] -> (r + 8, c), (r + 8, c + 1)
// Rows are this warp's 16 keys. Columns are queries in score_fragments and
// dp_fragments, then head channels in dk_accumulators and dv_accumulators.
struct Accumulator {
  float values[4];
};

// Convert an existing shared-memory pointer to the address used by PTX.
__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_m16n8k16(
    Accumulator& d,
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d.values[0]), "+f"(d.values[1]), "+f"(d.values[2]), "+f"(d.values[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Load a 16x16 A tile into four packed BF16 registers per thread.
// row_stride counts BF16 elements between row starts, including padding.
__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile,
    int row_stride,
    int column_start,
    int lane_id,
    uint32_t (&fragment)[4]) {
  const int row = lane_id & 15;
  const int column = column_start + ((lane_id >> 4) << 3);
  const uint32_t shared_address = shared_u32(tile + row * row_stride + column);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
               : "r"(shared_address));
}

// Load Q or dO for KQ^T or VdO^T: contraction over 16 head channels.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile,
    int row_stride,
    int row_start,
    int dim_start,
    int lane_id,
    uint32_t (&fragment)[2]) {
  const int row = row_start + (lane_id & 7);
  const int col = dim_start + (((lane_id >> 3) & 1) << 3);
  const uint32_t shared_address = shared_u32(tile + row * row_stride + col);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(fragment[0]), "=r"(fragment[1])
               : "r"(shared_address));
}

// Load Q or dO for dK/dV: contraction over 16 query rows, 8 output channels.
__device__ __forceinline__ void ldmatrix_b_rowtrans(
    const __nv_bfloat16* tile,
    int row_stride,
    int row_start,
    int dim_start,
    int lane_id,
    uint32_t (&fragment)[2]) {
  const int row = row_start + (lane_id & 15);
  const uint32_t shared_address = shared_u32(tile + row * row_stride + dim_start);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(fragment[0]), "=r"(fragment[1])
               : "r"(shared_address));
}

__device__ __forceinline__ void cp_async_16(__nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// -----------------------------------------------------------------------------
// TMA staging and WGMMA helpers for compact swizzled tiles
// -----------------------------------------------------------------------------
// TMA requires io_head_dim == HeadDim here. Its compact tiles have HeadDim
// elements per row; the swizzle changes each element's shared-memory address.

__device__ __forceinline__ void dkdv_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void dkdv_mbarrier_init(uint64_t* barrier, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(shared_u32(barrier)), "r"(count));
}

__device__ __forceinline__ void dkdv_mbarrier_expect_tx(uint64_t* barrier, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(shared_u32(barrier)),
      "r"(bytes));
}

__device__ __forceinline__ void dkdv_mbarrier_wait_parity(uint64_t* barrier, uint32_t parity) {
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "  wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "  @!done bra wait;\n"
      "}\n" ::"r"(shared_u32(barrier)),
      "r"(parity));
}

__device__ __forceinline__ void dkdv_tma_load_3d(
    const CUtensorMap* map,
    __nv_bfloat16* dst,
    uint64_t* barrier,
    int coord_row,
    int coord_plane) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(0),
      "r"(coord_row),
      "r"(coord_plane),
      "r"(shared_u32(barrier)));
}

__device__ __forceinline__ void dkdv_tma_load_4d(
    const CUtensorMap* map,
    __nv_bfloat16* dst,
    uint64_t* barrier,
    int coord_head,
    int coord_row,
    int coord_plane) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(0),
      "r"(coord_head),
      "r"(coord_row),
      "r"(coord_plane),
      "r"(shared_u32(barrier)));
}

// A/B-operand fragment loads over a COMPACT swizzled tile (stride ==
// HeadDim elements). Atom XOR = ((row*HeadDim) >> 6) & (HeadDim/8 - 1).
template <int HeadDim>
__device__ __forceinline__ int dkdv_swz_col(int row, int col) {
  constexpr int kAtoms = HeadDim / 8;
  const int atom = (col >> 3) ^ (((row * HeadDim) >> 6) & (kAtoms - 1));
  return atom * 8 + (col & 7);
}

template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_rowtrans_sw(
    const __nv_bfloat16* tile,
    int row_start,
    int dim_start,
    int lane_id,
    uint32_t (&fragment)[2]) {
  const int row = row_start + (lane_id & 15);
  const uint32_t shared_address =
      shared_u32(tile + row * HeadDim + dkdv_swz_col<HeadDim>(row, dim_start));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(fragment[0]), "=r"(fragment[1])
               : "r"(shared_address));
}

// Four warps collectively compute a 64-key by 32-query tile over 16 channels.
// Both inputs come from shared-memory descriptors. Each thread owns four
// m16n8 accumulator fragments with the same layout as mma.sync above.
__device__ __forceinline__ void dkdv_ss_wgmma_m64n32k16(
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

__device__ __forceinline__ void dkdv_wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void dkdv_wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void dkdv_wgmma_wait_zero() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// Round two FP32 values to BF16 and pack their bits into one register.
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l))
         | (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

// -----------------------------------------------------------------------------
// Shared kernel body: resident K/V, streamed Q/dO, optional delta_true partials
// -----------------------------------------------------------------------------

template <
    int HeadDim,
    int StaticSequence,
    bool TileMajorGrid,
    bool EmitDeltaTrue,
    bool OuterEqSeq,
    bool UseTma>
__device__ __forceinline__ void turbopairformer_triangle_dkdv_body(
    const CUtensorMap* q_tensor_map,
    const CUtensorMap* d_output_tensor_map,
    const CUtensorMap* k_tensor_map,
    const CUtensorMap* v_tensor_map,
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
    float* __restrict__ delta_true_partials,
    int key_blocks,
    int runtime_sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    float attention_scale) {
  // A. Map this block to one key tile, head, and outer plane.
  // StaticSequence == -1 removes query/key tail guards for full tiles while
  // sequence stays a runtime value. StaticSequence == 0 retains the guards.
  const int sequence = StaticSequence > 0 ? StaticSequence : runtime_sequence;
  constexpr bool kNoPartialTiles = StaticSequence != 0;
  constexpr int kPaddedRowStride = HeadDim + 8;
  // TMA stages compact swizzled Q/dO; cp.async uses padded row strides.
  constexpr int kQueryOperandStride = UseTma ? HeadDim : kPaddedRowStride;
  const int thread_id = threadIdx.x;
  const int lane_id = thread_id & (kWarpSize - 1);
  const int warp_id = thread_id / kWarpSize;
  const int lane_row = lane_id >> 2;
  const int lane_in_row_group = lane_id & 3;
  // Prefer key tiles on grid.x when grid.z can hold all B*O tasks; otherwise
  // swap these axes. Both layouts cover the same logical tiles.
  const int outer_extent = OuterEqSeq ? sequence : outer_count;
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / outer_extent;
  const int outer = outer_task - batch_index * outer_extent;
  const int key_block_start = (TileMajorGrid ? blockIdx.x : blockIdx.z) * kKeyBlock;
  const int head = blockIdx.y;
  const int warp_key_start = key_block_start + warp_id * kKeyRowsPerWarp;

  // Shared layout: resident K | V | two Q/dO buffers | optional delta scratch.
  // The base alignment also keeps swizzled tiles aligned for WGMMA descriptors.
  extern __shared__ __align__(1024) unsigned char shared_storage[];
  // TMA lane: shared COMPACT swizzled K/V tiles [64 x D] (whole key
  // block, loaded once). cp.async lane: per-warp padded K/V slices.
  constexpr int kKeyValueStride = UseTma ? HeadDim : kPaddedRowStride;
  __nv_bfloat16* shared_k_tile = reinterpret_cast<__nv_bfloat16*>(shared_storage);
  __nv_bfloat16* shared_v_tile = shared_k_tile + kKeyBlock * kKeyValueStride;
  __nv_bfloat16* shared_warp_k = UseTma ? shared_k_tile + warp_id * kKeyRowsPerWarp * HeadDim
                                        : reinterpret_cast<__nv_bfloat16*>(shared_storage)
                                              + warp_id * kKeyRowsPerWarp * kPaddedRowStride;
  __nv_bfloat16* shared_warp_v = UseTma ? shared_v_tile + warp_id * kKeyRowsPerWarp * HeadDim
                                        : reinterpret_cast<__nv_bfloat16*>(shared_storage)
                                              + kKeyBlock * kPaddedRowStride
                                              + warp_id * kKeyRowsPerWarp * kPaddedRowStride;
  __shared__ __align__(8) uint64_t qdo_barriers[2];
  __shared__ __align__(8) uint64_t kv_barrier;
  __nv_bfloat16* query_operand_buffers =
      reinterpret_cast<__nv_bfloat16*>(shared_storage) + 2 * kKeyBlock * kKeyValueStride;
  constexpr int kQueryBufferElements = 2 * kQueryTile * kQueryOperandStride;  // q + do
  // Optional delta_true scratch alternates [warp, query, lane_row] buffers.
  // Each cell holds one thread's P*dP sum over its two key rows. The next
  // query iteration folds these cells after its existing loop-head barrier;
  // only the final tile needs a separate barrier to publish its partials.
  constexpr int kDeltaScratchStride = 9;  // Eight lane rows plus one padding element.
  constexpr int kDeltaScratchWarp = kQueryTile * kDeltaScratchStride;
  constexpr int kDeltaScratchParity = kWarpsPerBlock * kDeltaScratchWarp;
  float* shared_delta_partials =
      reinterpret_cast<float*>(query_operand_buffers + 2 * kQueryBufferElements);

  const int64_t plane = static_cast<int64_t>(batch_index) * outer_extent + outer;
  const int64_t stats_base = (plane * heads + head) * static_cast<int64_t>(sequence);
  const int64_t head_base = stats_base * io_head_dim;
  // d_output arrives in the public [B, O, N, H, io_head_dim] layout.
  const int64_t dout_plane_base = plane * static_cast<int64_t>(sequence) * heads * io_head_dim
                                  + static_cast<int64_t>(head) * io_head_dim;
  const int64_t mask_base = plane * sequence;
  const int64_t bias_base = (static_cast<int64_t>(batch_index) * heads + head)
                            * static_cast<int64_t>(sequence) * sequence;
  const int64_t delta_true_base =
      EmitDeltaTrue
          ? stats_base * key_blocks + static_cast<int64_t>(key_block_start / kKeyBlock) * sequence
          : 0;

  // Fixed fold of one tile's staged scalars: per warp
  // ((g0+g1)+(g2+g3))+((g4+g5)+(g6+g7)), then (w0+w1)+(w2+w3); one
  // writer per (query, key-block) workspace cell.
  const auto fold_delta_tile_partials = [&](int fold_query_start, int parity) {
    if (thread_id < kQueryTile) {
      const int query_in_tile = thread_id;
      const float* base_slot = shared_delta_partials + parity * kDeltaScratchParity
                               + query_in_tile * kDeltaScratchStride;
      float warp_sums[kWarpsPerBlock];
#pragma unroll
      for (int source = 0; source < kWarpsPerBlock; ++source) {
        const float* slot = base_slot + source * kDeltaScratchWarp;
        const float left = __fadd_rn(__fadd_rn(slot[0], slot[1]), __fadd_rn(slot[2], slot[3]));
        const float right = __fadd_rn(__fadd_rn(slot[4], slot[5]), __fadd_rn(slot[6], slot[7]));
        warp_sums[source] = __fadd_rn(left, right);
      }
      const int query = fold_query_start + query_in_tile;
      if (kNoPartialTiles || query < sequence) {
        delta_true_partials[delta_true_base + query] =
            __fadd_rn(__fadd_rn(warp_sums[0], warp_sums[1]), __fadd_rn(warp_sums[2], warp_sums[3]));
      }
    }
  };

  // B. Load K/V once. TMA keeps them as shared operands; the fallback also
  // loads their per-warp register fragments before the query loop.
  if constexpr (UseTma) {
    // Whole-key-block K/V arrive by two bulk loads (OOB rows zero-fill;
    // guards keep them inert exactly like the clamped loads).
    if (thread_id == 0) {
      dkdv_mbarrier_init(&kv_barrier, 1);
      dkdv_mbarrier_expect_tx(
          &kv_barrier,
          2 * kKeyBlock * HeadDim * static_cast<int>(sizeof(__nv_bfloat16)));
      dkdv_tma_load_3d(
          k_tensor_map,
          shared_k_tile,
          &kv_barrier,
          key_block_start,
          static_cast<int>(plane * heads + head));
      dkdv_tma_load_3d(
          v_tensor_map,
          shared_v_tile,
          &kv_barrier,
          key_block_start,
          static_cast<int>(plane * heads + head));
    }
    __syncthreads();
    dkdv_mbarrier_wait_parity(&kv_barrier, 0U);
    dkdv_proxy_fence();
  } else {
    // Stage this warp's 16 K rows (clamped tails; contributions zeroed
    // later).
    for (int element = lane_id; element < kKeyRowsPerWarp * HeadDim; element += kWarpSize) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int global_key =
          kNoPartialTiles || warp_key_start + row < sequence ? warp_key_start + row : sequence - 1;
      const int64_t src = head_base + static_cast<int64_t>(global_key) * io_head_dim + col;
      shared_warp_k[row * kPaddedRowStride + col] =
          col < io_head_dim ? k[src] : __float2bfloat16_rn(0.0F);
      shared_warp_v[row * kPaddedRowStride + col] =
          col < io_head_dim ? v[src] : __float2bfloat16_rn(0.0F);
    }
    __syncwarp();
  }

  // cp.async lane keeps register A-fragments; the TMA lane feeds the
  // warpgroup MMA straight from the K/V descriptors.
  uint32_t k_fragments[UseTma ? 1 : HeadDim / 16][4];
  uint32_t v_fragments[UseTma ? 1 : HeadDim / 16][4];
  if constexpr (!UseTma) {
#pragma unroll
    for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
      ldmatrix_a16x16(
          shared_warp_k,
          kPaddedRowStride,
          channel_chunk * 16,
          lane_id,
          k_fragments[channel_chunk]);
      ldmatrix_a16x16(
          shared_warp_v,
          kPaddedRowStride,
          channel_chunk * 16,
          lane_id,
          v_fragments[channel_chunk]);
    }
  }

  // C. Initialize FP32 dK/dV sums. These registers survive every query tile.
  Accumulator dv_accumulators[HeadDim / 8];
  Accumulator dk_accumulators[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    dv_accumulators[tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
    dk_accumulators[tile] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const int key_row_low = warp_key_start + lane_row;
  const int key_row_high = key_row_low + 8;

  // D. Alternate two Q/dO buffers. UseTma chooses descriptor bulk loads;
  // otherwise threads issue 16-byte cp.async copies into padded rows.
  constexpr int kChunksPerRow = HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
  const auto issue_query_tile = [&](int query_start, int buffer) {
    __nv_bfloat16* staging_q_tile = query_operand_buffers + buffer * kQueryBufferElements;
    __nv_bfloat16* staging_d_output_tile = staging_q_tile + kQueryTile * kQueryOperandStride;
    if constexpr (UseTma) {
      if (thread_id == 0) {
        dkdv_mbarrier_expect_tx(
            &qdo_barriers[buffer],
            2 * kQueryTile * HeadDim * static_cast<int>(sizeof(__nv_bfloat16)));
        dkdv_tma_load_3d(
            q_tensor_map,
            staging_q_tile,
            &qdo_barriers[buffer],
            query_start,
            static_cast<int>(plane * heads + head));
        dkdv_tma_load_4d(
            d_output_tensor_map,
            staging_d_output_tile,
            &qdo_barriers[buffer],
            head,
            query_start,
            static_cast<int>(plane));
      }
      return;
    }
    for (int chunk = thread_id; chunk < kQueryTile * kChunksPerRow; chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * 8;
      const int global_query =
          kNoPartialTiles || query_start + row < sequence ? query_start + row : sequence - 1;
      if (col < io_head_dim) {
        const int64_t src = head_base + static_cast<int64_t>(global_query) * io_head_dim + col;
        const int64_t dout_src =
            dout_plane_base + static_cast<int64_t>(global_query) * heads * io_head_dim + col;
        cp_async_16(staging_q_tile + row * kPaddedRowStride + col, q + src);
        cp_async_16(staging_d_output_tile + row * kPaddedRowStride + col, d_output + dout_src);
      } else {
        *reinterpret_cast<uint4*>(staging_q_tile + row * kPaddedRowStride + col) =
            uint4{0U, 0U, 0U, 0U};
        *reinterpret_cast<uint4*>(staging_d_output_tile + row * kPaddedRowStride + col) =
            uint4{0U, 0U, 0U, 0U};
      }
    }
    cp_async_commit_group();
  };
  if constexpr (UseTma) {
    if (thread_id == 0) {
      dkdv_mbarrier_init(&qdo_barriers[0], 1);
      dkdv_mbarrier_init(&qdo_barriers[1], 1);
    }
    __syncthreads();  // barrier-init visibility before the first issue
  }
  issue_query_tile(0, 0);
#pragma unroll 1
  for (int query_start = 0; query_start < sequence; query_start += kQueryTile) {
    const int tile_index = query_start / kQueryTile;
    const int buffer = tile_index & 1;
    __nv_bfloat16* shared_q_tile = query_operand_buffers + buffer * kQueryBufferElements;
    __nv_bfloat16* shared_d_output_tile = shared_q_tile + kQueryTile * kQueryOperandStride;
    if constexpr (UseTma) {
      dkdv_mbarrier_wait_parity(
          &qdo_barriers[buffer],
          static_cast<uint32_t>((tile_index >> 1) & 1));
      dkdv_proxy_fence();
      __syncthreads();
    } else {
      cp_async_wait_all();
      __syncthreads();
    }
    if (query_start + kQueryTile < sequence) {
      issue_query_tile(query_start + kQueryTile, buffer ^ 1);
    }
    if constexpr (EmitDeltaTrue) {
      // The loop-head barrier makes the previous tile's partials visible.
      // The first warp folds them into the per-key-block workspace.
      if (query_start > 0) {
        fold_delta_tile_partials(query_start - kQueryTile, buffer ^ 1);
      }
    }

    // E. Compute S^T = KQ^T and dP^T = VdO^T in [key, query] order.
    // Each warp receives 16 keys by 32 queries, with four values per fragment.
    Accumulator score_fragments[kQueryTile / 8];
    Accumulator dp_fragments[kQueryTile / 8];
    // Per-thread P*dP partials for this tile. HeadDim == 16 stages them all
    // after the elementwise loop; other widths stage each column immediately.
    // Historical tuning (main 98eff406) found D16 issue-bound, favoring stores
    // outside the exp stream; memory-bound D32 favored per-column stores.
    // Both placements keep the same values and fixed fold order.
    constexpr bool kBunchedDeltaStores = HeadDim == 16;
    float weighted_dp_pairs[kQueryTile / 8][2];
    if constexpr (EmitDeltaTrue) {
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
        weighted_dp_pairs[column][0] = 0.0F;
        weighted_dp_pairs[column][1] = 0.0F;
      }
    }
    if constexpr (UseTma) {
      // WGMMA reads the full 64-key K/V tiles and 32-query Q/dO tiles from
      // shared memory. Its fragments match the per-warp mma.sync layout.
      constexpr uint64_t kSsMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
      constexpr uint64_t kSsLbo = 64;
      constexpr uint64_t kSsSbo = HeadDim;
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
        score_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
      dkdv_wgmma_fence();
#pragma unroll
      for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
        const uint64_t channel_byte_offset = static_cast<uint64_t>(channel_chunk) * 32;
        const uint64_t k_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_k_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t q_descriptor =
            ((static_cast<uint64_t>(shared_u32(shared_q_tile)) + channel_byte_offset) >> 4)
            | (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        dkdv_ss_wgmma_m64n32k16(score_fragments, k_descriptor, q_descriptor, channel_chunk != 0);
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
        dkdv_ss_wgmma_m64n32k16(
            dp_fragments,
            v_descriptor,
            d_output_descriptor,
            channel_chunk != 0);
      }
      dkdv_wgmma_commit();
      dkdv_wgmma_wait_zero();
    } else {
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
        score_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp_fragments[column] = Accumulator{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
        for (int channel_chunk = 0; channel_chunk < HeadDim / 16; ++channel_chunk) {
          uint32_t b_fragment[2];
          ldmatrix_b_dimpair(
              shared_q_tile,
              kPaddedRowStride,
              column * 8,
              channel_chunk * 16,
              lane_id,
              b_fragment);
          mma_m16n8k16(score_fragments[column], k_fragments[channel_chunk], b_fragment);
          ldmatrix_b_dimpair(
              shared_d_output_tile,
              kPaddedRowStride,
              column * 8,
              channel_chunk * 16,
              lane_id,
              b_fragment);
          mma_m16n8k16(dp_fragments[column], v_fragments[channel_chunk], b_fragment);
        }
      }
    }

    // F. Reconstruct P^T = exp(S^T*attention_scale + mask + bias - LSE[q])
    // and dS^T = P^T * (dP^T - delta[q]) in FP32. Columns are queries.
    // Reuse score_fragments for dS^T and dp_fragments for P^T. The optional
    // P*dP sum must read the old dP before its register is overwritten.
    // HeadDim == 16 factors the bias row address into tile and element
    // offsets; the other widths use a direct per-element address.
    // Historical tuning favored fewer address operations for issue-bound D16;
    // memory-bound D32 scheduled better with direct addresses (main 98eff406).
    constexpr bool kReduceBiasAddress = HeadDim == 16;
    const int64_t bias_query_base =
        kReduceBiasAddress ? bias_base + static_cast<int64_t>(query_start) * sequence : bias_base;
    const int pair_row_offset = kReduceBiasAddress ? lane_in_row_group * 2 * sequence : 0;
#pragma unroll
    for (int column = 0; column < kQueryTile / 8; ++column) {
      const int column_row_offset = kReduceBiasAddress ? column * 8 * sequence : 0;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key_row = half == 0 ? key_row_low : key_row_high;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query = query_start + column * 8 + lane_in_row_group * 2 + pair;
          const int index = half * 2 + pair;
          float score_gradient = 0.0F;
          float probability = 0.0F;
          if (kNoPartialTiles || (key_row < sequence && query < sequence)) {
            const int64_t stats = stats_base + query;
            const float value =
                score_fragments[column].values[index] * attention_scale + mask[mask_base + key_row]
                + __bfloat162float(
                    bias
                        [kReduceBiasAddress
                             ? bias_query_base + column_row_offset + pair_row_offset
                                   + pair * sequence + key_row
                             : bias_query_base + static_cast<int64_t>(query) * sequence + key_row]);
            const float row_logsumexp = logsumexp[stats];
            probability = row_logsumexp == -std::numeric_limits<float>::infinity()
                              ? 0.0F
                              : __expf(value - row_logsumexp);
            score_gradient = probability * (dp_fragments[column].values[index] - delta[stats]);
          }
          score_fragments[column].values[index] = score_gradient;
          if constexpr (EmitDeltaTrue) {
            // w * dP with the pre-overwrite dP; invalid pairs add zeros.
            weighted_dp_pairs[column][pair] += probability * dp_fragments[column].values[index];
          }
          dp_fragments[column].values[index] = probability;
        }
      }
      if constexpr (EmitDeltaTrue && !kBunchedDeltaStores) {
        // Single-writer per-column stores of this lane's two-key partials;
        // the fold happens behind the next tile's loop-head barrier.
        const int parity = (query_start / kQueryTile) & 1;
        float* shared_warp_delta_partials =
            shared_delta_partials + parity * kDeltaScratchParity + warp_id * kDeltaScratchWarp;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query_in_tile = column * 8 + lane_in_row_group * 2 + pair;
          shared_warp_delta_partials[query_in_tile * kDeltaScratchStride + lane_row] =
              weighted_dp_pairs[column][pair];
        }
      }
    }
    if constexpr (EmitDeltaTrue && kBunchedDeltaStores) {
      // Single-writer bunched stores of this lane's two-key partials; the
      // fold happens behind the next tile's loop-head barrier.
      const int parity = (query_start / kQueryTile) & 1;
      float* shared_warp_delta_partials =
          shared_delta_partials + parity * kDeltaScratchParity + warp_id * kDeltaScratchWarp;
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query_in_tile = column * 8 + lane_in_row_group * 2 + pair;
          shared_warp_delta_partials[query_in_tile * kDeltaScratchStride + lane_row] =
              weighted_dp_pairs[column][pair];
        }
      }
    }

    // G. Round P/dS to BF16 and accumulate dV += P^T dO, dK += dS^T Q.
    // The FP32 sums persist across tiles; only these MMA operands are rounded.
#pragma unroll
    for (int chunk = 0; chunk < kQueryTile / 16; ++chunk) {
      uint32_t probability_fragment[4];
      uint32_t score_gradient_fragment[4];
      probability_fragment[0] =
          pack_bf16_pair(dp_fragments[chunk * 2].values[0], dp_fragments[chunk * 2].values[1]);
      probability_fragment[1] =
          pack_bf16_pair(dp_fragments[chunk * 2].values[2], dp_fragments[chunk * 2].values[3]);
      probability_fragment[2] = pack_bf16_pair(
          dp_fragments[chunk * 2 + 1].values[0],
          dp_fragments[chunk * 2 + 1].values[1]);
      probability_fragment[3] = pack_bf16_pair(
          dp_fragments[chunk * 2 + 1].values[2],
          dp_fragments[chunk * 2 + 1].values[3]);
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
        uint32_t b_fragment[2];
        if constexpr (UseTma) {
          ldmatrix_b_rowtrans_sw<HeadDim>(
              shared_d_output_tile,
              chunk * 16,
              tile * 8,
              lane_id,
              b_fragment);
        } else {
          ldmatrix_b_rowtrans(
              shared_d_output_tile,
              kPaddedRowStride,
              chunk * 16,
              tile * 8,
              lane_id,
              b_fragment);
        }
        mma_m16n8k16(dv_accumulators[tile], probability_fragment, b_fragment);
        if constexpr (UseTma) {
          ldmatrix_b_rowtrans_sw<HeadDim>(shared_q_tile, chunk * 16, tile * 8, lane_id, b_fragment);
        } else {
          ldmatrix_b_rowtrans(
              shared_q_tile,
              kPaddedRowStride,
              chunk * 16,
              tile * 8,
              lane_id,
              b_fragment);
        }
        mma_m16n8k16(dk_accumulators[tile], score_gradient_fragment, b_fragment);
      }
    }
  }

  if constexpr (EmitDeltaTrue) {
    // The last tile's scalars have no following loop-head barrier.
    __syncthreads();
    const int last_query_start = ((sequence + kQueryTile - 1) / kQueryTile - 1) * kQueryTile;
    fold_delta_tile_partials(last_query_start, (last_query_start / kQueryTile) & 1);
  }

  // H. All queries are consumed. Scale dK once, store both gradients as BF16,
  // and skip tail key rows and any padded storage channels.
  const int64_t out_base = head_base + static_cast<int64_t>(warp_key_start) * io_head_dim;
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = lane_row + half * 8;
      if (!kNoPartialTiles && warp_key_start + row >= sequence) {
        continue;
      }
      const int column = tile * 8 + lane_in_row_group * 2;
      if (column >= io_head_dim) {
        continue;
      }
      out_dv[out_base + static_cast<int64_t>(row) * io_head_dim + column] =
          __float2bfloat16_rn(dv_accumulators[tile].values[half * 2]);
      out_dv[out_base + static_cast<int64_t>(row) * io_head_dim + column + 1] =
          __float2bfloat16_rn(dv_accumulators[tile].values[half * 2 + 1]);
      out_dk[out_base + static_cast<int64_t>(row) * io_head_dim + column] =
          __float2bfloat16_rn(dk_accumulators[tile].values[half * 2] * attention_scale);
      out_dk[out_base + static_cast<int64_t>(row) * io_head_dim + column + 1] =
          __float2bfloat16_rn(dk_accumulators[tile].values[half * 2 + 1] * attention_scale);
    }
  }
}

// -----------------------------------------------------------------------------
// Kernel entry points and optional delta_true reduction
// -----------------------------------------------------------------------------
// Both entries inline the same body. EmitDeltaTrue=false removes the optional
// scratch writes and folds; the plain launch allocates no delta_true scratch.
template <int HeadDim, int StaticSequence, bool TileMajorGrid, bool OuterEqSeq, bool UseTma>
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_dkdv_kernel(
    const __grid_constant__ CUtensorMap q_tensor_map,
    const __grid_constant__ CUtensorMap d_output_tensor_map,
    const __grid_constant__ CUtensorMap k_tensor_map,
    const __grid_constant__ CUtensorMap v_tensor_map,
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
    int runtime_sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    float attention_scale) {
  turbopairformer_triangle_dkdv_body<
      HeadDim, StaticSequence, TileMajorGrid, false, OuterEqSeq, UseTma>(
      &q_tensor_map, &d_output_tensor_map, &k_tensor_map, &v_tensor_map, q, k, v, d_output,
      logsumexp, delta, mask, bias, out_dk, out_dv, nullptr, 0, runtime_sequence, outer_count,
      heads, batch, io_head_dim, attention_scale);
}

template <int HeadDim, int StaticSequence, bool TileMajorGrid, bool OuterEqSeq, bool UseTma>
__global__ __launch_bounds__(kThreads) void turbopairformer_triangle_dkdv_delta_true_kernel(
    const __grid_constant__ CUtensorMap q_tensor_map,
    const __grid_constant__ CUtensorMap d_output_tensor_map,
    const __grid_constant__ CUtensorMap k_tensor_map,
    const __grid_constant__ CUtensorMap v_tensor_map,
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
    float* __restrict__ delta_true_partials,
    int key_blocks,
    int runtime_sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    float attention_scale) {
  turbopairformer_triangle_dkdv_body<
      HeadDim, StaticSequence, TileMajorGrid, true, OuterEqSeq, UseTma>(
      &q_tensor_map, &d_output_tensor_map, &k_tensor_map, &v_tensor_map, q, k, v, d_output,
      logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials, key_blocks,
      runtime_sequence, outer_count, heads, batch, io_head_dim, attention_scale);
}

// Fold FP32 P*dP partials in ascending key-block order, with one writer per row.
// Input layout is [B,O,H,key_blocks,N]; output matches delta's [B,O,H,N].
// The fold starts at zero and does not divide by a recomputed sum(P).
__global__ void turbopairformer_delta_true_reduce_kernel(
    const float* __restrict__ partials,
    float* __restrict__ delta_true,
    int64_t rows,
    int key_blocks,
    int sequence) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t query = row % sequence;
  const int64_t row_head = row / sequence;
  const int64_t base = row_head * key_blocks * static_cast<int64_t>(sequence) + query;
  float weighted_dp = 0.0F;
#pragma unroll 1
  for (int block = 0; block < key_blocks; ++block) {
    weighted_dp = __fadd_rn(weighted_dp, partials[base + static_cast<int64_t>(block) * sequence]);
  }
  delta_true[row] = weighted_dp;
}

// -----------------------------------------------------------------------------
// Host launch configuration, tensor maps, and dispatch
// -----------------------------------------------------------------------------

// One selected host specialization sets its attribute and launches its kernel.
// The ordinary and delta_true entries keep their distinct CUDA argument lists.
template <
    int HeadDim,
    int StaticSequence,
    bool EmitDeltaTrue,
    bool TileMajorGrid,
    bool OuterEqSeq,
    bool UseTma>
void launch_dkdv_kernel(
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
    float* delta_true_partials,
    const CUtensorMap& q_tensor_map,
    const CUtensorMap& d_output_tensor_map,
    const CUtensorMap& k_tensor_map,
    const CUtensorMap& v_tensor_map,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    cudaStream_t stream,
    dim3 grid,
    int kSharedBytes,
    int key_blocks,
    float attention_scale) {
  if constexpr (EmitDeltaTrue) {
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        turbopairformer_triangle_dkdv_delta_true_kernel<
            HeadDim, StaticSequence, TileMajorGrid, OuterEqSeq, UseTma>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    TORCH_CHECK(
        attribute_status == cudaSuccess, "failed to opt in TurboPairFormer dK/dV shared memory: ",
        cudaGetErrorString(attribute_status));
    turbopairformer_triangle_dkdv_delta_true_kernel<
        HeadDim, StaticSequence, TileMajorGrid, OuterEqSeq, UseTma>
        <<<grid, kThreads, kSharedBytes, stream>>>(
            q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map,
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
            logsumexp.data_ptr<float>(), delta.data_ptr<float>(), mask.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()), delta_true_partials, key_blocks,
            sequence, outer_count, heads, batch, io_head_dim, attention_scale);
  } else {
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        turbopairformer_triangle_dkdv_kernel<
            HeadDim, StaticSequence, TileMajorGrid, OuterEqSeq, UseTma>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    TORCH_CHECK(
        attribute_status == cudaSuccess, "failed to opt in TurboPairFormer dK/dV shared memory: ",
        cudaGetErrorString(attribute_status));
    turbopairformer_triangle_dkdv_kernel<HeadDim, StaticSequence, TileMajorGrid, OuterEqSeq, UseTma>
        <<<grid, kThreads, kSharedBytes, stream>>>(
            q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map,
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
            logsumexp.data_ptr<float>(), delta.data_ptr<float>(), mask.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()), sequence, outer_count, heads,
            batch, io_head_dim, attention_scale);
  }
}

template <int HeadDim, int StaticSequence, bool EmitDeltaTrue, bool OuterEqSeq, bool UseTma>
void launch(
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
    float* delta_true_partials,
    const CUtensorMap& q_tensor_map,
    const CUtensorMap& d_output_tensor_map,
    const CUtensorMap& k_tensor_map,
    const CUtensorMap& v_tensor_map,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    cudaStream_t stream) {
  constexpr int kPaddedRowStride = HeadDim + 8;
  constexpr int kSharedBytes =
      ((2 * kKeyBlock + 4 * kQueryTile) * (UseTma ? HeadDim : kPaddedRowStride))
          * static_cast<int>(sizeof(__nv_bfloat16))
      + (EmitDeltaTrue ? 2 * kWarpsPerBlock * kQueryTile * 9 * static_cast<int>(sizeof(float)) : 0);
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(io_head_dim));
  const int key_blocks = (sequence + kKeyBlock - 1) / kKeyBlock;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * outer_count;
  const bool tile_major_grid = outer_tasks <= kMaxGridYZ;
  const dim3 grid =
      tile_major_grid
          ? dim3(
                static_cast<unsigned int>(key_blocks), static_cast<unsigned int>(heads),
                static_cast<unsigned int>(outer_tasks))
          : dim3(
                static_cast<unsigned int>(outer_tasks), static_cast<unsigned int>(heads),
                static_cast<unsigned int>(key_blocks));
  const auto launch_kernel =
      tile_major_grid
          ? &launch_dkdv_kernel<HeadDim, StaticSequence, EmitDeltaTrue, true, OuterEqSeq, UseTma>
          : &launch_dkdv_kernel<HeadDim, StaticSequence, EmitDeltaTrue, false, OuterEqSeq, UseTma>;
  launch_kernel(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
      q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count, heads,
      batch, io_head_dim, stream, grid, kSharedBytes, key_blocks, attention_scale);
}

// cuTensorMapEncodeTiled via the runtime entry point (no -lcuda link).
using DkdvTensorMapEncodeFn = CUresult (*)(
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

DkdvTensorMapEncodeFn dkdv_tensor_map_encode_fn() {
  static const DkdvTensorMapEncodeFn encode = [] {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    const cudaError_t status =
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &symbol, cudaEnableDefault, &query);
    TORCH_CHECK(
        status == cudaSuccess && query == cudaDriverEntryPointSuccess && symbol != nullptr,
        "cuTensorMapEncodeTiled is unavailable in this driver");
    return reinterpret_cast<DkdvTensorMapEncodeFn>(symbol);
  }();
  return encode;
}

CUtensorMapSwizzle dkdv_swizzle_for(int head_dim) {
  return head_dim == 16 ? CU_TENSOR_MAP_SWIZZLE_32B
                        : (head_dim == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B);
}

// q: internal [B, O, H, N, D] as [planes, N, D]; box one query tile.
CUtensorMap dkdv_make_q_map(
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
  const cuuint32_t box_dim[3] = {static_cast<cuuint32_t>(head_dim), kQueryTile, 1};
  const cuuint32_t element_stride[3] = {1, 1, 1};
  const CUresult result = dkdv_tensor_map_encode_fn()(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, source.data_ptr(), global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      dkdv_swizzle_for(head_dim), CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "dkdv q map encode failed: ", static_cast<int>(result));
  return tensor_map;
}

// dO: public [B, O, N, H, D] as 4-D [B*O, N, H, D]; the box gathers one
// head's H-strided rows into a compact tile.
CUtensorMap dkdv_make_do_map(
    const torch::Tensor& source,
    int head_dim,
    int sequence,
    int heads,
    int64_t batch_outer) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[4] = {
      static_cast<cuuint64_t>(head_dim), static_cast<cuuint64_t>(heads),
      static_cast<cuuint64_t>(sequence), static_cast<cuuint64_t>(batch_outer)};
  const cuuint64_t global_stride[3] = {
      static_cast<cuuint64_t>(head_dim) * sizeof(__nv_bfloat16),
      static_cast<cuuint64_t>(head_dim) * heads * sizeof(__nv_bfloat16),
      static_cast<cuuint64_t>(head_dim) * heads * sequence * sizeof(__nv_bfloat16)};
  const cuuint32_t box_dim[4] = {static_cast<cuuint32_t>(head_dim), 1, kQueryTile, 1};
  const cuuint32_t element_stride[4] = {1, 1, 1, 1};
  const CUresult result = dkdv_tensor_map_encode_fn()(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, source.data_ptr(), global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      dkdv_swizzle_for(head_dim), CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "dkdv dO map encode failed: ", static_cast<int>(result));
  return tensor_map;
}

// K/V: internal layout, box = one whole key block [D x kKeyBlock rows].
CUtensorMap dkdv_make_kv_block_map(
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
  const cuuint32_t box_dim[3] = {static_cast<cuuint32_t>(head_dim), kKeyBlock, 1};
  const cuuint32_t element_stride[3] = {1, 1, 1};
  const CUresult result = dkdv_tensor_map_encode_fn()(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, source.data_ptr(), global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      dkdv_swizzle_for(head_dim), CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "dkdv kv map encode failed: ", static_cast<int>(result));
  return tensor_map;
}

template <int HeadDim, bool EmitDeltaTrue, bool UseTma>
void dispatch_sequence(
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
    float* delta_true_partials,
    const CUtensorMap& q_tensor_map,
    const CUtensorMap& d_output_tensor_map,
    const CUtensorMap& k_tensor_map,
    const CUtensorMap& v_tensor_map,
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_head_dim,
    cudaStream_t stream) {
  // Full key/query tiles remove tail guards at compile time. Other lengths
  // use the guarded specialization; sequence remains a runtime argument.
  const bool outer_eq = outer_count == sequence;
  if (sequence % kKeyBlock == 0 && sequence % kQueryTile == 0) {
    if (outer_eq) {
      launch<HeadDim, -1, EmitDeltaTrue, true, UseTma>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    } else {
      launch<HeadDim, -1, EmitDeltaTrue, false, UseTma>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    }
  } else if (outer_eq) {
    launch<HeadDim, 0, EmitDeltaTrue, true, UseTma>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
        q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count, heads,
        batch, io_head_dim, stream);
  } else {
    launch<HeadDim, 0, EmitDeltaTrue, false, UseTma>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
        q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count, heads,
        batch, io_head_dim, stream);
  }
}

template <bool EmitDeltaTrue>
void run_dkdv(
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
    float* delta_true_partials) {
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
          && batch64 * outer64 <= kMaxGridX && (sequence64 + kKeyBlock - 1) / kKeyBlock <= kMaxGridYZ,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kBFloat16
            && tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/d_output/out_dk/out_dv must be contiguous BF16 tensors "
        "matching q on one device");
  };
  check_q_like(k);
  check_q_like(v);
  TORCH_CHECK(
      d_output.is_cuda() && d_output.device() == q.device() && d_output.dtype() == torch::kBFloat16
          && d_output.is_contiguous() && d_output.dim() == 5 && d_output.size(0) == q.size(0)
          && d_output.size(1) == q.size(1) && d_output.size(2) == q.size(3)
          && d_output.size(3) == q.size(2) && d_output.size(4) == q.size(4),
      "d_output must be contiguous BF16 [B,O,N,H,D] on q.device");
  check_q_like(out_dk);
  check_q_like(out_dv);
  const auto check_row_tensor = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() && tensor.dtype() == torch::kFloat32
            && tensor.is_contiguous() && tensor.numel() == q.numel() / head_dim,
        "logsumexp/delta must be contiguous FP32 row tensors on q.device");
  };
  check_row_tensor(logsumexp);
  check_row_tensor(delta);
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() && mask.dtype() == torch::kFloat32
          && mask.is_contiguous()
          && mask.sizes() == torch::IntArrayRef({batch, outer_count, 1, 1, sequence}),
      "mask must be contiguous FP32 [B,N,1,1,N] on q.device");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() && bias.dtype() == torch::kBFloat16
          && bias.is_contiguous()
          && bias.sizes() == torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
      "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int io_head_dim = head_dim;
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 128, "head_dim must be a multiple of 8 and <= 128, got ",
      head_dim);
  const int storage = head_dim <= 16 ? 16 : (head_dim <= 32 ? 32 : (head_dim <= 64 ? 64 : 128));
  // TMA stages Q/dO/K/V together. Compact tiles require the tensor width to
  // equal HeadDim, and their plane coordinates must fit int32. Other shapes
  // keep the padded shared-memory path, including all HeadDim == 128 cases.
  const int64_t planes_i64 = q.size(0) * q.size(1) * q.size(2);
  const bool use_tma = io_head_dim == storage && storage <= 64 && planes_i64 <= kMaxTmaPlanes;
  CUtensorMap q_tensor_map{};
  CUtensorMap d_output_tensor_map{};
  CUtensorMap k_tensor_map{};
  CUtensorMap v_tensor_map{};
  if (use_tma) {
    q_tensor_map = dkdv_make_q_map(q, io_head_dim, sequence, planes_i64);
    d_output_tensor_map =
        dkdv_make_do_map(d_output, io_head_dim, sequence, heads, q.size(0) * q.size(1));
    k_tensor_map = dkdv_make_kv_block_map(k, io_head_dim, sequence, planes_i64);
    v_tensor_map = dkdv_make_kv_block_map(v, io_head_dim, sequence, planes_i64);
  }
  switch (storage) {
  case 16:
    if (use_tma) {
      dispatch_sequence<16, EmitDeltaTrue, true>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    } else {
      dispatch_sequence<16, EmitDeltaTrue, false>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    }
    break;
  case 32:
    if (use_tma) {
      dispatch_sequence<32, EmitDeltaTrue, true>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    } else {
      dispatch_sequence<32, EmitDeltaTrue, false>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    }
    break;
  case 64:
    if (use_tma) {
      dispatch_sequence<64, EmitDeltaTrue, true>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    } else {
      dispatch_sequence<64, EmitDeltaTrue, false>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
          q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count,
          heads, batch, io_head_dim, stream);
    }
    break;
  case 128:
    dispatch_sequence<128, EmitDeltaTrue, false>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, delta_true_partials,
        q_tensor_map, d_output_tensor_map, k_tensor_map, v_tensor_map, sequence, outer_count, heads,
        batch, io_head_dim, stream);
    break;
  default:
    TORCH_CHECK(false, "unsupported storage lane for TurboPairFormer dK/dV");
  }
  {
    const cudaError_t launch_status = cudaGetLastError();
    TORCH_CHECK(
        launch_status == cudaSuccess,
        "TurboPairFormer dK/dV launch failed: ", cudaGetErrorString(launch_status));
  }
}

// -----------------------------------------------------------------------------
// C++ entry points: plain gradients or gradients plus an independent row output
// -----------------------------------------------------------------------------

void turbopairformer_triangle_dkdv_out(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& d_output,
    const torch::Tensor& logsumexp,
    const torch::Tensor& delta,
    const torch::Tensor& mask,
    const torch::Tensor& bias,
    torch::Tensor& out_dk,
    torch::Tensor& out_dv) {
  run_dkdv<false>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv, nullptr);
}

void turbopairformer_triangle_dkdv_delta_true_out(
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
    torch::Tensor& delta_true_partials,
    torch::Tensor& out_delta_true) {
  const int64_t sequence = q.size(3);
  const int64_t rows = q.numel() / q.size(4);
  const int64_t key_blocks = (sequence + kKeyBlock - 1) / kKeyBlock;
  const int64_t required_partials = rows * key_blocks;
  TORCH_CHECK(
      delta_true_partials.is_cuda() && delta_true_partials.device() == q.device()
          && delta_true_partials.dtype() == torch::kFloat32 && delta_true_partials.is_contiguous()
          && delta_true_partials.numel() >= required_partials,
      "delta_true_partials must be contiguous FP32 with at least "
      "rows*key_blocks elements on q.device");
  TORCH_CHECK(
      out_delta_true.is_cuda() && out_delta_true.device() == q.device()
          && out_delta_true.dtype() == torch::kFloat32 && out_delta_true.is_contiguous()
          && out_delta_true.numel() == rows,
      "out_delta_true must be a contiguous FP32 row tensor on q.device");
  run_dkdv<true>(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
      delta_true_partials.data_ptr<float>());
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kReduceThreads = 256;
  const int64_t reduce_blocks = (rows + kReduceThreads - 1) / kReduceThreads;
  TORCH_CHECK(
      reduce_blocks > 0 && reduce_blocks <= kMaxGridX,
      "delta_true reduce exceeds the CUDA grid-x contract");
  turbopairformer_delta_true_reduce_kernel<<<
      static_cast<unsigned int>(reduce_blocks), kReduceThreads, 0, stream>>>(
      delta_true_partials.data_ptr<float>(), out_delta_true.data_ptr<float>(), rows,
      static_cast<int>(key_blocks), static_cast<int>(sequence));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TurboPairFormer delta_true reduce launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "turbopairformer_triangle_dkdv_out", &turbopairformer_triangle_dkdv_out,
      "TurboPairFormer dK/dV (key-block ownership, runtime H/D/N)");
  module.def(
      "turbopairformer_triangle_dkdv_delta_true_out", &turbopairformer_triangle_dkdv_delta_true_out,
      "TurboPairFormer dK/dV plus the recomputed delta_true = sum(P*dP) row "
      "scalars (fixed-order key-block fold, zero atomics)");
}
