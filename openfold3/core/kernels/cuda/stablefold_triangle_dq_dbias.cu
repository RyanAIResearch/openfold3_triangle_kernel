// SPDX-License-Identifier: Apache-2.0
// StableFold fused dQ + dBias with bounded deterministic waves.
//
// Production leaf under the StableFold contract (one fixed reduction tree,
// tolerance vs PyTorch, replay-bitwise vs self):
//   * raw mma.sync + ldmatrix fragments instead of WMMA with FP32 shared
//     round-trips and CTA-wide barriers per tile;
//   * dS = P * (dP - delta) is pure elementwise on identical accumulator
//     layouts, so the derivative never leaves registers;
//   * one CTA owns one (query-tile, head, 4-outer leaf). K/V depend on the
//     outer, so each warp stages its own K/V tile privately. Bias is shared
//     by the four outers and is staged once into unused Q-tile padding;
//     steady-state next-tile staging is published by the existing trailing
//     CTA barrier around the cross-warp dBias combine;
//   * the CTA's dBias plane slice has exactly one writer -- zero cross-CTA
//     coordination, zero atomics, and one leaf launch;
//   * a bounded [leaf,batch,head,query-slab,key] workspace is reused. Query
//     slabs and batch chunks partition independent outputs; leaf waves retain
//     one global fixed left-associated chain in FP32.
//
// Fixed dBias tree: plane[leaf] = (w0 + w1) + (w2 + w3);
// dbias = ((plane0 + plane1) + plane2) + ... in global leaf order. Wave
// boundaries only spill/reload the FP32 root and do not change arithmetic.
//
// Layouts match stablefold_triangle_forward: q/k/v/dO [B,N,H,N,D] BF16,
// mask [B,N,1,1,N] FP32 additive, bias [B,1,H,N,N] BF16, lse/delta
// [B,N,H,N] FP32, out_dq [B,N,H,N,D] BF16, out_dbias [B,1,H,N,N] FP32.

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
constexpr int kQueryTile = 16;
constexpr int kQuerySubTiles = 2;
constexpr int kQueryBlock = kQueryTile * kQuerySubTiles;
constexpr int kKeyTile = 32;
constexpr int kOuterLeaf = 4;
constexpr int kPasses = kOuterLeaf / kWarpsPerBlock;
constexpr int kMaxPrefixRank = 16;

struct PrefixBroadcastPlan {
  int rank;
  int target_dims[kMaxPrefixRank];
  int bias_dims[kMaxPrefixRank];
  int64_t bias_strides[kMaxPrefixRank];
};


struct Acc {
  float v[4];
};

__device__ __forceinline__ void mma_m16n8k16(
    Acc& d, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile, int ld, int column_start, int lane,
    uint32_t (&frag)[4]) {
  const int row = lane & 15;
  const int column = column_start + ((lane >> 4) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + column);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// B for QK^T and dO@V^T: contraction over the contiguous dim axis.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile, int ld, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(tile + key * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

// B for dS@K: contraction over 16 keys, n = 8 dims.
__device__ __forceinline__ void ldmatrix_b_keytrans(
    const __nv_bfloat16* tile, int ld, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 15);
  const uint32_t addr = shared_u32(tile + key * ld + dim_start);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l)) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

// The +8 columns in each warp's Q tile are intentionally outside every
// ldmatrix operand.  Across four warps they form an exact 32x32 BF16 tile:
// warp w owns bias key columns [8*w, 8*w+8).  Staging the batch/head bias
// once per CTA removes four identical L1/TEX reads (one from each outer
// warp) without increasing dynamic shared memory or changing arithmetic.
template <int HeadDim>
__device__ __forceinline__ __nv_bfloat16* shared_bias_cell(
    __nv_bfloat16* sq, int query, int key) {
  constexpr int kQStride = HeadDim + 8;
  const int owner_warp = key >> 3;
  return sq + owner_warp * kQueryBlock * kQStride +
      query * kQStride + HeadDim + (key & 7);
}

template <
    int HeadDim,
    int StaticSequence,
    bool FullQueryTiles,
    bool TaskInGridZ,
    bool OuterEqSeq>
__global__ __launch_bounds__(kThreads) void stablefold_dq_dbias_leaf_kernel(
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
    int io_dim,
    float scale) {
  // StaticSequence == -1 is the production lane: sequence stays a pure
  // runtime parameter under the facade's pad-to-tile contract (every tile
  // divides exactly), so all tail guards are compile-time true and vanish
  // -- one kernel per head dim, any length, unguarded hot loop.
  // StaticSequence == 0 keeps the fully guarded fallback for direct ABI
  // callers with unpadded lengths.
  const int sequence =
      StaticSequence > 0 ? StaticSequence : runtime_sequence;
  constexpr int outer_leaf = kOuterLeaf;
  constexpr int passes = kPasses;
  constexpr bool kSequenceExact = StaticSequence != 0;
  constexpr int kQStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
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
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdo = sq + kWarpsPerBlock * kQueryBlock * kQStride;
  __nv_bfloat16* skv = sdo + kWarpsPerBlock * kQueryBlock * kQStride;
  float* sbias_tiles = reinterpret_cast<float*>(
      skv + kWarpsPerBlock * 2 * kKeyTile * kKVStride);

  __nv_bfloat16* my_q = sq + warp * kQueryBlock * kQStride;
  __nv_bfloat16* my_do = sdo + warp * kQueryBlock * kQStride;
  __nv_bfloat16* my_k = skv + warp * 2 * kKeyTile * kKVStride;
  __nv_bfloat16* my_v = my_k + kKeyTile * kKVStride;
  float* my_bias_tile = sbias_tiles + warp * kQueryBlock * kKeyTile;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) *
            heads +
        head) *
           static_cast<int64_t>(query_capacity) +
       query_local_start) *
      sequence;
  const int64_t bias_base =
      ((static_cast<int64_t>(batch_index) * heads + head) * sequence +
       query_start) *
      static_cast<int64_t>(sequence);

#pragma unroll 1
  for (int pass = 0; pass < passes; ++pass) {
    const int oc = OuterEqSeq ? sequence : outer_count;
    const int outer = leaf * outer_leaf + pass * kWarpsPerBlock + warp;
    // Outer is a pure plane count (decoupled from sequence); its tail guard
    // must stay a runtime compare in every lane -- values are unchanged when
    // outer_count is a leaf multiple, and arbitrary outer stays in bounds.
    const bool outer_valid = outer < oc;
    // Multiplier form keeps the hot-loop condition compile-foldable on the
    // exact lanes: valid outers multiply by 1.0f (bitwise identity), tail
    // outers zero their (finite, clamped-load) contributions exactly.
    const float outer_mult = outer_valid ? 1.0F : 0.0F;
    const int outer_load = outer_valid ? outer : oc - 1;
    const int64_t plane =
        static_cast<int64_t>(batch_index) * oc + outer_load;
    const int64_t row_stats_base =
        (plane * heads + head) * static_cast<int64_t>(sequence);
    const int64_t tensor_base = row_stats_base * io_dim;
    // d_output arrives in the public [B, O, N, H, io_dim] layout.
    const int64_t dout_plane_base =
        plane * static_cast<int64_t>(sequence) * heads * io_dim +
        static_cast<int64_t>(head) * io_dim;
    const int64_t mask_base = plane * sequence;

    for (int element = lane; element < kQueryBlock * HeadDim;
         element += kWarpSize) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int gq = FullQueryTiles || query_start + row < query_end
          ? query_start + row
          : sequence - 1;
      const int64_t src = tensor_base +
          static_cast<int64_t>(gq) * io_dim + col;
      const int64_t dout_src = dout_plane_base +
          static_cast<int64_t>(gq) * heads * io_dim + col;
      my_q[row * kQStride + col] =
          col < io_dim ? q[src] : __float2bfloat16_rn(0.0F);
      my_do[row * kQStride + col] =
          col < io_dim ? d_output[dout_src] : __float2bfloat16_rn(0.0F);
    }
    __syncwarp();

    float lse_pair[kQuerySubTiles][2];
    float delta_pair[kQuerySubTiles][2];
#pragma unroll
    for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int qrow = query_start + sub * kQueryTile + group + half * 8;
        const int64_t stats = row_stats_base +
            (FullQueryTiles || qrow < query_end ? qrow : sequence - 1);
        lse_pair[sub][half] = logsumexp[stats];
        delta_pair[sub][half] = delta[stats];
      }
    }

    Acc dq_acc[kQuerySubTiles][HeadDim / 8];
#pragma unroll
    for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        dq_acc[sub][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
    }

    const auto stage_key_and_bias = [&](int key_start) {
      // K/V are outer-dependent, so each warp retains its private tile.
      for (int element = lane; element < kKeyTile * HeadDim;
           element += kWarpSize) {
        const int row = element / HeadDim;
        const int col = element - row * HeadDim;
        const int gk = kSequenceExact || key_start + row < sequence
            ? key_start + row
            : sequence - 1;
        const int64_t src = tensor_base +
            static_cast<int64_t>(gk) * io_dim + col;
        my_k[row * kKVStride + col] =
            col < io_dim ? k[src] : __float2bfloat16_rn(0.0F);
        my_v[row * kKVStride + col] =
            col < io_dim ? v[src] : __float2bfloat16_rn(0.0F);
      }
      // Bias is shared by all four outer warps.  The full CTA cooperatively
      // stages it into otherwise-unused Q padding.
      for (int element = threadIdx.x; element < kQueryBlock * kKeyTile;
           element += kThreads) {
        const int row = element / kKeyTile;
        const int key = element - row * kKeyTile;
        const bool valid =
            (FullQueryTiles || query_start + row < query_end) &&
            (kSequenceExact || key_start + key < sequence);
        *shared_bias_cell<HeadDim>(sq, row, key) = valid
            ? bias[bias_base + static_cast<int64_t>(row) * sequence +
                   key_start + key]
            : __float2bfloat16_rn(0.0F);
      }
    };

    stage_key_and_bias(0);
    __syncthreads();

#pragma unroll 1
    for (int key_start = 0; key_start < sequence; key_start += kKeyTile) {
#pragma unroll
      for (int sub = 0; sub < kQuerySubTiles; ++sub) {
        uint32_t q_frags[HeadDim / 16][4];
        uint32_t do_frags[HeadDim / 16][4];
#pragma unroll
        for (int chain = 0; chain < HeadDim / 16; ++chain) {
          ldmatrix_a16x16(
              my_q + sub * kQueryTile * kQStride, kQStride, chain * 16,
              lane, q_frags[chain]);
          ldmatrix_a16x16(
              my_do + sub * kQueryTile * kQStride, kQStride, chain * 16,
              lane, do_frags[chain]);
        }

        Acc score[kKeyTile / 8];
        Acc dp[kKeyTile / 8];
#pragma unroll
        for (int column = 0; column < kKeyTile / 8; ++column) {
          score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
          dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
          for (int chain = 0; chain < HeadDim / 16; ++chain) {
            uint32_t b_frag[2];
            ldmatrix_b_dimpair(
                my_k, kKVStride, column * 8, chain * 16, lane, b_frag);
            mma_m16n8k16(score[column], q_frags[chain], b_frag);
            ldmatrix_b_dimpair(
                my_v, kKVStride, column * 8, chain * 16, lane, b_frag);
            mma_m16n8k16(dp[column], do_frags[chain], b_frag);
          }
        }

#pragma unroll
        for (int column = 0; column < kKeyTile / 8; ++column) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const float lse = lse_pair[sub][half];
            const float dlt = delta_pair[sub][half];
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              const int key = key_start + column * 8 + thread_in_group * 2 +
                  pair;
              const int qrow =
                  query_start + sub * kQueryTile + group + half * 8;
              const int index = half * 2 + pair;
              float d_score = 0.0F;
              if ((kSequenceExact || key < sequence) &&
                  (FullQueryTiles || qrow < query_end)) {
                const float value = score[column].v[index] * scale +
                    mask[mask_base + key] +
                    __bfloat162float(
                        *shared_bias_cell<HeadDim>(
                            sq,
                            sub * kQueryTile + group + half * 8,
                            column * 8 + thread_in_group * 2 + pair));
                const float probability =
                    lse == -std::numeric_limits<float>::infinity()
                    ? 0.0F
                    : __expf(value - lse);
                d_score = probability * (dp[column].v[index] - dlt);
              }
              if constexpr (!(OuterEqSeq && kSequenceExact)) {
                // Exact-tiles + outer==sequence lane covers every outer:
                // the tail multiplier is compile-time dead there.
                d_score *= outer_mult;
              }
              score[column].v[index] = d_score;
              my_bias_tile[(sub * kQueryTile + group + half * 8) * kKeyTile +
                           column * 8 + thread_in_group * 2 + pair] =
                  d_score;
            }
          }
        }

#pragma unroll
        for (int chunk = 0; chunk < kKeyTile / 16; ++chunk) {
          uint32_t ds_frag[4];
          ds_frag[0] = pack_bf16_pair(
              score[chunk * 2].v[0], score[chunk * 2].v[1]);
          ds_frag[1] = pack_bf16_pair(
              score[chunk * 2].v[2], score[chunk * 2].v[3]);
          ds_frag[2] = pack_bf16_pair(
              score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
          ds_frag[3] = pack_bf16_pair(
              score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
          for (int tile = 0; tile < HeadDim / 8; ++tile) {
            uint32_t k_frag[2];
            ldmatrix_b_keytrans(
                my_k, kKVStride, chunk * 16, tile * 8, lane, k_frag);
            mma_m16n8k16(dq_acc[sub][tile], ds_frag, k_frag);
          }
        }
      }

      // Cross-warp dBias combine in the fixed (w0+w1)+(w2+w3) order, then
      // the single-writer plane accumulation for this leaf.
      __syncthreads();
      const int64_t plane_tile_base = plane_base + key_start;
      for (int index = threadIdx.x; index < kQueryBlock * kKeyTile;
           index += kThreads) {
        const int row = index / kKeyTile;
        const int key = index - row * kKeyTile;
        const float left = __fadd_rn(
            sbias_tiles[row * kKeyTile + key],
            sbias_tiles[kQueryBlock * kKeyTile + row * kKeyTile + key]);
        const float right = __fadd_rn(
            sbias_tiles[2 * kQueryBlock * kKeyTile + row * kKeyTile + key],
            sbias_tiles[3 * kQueryBlock * kKeyTile + row * kKeyTile + key]);
        const float tile_sum = __fadd_rn(left, right);
        if ((!FullQueryTiles && query_start + row >= query_end) ||
            (!kSequenceExact && key_start + key >= sequence)) {
          continue;
        }
        __nv_bfloat16* destination =
            workspace + plane_tile_base +
            static_cast<int64_t>(row) * sequence + key;
        if (pass == 0) {
          *destination = __float2bfloat16_rn(tile_sum);
        } else {
          *destination = __float2bfloat16_rn(
              __fadd_rn(__bfloat162float(*destination), tile_sum));
        }
      }
      if (key_start + kKeyTile < sequence) {
        // The first CTA barrier above proves every warp has finished reading
        // the current K/V and bias tiles.  Prefetch the next operands while
        // the same threads finish their disjoint workspace writes; the
        // existing trailing barrier publishes both, so steady state gains no
        // additional CTA barrier.
        stage_key_and_bias(key_start + kKeyTile);
      }
      __syncthreads();
    }

    // dQ epilogue for this warp's outer: scale once, store BF16.
    const int64_t dq_row_base = tensor_base +
        static_cast<int64_t>(query_start) * io_dim;
    if (outer_valid) {
#pragma unroll
      for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = sub * kQueryTile + group + half * 8;
            if (!FullQueryTiles && query_start + row >= query_end) {
              continue;
            }
            const int column = tile * 8 + thread_in_group * 2;
            if (column >= io_dim) {
              continue;
            }
            out_dq[dq_row_base + static_cast<int64_t>(row) * io_dim +
                   column] =
                __float2bfloat16_rn(dq_acc[sub][tile].v[half * 2] * scale);
            out_dq[dq_row_base + static_cast<int64_t>(row) * io_dim +
                   column + 1] =
                __float2bfloat16_rn(
                    dq_acc[sub][tile].v[half * 2 + 1] * scale);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Warpgroup lane: one (64-query block, head, 4-outer leaf) per CTA. The four
// warps cooperate on one outer at a time instead of owning one outer each, so
// K/V/bias staging traffic and CTA barriers per query row halve while every
// arithmetic chain (S/dP mma, softmax, dS, dQ accumulation) keeps the exact
// per-(outer, row, key) instruction order of the legacy leaf kernel.  The
// dBias tree accumulates (o0+o1) and (o2+o3) in two resident FP32 tiles --
// the same cell values, rounding points, and workspace order as the legacy
// cross-warp combine.  Exact-tiles contract only: sequence % 64 == 0 and
// query slabs in 64-row blocks (the facade pad quantum); runtime outer tails
// keep the multiplier guard.
constexpr int kQueryBlockWg = 64;
// Two warpgroups per CTA: warpgroup 0 owns leaf slots {0,1}, warpgroup 1
// owns {2,3}.  Each warpgroup folds its slot pair into its own resident
// FP32 tree tile, so the (o0+o1)+(o2+o3) pairing runs concurrently instead
// of serially, and SM residency matches the legacy kernel (2 CTAs x 256
// threads = 16 warps) while staging traffic per query row stays halved.
constexpr int kThreadsWg = 2 * kThreads;
constexpr int kSlotsPerWarpgroup = 2;
// Padded shared strides: the FP32 tree at a raw kKeyTile stride wraps the
// 32 banks exactly (128B) so every row hits one bank; +1 float staggers
// rows across banks.  The BF16 bias slab gets +4 halves for the same
// reason (64B rows alias two bank halves).
constexpr int kTreeStride = kKeyTile + 1;
constexpr int kBiasStride = kKeyTile + 4;

// Compact swizzled tiles (stride == HeadDim elements) feed the ss-wgmma
// descriptors.  The universal swizzle law (dump-probe verified in the dkdv
// campaign): the 16B-atom XOR advances once per 128 bytes of tile offset.
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
    const __nv_bfloat16* tile, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 15);
  const uint32_t addr = shared_u32(
      tile + key * HeadDim + wg_swz_col<HeadDim>(key, dim_start));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

// ss m64n32k16 with both operands from compact swizzled-tile descriptors;
// the universal (LBO=64, SBO=HeadDim/16B, trans=0) encoding serves the
// A-m64 and B-k16n32 interpretations (probe: bitwise vs the mma.sync
// chain).  Accumulator layout equals the per-warp m16n8 fragments.
__device__ __forceinline__ void wg_ss_wgmma_m64n32k16(
    Acc (&o)[4], uint64_t descriptor_a, uint64_t descriptor_b,
    int accumulate) {
  asm volatile(
      "{\n"
      "  .reg .pred use_acc;\n"
      "  setp.ne.b32 use_acc, %18, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
      "  %16, %17, use_acc, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(o[0].v[0]), "+f"(o[0].v[1]), "+f"(o[0].v[2]), "+f"(o[0].v[3]),
        "+f"(o[1].v[0]), "+f"(o[1].v[1]), "+f"(o[1].v[2]), "+f"(o[1].v[3]),
        "+f"(o[2].v[0]), "+f"(o[2].v[1]), "+f"(o[2].v[2]), "+f"(o[2].v[3]),
        "+f"(o[3].v[0]), "+f"(o[3].v[1]), "+f"(o[3].v[2]), "+f"(o[3].v[3])
      : "l"(descriptor_a), "l"(descriptor_b), "r"(accumulate));
}

// Generic-proxy shared writes (plain stores, non-bulk cp.async) must be
// fenced into the async proxy before wgmma descriptor reads: the PTX
// ordering contract, not optional (racecheck cannot see it).
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

// minBlocksPerMultiprocessor=2 keeps ptxas at <=128 registers: without it
// the wgmma async live-ranges push allocation to 164 regs and occupancy
// collapses to one CTA per SM (measured).
template <int HeadDim, bool TaskInGridZ, bool OuterEqSeq>
__global__ __launch_bounds__(kThreadsWg, 2) void stablefold_dq_dbias_wg_kernel(
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
    int io_dim,
    float scale) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warpgroup = warp / kWarpsPerBlock;
  const int warp_in_group = warp & (kWarpsPerBlock - 1);
  const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
  const int local_batch = wave_task % active_batch;
  const int batch_index = batch_base + local_batch;
  const int wave_leaf = wave_task / active_batch;
  const int query_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
  const int query_local_start = query_block * kQueryBlockWg;
  const int query_start = query_base + query_local_start;
  const int head = blockIdx.y;
  const int leaf = leaf_base + wave_leaf;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;

  // Compact swizzled tiles (stride == HeadDim): the ss-wgmma descriptors
  // and their ldmatrix co-consumers share one layout.  1024-byte base
  // alignment keeps the written swizzle pattern congruent with the
  // hardware's descriptor view (the dkdv misalignment lesson).
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdo = sq + kOuterLeaf * kQueryBlockWg * HeadDim;
  __nv_bfloat16* skv = sdo + kOuterLeaf * kQueryBlockWg * HeadDim;
  __nv_bfloat16* sbias_tile = skv + kOuterLeaf * 2 * kKeyTile * HeadDim;
  float* smask_tiles = reinterpret_cast<float*>(
      sbias_tile + kQueryBlockWg * kBiasStride);
  float* sbias_acc = smask_tiles + kOuterLeaf * kKeyTile;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) *
            heads +
        head) *
           static_cast<int64_t>(query_capacity) +
       query_local_start) *
      sequence;
  const int64_t bias_base =
      ((static_cast<int64_t>(batch_index) * heads + head) * sequence +
       query_start) *
      static_cast<int64_t>(sequence);

  // Per-slot global bases are recomputed at each (infrequent) staging use:
  // four resident int64[4] arrays cost ~32 registers against the 128-cap
  // and buy nothing in the hot loop.
  const int oc = OuterEqSeq ? sequence : outer_count;
  const auto slot_plane = [&](int slot) -> int64_t {
    const int outer = leaf * kOuterLeaf + slot;
    const int outer_load = outer < oc ? outer : oc - 1;
    return static_cast<int64_t>(batch_index) * oc + outer_load;
  };
  float outer_mult[kOuterLeaf];
#pragma unroll
  for (int slot = 0; slot < kOuterLeaf; ++slot) {
    outer_mult[slot] = leaf * kOuterLeaf + slot < oc ? 1.0F : 0.0F;
  }

  // q/dO are key-invariant: stage all four outers once, block-cooperatively.
#pragma unroll 1
  for (int slot = 0; slot < kOuterLeaf; ++slot) {
    __nv_bfloat16* dst_q = sq + slot * kQueryBlockWg * HeadDim;
    __nv_bfloat16* dst_do = sdo + slot * kQueryBlockWg * HeadDim;
    const int64_t plane = slot_plane(slot);
    const int64_t tensor_base =
        (plane * heads + head) * static_cast<int64_t>(sequence) * io_dim;
    const int64_t dout_base =
        plane * static_cast<int64_t>(sequence) * heads * io_dim +
        static_cast<int64_t>(head) * io_dim;
    for (int element = threadIdx.x; element < kQueryBlockWg * HeadDim;
         element += kThreadsWg) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int gq = query_start + row;
      const int64_t src = tensor_base +
          static_cast<int64_t>(gq) * io_dim + col;
      const int64_t dout_src = dout_base +
          static_cast<int64_t>(gq) * heads * io_dim + col;
      const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
      dst_q[dst_index] =
          col < io_dim ? q[src] : __float2bfloat16_rn(0.0F);
      dst_do[dst_index] =
          col < io_dim ? d_output[dout_src] : __float2bfloat16_rn(0.0F);
    }
  }

  // Each warpgroup carries only its own slot pair's row statistics and dQ
  // accumulators; the local index stays a compile-time constant so the
  // state lives in registers.
  float lse_arr[kSlotsPerWarpgroup][2];
  float delta_arr[kSlotsPerWarpgroup][2];
#pragma unroll
  for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
    const int slot = warpgroup * kSlotsPerWarpgroup + local;
    const int64_t row_stats_base = (slot_plane(slot) * heads + head) *
        static_cast<int64_t>(sequence);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int qrow =
          query_start + warp_in_group * kQueryTile + group + half * 8;
      const int64_t stats = row_stats_base + qrow;
      lse_arr[local][half] = logsumexp[stats];
      delta_arr[local][half] = delta[stats];
    }
  }

  Acc dq_acc[kSlotsPerWarpgroup][HeadDim / 8];
#pragma unroll
  for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
      dq_acc[local][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
    }
  }

  const auto stage_kv_and_bias = [&](int key_start) {
#pragma unroll 1
    for (int slot = 0; slot < kOuterLeaf; ++slot) {
      __nv_bfloat16* dst_k = skv + slot * 2 * kKeyTile * HeadDim;
      __nv_bfloat16* dst_v = dst_k + kKeyTile * HeadDim;
      const int64_t tensor_base = (slot_plane(slot) * heads + head) *
          static_cast<int64_t>(sequence) * io_dim;
      for (int element = threadIdx.x; element < kKeyTile * HeadDim;
           element += kThreadsWg) {
        const int row = element / HeadDim;
        const int col = element - row * HeadDim;
        const int gk = key_start + row;
        const int64_t src = tensor_base +
            static_cast<int64_t>(gk) * io_dim + col;
        const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
        dst_k[dst_index] =
            col < io_dim ? k[src] : __float2bfloat16_rn(0.0F);
        dst_v[dst_index] =
            col < io_dim ? v[src] : __float2bfloat16_rn(0.0F);
      }
    }
    for (int element = threadIdx.x; element < kQueryBlockWg * kKeyTile;
         element += kThreadsWg) {
      const int row = element / kKeyTile;
      const int key = element - row * kKeyTile;
      sbias_tile[row * kBiasStride + key] =
          bias[bias_base + static_cast<int64_t>(row) * sequence +
               key_start + key];
    }
    // Additive mask depends only on (outer plane, key): one FP32 row per
    // slot replaces sixteen per-thread global reads per slot-tile.
    for (int element = threadIdx.x; element < kOuterLeaf * kKeyTile;
         element += kThreadsWg) {
      const int slot = element / kKeyTile;
      const int key = element - slot * kKeyTile;
      smask_tiles[element] =
          mask[slot_plane(slot) * sequence + key_start + key];
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
    // Fully unrolled so dq_acc/lse/delta local indices stay compile-time
    // constants -- a rolled loop demotes the accumulators to local memory
    // (the archived dynamic-indexed-state-array trap).  Each warpgroup
    // walks its own slot pair; the two pairs run concurrently.
#pragma unroll
    for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
      const int slot = warpgroup * kSlotsPerWarpgroup + local;
      const __nv_bfloat16* tile_q = sq + slot * kQueryBlockWg * HeadDim;
      const __nv_bfloat16* tile_do = sdo + slot * kQueryBlockWg * HeadDim;
      const __nv_bfloat16* tile_k = skv + slot * 2 * kKeyTile * HeadDim;
      const __nv_bfloat16* tile_v = tile_k + kKeyTile * HeadDim;

      // S = Q.K^T and dP = dO.V^T as chained ss m64n32k16 issued by this
      // warpgroup: A = the slot's 64-row q/dO tile, B = its 32-key K/V
      // tile, all via the universal descriptor encoding.  The accumulator
      // rows land on this warp's 16 query rows, matching the legacy
      // per-warp m16n8 fragment layout, so the elementwise recompute and
      // the dQ mma below are untouched.
      constexpr uint64_t kSsMode =
          HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
      constexpr uint64_t kSsLbo = 64;
      constexpr uint64_t kSsSbo = HeadDim;
      Acc score[kKeyTile / 8];
      Acc dp[kKeyTile / 8];
#pragma unroll
      for (int column = 0; column < kKeyTile / 8; ++column) {
        score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
      wg_wgmma_fence();
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
        const uint64_t desc_q =
            ((static_cast<uint64_t>(shared_u32(tile_q)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t desc_k =
            ((static_cast<uint64_t>(shared_u32(tile_k)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(score, desc_q, desc_k, chain != 0);
      }
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
        const uint64_t desc_do =
            ((static_cast<uint64_t>(shared_u32(tile_do)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t desc_v =
            ((static_cast<uint64_t>(shared_u32(tile_v)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(dp, desc_do, desc_v, chain != 0);
      }
      wg_wgmma_commit();
      wg_wgmma_wait_zero();

      float* tree = sbias_acc + warpgroup * (kQueryBlockWg * kTreeStride);
      const float slot_mult = outer_mult[slot];
      const float* slot_mask = smask_tiles + slot * kKeyTile;
#pragma unroll
      for (int column = 0; column < kKeyTile / 8; ++column) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float lse = lse_arr[local][half];
          const float dlt = delta_arr[local][half];
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const int tile_key = column * 8 + thread_in_group * 2 + pair;
            const int block_row =
                warp_in_group * kQueryTile + group + half * 8;
            const int index = half * 2 + pair;
            const float value = score[column].v[index] * scale +
                slot_mask[tile_key] +
                __bfloat162float(
                    sbias_tile[block_row * kBiasStride + tile_key]);
            const float probability =
                lse == -std::numeric_limits<float>::infinity()
                ? 0.0F
                : __expf(value - lse);
            float d_score = probability * (dp[column].v[index] - dlt);
            if constexpr (!OuterEqSeq) {
              d_score *= slot_mult;
            }
            score[column].v[index] = d_score;
            const int cell = block_row * kTreeStride + tile_key;
            if (local == 0) {
              tree[cell] = d_score;
            } else {
              tree[cell] = __fadd_rn(tree[cell], d_score);
            }
          }
        }
      }

#pragma unroll
      for (int chunk = 0; chunk < kKeyTile / 16; ++chunk) {
        uint32_t ds_frag[4];
        ds_frag[0] = pack_bf16_pair(
            score[chunk * 2].v[0], score[chunk * 2].v[1]);
        ds_frag[1] = pack_bf16_pair(
            score[chunk * 2].v[2], score[chunk * 2].v[3]);
        ds_frag[2] = pack_bf16_pair(
            score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
        ds_frag[3] = pack_bf16_pair(
            score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          uint32_t k_frag[2];
          ldmatrix_b_keytrans_sw<HeadDim>(
              tile_k, chunk * 16, tile * 8, lane, k_frag);
          mma_m16n8k16(dq_acc[local][tile], ds_frag, k_frag);
        }
      }
    }

    // All four outers folded; publish the tree, write the leaf plane, and
    // stage the next tile behind the same barrier (legacy steady-state
    // pattern: no extra CTA barrier).
    __syncthreads();
    const int64_t plane_tile_base = plane_base + key_start;
    for (int index = threadIdx.x; index < kQueryBlockWg * kKeyTile;
         index += kThreadsWg) {
      const int row = index / kKeyTile;
      const int key = index - row * kKeyTile;
      const int cell = row * kTreeStride + key;
      const float tile_sum = __fadd_rn(
          sbias_acc[cell],
          sbias_acc[kQueryBlockWg * kTreeStride + cell]);
      workspace[plane_tile_base + static_cast<int64_t>(row) * sequence +
                key] = __float2bfloat16_rn(tile_sum);
    }
    if (key_start + kKeyTile < sequence) {
      stage_kv_and_bias(key_start + kKeyTile);
    }
  }

  // dQ epilogue: each warp owns 16 query rows for its warpgroup's slots.
#pragma unroll
  for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
    const int slot = warpgroup * kSlotsPerWarpgroup + local;
    if (outer_mult[slot] == 0.0F) {
      continue;
    }
    const int64_t dq_row_base = (slot_plane(slot) * heads + head) *
        static_cast<int64_t>(sequence) * io_dim +
        static_cast<int64_t>(query_start) * io_dim;
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp_in_group * kQueryTile + group + half * 8;
        const int column = tile * 8 + thread_in_group * 2;
        if (column >= io_dim) {
          continue;
        }
        out_dq[dq_row_base + static_cast<int64_t>(row) * io_dim + column] =
            __float2bfloat16_rn(dq_acc[local][tile].v[half * 2] * scale);
        out_dq[dq_row_base + static_cast<int64_t>(row) * io_dim + column +
               1] =
            __float2bfloat16_rn(dq_acc[local][tile].v[half * 2 + 1] * scale);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SR (single-reconstruction) backward leaf: ONE kernel reconstructs
// {S^T, P^T, dP^T, dS^T} per (outer, q-tile) and feeds all four gradients.
//   * CTA = (64-key block, head, 4-outer leaf), 256 threads = 2 warpgroups
//     owning slot pairs {0,1}/{2,3} (the proven dq-wg platform, mirrored to
//     K residency).  K/V and the additive mask are key-invariant per CTA and
//     staged once; the CTA sweeps the query slab in 32-row tiles.
//   * dK/dV accumulate in registers with the dkdv chain order (A = packed
//     dS^T/P^T accumulators, B = rowtrans over the staged q/dO tiles) --
//     bit-identical to the two-pass dkdv by construction.
//   * dBias keeps the exact workspace contract: per q-tile the two
//     warpgroups fold (o0+o1)/(o2+o3) into FP32 tree tiles, combine, and
//     single-write the leaf plane cells [q, kb-keys]; leaf-wave residency
//     and the finalize kernel are untouched.
//   * dQ is the one scattered gradient: per (kb, q-tile, slot) the dS tile
//     is transposed through shared memory, multiplied by K, and written to
//     a flat FP32 partial [kb][plane][q][D]; a fixed ascending-kb reduce
//     folds partials into BF16 dQ.  delta = classic dO.O (v1 contract).
constexpr int kSrKeyBlock = 64;
constexpr int kSrQueryTile = 32;
constexpr int kSrTreeStride = kSrKeyBlock + 1;
constexpr int kSrBiasStride = kSrKeyBlock + 8;
constexpr int kSrDsStride = kSrKeyBlock + 8;

__device__ __forceinline__ void sr_cp_async_16(
    __nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
      "l"(src));
}

__device__ __forceinline__ void sr_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void sr_cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_rowtrans_sw_sr(
    const __nv_bfloat16* tile, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 15);
  const uint32_t addr = shared_u32(
      tile + row * HeadDim + wg_swz_col<HeadDim>(row, dim_start));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

template <int HeadDim, bool TaskInGridZ, bool OuterEqSeq>
__global__ __launch_bounds__(kThreadsWg, 2) void stablefold_sr_leaf_kernel(
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
    int io_dim,
    float scale) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warpgroup = warp / kWarpsPerBlock;
  const int warp_in_group = warp & (kWarpsPerBlock - 1);
  const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
  const int local_batch = wave_task % active_batch;
  const int batch_index = batch_base + local_batch;
  const int wave_leaf = wave_task / active_batch;
  const int key_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
  const int key_start = key_block * kSrKeyBlock;
  const int head = blockIdx.y;
  const int leaf = leaf_base + wave_leaf;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int warp_key = warp_in_group * kQueryTile;

  // q/dO/bias staging is double-buffered behind cp.async so the global
  // latency of the next query tile hides under the current tile's math.
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sv = sk + kOuterLeaf * kSrKeyBlock * HeadDim;
  __nv_bfloat16* sq = sv + kOuterLeaf * kSrKeyBlock * HeadDim;
  __nv_bfloat16* sdo = sq + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
  __nv_bfloat16* sds = sdo + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
  __nv_bfloat16* sbias_tile =
      sds + kOuterLeaf * kSrQueryTile * kSrDsStride;
  float* smask = reinterpret_cast<float*>(
      sbias_tile + 2 * kSrQueryTile * kSrBiasStride);
  float* sbias_acc = smask + kOuterLeaf * kSrKeyBlock;

  const int64_t plane_base =
      (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) *
            heads +
        head) *
           static_cast<int64_t>(query_capacity)) *
      sequence;
  const int64_t bias_base =
      ((static_cast<int64_t>(batch_index) * heads + head) * sequence) *
      static_cast<int64_t>(sequence);
  const bool single_outer = !OuterEqSeq && outer_count == 1;

  const int oc = OuterEqSeq ? sequence : outer_count;
  const auto slot_plane = [&](int slot) -> int64_t {
    const int outer = leaf * kOuterLeaf + slot;
    const int outer_load = outer < oc ? outer : oc - 1;
    return static_cast<int64_t>(batch_index) * oc + outer_load;
  };
  float outer_mult[kOuterLeaf];
#pragma unroll
  for (int slot = 0; slot < kOuterLeaf; ++slot) {
    outer_mult[slot] = leaf * kOuterLeaf + slot < oc ? 1.0F : 0.0F;
  }

  // K/V and the additive mask are key-invariant: staged once.
#pragma unroll 1
  const int staged_slots = single_outer ? 1 : kOuterLeaf;
  for (int slot = 0; slot < staged_slots; ++slot) {
    __nv_bfloat16* dst_k = sk + slot * kSrKeyBlock * HeadDim;
    __nv_bfloat16* dst_v = sv + slot * kSrKeyBlock * HeadDim;
    const int64_t tensor_base = (slot_plane(slot) * heads + head) *
        static_cast<int64_t>(sequence) * io_dim;
    for (int element = threadIdx.x; element < kSrKeyBlock * HeadDim;
         element += kThreadsWg) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int64_t src = tensor_base +
          static_cast<int64_t>(key_start + row) * io_dim + col;
      const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
      dst_k[dst_index] =
          col < io_dim ? k[src] : __float2bfloat16_rn(0.0F);
      dst_v[dst_index] =
          col < io_dim ? v[src] : __float2bfloat16_rn(0.0F);
    }
  }
  for (int element = threadIdx.x; element < staged_slots * kSrKeyBlock;
       element += kThreadsWg) {
    const int slot = element / kSrKeyBlock;
    const int key = element - slot * kSrKeyBlock;
    smask[element] = mask[slot_plane(slot) * sequence + key_start + key];
  }

  Acc dk_acc[kSlotsPerWarpgroup][HeadDim / 8];
  Acc dv_acc[kSlotsPerWarpgroup][HeadDim / 8];
#pragma unroll
  for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
      dk_acc[local][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      dv_acc[local][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
    }
  }
  if (query_base != 0) {
#pragma unroll
    for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
      const int slot = warpgroup * kSlotsPerWarpgroup + local;
      if (outer_mult[slot] == 0.0F) {
        continue;
      }
      const int64_t out_base = (slot_plane(slot) * heads + head) *
              static_cast<int64_t>(sequence) * io_dim +
          static_cast<int64_t>(key_start + warp_key) * io_dim;
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = group + half * 8;
          const int column = tile * 8 + thread_in_group * 2;
          if (column >= io_dim) {
            continue;
          }
          const int64_t index =
              out_base + static_cast<int64_t>(row) * io_dim + column;
          dv_acc[local][tile].v[half * 2] = dv_continuation[index];
          dv_acc[local][tile].v[half * 2 + 1] =
              dv_continuation[index + 1];
          dk_acc[local][tile].v[half * 2] = dk_continuation[index];
          dk_acc[local][tile].v[half * 2 + 1] =
              dk_continuation[index + 1];
        }
      }
    }
  }

  constexpr int kSrChunks = HeadDim / 8;
  const auto stage_qdo_and_bias = [&](int query_start, int parity) {
    __nv_bfloat16* buf_q =
        sq + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    __nv_bfloat16* buf_do =
        sdo + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    __nv_bfloat16* buf_bias =
        sbias_tile + parity * kSrQueryTile * kSrBiasStride;
#pragma unroll 1
    for (int slot = 0; slot < staged_slots; ++slot) {
      __nv_bfloat16* dst_q = buf_q + slot * kSrQueryTile * HeadDim;
      __nv_bfloat16* dst_do = buf_do + slot * kSrQueryTile * HeadDim;
      const int64_t plane = slot_plane(slot);
      const int64_t tensor_base = (plane * heads + head) *
          static_cast<int64_t>(sequence) * io_dim;
      const int64_t dout_base =
          plane * static_cast<int64_t>(sequence) * heads * io_dim +
          static_cast<int64_t>(head) * io_dim;
      for (int chunk = threadIdx.x; chunk < kSrQueryTile * kSrChunks;
           chunk += kThreadsWg) {
        const int row = chunk / kSrChunks;
        const int col = (chunk - row * kSrChunks) * 8;
        const int gq = query_start + row;
        const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
        if (col < io_dim) {
          sr_cp_async_16(
              dst_q + dst_index,
              q + tensor_base + static_cast<int64_t>(gq) * io_dim + col);
          sr_cp_async_16(
              dst_do + dst_index,
              d_output + dout_base +
                  static_cast<int64_t>(gq) * heads * io_dim + col);
        } else {
          *reinterpret_cast<uint4*>(dst_q + dst_index) =
              uint4{0U, 0U, 0U, 0U};
          *reinterpret_cast<uint4*>(dst_do + dst_index) =
              uint4{0U, 0U, 0U, 0U};
        }
      }
    }
    for (int chunk = threadIdx.x; chunk < kSrQueryTile * (kSrKeyBlock / 8);
         chunk += kThreadsWg) {
      const int row = chunk / (kSrKeyBlock / 8);
      const int key = (chunk - row * (kSrKeyBlock / 8)) * 8;
      sr_cp_async_16(
          buf_bias + row * kSrBiasStride + key,
          bias + bias_base +
              static_cast<int64_t>(query_start + row) * sequence +
              key_start + key);
    }
    sr_cp_async_commit();
  };

  stage_qdo_and_bias(query_base, 0);

  const int query_end = query_base + query_rows;
#pragma unroll 1
  for (int query_start = query_base; query_start < query_end;
       query_start += kSrQueryTile) {
    const int parity = ((query_start - query_base) / kSrQueryTile) & 1;
    sr_cp_async_wait_all();
    wg_proxy_fence();
    __syncthreads();
    if (query_start + kSrQueryTile < query_end) {
      stage_qdo_and_bias(query_start + kSrQueryTile, parity ^ 1);
    }
    const __nv_bfloat16* cur_q =
        sq + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    const __nv_bfloat16* cur_do =
        sdo + parity * kOuterLeaf * kSrQueryTile * HeadDim;
    const __nv_bfloat16* cur_bias =
        sbias_tile + parity * kSrQueryTile * kSrBiasStride;
#pragma unroll
    for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
      const int slot = warpgroup * kSlotsPerWarpgroup + local;
      if (single_outer && slot != 0) {
        continue;
      }
      const __nv_bfloat16* tile_k = sk + slot * kSrKeyBlock * HeadDim;
      const __nv_bfloat16* tile_v = sv + slot * kSrKeyBlock * HeadDim;
      const __nv_bfloat16* tile_q = cur_q + slot * kSrQueryTile * HeadDim;
      const __nv_bfloat16* tile_do = cur_do + slot * kSrQueryTile * HeadDim;

      // S^T = K.Q^T and dP^T = V.dO^T: A = the slot's 64-key K/V tile,
      // B = its 32-query q/dO tile (the dkdv ss encoding, proven bitwise
      // vs the mma.sync chain).  Accumulator rows = this warp's 16 keys.
      constexpr uint64_t kSsMode =
          HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
      constexpr uint64_t kSsLbo = 64;
      constexpr uint64_t kSsSbo = HeadDim;
      Acc score[kSrQueryTile / 8];
      Acc dp[kSrQueryTile / 8];
#pragma unroll
      for (int column = 0; column < kSrQueryTile / 8; ++column) {
        score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
        dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      }
      wg_wgmma_fence();
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
        const uint64_t desc_k =
            ((static_cast<uint64_t>(shared_u32(tile_k)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t desc_q =
            ((static_cast<uint64_t>(shared_u32(tile_q)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(score, desc_k, desc_q, chain != 0);
      }
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
        const uint64_t desc_v =
            ((static_cast<uint64_t>(shared_u32(tile_v)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        const uint64_t desc_do =
            ((static_cast<uint64_t>(shared_u32(tile_do)) + chain_off) >> 4) |
            (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
        wg_ss_wgmma_m64n32k16(dp, desc_v, desc_do, chain != 0);
      }
      wg_wgmma_commit();
      wg_wgmma_wait_zero();

      // Elementwise recompute in the [key, query] accumulator layout:
      // P^T = exp(S^T*scale + mask[key] + bias[q,k] - LSE[q]);
      // dS^T = P^T * (dP^T - delta[q]).  Column index = query.
      float* tree = sbias_acc +
          warpgroup * (kSrQueryTile * kSrTreeStride);
      __nv_bfloat16* ds_tile = sds + slot * kSrQueryTile * kSrDsStride;
      const float slot_mult = outer_mult[slot];
      const float* slot_mask = smask + slot * kSrKeyBlock;
      const int64_t stats_base = (slot_plane(slot) * heads + head) *
          static_cast<int64_t>(sequence);
#pragma unroll
      for (int column = 0; column < kSrQueryTile / 8; ++column) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int key_row = warp_key + group + half * 8;
          const float mask_value = slot_mask[key_row];
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const int query_in_tile = column * 8 + thread_in_group * 2 +
                pair;
            const int query = query_start + query_in_tile;
            const int index = half * 2 + pair;
            const int64_t stats = stats_base + query;
            const float lse = logsumexp[stats];
            const float value = score[column].v[index] * scale +
                mask_value +
                __bfloat162float(
                    cur_bias[query_in_tile * kSrBiasStride + key_row]);
            const float probability =
                lse == -std::numeric_limits<float>::infinity()
                ? 0.0F
                : __expf(value - lse);
            float d_score =
                probability * (dp[column].v[index] - delta[stats]);
            if constexpr (!OuterEqSeq) {
              d_score *= slot_mult;
            }
            float p_value = probability;
            if constexpr (!OuterEqSeq) {
              p_value *= slot_mult;
            }
            dp[column].v[index] = p_value;
            score[column].v[index] = d_score;
            const int cell = query_in_tile * kSrTreeStride + key_row;
            if (local == 0) {
              tree[cell] = d_score;
            } else {
              tree[cell] = __fadd_rn(tree[cell], d_score);
            }
            ds_tile[query_in_tile * kSrDsStride + key_row] =
                __float2bfloat16_rn(d_score);
          }
        }
      }

      // dV += P^T @ dO; dK += dS^T @ Q -- contraction over the 32 queries,
      // the exact dkdv chain (A = packed accumulators, B = rowtrans).
#pragma unroll
      for (int chunk = 0; chunk < kSrQueryTile / 16; ++chunk) {
        uint32_t p_frag[4];
        uint32_t ds_frag[4];
        p_frag[0] = pack_bf16_pair(
            dp[chunk * 2].v[0], dp[chunk * 2].v[1]);
        p_frag[1] = pack_bf16_pair(
            dp[chunk * 2].v[2], dp[chunk * 2].v[3]);
        p_frag[2] = pack_bf16_pair(
            dp[chunk * 2 + 1].v[0], dp[chunk * 2 + 1].v[1]);
        p_frag[3] = pack_bf16_pair(
            dp[chunk * 2 + 1].v[2], dp[chunk * 2 + 1].v[3]);
        ds_frag[0] = pack_bf16_pair(
            score[chunk * 2].v[0], score[chunk * 2].v[1]);
        ds_frag[1] = pack_bf16_pair(
            score[chunk * 2].v[2], score[chunk * 2].v[3]);
        ds_frag[2] = pack_bf16_pair(
            score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
        ds_frag[3] = pack_bf16_pair(
            score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          uint32_t b_frag[2];
          ldmatrix_b_rowtrans_sw_sr<HeadDim>(
              tile_do, chunk * 16, tile * 8, lane, b_frag);
          mma_m16n8k16(dv_acc[local][tile], p_frag, b_frag);
          ldmatrix_b_rowtrans_sw_sr<HeadDim>(
              tile_q, chunk * 16, tile * 8, lane, b_frag);
          mma_m16n8k16(dk_acc[local][tile], ds_frag, b_frag);
        }
      }
    }

    // Tree tiles and dS tiles complete for this q-tile.
    __syncthreads();

    // dBias: combine the warpgroup pair trees and single-write the leaf
    // plane cells [q, kb-keys] (same values, cells, and order as the
    // two-pass kernel's fixed tree).
    const int64_t plane_tile_base = plane_base +
        static_cast<int64_t>(query_start - query_base) * sequence +
        key_start;
    for (int element = threadIdx.x; element < kSrQueryTile * kSrKeyBlock;
         element += kThreadsWg) {
      const int row = element / kSrKeyBlock;
      const int key = element - row * kSrKeyBlock;
      const int cell = row * kSrTreeStride + key;
      const float tile_sum = single_outer
          ? sbias_acc[cell]
          : __fadd_rn(
                sbias_acc[cell],
                sbias_acc[kSrQueryTile * kSrTreeStride + cell]);
      workspace[plane_tile_base + static_cast<int64_t>(row) * sequence +
                key] = __float2bfloat16_rn(tile_sum);
    }

    // dQ partial for this (kb, q-tile): dq += dS @ K over the CTA's 64
    // keys.  Each warpgroup serves its own slot pair; warps 0/1 take the
    // first slot's row halves, warps 2/3 the second's.
    {
      const int local = warp_in_group >> 1;
      const int slot = warpgroup * kSlotsPerWarpgroup + local;
      const int row_half = warp_in_group & 1;
      if (outer_mult[slot] != 0.0F) {
        const __nv_bfloat16* ds_tile =
            sds + slot * kSrQueryTile * kSrDsStride +
            row_half * kQueryTile * kSrDsStride;
        const __nv_bfloat16* tile_k = sk + slot * kSrKeyBlock * HeadDim;
        Acc dq_acc[HeadDim / 8];
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
          dq_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
        }
#pragma unroll
        for (int chunk = 0; chunk < kSrKeyBlock / 16; ++chunk) {
          uint32_t a_frag[4];
          ldmatrix_a16x16(ds_tile, kSrDsStride, chunk * 16, lane, a_frag);
          #pragma unroll
          for (int tile = 0; tile < HeadDim / 8; ++tile) {
            uint32_t k_frag[2];
            ldmatrix_b_keytrans_sw<HeadDim>(
                tile_k, chunk * 16, tile * 8, lane, k_frag);
            mma_m16n8k16(dq_acc[tile], a_frag, k_frag);
          }
        }
        // Partial layout matches the contiguous [B,O,H,N,D] output per
        // key block: idx = ((kb * total_planes + plane) * heads + head)
        //                  * sequence + q) * io_dim + col
        const int64_t plane = slot_plane(slot);
        const int64_t idx_base =
            (((static_cast<int64_t>(key_block) * total_planes + plane) *
                  heads +
              head) *
                 static_cast<int64_t>(sequence) +
             (query_start + row_half * kQueryTile)) *
            io_dim;
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = group + half * 8;
            const int column = tile * 8 + thread_in_group * 2;
            if (column >= io_dim) {
              continue;
            }
            dq_partials[idx_base + static_cast<int64_t>(row) * io_dim +
                        column] =
                __float2bfloat16_rn(dq_acc[tile].v[half * 2]);
            dq_partials[idx_base + static_cast<int64_t>(row) * io_dim +
                        column + 1] =
                __float2bfloat16_rn(dq_acc[tile].v[half * 2 + 1]);
          }
        }
      }
    }

  }

  // Epilogue: dK/dV for this warp's 16 key rows, both slots of its
  // warpgroup.  Intermediate query stripes preserve the raw FP32 MMA
  // accumulators; the final stripe scales dK once and stores both BF16.
  const bool final_query_stripe = query_base + query_rows == sequence;
#pragma unroll
  for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
    const int slot = warpgroup * kSlotsPerWarpgroup + local;
    if (outer_mult[slot] == 0.0F) {
      continue;
    }
    const int64_t out_base = (slot_plane(slot) * heads + head) *
        static_cast<int64_t>(sequence) * io_dim +
        static_cast<int64_t>(key_start + warp_key) * io_dim;
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = group + half * 8;
        const int column = tile * 8 + thread_in_group * 2;
        if (column >= io_dim) {
          continue;
        }
        const int64_t index =
            out_base + static_cast<int64_t>(row) * io_dim + column;
        if (final_query_stripe) {
          out_dv[index] =
              __float2bfloat16_rn(dv_acc[local][tile].v[half * 2]);
          out_dv[index + 1] =
              __float2bfloat16_rn(dv_acc[local][tile].v[half * 2 + 1]);
          out_dk[index] = __float2bfloat16_rn(
              dk_acc[local][tile].v[half * 2] * scale);
          out_dk[index + 1] = __float2bfloat16_rn(
              dk_acc[local][tile].v[half * 2 + 1] * scale);
        } else {
          dv_continuation[index] = dv_acc[local][tile].v[half * 2];
          dv_continuation[index + 1] =
              dv_acc[local][tile].v[half * 2 + 1];
          dk_continuation[index] = dk_acc[local][tile].v[half * 2];
          dk_continuation[index + 1] =
              dk_acc[local][tile].v[half * 2 + 1];
        }
      }
    }
  }
}

// Fixed ascending-kb fold of the SR dQ partials into BF16 dQ.  Each
// thread owns eight consecutive elements (one 16-byte vector) so the
// per-kb loads coalesce to full lines and the eight per-element fadd
// chains stay independent; per-element fold order (ascending kb) and
// therefore the output bits match the scalar form exactly.
__global__ void stablefold_sr_dq_reduce_kernel(
    const __nv_bfloat16* __restrict__ dq_partials,
    __nv_bfloat16* __restrict__ out_dq,
    int64_t output_elements,
    int64_t partial_stride,
    int key_blocks,
    float scale) {
  const int64_t vectors = output_elements / 8;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vec =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vec < vectors;
       vec += grid_stride) {
    const int64_t linear = vec * 8;
    float root[8];
    {
      const uint4 first = *reinterpret_cast<const uint4*>(
          dq_partials + linear);
      const __nv_bfloat16* half_values =
          reinterpret_cast<const __nv_bfloat16*>(&first);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        root[element] = __bfloat162float(half_values[element]);
      }
    }
#pragma unroll 1
    for (int kb = 1; kb < key_blocks; ++kb) {
      const uint4 chunk = *reinterpret_cast<const uint4*>(
          dq_partials + static_cast<int64_t>(kb) * partial_stride + linear);
      const __nv_bfloat16* half_values =
          reinterpret_cast<const __nv_bfloat16*>(&chunk);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        root[element] = __fadd_rn(
            root[element], __bfloat162float(half_values[element]));
      }
    }
    uint4 packed;
    __nv_bfloat16* out_values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      out_values[element] = __float2bfloat16_rn(root[element] * scale);
    }
    *reinterpret_cast<uint4*>(out_dq + linear) = packed;
  }
}

// dbias[h,q,k] = ((plane0 + plane1) + plane2) + ... in leaf order.
__global__ void stablefold_dq_dbias_finalize_kernel(
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
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= active_elements) {
    return;
  }
  int64_t remaining = linear;
  const int key = static_cast<int>(remaining % sequence);
  remaining /= sequence;
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
      (((static_cast<int64_t>(local_batch) * heads + head) * query_capacity +
        query_local) *
           sequence +
       key);
  const int64_t output_linear =
      (((static_cast<int64_t>(batch_base + local_batch) * heads + head) *
            sequence +
        query_base + query_local) *
           sequence +
       key);
  float root = continue_root
      ? out_dbias[output_linear]
      : __bfloat162float(workspace[workspace_linear]);
  const int first_leaf = continue_root ? 0 : 1;
#pragma unroll 1
  for (int leaf = first_leaf; leaf < wave_leaves; ++leaf) {
    root = __fadd_rn(
        root,
        __bfloat162float(
            workspace[static_cast<int64_t>(leaf) * workspace_plane_elements +
                      workspace_linear]));
  }
  out_dbias[output_linear] = root;
}

// Reduce the public per-batch BF16 gradient in flattened batch order. Every
// FP32 partial is first rounded to the public BF16 per-batch contract, then
// accumulated with explicit FP32 round-to-nearest additions. One thread owns
// one output element; no atomics or framework reduction tree participate.
__global__ void stablefold_reduce_broadcast_dbias_kernel(
    const float* __restrict__ per_batch_dbias,
    __nv_bfloat16* __restrict__ out,
    int64_t output_elements,
    int64_t plane_elements,
    int target_batch,
    PrefixBroadcastPlan plan) {
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
          mapped_bias_index +=
              static_cast<int64_t>(coordinate) * plan.bias_strides[axis];
        }
      }
      if (mapped_bias_index != bias_batch_index) {
        continue;
      }
      const float public_partial = __bfloat162float(__float2bfloat16_rn(
          per_batch_dbias[
              static_cast<int64_t>(target_index) * plane_elements + element]));
      root = initialized ? __fadd_rn(root, public_partial) : public_partial;
      initialized = true;
    }
    out[linear] = __float2bfloat16_rn(initialized ? root : 0.0F);
  }
}

int64_t checked_positive_product(
    int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left > 0 && right > 0, label, " dimensions must be positive");
  TORCH_CHECK(
      left <= std::numeric_limits<int64_t>::max() / right,
      label,
      " element count overflow");
  return left * right;
}

int64_t positive_ceil_div(int64_t numerator, int64_t denominator) {
  TORCH_CHECK(
      numerator > 0 && denominator > 0,
      "ceil-div inputs must be positive");
  return numerator / denominator + (numerator % denominator != 0);
}

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
    int io_dim,
    cudaStream_t stream) {
  constexpr int kQStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  constexpr int kSharedBytes =
      (2 * kWarpsPerBlock * kQueryBlock * kQStride +
       kWarpsPerBlock * 2 * kKeyTile * kKVStride) *
          static_cast<int>(sizeof(__nv_bfloat16)) +
      kWarpsPerBlock * kQueryBlock * kKeyTile *
          static_cast<int>(sizeof(float));
  const float scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  TORCH_CHECK(
      query_span > 0 && query_span <= std::numeric_limits<int>::max() &&
          resident_leaves > 0 && resident_leaves <= leaves &&
          batch_chunk > 0 && batch_chunk <= batch,
      "invalid query-slab workspace plan");
  int64_t required_workspace = checked_positive_product(
      resident_leaves, batch_chunk, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, heads, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace,
      "workspace is smaller than the query-slab plan");
#define SET_DQ_DBIAS_SMEM_ATTRIBUTE(FullTiles, TaskInZ)                    \
  do {                                                                     \
    const cudaError_t status = cudaFuncSetAttribute(                       \
        stablefold_dq_dbias_leaf_kernel<                                    \
            HeadDim, StaticSequence, FullTiles, TaskInZ, true>,            \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytes);                                                     \
    const cudaError_t status_g = cudaFuncSetAttribute(                     \
        stablefold_dq_dbias_leaf_kernel<                                    \
            HeadDim, StaticSequence, FullTiles, TaskInZ, false>,           \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytes);                                                     \
    TORCH_CHECK(                                                           \
        status == cudaSuccess && status_g == cudaSuccess,                  \
        "failed to opt in StableFold dQ/dBias shared memory");             \
  } while (false)
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(true, true);
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(true, false);
  if constexpr (StaticSequence != -1) {
    SET_DQ_DBIAS_SMEM_ATTRIBUTE(false, true);
    SET_DQ_DBIAS_SMEM_ATTRIBUTE(false, false);
  }
#undef SET_DQ_DBIAS_SMEM_ATTRIBUTE
  const int finalize_threads = 256;
  for (int batch_base = 0; batch_base < batch;) {
    const int active_batch = std::min(batch_chunk, batch - batch_base);
    for (int query_base = 0; query_base < sequence;) {
      const int query_rows = std::min(query_span, sequence - query_base);
      const int query_blocks =
          (query_rows + kQueryBlock - 1) / kQueryBlock;
      TORCH_CHECK(
          query_blocks > 0 && query_blocks <= 65535,
          "query slab exceeds the CUDA grid-z contract");
      int64_t active_elements = checked_positive_product(
          active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, sequence, "active dBias slab");
      const int64_t finalize_blocks =
          positive_ceil_div(active_elements, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= 2147483647LL,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves =
            std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks =
            static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= 2147483647LL,
            "dBias leaf wave exceeds the CUDA grid-x contract");
        // Preserve the old query-major launch geometry whenever the task
        // axis fits CUDA grid-z.  Large prefix batches use the outer-major
        // fallback.  Both layouts decode the same logical CTA coordinates.
        const bool task_in_grid_z = leaf_tasks <= 65535;
        const dim3 grid = task_in_grid_z
            ? dim3(
                  static_cast<unsigned int>(query_blocks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(leaf_tasks))
            : dim3(
                  static_cast<unsigned int>(leaf_tasks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(query_blocks));
#define LAUNCH_DQ_DBIAS_LEAF(FullTiles, TaskInZ, OuterEq)                  \
  stablefold_dq_dbias_leaf_kernel<                                          \
      HeadDim, StaticSequence, FullTiles, TaskInZ, OuterEq>                \
      <<<grid, kThreads, kSharedBytes, stream>>>(                          \
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),    \
          logsumexp.data_ptr<float>(),                                    \
          delta.data_ptr<float>(),                                        \
          mask.data_ptr<float>(),                                         \
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),        \
          reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),            \
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),         \
          sequence,                                                       \
          outer_count,                                                    \
          heads,                                                          \
          batch_base,                                                     \
          active_batch,                                                   \
          batch_chunk,                                                    \
          query_base,                                                     \
          query_rows,                                                     \
          query_span,                                                     \
          leaf_base,                                                      \
          io_dim,                                                          \
          scale)
        if constexpr (StaticSequence == -1) {
          // The exact-tiles contract guarantees full query blocks; the
          // guarded row-tail lane is never instantiated for this lane.
          TORCH_CHECK(
              query_rows % kQueryBlock == 0,
              "exact-tiles lane requires query rows in full blocks");
          if (task_in_grid_z) {
            if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(true, true, true); } else { LAUNCH_DQ_DBIAS_LEAF(true, true, false); }
          } else {
            if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(true, false, true); } else { LAUNCH_DQ_DBIAS_LEAF(true, false, false); }
          }
        } else {
          if (query_rows % kQueryBlock == 0) {
            if (task_in_grid_z) {
              if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(true, true, true); } else { LAUNCH_DQ_DBIAS_LEAF(true, true, false); }
            } else {
              if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(true, false, true); } else { LAUNCH_DQ_DBIAS_LEAF(true, false, false); }
            }
          } else {
            if (task_in_grid_z) {
              if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(false, true, true); } else { LAUNCH_DQ_DBIAS_LEAF(false, true, false); }
            } else {
              if (outer_eq) { LAUNCH_DQ_DBIAS_LEAF(false, false, true); } else { LAUNCH_DQ_DBIAS_LEAF(false, false, false); }
            }
          }
        }
#undef LAUNCH_DQ_DBIAS_LEAF
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold dQ/dBias slab leaf launch failed");
        stablefold_dq_dbias_finalize_kernel
            <<<static_cast<unsigned int>(finalize_blocks),
               finalize_threads,
               0,
               stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
                out_dbias.data_ptr<float>(),
                active_elements,
                sequence,
                heads,
                batch_base,
                active_batch,
                batch_chunk,
                query_base,
                query_rows,
                query_span,
                wave_leaves,
                leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold dQ/dBias slab finalizer launch failed");
        // Increment by the validated active extent, so even the theoretical
        // INT_MAX boundary cannot overflow a signed host loop counter.
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
}

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
    int io_dim,
    cudaStream_t stream) {
  constexpr int kSharedBytesWg =
      (2 * kOuterLeaf * kQueryBlockWg * HeadDim +
       kOuterLeaf * 2 * kKeyTile * HeadDim +
       kQueryBlockWg * kBiasStride) *
          static_cast<int>(sizeof(__nv_bfloat16)) +
      (kOuterLeaf * kKeyTile +
       2 * kQueryBlockWg * kTreeStride) *
          static_cast<int>(sizeof(float));
  const float scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  TORCH_CHECK(
      query_span > 0 && query_span % kQueryBlockWg == 0 &&
          sequence % kQueryBlockWg == 0 && resident_leaves > 0 &&
          resident_leaves <= leaves && batch_chunk > 0 &&
          batch_chunk <= batch,
      "invalid warpgroup query-slab plan");
  int64_t required_workspace = checked_positive_product(
      resident_leaves, batch_chunk, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, heads, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace,
      "workspace is smaller than the query-slab plan");
#define SET_DQ_DBIAS_WG_SMEM_ATTRIBUTE(TaskInZ)                             \
  do {                                                                     \
    const cudaError_t status = cudaFuncSetAttribute(                       \
        stablefold_dq_dbias_wg_kernel<HeadDim, TaskInZ, true>,             \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytesWg);                                                   \
    const cudaError_t status_g = cudaFuncSetAttribute(                     \
        stablefold_dq_dbias_wg_kernel<HeadDim, TaskInZ, false>,            \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytesWg);                                                   \
    TORCH_CHECK(                                                           \
        status == cudaSuccess && status_g == cudaSuccess,                  \
        "failed to opt in StableFold dQ/dBias warpgroup shared memory");   \
  } while (false)
  SET_DQ_DBIAS_WG_SMEM_ATTRIBUTE(true);
  SET_DQ_DBIAS_WG_SMEM_ATTRIBUTE(false);
#undef SET_DQ_DBIAS_WG_SMEM_ATTRIBUTE
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
          query_blocks > 0 && query_blocks <= 65535,
          "query slab exceeds the CUDA grid-z contract");
      int64_t active_elements = checked_positive_product(
          active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, sequence, "active dBias slab");
      const int64_t finalize_blocks =
          positive_ceil_div(active_elements, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= 2147483647LL,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves =
            std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks =
            static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= 2147483647LL,
            "dBias leaf wave exceeds the CUDA grid-x contract");
        const bool task_in_grid_z = leaf_tasks <= 65535;
        const dim3 grid = task_in_grid_z
            ? dim3(
                  static_cast<unsigned int>(query_blocks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(leaf_tasks))
            : dim3(
                  static_cast<unsigned int>(leaf_tasks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(query_blocks));
#define LAUNCH_DQ_DBIAS_WG(TaskInZ, OuterEq)                                \
  stablefold_dq_dbias_wg_kernel<HeadDim, TaskInZ, OuterEq>                  \
      <<<grid, kThreadsWg, kSharedBytesWg, stream>>>(                      \
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),    \
          logsumexp.data_ptr<float>(),                                    \
          delta.data_ptr<float>(),                                        \
          mask.data_ptr<float>(),                                         \
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),        \
          reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),            \
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),         \
          sequence,                                                       \
          outer_count,                                                    \
          heads,                                                          \
          batch_base,                                                     \
          active_batch,                                                   \
          batch_chunk,                                                    \
          query_base,                                                     \
          query_span,                                                     \
          leaf_base,                                                      \
          io_dim,                                                         \
          scale)
        if (task_in_grid_z) {
          if (outer_eq) { LAUNCH_DQ_DBIAS_WG(true, true); } else { LAUNCH_DQ_DBIAS_WG(true, false); }
        } else {
          if (outer_eq) { LAUNCH_DQ_DBIAS_WG(false, true); } else { LAUNCH_DQ_DBIAS_WG(false, false); }
        }
#undef LAUNCH_DQ_DBIAS_WG
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold dQ/dBias warpgroup leaf launch failed");
        stablefold_dq_dbias_finalize_kernel
            <<<static_cast<unsigned int>(finalize_blocks),
               finalize_threads,
               0,
               stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
                out_dbias.data_ptr<float>(),
                active_elements,
                sequence,
                heads,
                batch_base,
                active_batch,
                batch_chunk,
                query_base,
                query_rows,
                query_span,
                wave_leaves,
                leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold dQ/dBias slab finalizer launch failed");
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
}

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
    int io_dim,
    cudaStream_t stream) {
  constexpr int kSharedBytesSr =
      (2 * kOuterLeaf * kSrKeyBlock * HeadDim +
       4 * kOuterLeaf * kSrQueryTile * HeadDim +
       kOuterLeaf * kSrQueryTile * kSrDsStride +
       2 * kSrQueryTile * kSrBiasStride) *
          static_cast<int>(sizeof(__nv_bfloat16)) +
      (kOuterLeaf * kSrKeyBlock +
       2 * kSrQueryTile * kSrTreeStride) *
          static_cast<int>(sizeof(float));
  const float scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
  const int leaves = (outer_count + kOuterLeaf - 1) / kOuterLeaf;
  const bool outer_eq = outer_count == sequence;
  const int key_blocks = sequence / kSrKeyBlock;
  const int total_planes = batch * outer_count;
  TORCH_CHECK(
      sequence % kSrKeyBlock == 0 && key_blocks <= 65535 &&
          query_span > 0 && query_span % kSrQueryTile == 0 &&
          resident_leaves > 0 && resident_leaves <= leaves &&
          batch_chunk > 0 && batch_chunk <= batch,
      "invalid SR backward plan");
  int64_t required_workspace = checked_positive_product(
      resident_leaves, batch_chunk, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, heads, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, query_span, "workspace");
  required_workspace =
      checked_positive_product(required_workspace, sequence, "workspace");
  TORCH_CHECK(
      workspace.numel() >= required_workspace,
      "workspace is smaller than the query-slab plan");
  int64_t required_partials = checked_positive_product(
      key_blocks, total_planes, "dq partials");
  required_partials =
      checked_positive_product(required_partials, heads, "dq partials");
  required_partials =
      checked_positive_product(required_partials, sequence, "dq partials");
  required_partials =
      checked_positive_product(required_partials, io_dim, "dq partials");
  TORCH_CHECK(
      dq_partials.numel() >= required_partials,
      "dq_partials is smaller than the key-block plan");
  const int64_t required_continuation = query_span < sequence ? q.numel() : 0;
  TORCH_CHECK(
      dk_continuation.numel() >= required_continuation &&
          dv_continuation.numel() >= required_continuation,
      "dK/dV continuation state is smaller than the query-stripe plan");
#define SET_SR_SMEM_ATTRIBUTE(TaskInZ)                                      \
  do {                                                                     \
    const cudaError_t status = cudaFuncSetAttribute(                       \
        stablefold_sr_leaf_kernel<HeadDim, TaskInZ, true>,                 \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytesSr);                                                   \
    const cudaError_t status_g = cudaFuncSetAttribute(                     \
        stablefold_sr_leaf_kernel<HeadDim, TaskInZ, false>,                \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytesSr);                                                   \
    TORCH_CHECK(                                                           \
        status == cudaSuccess && status_g == cudaSuccess,                  \
        "failed to opt in StableFold SR shared memory");                   \
  } while (false)
  SET_SR_SMEM_ATTRIBUTE(true);
  SET_SR_SMEM_ATTRIBUTE(false);
#undef SET_SR_SMEM_ATTRIBUTE
  const int finalize_threads = 256;
  for (int batch_base = 0; batch_base < batch;) {
    const int active_batch = std::min(batch_chunk, batch - batch_base);
    for (int query_base = 0; query_base < sequence;) {
      const int query_rows = std::min(query_span, sequence - query_base);
      TORCH_CHECK(
          query_rows % kSrQueryTile == 0,
          "SR lane requires query rows in 32-row tiles");
      int64_t active_elements = checked_positive_product(
          active_batch, heads, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, query_rows, "active dBias slab");
      active_elements = checked_positive_product(
          active_elements, sequence, "active dBias slab");
      const int64_t finalize_blocks =
          positive_ceil_div(active_elements, finalize_threads);
      TORCH_CHECK(
          finalize_blocks > 0 && finalize_blocks <= 2147483647LL,
          "dBias slab finalizer exceeds the CUDA grid-x contract");
      for (int leaf_base = 0; leaf_base < leaves;) {
        const int wave_leaves =
            std::min(resident_leaves, leaves - leaf_base);
        const int64_t leaf_tasks =
            static_cast<int64_t>(active_batch) * wave_leaves;
        TORCH_CHECK(
            leaf_tasks > 0 && leaf_tasks <= 2147483647LL,
            "SR leaf wave exceeds the CUDA grid contract");
        const bool task_in_grid_z = leaf_tasks <= 65535;
        const dim3 grid = task_in_grid_z
            ? dim3(
                  static_cast<unsigned int>(key_blocks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(leaf_tasks))
            : dim3(
                  static_cast<unsigned int>(leaf_tasks),
                  static_cast<unsigned int>(heads),
                  static_cast<unsigned int>(key_blocks));
#define LAUNCH_SR_LEAF(TaskInZ, OuterEq)                                    \
  stablefold_sr_leaf_kernel<HeadDim, TaskInZ, OuterEq>                      \
      <<<grid, kThreadsWg, kSharedBytesSr, stream>>>(                      \
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),           \
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),    \
          logsumexp.data_ptr<float>(),                                    \
          delta.data_ptr<float>(),                                        \
          mask.data_ptr<float>(),                                         \
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),        \
          reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),            \
          reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),            \
          dk_continuation.data_ptr<float>(),                              \
          dv_continuation.data_ptr<float>(),                              \
          reinterpret_cast<__nv_bfloat16*>(                               \
              dq_partials.data_ptr()),                                    \
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),         \
          sequence,                                                       \
          outer_count,                                                    \
          total_planes,                                                   \
          heads,                                                          \
          batch_base,                                                     \
          active_batch,                                                   \
          batch_chunk,                                                    \
          query_base,                                                     \
          query_rows,                                                     \
          query_span,                                                     \
          leaf_base,                                                      \
          io_dim,                                                         \
          scale)
        if (task_in_grid_z) {
          if (outer_eq) { LAUNCH_SR_LEAF(true, true); } else { LAUNCH_SR_LEAF(true, false); }
        } else {
          if (outer_eq) { LAUNCH_SR_LEAF(false, true); } else { LAUNCH_SR_LEAF(false, false); }
        }
#undef LAUNCH_SR_LEAF
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold SR leaf launch failed");
        stablefold_dq_dbias_finalize_kernel
            <<<static_cast<unsigned int>(finalize_blocks),
               finalize_threads,
               0,
               stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
                out_dbias.data_ptr<float>(),
                active_elements,
                sequence,
                heads,
                batch_base,
                active_batch,
                batch_chunk,
                query_base,
                query_rows,
                query_span,
                wave_leaves,
                leaf_base != 0);
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess,
            "StableFold SR finalizer launch failed");
        leaf_base += wave_leaves;
      }
      query_base += query_rows;
    }
    batch_base += active_batch;
  }
  const int64_t output_elements = out_dq.numel();
  TORCH_CHECK(
      output_elements % 8 == 0,
      "SR dQ reduce requires an 8-element-aligned output");
  const int64_t partial_stride = static_cast<int64_t>(total_planes) *
      heads * sequence * io_dim;
  constexpr int reduce_threads = 256;
  const int64_t reduce_blocks_needed =
      positive_ceil_div(output_elements / 8, reduce_threads);
  const unsigned int reduce_blocks = static_cast<unsigned int>(
      std::min<int64_t>(reduce_blocks_needed, 65535));
  stablefold_sr_dq_reduce_kernel<<<reduce_blocks, reduce_threads, 0,
                                   stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(dq_partials.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),
      output_elements,
      partial_stride,
      key_blocks,
      scale);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "StableFold SR dQ reduce launch failed");
}

void stablefold_triangle_sr_backward_out(
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
      sequence % kSrKeyBlock == 0,
      "SR backward requires the 64-padded exact-tiles contract");
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 64,
      "SR backward serves D<=64 (D=128 keeps the two-pass path)");
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/out_dq/out_dk/out_dv must be contiguous BF16 matching q");
  };
  check_q_like(k);
  check_q_like(v);
  check_q_like(out_dq);
  check_q_like(out_dk);
  check_q_like(out_dv);
  TORCH_CHECK(
      d_output.is_cuda() && d_output.dtype() == torch::kBFloat16 &&
          d_output.is_contiguous() && d_output.dim() == 5 &&
          d_output.size(0) == q.size(0) && d_output.size(1) == q.size(1) &&
          d_output.size(2) == q.size(3) && d_output.size(3) == q.size(2) &&
          d_output.size(4) == q.size(4),
      "d_output must be contiguous BF16 [B,O,N,H,D]");
  TORCH_CHECK(
      logsumexp.is_cuda() && logsumexp.dtype() == torch::kFloat32 &&
      logsumexp.is_contiguous() && logsumexp.numel() == q.numel() / head_dim);
  TORCH_CHECK(
      delta.is_cuda() && delta.dtype() == torch::kFloat32 &&
      delta.is_contiguous() && delta.sizes() == logsumexp.sizes());
  TORCH_CHECK(
      mask.is_cuda() && mask.dtype() == torch::kFloat32 &&
      mask.is_contiguous() &&
      mask.sizes() ==
          torch::IntArrayRef({batch, outer_count, 1, 1, sequence}));
  TORCH_CHECK(
      bias.is_cuda() && bias.dtype() == torch::kBFloat16 &&
      bias.is_contiguous() &&
      bias.sizes() ==
          torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      out_dbias.is_cuda() && out_dbias.dtype() == torch::kFloat32 &&
      out_dbias.is_contiguous() &&
      out_dbias.sizes() ==
          torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      workspace.is_cuda() && workspace.dtype() == torch::kBFloat16 &&
      workspace.is_contiguous());
  TORCH_CHECK(
      dq_partials.is_cuda() && dq_partials.dtype() == torch::kBFloat16 &&
      dq_partials.is_contiguous());
  const auto check_continuation = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kFloat32 && tensor.is_contiguous(),
        "dK/dV continuation state must be contiguous FP32 on q.device");
  };
  check_continuation(dk_continuation);
  check_continuation(dv_continuation);
  const int query_span = static_cast<int>(query_span64);
  const int resident_leaves = static_cast<int>(resident_leaves64);
  const int batch_chunk = static_cast<int>(batch_chunk64);

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int io_dim = head_dim;
  const int storage = head_dim <= 16 ? 16 : (head_dim <= 32 ? 32 : 64);
  switch (storage) {
    case 16:
      launch_sr<16>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                    out_dk, out_dv, out_dbias, workspace, dq_partials,
                    dk_continuation, dv_continuation,
                    sequence, outer_count, heads, batch, query_span,
                    resident_leaves, batch_chunk, io_dim, stream);
      break;
    case 32:
      launch_sr<32>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                    out_dk, out_dv, out_dbias, workspace, dq_partials,
                    dk_continuation, dv_continuation,
                    sequence, outer_count, heads, batch, query_span,
                    resident_leaves, batch_chunk, io_dim, stream);
      break;
    default:
      launch_sr<64>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                    out_dk, out_dv, out_dbias, workspace, dq_partials,
                    dk_continuation, dv_continuation,
                    sequence, outer_count, heads, batch, query_span,
                    resident_leaves, batch_chunk, io_dim, stream);
      break;
  }
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess, "StableFold SR launch failed");
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
    int io_dim,
    cudaStream_t stream) {
  // One lane for every length: the exact-tiles contract (sequence padded
  // to the tile quantum by the facade) folds all guards at compile time
  // with sequence still a pure runtime value. Unpadded lengths from direct
  // ABI callers take the guarded fallback.
  if (HeadDim <= 64 && sequence % kQueryBlockWg == 0 &&
      query_span % kQueryBlockWg == 0) {
    // D=128 keeps the legacy lanes: the warpgroup lane's swizzle-descriptor
    // encoding covers the shipped {16,32,64} storage dims (capability
    // routing, not a shape specialization).
    launch_wg<HeadDim>(q, k, v, d_output, logsumexp, delta, mask, bias,
                       out_dq, out_dbias, workspace, sequence, outer_count,
                       heads, batch, query_span, resident_leaves,
                       batch_chunk, io_dim, stream);
  } else if (sequence % kQueryBlock == 0 && sequence % kOuterLeaf == 0) {
    launch<HeadDim, -1>(q, k, v, d_output, logsumexp, delta, mask, bias,
                        out_dq, out_dbias, workspace, sequence, outer_count,
                        heads, batch, query_span, resident_leaves,
                        batch_chunk, io_dim, stream);
  } else {
    launch<HeadDim, 0>(q, k, v, d_output, logsumexp, delta, mask, bias,
                       out_dq, out_dbias, workspace, sequence, outer_count,
                       heads, batch, query_span, resident_leaves, batch_chunk,
                       io_dim, stream);
  }
}

void stablefold_triangle_dq_dbias_out(
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
      batch64 > 0 && batch64 <= std::numeric_limits<int>::max() &&
          outer64 > 0 && outer64 <= std::numeric_limits<int>::max() &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          heads64 > 0 && heads64 <= 65535 && head_dim64 > 0 &&
          head_dim64 <= std::numeric_limits<int>::max() &&
          batch64 * outer64 <= 2147483647LL &&
          (sequence64 + kQueryBlock - 1) / kQueryBlock <= 65535,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const int64_t leaves64 =
      (outer64 + kOuterLeaf - 1) / kOuterLeaf;
  TORCH_CHECK(
      query_span64 > 0 &&
          query_span64 <= std::numeric_limits<int>::max() &&
          resident_leaves64 > 0 && resident_leaves64 <= leaves64 &&
          resident_leaves64 <= std::numeric_limits<int>::max() &&
          batch_chunk64 > 0 && batch_chunk64 <= batch64 &&
          batch_chunk64 <= std::numeric_limits<int>::max(),
      "invalid query-slab plan dimensions");
  const int query_span = static_cast<int>(query_span64);
  const int resident_leaves = static_cast<int>(resident_leaves64);
  const int batch_chunk = static_cast<int>(batch_chunk64);
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/d_output/out_dq must be contiguous BF16 tensors matching q "
        "on one device");
  };
  check_q_like(k);
  check_q_like(v);
  TORCH_CHECK(
      d_output.is_cuda() && d_output.device() == q.device() &&
          d_output.dtype() == torch::kBFloat16 && d_output.is_contiguous() &&
          d_output.dim() == 5 && d_output.size(0) == q.size(0) &&
          d_output.size(1) == q.size(1) && d_output.size(2) == q.size(3) &&
          d_output.size(3) == q.size(2) && d_output.size(4) == q.size(4),
      "d_output must be contiguous BF16 [B,O,N,H,D] on q.device");
  check_q_like(out_dq);
  TORCH_CHECK(
      logsumexp.is_cuda() && logsumexp.device() == q.device() &&
          logsumexp.dtype() == torch::kFloat32 && logsumexp.is_contiguous() &&
          logsumexp.numel() == q.numel() / head_dim);
  TORCH_CHECK(
      delta.is_cuda() && delta.device() == q.device() &&
          delta.dtype() == torch::kFloat32 && delta.is_contiguous() &&
          delta.sizes() == logsumexp.sizes());
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() &&
          mask.dtype() == torch::kFloat32 && mask.is_contiguous() &&
          mask.sizes() ==
              torch::IntArrayRef({batch, outer_count, 1, 1, sequence}));
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      out_dbias.is_cuda() && out_dbias.device() == q.device() &&
          out_dbias.dtype() == torch::kFloat32 && out_dbias.is_contiguous() &&
          out_dbias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}));
  int64_t required_workspace = checked_positive_product(
      resident_leaves64, batch_chunk64, "workspace plan");
  required_workspace = checked_positive_product(
      required_workspace, heads64, "workspace plan");
  required_workspace = checked_positive_product(
      required_workspace, query_span64, "workspace plan");
  required_workspace = checked_positive_product(
      required_workspace, sequence64, "workspace plan");
  TORCH_CHECK(
      workspace.is_cuda() && workspace.device() == q.device() &&
      workspace.dtype() == torch::kBFloat16 && workspace.is_contiguous() &&
      workspace.numel() >= required_workspace,
      "workspace is smaller than the validated query-slab plan");

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int io_dim = head_dim;
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 128,
      "head_dim must be a multiple of 8 and <= 128, got ", head_dim);
  const int storage = head_dim <= 16
      ? 16
      : (head_dim <= 32 ? 32 : (head_dim <= 64 ? 64 : 128));
  switch (storage) {
    case 16:
      dispatch_sequence<16>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, outer_count, heads, batch,
                 query_span, resident_leaves, batch_chunk, io_dim, stream);
      break;
    case 32:
      dispatch_sequence<32>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, outer_count, heads, batch,
                 query_span, resident_leaves, batch_chunk, io_dim, stream);
      break;
    case 64:
      dispatch_sequence<64>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, outer_count, heads, batch,
                 query_span, resident_leaves, batch_chunk, io_dim, stream);
      break;
    default:
      dispatch_sequence<128>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, outer_count, heads, batch,
                 query_span, resident_leaves, batch_chunk, io_dim, stream);
      break;
  }
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess, "StableFold dQ/dBias launch failed");
}

void stablefold_reduce_broadcast_dbias_out(
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
  const int rank_delta =
      static_cast<int>(target_prefix.size() - bias_prefix.size());
  for (int axis = 0; axis < plan.rank; ++axis) {
    const int64_t target = target_prefix[axis];
    const int64_t bias = axis < rank_delta ? 1 : bias_prefix[axis - rank_delta];
    TORCH_CHECK(
        target > 0 && target <= std::numeric_limits<int>::max() &&
            bias > 0 && bias <= std::numeric_limits<int>::max() &&
            (bias == 1 || bias == target),
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
        bias_batch <= std::numeric_limits<int64_t>::max() /
            plan.bias_dims[axis],
        "bias prefix product overflow");
    bias_batch *= plan.bias_dims[axis];
  }
  const int target_batch = static_cast<int>(target_batch64);
  TORCH_CHECK(
      per_batch_dbias.is_cuda() &&
          per_batch_dbias.dtype() == torch::kFloat32 &&
          per_batch_dbias.is_contiguous() && per_batch_dbias.dim() == 5 &&
          per_batch_dbias.size(0) == target_batch &&
          per_batch_dbias.size(1) == 1,
      "per_batch_dbias must be contiguous FP32 [B,1,H,N,N]");
  const int64_t heads = per_batch_dbias.size(2);
  const int64_t sequence = per_batch_dbias.size(3);
  TORCH_CHECK(
      heads > 0 && sequence > 0 && per_batch_dbias.size(4) == sequence,
      "per_batch_dbias must have positive square pair dimensions");
  TORCH_CHECK(
      out.is_cuda() && out.device() == per_batch_dbias.device() &&
          out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
          out.dim() == static_cast<int64_t>(bias_prefix.size()) + 4,
      "out must be contiguous BF16 with the source bias rank");
  for (int axis = 0; axis < static_cast<int>(bias_prefix.size()); ++axis) {
    TORCH_CHECK(out.size(axis) == bias_prefix[axis], "out bias prefix mismatch");
  }
  const int tail = static_cast<int>(bias_prefix.size());
  TORCH_CHECK(
      out.size(tail) == 1 && out.size(tail + 1) == heads &&
          out.size(tail + 2) == sequence && out.size(tail + 3) == sequence,
      "out bias tail must be [1,H,N,N]");
  int64_t plane_elements =
      checked_positive_product(heads, sequence, "broadcast dBias plane");
  plane_elements = checked_positive_product(
      plane_elements, sequence, "broadcast dBias plane");
  const int64_t expected_output_elements = checked_positive_product(
      bias_batch, plane_elements, "broadcast dBias output");
  TORCH_CHECK(
      out.numel() == expected_output_elements,
      "out element count does not match the bias prefix");

  const c10::cuda::CUDAGuard device_guard(per_batch_dbias.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int64_t required_blocks = positive_ceil_div(out.numel(), threads);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min<int64_t>(required_blocks, 65535));
  stablefold_reduce_broadcast_dbias_kernel<<<blocks, threads, 0, stream>>>(
      per_batch_dbias.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.numel(),
      plane_elements,
      target_batch,
      plan);
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "fixed broadcast dBias reduction launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "stablefold_triangle_dq_dbias_out",
      &stablefold_triangle_dq_dbias_out,
      "StableFold fused dQ+dBias (query-slab deterministic waves)");
  module.def(
      "stablefold_reduce_broadcast_dbias_out",
      &stablefold_reduce_broadcast_dbias_out,
      "Fixed-order broadcast dBias reduction");
  module.def(
      "stablefold_triangle_sr_backward_out",
      &stablefold_triangle_sr_backward_out,
      "StableFold single-reconstruction backward (dQ/dK/dV/dBias)");
}
