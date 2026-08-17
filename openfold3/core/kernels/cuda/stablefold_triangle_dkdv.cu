// SPDX-License-Identifier: Apache-2.0
// StableFold dK/dV with key-block ownership and runtime shape dispatch.
//
// One CTA owns one (64-key block, head, outer) and its dK/dV accumulators
// outright: dK[k]/dV[k] depend only on column k of dS, so key-block
// ownership is contention-free by construction (zero atomics, no
// workspace). The kernel streams query tiles in fixed ascending order,
// recomputes S^T = K@Q^T directly in the [key, query] accumulator layout
// (P^T and dS^T then follow elementwise from LSE/delta), and accumulates
//   dV += P^T  @ dO   and   dK += dS^T @ Q, scaled once at the epilogue.
//
// Generality by construction: HeadDim templated {16,32,64}; heads and
// sequence runtime (tail rows clamped for loads, contributions zeroed).

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

struct Acc {
  float v[4];
};

__device__ __forceinline__ uint32_t shared_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mma_m16n8k16(
    Acc& d, const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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

// B for K@Q^T (contraction over dims) and for the epilogue products'
// contraction over queries via the transposed 16-row load.
__device__ __forceinline__ void ldmatrix_b_dimpair(
    const __nv_bfloat16* tile, int ld, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b_rowtrans(
    const __nv_bfloat16* tile, int ld, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 15);
  const uint32_t addr = shared_u32(tile + row * ld + dim_start);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

__device__ __forceinline__ void cp_async_16(
    __nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
      "l"(src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}


// ---- TMA staging lane (io_dim == HeadDim): swizzle-aware consumers ----
// The universal swizzle law (dump-probe verified): the 16B-atom XOR
// advances once per 128 bytes of tile offset. Compact tiles have
// stride == HeadDim elements per row.

__device__ __forceinline__ void dkdv_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void dkdv_mbarrier_init(uint64_t* bar, int count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(shared_u32(bar)),
      "r"(count));
}

__device__ __forceinline__ void dkdv_mbarrier_expect_tx(
    uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          shared_u32(bar)),
      "r"(bytes));
}

__device__ __forceinline__ void dkdv_mbarrier_wait_parity(
    uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      "  .reg .pred done;\n"
      "  wait:\n"
      "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "  @!done bra wait;\n"
      "}\n" ::"r"(shared_u32(bar)),
      "r"(parity));
}

__device__ __forceinline__ void dkdv_tma_load_3d(
    const CUtensorMap* map, __nv_bfloat16* dst, uint64_t* bar,
    int coord_row, int coord_plane) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(0), "r"(coord_row), "r"(coord_plane),
      "r"(shared_u32(bar)));
}

__device__ __forceinline__ void dkdv_tma_load_4d(
    const CUtensorMap* map, __nv_bfloat16* dst, uint64_t* bar,
    int coord_head, int coord_row, int coord_plane) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(0), "r"(coord_head), "r"(coord_row), "r"(coord_plane),
      "r"(shared_u32(bar)));
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
__device__ __forceinline__ void ldmatrix_b_dimpair_sw(
    const __nv_bfloat16* tile, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int key = key_start + (lane & 7);
  const int col = dim_start + (((lane >> 3) & 1) << 3);
  const uint32_t addr = shared_u32(
      tile + key * HeadDim + dkdv_swz_col<HeadDim>(key, col));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_rowtrans_sw(
    const __nv_bfloat16* tile, int row_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  const int row = row_start + (lane & 15);
  const uint32_t addr = shared_u32(
      tile + row * HeadDim + dkdv_swz_col<HeadDim>(row, dim_start));
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

template <int HeadDim, int StaticSequence, bool TileMajorGrid,
          bool EmitDeltaTrue, bool OuterEqSeq>
__device__ __forceinline__ void stablefold_triangle_dkdv_body(
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
    int io_dim,
    float scale) {
  // StaticSequence == -1: the production exact-tiles lane -- sequence is
  // a pure runtime parameter under the facade's pad-to-tile contract, so
  // every tail guard is compile-time true and vanishes. StaticSequence
  // == 0 keeps the fully guarded fallback for unpadded direct callers.
  const int sequence =
      StaticSequence > 0 ? StaticSequence : runtime_sequence;
  constexpr bool kExact = StaticSequence != 0;
  constexpr int kStride = HeadDim + 8;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  // Prefer key tiles on grid.x whenever z can represent every (batch, outer)
  // task.  This preserves the cache-friendly legacy ordering for all normal
  // shapes while retaining an outer-major fallback for larger B*N.  The two
  // layouts run identical tiles and reductions.
  const int oc = OuterEqSeq ? sequence : outer_count;
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / oc;
  const int outer = outer_task - batch_index * oc;
  const int key_block_start =
      (TileMajorGrid ? blockIdx.x : blockIdx.z) * kKeyBlock;
  const int head = blockIdx.y;
  const int warp_key = key_block_start + warp * kKeyRowsPerWarp;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  // Per-warp K and V slices [16 x D]; shared Q and dO tiles [32 x D].
  __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kKeyRowsPerWarp * kStride;
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kKeyBlock * kStride + warp * kKeyRowsPerWarp * kStride;
  __nv_bfloat16* q_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      2 * kKeyBlock * kStride;
  constexpr int kQBufferElements = 2 * kQueryTile * kStride;  // q + do
  // Parity-double-buffered [warp][query][group] staging for the delta-true
  // row scalar T = sum_k w_k * dP_k. The Z = sum_k w_k plane is dropped:
  // Z == 1 +- 1.3e-6 on the real capture (LSE comes from the same score
  // recompute), so delta_true = T without the divide. Each lane writes its
  // own cells with no shuffles or extra barriers; the fold for tile t runs
  // at the top of tile t+1 behind the existing loop-head barrier,
  // overlapping the MMAs.
  constexpr int kDeltaScratchStride = 9;  // groups + conflict pad
  constexpr int kDeltaScratchWarp = kQueryTile * kDeltaScratchStride;
  constexpr int kDeltaScratchParity = kWarpsPerBlock * kDeltaScratchWarp;
  float* delta_scratch = reinterpret_cast<float*>(
      q_buffers + 2 * kQBufferElements);

  const int64_t plane =
      static_cast<int64_t>(batch_index) * oc + outer;
  const int64_t stats_base =
      (plane * heads + head) * static_cast<int64_t>(sequence);
  const int64_t head_base = stats_base * io_dim;
  // d_output arrives in the public [B, O, N, H, io_dim] layout.
  const int64_t dout_plane_base =
      plane * static_cast<int64_t>(sequence) * heads * io_dim +
      static_cast<int64_t>(head) * io_dim;
  const int64_t mask_base = plane * sequence;
  const int64_t bias_base =
      (static_cast<int64_t>(batch_index) * heads + head) *
      static_cast<int64_t>(sequence) * sequence;
  const int64_t delta_true_base = EmitDeltaTrue
      ? stats_base * key_blocks +
          static_cast<int64_t>(key_block_start / kKeyBlock) * sequence
      : 0;

  // Fixed fold of one tile's staged scalars: per warp
  // ((g0+g1)+(g2+g3))+((g4+g5)+(g6+g7)), then (w0+w1)+(w2+w3); one
  // writer per (query, key-block) workspace cell.
  const auto fold_delta_scratch = [&](int fold_query_start, int parity) {
    if (tid < kQueryTile) {
      const int query_in_tile = tid;
      const float* base_slot = delta_scratch +
          parity * kDeltaScratchParity +
          query_in_tile * kDeltaScratchStride;
      float warp_sums[kWarpsPerBlock];
#pragma unroll
      for (int source = 0; source < kWarpsPerBlock; ++source) {
        const float* slot = base_slot + source * kDeltaScratchWarp;
        const float left = __fadd_rn(
            __fadd_rn(slot[0], slot[1]), __fadd_rn(slot[2], slot[3]));
        const float right = __fadd_rn(
            __fadd_rn(slot[4], slot[5]), __fadd_rn(slot[6], slot[7]));
        warp_sums[source] = __fadd_rn(left, right);
      }
      const int query = fold_query_start + query_in_tile;
      if (kExact || query < sequence) {
        delta_true_partials[delta_true_base + query] = __fadd_rn(
            __fadd_rn(warp_sums[0], warp_sums[1]),
            __fadd_rn(warp_sums[2], warp_sums[3]));
      }
    }
  };

  // Stage this warp's 16 K rows (clamped tails; contributions zeroed later).
  for (int element = lane; element < kKeyRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gk = kExact || warp_key + row < sequence ? warp_key + row
                                                        : sequence - 1;
    const int64_t src = head_base + static_cast<int64_t>(gk) * io_dim + col;
    sk[row * kStride + col] =
        col < io_dim ? k[src] : __float2bfloat16_rn(0.0F);
    sv[row * kStride + col] =
        col < io_dim ? v[src] : __float2bfloat16_rn(0.0F);
  }
  __syncwarp();

  uint32_t k_frags[HeadDim / 16][4];
  uint32_t v_frags[HeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < HeadDim / 16; ++chain) {
    ldmatrix_a16x16(sk, kStride, chain * 16, lane, k_frags[chain]);
    ldmatrix_a16x16(sv, kStride, chain * 16, lane, v_frags[chain]);
  }

  Acc dv_acc[HeadDim / 8];
  Acc dk_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    dv_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
    dk_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const int key_row_low = warp_key + group;
  const int key_row_high = key_row_low + 8;

  // Two-stage cp.async pipeline over the query tiles.
  constexpr int kChunksPerRow =
      HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
  const auto issue_query_tile = [&](int query_start, int buffer) {
    __nv_bfloat16* bq = q_buffers + buffer * kQBufferElements;
    __nv_bfloat16* bdo = bq + kQueryTile * kStride;
    for (int chunk = tid; chunk < kQueryTile * kChunksPerRow;
         chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * 8;
      const int gq = kExact || query_start + row < sequence
          ? query_start + row
          : sequence - 1;
      if (col < io_dim) {
        const int64_t src = head_base +
            static_cast<int64_t>(gq) * io_dim + col;
        const int64_t dout_src = dout_plane_base +
            static_cast<int64_t>(gq) * heads * io_dim + col;
        cp_async_16(bq + row * kStride + col, q + src);
        cp_async_16(bdo + row * kStride + col, d_output + dout_src);
      } else {
        *reinterpret_cast<uint4*>(bq + row * kStride + col) =
            uint4{0U, 0U, 0U, 0U};
        *reinterpret_cast<uint4*>(bdo + row * kStride + col) =
            uint4{0U, 0U, 0U, 0U};
      }
    }
    cp_async_commit_group();
  };
  issue_query_tile(0, 0);
#pragma unroll 1
  for (int query_start = 0; query_start < sequence;
       query_start += kQueryTile) {
    const int buffer = (query_start / kQueryTile) & 1;
    __nv_bfloat16* sq = q_buffers + buffer * kQBufferElements;
    __nv_bfloat16* sdo = sq + kQueryTile * kStride;
    cp_async_wait_all();
    __syncthreads();
    if (query_start + kQueryTile < sequence) {
      issue_query_tile(query_start + kQueryTile, buffer ^ 1);
    }
    if constexpr (EmitDeltaTrue) {
      // Fold the previous tile's staged scalars behind the barrier that
      // already separates its writes; overlaps with this tile's MMAs.
      if (query_start > 0) {
        fold_delta_scratch(query_start - kQueryTile, buffer ^ 1);
      }
    }

    // S^T tile [16 keys x 32 queries]: A = K rows, B = Q dim-pairs.
    Acc score[kQueryTile / 8];
    Acc dp[kQueryTile / 8];
    // Per-lane T accumulators for the whole tile. D16 is issue-bound and
    // wants the scratch stores bunched after the elementwise loop, out of
    // the exp stream; D32 is memory-bound and schedules better with the
    // stores interleaved per column. Same values, same order either way.
    constexpr bool kBunchedDeltaStores = HeadDim == 16;
    float t_tile[kQueryTile / 8][2];
    if constexpr (EmitDeltaTrue) {
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
        t_tile[column][0] = 0.0F;
        t_tile[column][1] = 0.0F;
      }
    }
#pragma unroll
    for (int column = 0; column < kQueryTile / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
      dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        uint32_t b_frag[2];
        ldmatrix_b_dimpair(
            sq, kStride, column * 8, chain * 16, lane, b_frag);
        mma_m16n8k16(score[column], k_frags[chain], b_frag);
        ldmatrix_b_dimpair(
            sdo, kStride, column * 8, chain * 16, lane, b_frag);
        mma_m16n8k16(dp[column], v_frags[chain], b_frag);
      }
    }

    // Elementwise: P^T = exp(S^T*scale + mask + bias - LSE[q]);
    // dS^T = P^T * (dP^T - delta[q]). Column index = query. The bias row
    // offset query*sequence is strength-reduced to one multiply per tile
    // plus per-element adds (the static lanes used to get this for free
    // from the constant length; the runtime lane must do it by hand --
    // the address values are unchanged, so results stay bit-identical).
    // D16 is issue-bound and needs the reduction; D32 is memory-bound and
    // schedules better with the plain per-element form. Address values are
    // identical either way.
    constexpr bool kReduceBiasAddress = HeadDim == 16;
    const int64_t bias_query_base = kReduceBiasAddress
        ? bias_base + static_cast<int64_t>(query_start) * sequence
        : bias_base;
    const int pair_row_offset =
        kReduceBiasAddress ? thread_in_group * 2 * sequence : 0;
#pragma unroll
    for (int column = 0; column < kQueryTile / 8; ++column) {
      const int column_row_offset =
          kReduceBiasAddress ? column * 8 * sequence : 0;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key_row = half == 0 ? key_row_low : key_row_high;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query = query_start + column * 8 + thread_in_group * 2 +
              pair;
          const int index = half * 2 + pair;
          float d_score = 0.0F;
          float probability = 0.0F;
          if (kExact || (key_row < sequence && query < sequence)) {
            const int64_t stats = stats_base + query;
            const float value = score[column].v[index] * scale +
                mask[mask_base + key_row] +
                __bfloat162float(
                    bias[kReduceBiasAddress
                             ? bias_query_base + column_row_offset +
                                   pair_row_offset + pair * sequence +
                                   key_row
                             : bias_query_base +
                                   static_cast<int64_t>(query) * sequence +
                                   key_row]);
            const float lse = logsumexp[stats];
            probability = lse == -std::numeric_limits<float>::infinity()
                ? 0.0F
                : __expf(value - lse);
            d_score = probability * (dp[column].v[index] - delta[stats]);
          }
          score[column].v[index] = d_score;
          if constexpr (EmitDeltaTrue) {
            // w * dP with the pre-overwrite dP; invalid pairs add zeros.
            t_tile[column][pair] += probability * dp[column].v[index];
          }
          dp[column].v[index] = probability;
        }
      }
      if constexpr (EmitDeltaTrue && !kBunchedDeltaStores) {
        // Single-writer per-column stores of this lane's two-key partials;
        // the fold happens behind the next tile's loop-head barrier.
        const int parity = (query_start / kQueryTile) & 1;
        float* warp_slot = delta_scratch +
            parity * kDeltaScratchParity + warp * kDeltaScratchWarp;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query_in_tile = column * 8 + thread_in_group * 2 + pair;
          warp_slot[query_in_tile * kDeltaScratchStride + group] =
              t_tile[column][pair];
        }
      }
    }
    if constexpr (EmitDeltaTrue && kBunchedDeltaStores) {
      // Single-writer bunched stores of this lane's two-key partials; the
      // fold happens behind the next tile's loop-head barrier.
      const int parity = (query_start / kQueryTile) & 1;
      float* warp_slot = delta_scratch +
          parity * kDeltaScratchParity + warp * kDeltaScratchWarp;
#pragma unroll
      for (int column = 0; column < kQueryTile / 8; ++column) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int query_in_tile = column * 8 + thread_in_group * 2 + pair;
          warp_slot[query_in_tile * kDeltaScratchStride + group] =
              t_tile[column][pair];
        }
      }
    }

    // dV += P^T @ dO; dK += dS^T @ Q — contraction over the 32 queries.
#pragma unroll
    for (int chunk = 0; chunk < kQueryTile / 16; ++chunk) {
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
        ldmatrix_b_rowtrans(
            sdo, kStride, chunk * 16, tile * 8, lane, b_frag);
        mma_m16n8k16(dv_acc[tile], p_frag, b_frag);
        ldmatrix_b_rowtrans(
            sq, kStride, chunk * 16, tile * 8, lane, b_frag);
        mma_m16n8k16(dk_acc[tile], ds_frag, b_frag);
      }
    }
  }

  if constexpr (EmitDeltaTrue) {
    // The last tile's scalars have no following loop-head barrier.
    __syncthreads();
    const int last_query_start =
        ((sequence + kQueryTile - 1) / kQueryTile - 1) * kQueryTile;
    fold_delta_scratch(
        last_query_start, (last_query_start / kQueryTile) & 1);
  }

  // Epilogue: dK scaled once; both stored BF16, tail keys skipped.
  const int64_t out_base = head_base +
      static_cast<int64_t>(warp_key) * io_dim;
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = group + half * 8;
      if (!kExact && warp_key + row >= sequence) {
        continue;
      }
      const int column = tile * 8 + thread_in_group * 2;
      if (column >= io_dim) {
        continue;
      }
      out_dv[out_base + static_cast<int64_t>(row) * io_dim + column] =
          __float2bfloat16_rn(dv_acc[tile].v[half * 2]);
      out_dv[out_base + static_cast<int64_t>(row) * io_dim + column + 1] =
          __float2bfloat16_rn(dv_acc[tile].v[half * 2 + 1]);
      out_dk[out_base + static_cast<int64_t>(row) * io_dim + column] =
          __float2bfloat16_rn(dk_acc[tile].v[half * 2] * scale);
      out_dk[out_base + static_cast<int64_t>(row) * io_dim + column + 1] =
          __float2bfloat16_rn(dk_acc[tile].v[half * 2 + 1] * scale);
    }
  }
}

// The plain kernel keeps the exact historical signature, attributes, and
// codegen (the EmitDeltaTrue=false body folds every delta-true statement
// away and the dead nullptr/0 arguments vanish under inlining).
template <int HeadDim, int StaticSequence, bool TileMajorGrid,
          bool OuterEqSeq>
__global__ __launch_bounds__(kThreads) void stablefold_triangle_dkdv_kernel(
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
    int io_dim,
    float scale) {
  stablefold_triangle_dkdv_body<HeadDim, StaticSequence, TileMajorGrid,
                                false, OuterEqSeq>(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
      nullptr, 0, runtime_sequence, outer_count, heads, batch, io_dim,
      scale);
}

template <int HeadDim, int StaticSequence, bool TileMajorGrid,
          bool OuterEqSeq>
__global__ __launch_bounds__(kThreads)
void stablefold_triangle_dkdv_delta_true_kernel(
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
    int io_dim,
    float scale) {
  stablefold_triangle_dkdv_body<HeadDim, StaticSequence, TileMajorGrid,
                                true, OuterEqSeq>(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
      delta_true_partials, key_blocks, runtime_sequence, outer_count, heads,
      batch, io_dim, scale);
}

// Fixed ascending fold of the per-key-block row scalars into
// delta_true = T = sum_k w_k * dP_k (the Z divide is dropped; Z == 1 up
// to LSE rounding, measured 1 +- 1.3e-6 on the real capture). One thread
// owns one row; fully masked rows emit exact zeros because every w_k is
// zero. Row layout matches lse/delta [B,N,H,N].
__global__ void stablefold_delta_true_reduce_kernel(
    const float* __restrict__ partials,
    float* __restrict__ delta_true,
    int64_t rows,
    int key_blocks,
    int sequence) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t query = row % sequence;
  const int64_t row_head = row / sequence;
  const int64_t base =
      row_head * key_blocks * static_cast<int64_t>(sequence) + query;
  float weighted_dp = 0.0F;
#pragma unroll 1
  for (int block = 0; block < key_blocks; ++block) {
    weighted_dp = __fadd_rn(
        weighted_dp,
        partials[base + static_cast<int64_t>(block) * sequence]);
  }
  delta_true[row] = weighted_dp;
}

template <int HeadDim, int StaticSequence, bool EmitDeltaTrue,
          bool OuterEqSeq>
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
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_dim,
    cudaStream_t stream) {
  constexpr int kStride = HeadDim + 8;
  constexpr int kSharedBytes =
      (2 * kKeyBlock * kStride + 4 * kQueryTile * kStride) *
          static_cast<int>(sizeof(__nv_bfloat16)) +
      (EmitDeltaTrue
           ? 2 * kWarpsPerBlock * kQueryTile * 9 *
                 static_cast<int>(sizeof(float))
           : 0);
  const float scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
  const int key_blocks = (sequence + kKeyBlock - 1) / kKeyBlock;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * outer_count;
  const bool tile_major_grid = outer_tasks <= 65535;
  const dim3 grid = tile_major_grid
      ? dim3(
            static_cast<unsigned int>(key_blocks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(outer_tasks))
      : dim3(
            static_cast<unsigned int>(outer_tasks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(key_blocks));
  if constexpr (EmitDeltaTrue) {
    const cudaError_t attribute_status = tile_major_grid
        ? cudaFuncSetAttribute(
              stablefold_triangle_dkdv_delta_true_kernel<
                  HeadDim, StaticSequence, true, OuterEqSeq>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kSharedBytes)
        : cudaFuncSetAttribute(
              stablefold_triangle_dkdv_delta_true_kernel<
                  HeadDim, StaticSequence, false, OuterEqSeq>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kSharedBytes);
    TORCH_CHECK(
        attribute_status == cudaSuccess,
        "failed to opt in StableFold dK/dV shared memory: ",
        cudaGetErrorString(attribute_status));
    if (tile_major_grid) {
      stablefold_triangle_dkdv_delta_true_kernel<HeadDim, StaticSequence,
                                                 true, OuterEqSeq>
          <<<grid, kThreads, kSharedBytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
              logsumexp.data_ptr<float>(),
              delta.data_ptr<float>(),
              mask.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
              delta_true_partials,
              key_blocks,
              sequence,
              outer_count,
              heads,
              batch,
              io_dim,
              scale);
    } else {
      stablefold_triangle_dkdv_delta_true_kernel<HeadDim, StaticSequence,
                                                 false, OuterEqSeq>
          <<<grid, kThreads, kSharedBytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
              logsumexp.data_ptr<float>(),
              delta.data_ptr<float>(),
              mask.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
              delta_true_partials,
              key_blocks,
              sequence,
              outer_count,
              heads,
              batch,
              io_dim,
              scale);
    }
  } else {
    const cudaError_t attribute_status = tile_major_grid
        ? cudaFuncSetAttribute(
              stablefold_triangle_dkdv_kernel<
                  HeadDim, StaticSequence, true, OuterEqSeq>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kSharedBytes)
        : cudaFuncSetAttribute(
              stablefold_triangle_dkdv_kernel<
                  HeadDim, StaticSequence, false, OuterEqSeq>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kSharedBytes);
    TORCH_CHECK(
        attribute_status == cudaSuccess,
        "failed to opt in StableFold dK/dV shared memory: ",
        cudaGetErrorString(attribute_status));
    if (tile_major_grid) {
      stablefold_triangle_dkdv_kernel<HeadDim, StaticSequence, true,
                                      OuterEqSeq>
          <<<grid, kThreads, kSharedBytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
              logsumexp.data_ptr<float>(),
              delta.data_ptr<float>(),
              mask.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
              sequence,
              outer_count,
              heads,
              batch,
              io_dim,
              scale);
    } else {
      stablefold_triangle_dkdv_kernel<HeadDim, StaticSequence, false,
                                      OuterEqSeq>
          <<<grid, kThreads, kSharedBytes, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
              logsumexp.data_ptr<float>(),
              delta.data_ptr<float>(),
              mask.data_ptr<float>(),
              reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dk.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out_dv.data_ptr()),
              sequence,
              outer_count,
              heads,
              batch,
              io_dim,
              scale);
    }
  }
}

template <int HeadDim, bool EmitDeltaTrue>
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
    int sequence,
    int outer_count,
    int heads,
    int batch,
    int io_dim,
    cudaStream_t stream) {
  // One lane for every length: the exact-tiles contract (sequence padded
  // to the tile quantum by the facade) folds all guards at compile time
  // with sequence still runtime. Unpadded lengths take the guarded lane.
  const bool outer_eq = outer_count == sequence;
  if (sequence % kKeyBlock == 0 && sequence % kQueryTile == 0) {
    if (outer_eq) {
      launch<HeadDim, -1, EmitDeltaTrue, true>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
    } else {
      launch<HeadDim, -1, EmitDeltaTrue, false>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
    }
  } else if (outer_eq) {
    launch<HeadDim, 0, EmitDeltaTrue, true>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
        delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
  } else {
    launch<HeadDim, 0, EmitDeltaTrue, false>(
        q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
        delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
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
      batch64 > 0 && batch64 <= std::numeric_limits<int>::max() &&
          outer64 > 0 && outer64 <= std::numeric_limits<int>::max() &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          heads64 > 0 && heads64 <= 65535 && head_dim64 > 0 &&
          head_dim64 <= std::numeric_limits<int>::max() &&
          batch64 * outer64 <= 2147483647LL &&
          (sequence64 + kKeyBlock - 1) / kKeyBlock <= 65535,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  TORCH_CHECK(q.is_contiguous());
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/d_output/out_dk/out_dv must be contiguous BF16 tensors "
        "matching q on one device");
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
  check_q_like(out_dk);
  check_q_like(out_dv);
  const auto check_row_tensor = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kFloat32 && tensor.is_contiguous() &&
            tensor.numel() == q.numel() / head_dim,
        "logsumexp/delta must be contiguous FP32 row tensors on q.device");
  };
  check_row_tensor(logsumexp);
  check_row_tensor(delta);
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() &&
          mask.dtype() == torch::kFloat32 && mask.is_contiguous() &&
          mask.sizes() ==
              torch::IntArrayRef({batch, outer_count, 1, 1, sequence}),
      "mask must be contiguous FP32 [B,N,1,1,N] on q.device");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
      "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
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
      dispatch_sequence<16, EmitDeltaTrue>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
      break;
    case 32:
      dispatch_sequence<32, EmitDeltaTrue>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
      break;
    case 64:
      dispatch_sequence<64, EmitDeltaTrue>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
      break;
    case 128:
      dispatch_sequence<128, EmitDeltaTrue>(
          q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
          delta_true_partials, sequence, outer_count, heads, batch, io_dim,
          stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported storage lane for StableFold dK/dV");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "StableFold dK/dV launch failed");
}

void stablefold_triangle_dkdv_out(
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
  run_dkdv<false>(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
      nullptr);
}

void stablefold_triangle_dkdv_delta_true_out(
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
      delta_true_partials.is_cuda() &&
          delta_true_partials.device() == q.device() &&
          delta_true_partials.dtype() == torch::kFloat32 &&
          delta_true_partials.is_contiguous() &&
          delta_true_partials.numel() >= required_partials,
      "delta_true_partials must be contiguous FP32 with at least "
      "rows*key_blocks elements on q.device");
  TORCH_CHECK(
      out_delta_true.is_cuda() && out_delta_true.device() == q.device() &&
          out_delta_true.dtype() == torch::kFloat32 &&
          out_delta_true.is_contiguous() && out_delta_true.numel() == rows,
      "out_delta_true must be a contiguous FP32 row tensor on q.device");
  run_dkdv<true>(
      q, k, v, d_output, logsumexp, delta, mask, bias, out_dk, out_dv,
      delta_true_partials.data_ptr<float>());
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kReduceThreads = 256;
  const int64_t reduce_blocks =
      (rows + kReduceThreads - 1) / kReduceThreads;
  TORCH_CHECK(
      reduce_blocks > 0 && reduce_blocks <= 2147483647LL,
      "delta_true reduce exceeds the CUDA grid-x contract");
  stablefold_delta_true_reduce_kernel<<<
      static_cast<unsigned int>(reduce_blocks), kReduceThreads, 0, stream>>>(
      delta_true_partials.data_ptr<float>(),
      out_delta_true.data_ptr<float>(),
      rows,
      static_cast<int>(key_blocks),
      static_cast<int>(sequence));
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "StableFold delta_true reduce launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "stablefold_triangle_dkdv_out",
      &stablefold_triangle_dkdv_out,
      "StableFold dK/dV (key-block ownership, runtime H/D/N)");
  module.def(
      "stablefold_triangle_dkdv_delta_true_out",
      &stablefold_triangle_dkdv_delta_true_out,
      "StableFold dK/dV plus the consistent delta_true = sum(P*dP) row "
      "scalars (fixed-order key-block fold, zero atomics)");
}
