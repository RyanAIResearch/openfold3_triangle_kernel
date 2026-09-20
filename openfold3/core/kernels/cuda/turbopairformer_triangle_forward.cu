// SPDX-License-Identifier: Apache-2.0
// TurboPairFormer Triangle Attention forward with online softmax.
//
// FA2-style online softmax: each warp owns 16 query rows and streams KV in
// 64-key tiles; scores never touch HBM. Deterministic by construction in the
// forward (a single CTA owns each row; the KV tile order is fixed), replay
// bitwise vs itself. Accuracy contract is the error-ratio bound vs fp64, NOT
// bitwise vs PyTorch (see PRODUCTION_MANIFEST.md).
//
// Goals: correctness, replay stability, and arbitrary N. Perf work (cp.async double
// buffering, register-layout P reuse, WGMMA) comes after the gate is green.
//
// Internal layout contract used by the public TurboPairFormer facade:
//   q, k, v : [1, N, H=4, N, D=32] BF16, contiguous
//   mask    : [1, N, 1, 1, N] FP32 additive (0 or -inf-like)
//   bias    : [1, 1, H, N, N] BF16 additive
//   out     : [1, N, H, N, D] BF16
//   lse     : [1, N, H, N] FP32 (m + log l, natural log)

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
constexpr int kBlockM = kWarpsPerBlock * kRowsPerWarp;  // 64 query rows / CTA
constexpr int kBlockN = 64;                             // keys per streamed tile

// Repack the projection producer's physical [batch_outer, N, H, D] rows into
// the attention kernels' contiguous [batch_outer, H, N, D] rows. The Python
// facade passes flattened physical-order views, so this is one fused launch
// for q/k/v with no runtime compiler dependency.
// CopyT is selected only from row-byte divisibility and pointer alignment.
// Every vector fits within one logical row; the scalar BF16-sized lane is
// the general fallback. There is no D/N/crop-specific kernel selection.
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
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total_vectors;
       linear += grid_stride) {
    const int64_t row = linear / vectors_per_row;
    const int64_t column = linear % vectors_per_row;
    const int64_t rows_per_outer = static_cast<int64_t>(heads) * sequence;
    const int64_t batch_outer = row / rows_per_outer;
    const int64_t remainder = row % rows_per_outer;
    const int head = static_cast<int>(remainder / sequence);
    const int query = static_cast<int>(remainder % sequence);
    const int64_t source_row =
        (batch_outer * sequence + query) * heads + head;
    const int64_t source = source_row * vectors_per_row + column;
    // Integer assignments preserve every BF16 payload bit without conversion.
    out_q[linear] = q[source];
    out_k[linear] = k[source];
    out_v[linear] = v[source];
  }
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  const __nv_bfloat16 l = __float2bfloat16_rn(low);
  const __nv_bfloat16 h = __float2bfloat16_rn(high);
  return static_cast<uint32_t>(__bfloat16_as_ushort(l)) |
      (static_cast<uint32_t>(__bfloat16_as_ushort(h)) << 16);
}

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

__device__ __forceinline__ void cp_async_16(
    __nv_bfloat16* dst, const __nv_bfloat16* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
      "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all_committed() {
  asm volatile("cp.async.wait_group 0;\n");
}

// A-operand fragment of a 16xD row-major shared tile at (row 0, col dim_start).
__device__ __forceinline__ void ldmatrix_a16x16(
    const __nv_bfloat16* tile, int ld, int dim_start, int lane,
    uint32_t (&frag)[4]) {
  const int row = lane & 15;
  const int col = dim_start + ((lane >> 4) << 3);
  const uint32_t addr = shared_u32(tile + row * ld + col);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// B-operand for QK^T: contraction over D, n = 8 keys. From the row-major
// [keys, D] tile this is the NON-transposed load whose pairs run along the
// contiguous dim axis (byte-identical twin proven in the fused-dP gate).
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

// B-operand for PV: contraction over 16 keys, n = 8 dims. From the row-major
// [keys, D] tile this is the 16-key-row TRANSPOSED load (byte-identical twin
// proven in the compact_cta phase-3 gate).
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


// ---------------------------------------------------------------------------
// TMA + WGMMA lane (HeadDim <= 64). K and V arrive by TMA bulk copies into
// swizzled shared boxes; QK^T is one wide m64n64k16 warpgroup MMA per
// 16-dim chain (probe-verified descriptor: LBO=64, SBO=HeadDim in 16-byte
// units = the 8-key group stride, mode by swizzle span); PV keeps the
// mma.sync path with ldmatrix applying the swizzle XOR in its address.
// Bitwise equal to the legacy lane (wgmma == chained mma.sync, gated).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(shared_u32(bar)),
      "r"(count));
}

__device__ __forceinline__ void mbarrier_expect_tx(
    uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          shared_u32(bar)),
      "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
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

__device__ __forceinline__ void tma_load_3d_slice(
    const CUtensorMap* map, __nv_bfloat16* dst, uint64_t* bar,
    int coord_key, int coord_plane) {
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

// One wide QK atom over all eight key groups; operand order score[0].v[0..3]
// ... score[7].v[0..3] matches the accumulator layout d[group*4+quad]
// (probe-verified against the chained-n8 reference).
__device__ __forceinline__ void wgmma_m64n64k16(
    Acc (&score)[8], const uint32_t (&a)[4], uint64_t descriptor_b,
    int accumulate) {
  asm volatile(
      "{\n"
      "  .reg .pred use_acc;\n"
      "  setp.ne.b32 use_acc, %37, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "   %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "  {%32,%33,%34,%35}, %36, use_acc, 1, 1, 0;\n"
      "}\n"
      : "+f"(score[0].v[0]), "+f"(score[0].v[1]), "+f"(score[0].v[2]),
        "+f"(score[0].v[3]), "+f"(score[1].v[0]), "+f"(score[1].v[1]),
        "+f"(score[1].v[2]), "+f"(score[1].v[3]), "+f"(score[2].v[0]),
        "+f"(score[2].v[1]), "+f"(score[2].v[2]), "+f"(score[2].v[3]),
        "+f"(score[3].v[0]), "+f"(score[3].v[1]), "+f"(score[3].v[2]),
        "+f"(score[3].v[3]), "+f"(score[4].v[0]), "+f"(score[4].v[1]),
        "+f"(score[4].v[2]), "+f"(score[4].v[3]), "+f"(score[5].v[0]),
        "+f"(score[5].v[1]), "+f"(score[5].v[2]), "+f"(score[5].v[3]),
        "+f"(score[6].v[0]), "+f"(score[6].v[1]), "+f"(score[6].v[2]),
        "+f"(score[6].v[3]), "+f"(score[7].v[0]), "+f"(score[7].v[1]),
        "+f"(score[7].v[2]), "+f"(score[7].v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(descriptor_b),
        "r"(accumulate));
}

// Keytrans ldmatrix over a TMA-swizzled tile: each key row spans
// HeadDim/8 16-byte atoms and the swizzle XOR advances once per 128
// BYTES of tile offset (dump-probe verified for SW32/SW64/SW128), so the
// atom permutation is chunk XOR ((key*HeadDim) >> 6, masked to the row).
template <int HeadDim>
__device__ __forceinline__ void ldmatrix_b_keytrans_sw(
    const __nv_bfloat16* tile, int key_start, int dim_start, int lane,
    uint32_t (&frag)[2]) {
  constexpr int kAtoms = HeadDim / 8;
  const int key = key_start + (lane & 15);
  const int atom =
      (dim_start >> 3) ^ (((key * HeadDim) >> 6) & (kAtoms - 1));
  const uint32_t addr = shared_u32(tile + key * HeadDim + atom * 8);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(frag[0]), "=r"(frag[1])
      : "r"(addr));
}

// Generic 3-D TMA box load: coordinates (innermost first) into a swizzled box.
__device__ __forceinline__ void tma_load_3d_box(
    const CUtensorMap* map, __nv_bfloat16* dst, uint64_t* bar,
    int coord0, int coord1, int coord2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx"
      "::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(shared_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)),
      "r"(coord0),
      "r"(coord1),
      "r"(coord2),
      "r"(shared_u32(bar)));
}

// Pair-bias box: TMA [kBlockM rows][kBlockN keys] BF16 with the 128-byte
// swizzle, so each 128-byte row stores 16-byte chunk c at c ^ (row & 7).
// One x4 ldmatrix yields, for the 16 rows of a warp and keys 16*key_pair..+15,
// the exact m16n8 accumulator fragment layout: frag[0]/frag[1] are rows r and
// r+8 of the lower 8 keys, frag[2]/frag[3] of the upper 8 keys, each register
// holding the two adjacent keys 2*(lane&3), 2*(lane&3)+1 in memory order.
__device__ __forceinline__ void ldmatrix_bias_x4(
    const __nv_bfloat16* tile, int row_base, int key_pair, int lane,
    uint32_t (&frag)[4]) {
  static_assert(
      kBlockN * sizeof(__nv_bfloat16) == 128,
      "bias box rows must span exactly the 128-byte swizzle");
  const int row = row_base + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int chunk = key_pair * 2 + ((lane >> 4) & 1);
  const uint32_t addr =
      shared_u32(tile + row * kBlockN + ((chunk ^ (row & 7)) << 3));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
      : "r"(addr));
}

// Reduce a value across the 4 lanes of one accumulator row quad.
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

// ExactTiles is the production lane: under the facade's pad-to-tile
// contract every tile divides the sequence exactly, so all tail guards
// fold away at compile time with sequence still a runtime parameter.
// io_dim <= HeadDim: global tensors carry io_dim head channels; the kernel
// zero-extends them to the HeadDim compute lane at staging time (padded dot
// terms are +0.0 — exact) and reads/writes global memory at io_dim strides.
// io_dim == HeadDim is the plain lane; the ternaries stay out of hot loops.
template <int HeadDim, bool TileMajorGrid, bool ExactTiles, bool OuterEqSeq>
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
    float scale) {
  constexpr int kKVStride = HeadDim + 8;
  static_assert(HeadDim % 16 == 0 && (kKVStride * 2) % 16 == 0);
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  // Keep query tiles on grid.x whenever the CUDA z dimension can hold all
  // (batch, outer) tasks.  This is the cache-friendly layout used by the
  // original single-prefix kernel, generalized to every batch/sequence that
  // fits the hardware grid.  The alternate layout is a functional fallback
  // for larger B*N; both execute the same tiles and reduction order.
  // Dual-lane: outer == sequence (triangle) folds back to the original
  // single-variable index math; decoupled shapes take the general lane.
  const int oc = OuterEqSeq ? sequence : outer_count;
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / oc;
  const int outer = outer_task - batch_index * oc;
  const int query_block =
      (TileMajorGrid ? blockIdx.x : blockIdx.z) * kBlockM;
  const int head = blockIdx.y;
  // Tail-CTA warps whose rows all fall beyond the sequence stay resident:
  // they must keep participating in the cooperative K/V staging loop and its
  // __syncthreads(). Their scores are forced to -inf and their stores are
  // row-guarded, so their state is inert.
  const int warp_query = query_block + warp * kRowsPerWarp;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  // Per-warp Q tile [16 x D], then shared K tile, V tile, per-warp P tile.
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kRowsPerWarp * kKVStride;
  __nv_bfloat16* kv_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kBlockM * kKVStride;

  // One shared plane index: keeps the three bases on a single CSE chain so
  // ptxas never rematerializes the 64-bit products inside the hot loops.
  const int64_t plane =
      static_cast<int64_t>(batch_index) * oc + outer;
  const int64_t plane_row_base =
      (plane * heads + head) * static_cast<int64_t>(sequence);
  const int64_t head_base = plane_row_base * io_dim;
  const int64_t bias_base =
      (static_cast<int64_t>(batch_index) * heads + head) *
      static_cast<int64_t>(sequence) * sequence;
  const int64_t mask_base = plane * sequence;
  // Output is written in the PUBLIC [B, O, N, H, io_dim] layout directly.
  const int64_t out_plane_base =
      plane * static_cast<int64_t>(sequence) * heads * io_dim +
      static_cast<int64_t>(head) * io_dim;

  // Stage this warp's Q rows (guarded rows load row 0; their stores are
  // guarded at the epilogue, and their softmax state is discarded).
  for (int element = lane; element < kRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gq = ExactTiles || warp_query + row < sequence
        ? warp_query + row
        : 0;
    sq[row * kKVStride + col] = col < io_dim
        ? q[head_base + static_cast<int64_t>(gq) * io_dim + col]
        : __float2bfloat16_rn(0.0F);
  }
  __syncwarp();

  uint32_t q_frags[HeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < HeadDim / 16; ++chain) {
    ldmatrix_a16x16(sq, kKVStride, chain * 16, lane, q_frags[chain]);
  }

  // Online softmax state: two accumulator rows per lane (r and r+8).
  float m_state[2] = {-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()};
  float l_state[2] = {0.0F, 0.0F};
  Acc o_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    o_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const int key_tiles = (sequence + kBlockN - 1) / kBlockN;
  // Two-stage cp.async pipeline over the K/V tiles: tile t+1 copies while
  // tile t computes. Guarded tail rows stage key 0 and are neutralized by
  // the -inf score mask below. 16-byte chunks; HeadDim*2B = 4 chunks/row.
  constexpr int kChunksPerRow =
      HeadDim * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
  constexpr int kChunkElements = 16 / static_cast<int>(sizeof(__nv_bfloat16));
  const auto issue_tile = [&](int key_tile, int buffer) {
    __nv_bfloat16* bk = kv_buffers + buffer * 2 * kBlockN * kKVStride;
    __nv_bfloat16* bv = bk + kBlockN * kKVStride;
    const int key_start = key_tile * kBlockN;
    for (int chunk = threadIdx.x; chunk < kBlockN * kChunksPerRow;
         chunk += kThreads) {
      const int row = chunk / kChunksPerRow;
      const int col = (chunk - row * kChunksPerRow) * kChunkElements;
      const int gk = ExactTiles || key_start + row < sequence
          ? key_start + row
          : 0;
      if (col < io_dim) {
        const int64_t src =
            head_base + static_cast<int64_t>(gk) * io_dim + col;
        cp_async_16(bk + row * kKVStride + col, k + src);
        cp_async_16(bv + row * kKVStride + col, v + src);
      } else {
        *reinterpret_cast<uint4*>(bk + row * kKVStride + col) =
            uint4{0U, 0U, 0U, 0U};
        *reinterpret_cast<uint4*>(bv + row * kKVStride + col) =
            uint4{0U, 0U, 0U, 0U};
      }
    }
    cp_async_commit();
  };
  issue_tile(0, 0);
#pragma unroll 1
  for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const int key_start = key_tile * kBlockN;
    const int buffer = key_tile & 1;
    __nv_bfloat16* sk = kv_buffers + buffer * 2 * kBlockN * kKVStride;
    __nv_bfloat16* sv = sk + kBlockN * kKVStride;
    cp_async_wait_all_committed();
    __syncthreads();
    if (key_tile + 1 < key_tiles) {
      issue_tile(key_tile + 1, buffer ^ 1);
    }

    // Scores for this warp's 16 rows x 64 keys: 8 key-columns of 16x8 HMMA.
    Acc score[kBlockN / 8];
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
      for (int chain = 0; chain < HeadDim / 16; ++chain) {
        uint32_t k_frag[2];
        ldmatrix_b_dimpair(
            sk, kKVStride, column * 8, chain * 16, lane, k_frag);
        mma_m16n8k16(score[column], q_frags[chain], k_frag);
      }
    }

    // scale + mask + bias in FP32, tail keys forced to -inf.
    float tile_max[2] = {-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = warp_query + group + half * 8;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          const int key = key_start + column * 8 + thread_in_group * 2 + pair;
          float value = score[column].v[half * 2 + pair];
          if (ExactTiles || (key < sequence && row < sequence)) {
            value = value * scale + mask[mask_base + key] +
                __bfloat162float(
                    bias[bias_base +
                         static_cast<int64_t>(row) * sequence + key]);
          } else {
            value = -std::numeric_limits<float>::infinity();
          }
          score[column].v[half * 2 + pair] = value;
          tile_max[half] = fmaxf(tile_max[half], value);
        }
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      tile_max[half] = quad_max(tile_max[half]);
    }

    // Online rescale.
    float rescale[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float m_new = fmaxf(m_state[half], tile_max[half]);
      rescale[half] =
          m_state[half] == -std::numeric_limits<float>::infinity()
          ? 0.0F
          : __expf(m_state[half] - m_new);
      m_state[half] = m_new;
      l_state[half] *= rescale[half];
    }
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        o_acc[tile].v[half * 2] *= rescale[half];
        o_acc[tile].v[half * 2 + 1] *= rescale[half];
      }
    }

    // P = exp(S - m), accumulate l, stage P to shared for the PV HMMA.
    float row_sum[2] = {0.0F, 0.0F};
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float p0 = score[column].v[half * 2];
        float p1 = score[column].v[half * 2 + 1];
        p0 = p0 == -std::numeric_limits<float>::infinity()
            ? 0.0F
            : __expf(p0 - m_state[half]);
        p1 = p1 == -std::numeric_limits<float>::infinity()
            ? 0.0F
            : __expf(p1 - m_state[half]);
        row_sum[half] += p0 + p1;
        score[column].v[half * 2] = p0;
        score[column].v[half * 2 + 1] = p1;
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      l_state[half] += quad_sum(row_sum[half]);
    }

    // O += P @ V: pack adjacent probability accumulator columns straight
    // into A fragments (identical bytes to the former shared round-trip),
    // ascending 16-key chunks -> fixed order, deterministic.
#pragma unroll
    for (int chunk = 0; chunk < kBlockN / 16; ++chunk) {
      uint32_t p_frag[4];
      p_frag[0] = pack_bf16_pair(
          score[chunk * 2].v[0], score[chunk * 2].v[1]);
      p_frag[1] = pack_bf16_pair(
          score[chunk * 2].v[2], score[chunk * 2].v[3]);
      p_frag[2] = pack_bf16_pair(
          score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
      p_frag[3] = pack_bf16_pair(
          score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        uint32_t v_frag[2];
        ldmatrix_b_keytrans(
            sv, kKVStride, chunk * 16, tile * 8, lane, v_frag);
        mma_m16n8k16(o_acc[tile], p_frag, v_frag);
      }
    }
  }

  // Epilogue: O / l, LSE = m + log l.
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = warp_query + group + half * 8;
    if (!ExactTiles && row >= sequence) {
      continue;
    }
    const float inv_l = l_state[half] == 0.0F ? 0.0F : 1.0F / l_state[half];
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int dim = tile * 8 + thread_in_group * 2 + pair;
        if (dim < io_dim) {
          const float o_value = o_acc[tile].v[half * 2 + pair] * inv_l;
          const __nv_bfloat16 o_rounded = __float2bfloat16_rn(o_value);
          const int64_t out_index = out_plane_base +
              static_cast<int64_t>(row) * heads * io_dim + dim;
          out[out_index] = o_rounded;
          if (shadow_residual != nullptr) {
            shadow_residual[out_index] = __float2bfloat16_rn(
                o_value - __bfloat162float(o_rounded));
          }
        }
      }
    }
    if (thread_in_group == 0) {
      lse[plane_row_base + row] =
          m_state[half] + __logf(l_state[half]);
    }
  }
}


// TMA + WGMMA production lane (HeadDim <= 64): identical math and reduction
// order to the legacy kernel (wgmma proved bitwise equal to the chained
// mma.sync; the swizzled keytrans reads the same bytes). K, V and the pair
// bias box are staged by the bulk tensor engine; the additive mask tile is
// staged one key tile ahead in shared memory. Bias values are read from the
// swizzled shared box straight into the accumulator fragment layout, so the
// score expression (scale, mask, bias, max, exp, sum) is evaluated in the
// same order as the global-load lane.
template <int HeadDim, bool TileMajorGrid, bool ExactTiles, bool OuterEqSeq,
          bool BiasTma>
__global__ __launch_bounds__(kThreads) void
turbopairformer_triangle_forward_tma_kernel(
    const __grid_constant__ CUtensorMap k_map,
    const __grid_constant__ CUtensorMap v_map,
    const __grid_constant__ CUtensorMap b_map,
    const __nv_bfloat16* __restrict__ q,
    const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    __nv_bfloat16* __restrict__ shadow_residual,
    int sequence,
    int outer_count,
    int heads,
    float scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int kQStride = HeadDim + 8;
  constexpr int kTileElements = kBlockN * HeadDim;  // K/V TMA box, swizzled
  constexpr int kBiasTileElements = kBlockM * kBlockN;  // bias box, 128B swizzle
  // Every TMA box must start on a 1024-byte swizzle atom.
  static_assert(
      ((kBlockM * kQStride) * static_cast<int>(sizeof(__nv_bfloat16))) %
              1024 == 0 &&
          (kTileElements * static_cast<int>(sizeof(__nv_bfloat16))) % 1024 ==
              0 &&
          (kBiasTileElements * static_cast<int>(sizeof(__nv_bfloat16))) %
                  1024 == 0,
      "TMA boxes must be 1024-byte aligned in shared memory");
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int oc = OuterEqSeq ? sequence : outer_count;
  const int outer_task = TileMajorGrid ? blockIdx.z : blockIdx.x;
  const int batch_index = outer_task / oc;
  const int outer = outer_task - batch_index * oc;
  const int query_block =
      (TileMajorGrid ? blockIdx.x : blockIdx.z) * kBlockM;
  const int head = blockIdx.y;
  const int warp_query = query_block + warp * kRowsPerWarp;

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      warp * kRowsPerWarp * kQStride;
  __nv_bfloat16* k_buffers = reinterpret_cast<__nv_bfloat16*>(smem_raw) +
      kBlockM * kQStride;
  __nv_bfloat16* v_buffers = k_buffers + 2 * kTileElements;
  __nv_bfloat16* b_buffers = v_buffers + 2 * kTileElements;
  float* smask = reinterpret_cast<float*>(
      b_buffers + (BiasTma ? 2 * kBiasTileElements : 0));
  __shared__ __align__(8) uint64_t k_barriers[2];
  __shared__ __align__(8) uint64_t v_barriers[2];
  __shared__ __align__(8) uint64_t b_barriers[2];

  const int64_t plane =
      static_cast<int64_t>(batch_index) * oc + outer;
  const int64_t plane_row_base =
      (plane * heads + head) * static_cast<int64_t>(sequence);
  const int64_t head_base = plane_row_base * HeadDim;
  const int64_t mask_base = plane * sequence;
  const int64_t bias_base =
      (static_cast<int64_t>(batch_index) * heads + head) *
      static_cast<int64_t>(sequence) * sequence;
  // Output is written in the PUBLIC [B, O, N, H, D] layout directly, so the
  // facade returns it with no transpose/contiguous pass.
  const int64_t out_plane_base =
      plane * static_cast<int64_t>(sequence) * heads * HeadDim +
      static_cast<int64_t>(head) * HeadDim;
  // The launch contract guarantees batch*outer*heads fits int32.
  const int tma_plane = static_cast<int>(plane * heads + head);
  // The pair bias is shared across the outer dimension: one [N, N] plane per
  // (batch, head), so its box coordinate ignores `outer`.
  const int bias_plane = batch_index * heads + head;

  if (threadIdx.x == 0) {
    mbarrier_init(&k_barriers[0], 1);
    mbarrier_init(&k_barriers[1], 1);
    mbarrier_init(&v_barriers[0], 1);
    mbarrier_init(&v_barriers[1], 1);
    if constexpr (BiasTma) {
      mbarrier_init(&b_barriers[0], 1);
      mbarrier_init(&b_barriers[1], 1);
    }
  }

  for (int element = lane; element < kRowsPerWarp * HeadDim;
       element += kWarpSize) {
    const int row = element / HeadDim;
    const int col = element - row * HeadDim;
    const int gq = ExactTiles || warp_query + row < sequence
        ? warp_query + row
        : 0;
    sq[row * kQStride + col] =
        q[head_base + static_cast<int64_t>(gq) * HeadDim + col];
  }
  // Mask tile 0: the additive key mask is one row per plane, shared by every
  // query row and head, so the CTA stages 64 floats per key tile. Later
  // tiles are staged one tile ahead inside the loop.
  if (threadIdx.x < kBlockN) {
    const int key = threadIdx.x;
    smask[key] =
        ExactTiles || key < sequence ? mask[mask_base + key] : 0.0F;
  }
  __syncthreads();  // covers mbarrier_init and mask tile 0 visibility too

  uint32_t q_frags[HeadDim / 16][4];
#pragma unroll
  for (int chain = 0; chain < HeadDim / 16; ++chain) {
    ldmatrix_a16x16(sq, kQStride, chain * 16, lane, q_frags[chain]);
  }

  float m_state[2] = {-std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()};
  float l_state[2] = {0.0F, 0.0F};
  Acc o_acc[HeadDim / 8];
#pragma unroll
  for (int tile = 0; tile < HeadDim / 8; ++tile) {
    o_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const auto issue_k = [&](int key_tile, int buffer) {
    if (threadIdx.x == 0) {
      mbarrier_expect_tx(
          &k_barriers[buffer],
          kTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
      tma_load_3d_slice(
          &k_map, k_buffers + buffer * kTileElements, &k_barriers[buffer],
          key_tile * kBlockN, tma_plane);
    }
  };
  const auto issue_v = [&](int key_tile, int buffer) {
    if (threadIdx.x == 0) {
      mbarrier_expect_tx(
          &v_barriers[buffer],
          kTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
      tma_load_3d_slice(
          &v_map, v_buffers + buffer * kTileElements, &v_barriers[buffer],
          key_tile * kBlockN, tma_plane);
    }
  };
  const auto issue_b = [&](int key_tile, int buffer) {
    if constexpr (BiasTma) {
      if (threadIdx.x == 0) {
        mbarrier_expect_tx(
            &b_barriers[buffer],
            kBiasTileElements * static_cast<int>(sizeof(__nv_bfloat16)));
        tma_load_3d_box(
            &b_map, b_buffers + buffer * kBiasTileElements,
            &b_barriers[buffer], key_tile * kBlockN, query_block, bias_plane);
      }
    }
  };

  const int key_tiles = (sequence + kBlockN - 1) / kBlockN;
  issue_k(0, 0);
  issue_v(0, 0);
  issue_b(0, 0);
#pragma unroll 1
  for (int key_tile = 0; key_tile < key_tiles; ++key_tile) {
    const int key_start = key_tile * kBlockN;
    const int buffer = key_tile & 1;
    const uint32_t parity = (key_tile >> 1) & 1;
    __nv_bfloat16* sb_k = k_buffers + buffer * kTileElements;
    __nv_bfloat16* sv = v_buffers + buffer * kTileElements;
    const __nv_bfloat16* sb =
        b_buffers + (BiasTma ? buffer * kBiasTileElements : 0);
    const float* tile_mask = smask + buffer * kBlockN;
    const bool has_next = key_tile + 1 < key_tiles;

    // Next tile's mask segment: loaded now, stored into the other mask
    // buffer before this tile's trailing barrier. That buffer's readers all
    // finished before the previous trailing barrier.
    float next_mask = 0.0F;
    if (has_next && threadIdx.x < kBlockN) {
      const int key = key_start + kBlockN + threadIdx.x;
      next_mask =
          ExactTiles || key < sequence ? mask[mask_base + key] : 0.0F;
    }

    // QK only needs K; V's completion wait is deferred to just before PV.
    mbarrier_wait_parity(&k_barriers[buffer], parity);
    proxy_fence();

    // One wide m64n64k16 per 16-dim chain over the swizzled K box.
    // Descriptor: LBO=64, SBO=HeadDim (in 16B units: 8 keys * HeadDim*2B
    // per row / 16), mode by swizzle span, chain offset 32B.
    constexpr uint64_t kMode = HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
    constexpr uint64_t kWideLbo = 64;
    constexpr uint64_t kWideSbo = HeadDim;
    Acc score[kBlockN / 8];
#pragma unroll
    for (int column = 0; column < kBlockN / 8; ++column) {
      score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
    }
    wgmma_fence();
#pragma unroll
    for (int chain = 0; chain < HeadDim / 16; ++chain) {
      const uint64_t descriptor =
          ((static_cast<uint64_t>(shared_u32(sb_k)) + chain * 32) >> 4) |
          (kWideLbo << 16) | (kWideSbo << 32) | (kMode << 62);
      wgmma_m64n64k16(score, q_frags[chain], descriptor, chain != 0);
    }
    wgmma_commit();
    // Overlap window: stage the NEXT tile's K, V and bias while the QK MMAs
    // run.
    if (has_next) {
      issue_k(key_tile + 1, buffer ^ 1);
      issue_v(key_tile + 1, buffer ^ 1);
      issue_b(key_tile + 1, buffer ^ 1);
    }
    wgmma_wait_zero();

    // This tile's bias box (issued one tile ahead), then the async->generic
    // proxy fence so ldmatrix sees it.
    if constexpr (BiasTma) {
      mbarrier_wait_parity(&b_barriers[buffer], parity);
      proxy_fence();
    }

    float tile_max[2] = {-std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()};
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int key_pair = 0; key_pair < kBlockN / 16; ++key_pair) {
      // 16 rows x 16 keys of bias in the accumulator fragment layout.
      uint32_t bias_frag[4];
      if constexpr (BiasTma) {
        ldmatrix_bias_x4(sb, warp * kRowsPerWarp, key_pair, lane, bias_frag);
      }
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int column = key_pair * 2 + sub;
        const float2 mask_pair = *reinterpret_cast<const float2*>(
            tile_mask + column * 8 + thread_in_group * 2);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = warp_query + group + half * 8;
          const int first_key = key_start + column * 8 + thread_in_group * 2;
#pragma unroll
          for (int pair = 0; pair < 2; ++pair) {
            const int key = first_key + pair;
            float value = score[column].v[half * 2 + pair];
            if (ExactTiles || (key < sequence && row < sequence)) {
              float bias_value;
              if constexpr (BiasTma) {
                const __nv_bfloat16* pair_values =
                    reinterpret_cast<const __nv_bfloat16*>(
                        &bias_frag[sub * 2 + half]);
                bias_value = __bfloat162float(pair_values[pair]);
              } else {
                // Hardware-ineligible bias layouts retain the original
                // scalar load, including odd row strides and offset views.
                bias_value = __bfloat162float(
                    bias[bias_base + static_cast<int64_t>(row) * sequence + key]);
              }
              value = value * scale +
                  (pair == 0 ? mask_pair.x : mask_pair.y) + bias_value;
            } else {
              value = -std::numeric_limits<float>::infinity();
            }
            score[column].v[half * 2 + pair] = value;
            tile_max[half] = fmaxf(tile_max[half], value);
          }
        }
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      tile_max[half] = quad_max(tile_max[half]);
    }

    float rescale[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float m_new = fmaxf(m_state[half], tile_max[half]);
      rescale[half] =
          m_state[half] == -std::numeric_limits<float>::infinity()
          ? 0.0F
          : __expf(m_state[half] - m_new);
      m_state[half] = m_new;
      l_state[half] *= rescale[half];
    }
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        o_acc[tile].v[half * 2] *= rescale[half];
        o_acc[tile].v[half * 2 + 1] *= rescale[half];
      }
    }

    // Evaluate each original probability expression immediately before its
    // 16-key PV chunk. Each score keeps its own -inf guard, including when
    // another score is NaN and fmaxf left the row maximum at -inf. Row-sum
    // additions and each MMA accumulator still visit columns in original order.
    float row_sum[2] = {0.0F, 0.0F};

    // V's TMA completion, then the async->generic proxy fence so ldmatrix
    // sees the tile; PV keeps the proven mma.sync path over the swizzled
    // layout.
    mbarrier_wait_parity(&v_barriers[buffer], parity);
    proxy_fence();
#pragma unroll
    for (int chunk = 0; chunk < kBlockN / 16; ++chunk) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int column = chunk * 2 + sub;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          float p0 = score[column].v[half * 2];
          float p1 = score[column].v[half * 2 + 1];
          p0 = p0 == -std::numeric_limits<float>::infinity()
              ? 0.0F
              : __expf(p0 - m_state[half]);
          p1 = p1 == -std::numeric_limits<float>::infinity()
              ? 0.0F
              : __expf(p1 - m_state[half]);
          row_sum[half] += p0 + p1;
          score[column].v[half * 2] = p0;
          score[column].v[half * 2 + 1] = p1;
        }
      }
      uint32_t p_frag[4];
      p_frag[0] = pack_bf16_pair(
          score[chunk * 2].v[0], score[chunk * 2].v[1]);
      p_frag[1] = pack_bf16_pair(
          score[chunk * 2].v[2], score[chunk * 2].v[3]);
      p_frag[2] = pack_bf16_pair(
          score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
      p_frag[3] = pack_bf16_pair(
          score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
#pragma unroll
      for (int tile = 0; tile < HeadDim / 8; ++tile) {
        uint32_t v_frag[2];
        ldmatrix_b_keytrans_sw<HeadDim>(
            sv, chunk * 16, tile * 8, lane, v_frag);
        mma_m16n8k16(o_acc[tile], p_frag, v_frag);
      }
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      l_state[half] += quad_sum(row_sum[half]);
    }

    if (has_next && threadIdx.x < kBlockN) {
      smask[(buffer ^ 1) * kBlockN + threadIdx.x] = next_mask;
    }
    __syncthreads();  // all reads of this tile's buffers done before reuse
  }

  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = warp_query + group + half * 8;
    if (!ExactTiles && row >= sequence) {
      continue;
    }
    const float inv_l = l_state[half] == 0.0F ? 0.0F : 1.0F / l_state[half];
#pragma unroll
    for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int dim = tile * 8 + thread_in_group * 2 + pair;
        const float o_value = o_acc[tile].v[half * 2 + pair] * inv_l;
        const __nv_bfloat16 o_rounded = __float2bfloat16_rn(o_value);
        const int64_t out_index = out_plane_base +
            static_cast<int64_t>(row) * heads * HeadDim + dim;
        out[out_index] = o_rounded;
        if (shadow_residual != nullptr) {
          // Rounding residual of the public BF16 output; consumed by the
          // shadow delta so delta ~= sum P*dP without a second recompute.
          shadow_residual[out_index] = __float2bfloat16_rn(
              o_value - __bfloat162float(o_rounded));
        }
      }
    }
    if (thread_in_group == 0) {
      lse[plane_row_base + row] =
          m_state[half] + __logf(l_state[half]);
    }
  }
#endif
}

template <int HeadDim>
constexpr int shared_bytes() {
  return (kBlockM * (HeadDim + 8) + 4 * kBlockN * (HeadDim + 8)) *
      static_cast<int>(sizeof(__nv_bfloat16));
}

template <int HeadDim, bool BiasTma>
constexpr int shared_bytes_tma() {
  // Per-warp Q, double-buffered K/V, optional double-buffered bias, and
  // a double-buffered FP32 mask row. Each TMA box is 1024-byte aligned.
  return (kBlockM * (HeadDim + 8) + 4 * kBlockN * HeadDim +
          (BiasTma ? 2 * kBlockM * kBlockN : 0)) *
      static_cast<int>(sizeof(__nv_bfloat16)) +
      2 * kBlockN * static_cast<int>(sizeof(float));
}

// cuTensorMapEncodeTiled resolved through the runtime so the extension
// never links libcuda directly.
using TensorMapEncodeFn = CUresult (*)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

TensorMapEncodeFn tensor_map_encode_fn() {
  static const TensorMapEncodeFn encode = [] {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query =
        cudaDriverEntryPointSymbolNotFound;
    const cudaError_t status = cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &symbol, cudaEnableDefault, &query);
    TORCH_CHECK(
        status == cudaSuccess && query == cudaDriverEntryPointSuccess &&
            symbol != nullptr,
        "cuTensorMapEncodeTiled is unavailable in this driver");
    return reinterpret_cast<TensorMapEncodeFn>(symbol);
  }();
  return encode;
}

// [planes, sequence, HeadDim] view of a contiguous [B, O, H, N, D] tensor,
// boxed per key tile with the swizzle matching the row span.
CUtensorMap make_kv_tensor_map(
    const torch::Tensor& source, int head_dim, int sequence, int64_t planes) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[3] = {
      static_cast<cuuint64_t>(head_dim),
      static_cast<cuuint64_t>(sequence),
      static_cast<cuuint64_t>(planes)};
  const cuuint64_t global_stride[2] = {
      static_cast<cuuint64_t>(head_dim) * sizeof(__nv_bfloat16),
      static_cast<cuuint64_t>(head_dim) * sequence * sizeof(__nv_bfloat16)};
  const cuuint32_t box_dim[3] = {
      static_cast<cuuint32_t>(head_dim), kBlockN, 1};
  const cuuint32_t element_stride[3] = {1, 1, 1};
  const CUtensorMapSwizzle swizzle = head_dim == 16
      ? CU_TENSOR_MAP_SWIZZLE_32B
      : (head_dim == 32 ? CU_TENSOR_MAP_SWIZZLE_64B
                        : CU_TENSOR_MAP_SWIZZLE_128B);
  const CUresult result = tensor_map_encode_fn()(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      source.data_ptr(),
      global_dim,
      global_stride,
      box_dim,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(
      result == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return tensor_map;
}

// [planes, rows, keys] view of the contiguous [B, 1, H, N, N] bias, boxed per
// (query block, key tile) with the 128-byte swizzle used by the
// accumulator-layout ldmatrix reads. The caller checks
// the 16-byte-aligned base and 16-byte-multiple row stride TMA requires.
CUtensorMap make_bias_tensor_map(
    const torch::Tensor& bias, int sequence, int64_t planes) {
  CUtensorMap tensor_map;
  const cuuint64_t global_dim[3] = {
      static_cast<cuuint64_t>(sequence),
      static_cast<cuuint64_t>(sequence),
      static_cast<cuuint64_t>(planes)};
  const cuuint64_t global_stride[2] = {
      static_cast<cuuint64_t>(sequence) * sizeof(__nv_bfloat16),
      static_cast<cuuint64_t>(sequence) * sequence * sizeof(__nv_bfloat16)};
  const cuuint32_t box_dim[3] = {kBlockN, kBlockM, 1};
  const cuuint32_t element_stride[3] = {1, 1, 1};
  const CUresult result = tensor_map_encode_fn()(
      &tensor_map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      bias.data_ptr(),
      global_dim,
      global_stride,
      box_dim,
      element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(
      result == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled (bias) failed: ", static_cast<int>(result));
  return tensor_map;
}

template <int HeadDim, bool BiasTma>
void launch_forward_tma_lane(
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
  const int64_t planes =
      static_cast<int64_t>(batch) * outer_count * heads;
  const CUtensorMap k_map =
      make_kv_tensor_map(k, HeadDim, sequence, planes);
  const CUtensorMap v_map =
      make_kv_tensor_map(v, HeadDim, sequence, planes);
  CUtensorMap b_map{};
  if constexpr (BiasTma) {
    b_map = make_bias_tensor_map(
        bias, sequence, static_cast<int64_t>(batch) * heads);
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
  const int query_blocks = (sequence + kBlockM - 1) / kBlockM;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * outer_count;
  const bool tile_major_grid = outer_tasks <= 65535;
  const dim3 grid = tile_major_grid
      ? dim3(
            static_cast<unsigned int>(query_blocks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(outer_tasks))
      : dim3(
            static_cast<unsigned int>(outer_tasks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(query_blocks));
  const bool exact_tiles =
      sequence % kBlockM == 0 && sequence % kBlockN == 0;
#define LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(TileMajor, Exact, OuterEq)            \
  do {                                                                      \
    const cudaError_t status = cudaFuncSetAttribute(                        \
        turbopairformer_triangle_forward_tma_kernel<HeadDim, TileMajor, Exact,   \
                                               OuterEq, BiasTma>,                    \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                        \
        shared_bytes_tma<HeadDim, BiasTma>());                                       \
    TORCH_CHECK(                                                            \
        status == cudaSuccess,                                              \
        "failed to opt in TurboPairFormer TMA forward shared memory: ",          \
        cudaGetErrorString(status));                                        \
    turbopairformer_triangle_forward_tma_kernel<HeadDim, TileMajor, Exact,       \
                                           OuterEq, BiasTma>                         \
        <<<grid, kThreads, shared_bytes_tma<HeadDim, BiasTma>(), stream>>>(          \
            k_map,                                                          \
            v_map,                                                          \
            b_map,                                                          \
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),           \
            mask.data_ptr<float>(),                                         \
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),        \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),               \
            lse.data_ptr<float>(),                                          \
            shadow_residual,                                          \
            sequence,                                                       \
            outer_count,                                                    \
            heads,                                                          \
            scale);                                                         \
  } while (false)
  const bool outer_eq = outer_count == sequence;
  if (tile_major_grid) {
    if (exact_tiles) {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(true, true, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(true, true, false);
    } else {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(true, false, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(true, false, false);
    }
  } else {
    if (exact_tiles) {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(false, true, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(false, true, false);
    } else {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(false, false, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD_TMA(false, false, false);
    }
  }
#undef LAUNCH_TURBOPAIRFORMER_FORWARD_TMA
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
  // These are tensor-map hardware constraints, not crop/data special cases.
  // Original TMA K/V handling remains active for every other bias layout;
  // only bias reads use the scalar fallback there.
  const bool bias_tma_ok = sequence % 8 == 0 &&
      reinterpret_cast<std::uintptr_t>(bias.data_ptr()) % 16 == 0;
  if (bias_tma_ok) {
    launch_forward_tma_lane<HeadDim, true>(
        q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
        shadow_residual, stream);
  } else {
    launch_forward_tma_lane<HeadDim, false>(
        q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
        shadow_residual, stream);
  }
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
  const float scale = 1.0F / std::sqrt(static_cast<float>(io_dim));
  const int query_blocks = (sequence + kBlockM - 1) / kBlockM;
  const int64_t outer_tasks = static_cast<int64_t>(batch) * outer_count;
  const bool tile_major_grid = outer_tasks <= 65535;
  const dim3 grid = tile_major_grid
      ? dim3(
            static_cast<unsigned int>(query_blocks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(outer_tasks))
      : dim3(
            static_cast<unsigned int>(outer_tasks),
            static_cast<unsigned int>(heads),
            static_cast<unsigned int>(query_blocks));
  const bool exact_tiles =
      sequence % kBlockM == 0 && sequence % kBlockN == 0;
#define LAUNCH_TURBOPAIRFORMER_FORWARD(TileMajor, Exact, OuterEq)                \
  do {                                                                      \
    const cudaError_t status = cudaFuncSetAttribute(                        \
        turbopairformer_triangle_forward_kernel<HeadDim, TileMajor, Exact,       \
                                           OuterEq>,                        \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                        \
        shared_bytes<HeadDim>());                                           \
    TORCH_CHECK(                                                            \
        status == cudaSuccess,                                              \
        "failed to opt in TurboPairFormer forward shared memory: ",              \
        cudaGetErrorString(status));                                        \
    turbopairformer_triangle_forward_kernel<HeadDim, TileMajor, Exact, OuterEq>  \
        <<<grid, kThreads, shared_bytes<HeadDim>(), stream>>>(              \
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),          \
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),          \
            reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),          \
            mask.data_ptr<float>(),                                        \
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),       \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),              \
            lse.data_ptr<float>(),                                         \
            shadow_residual,                                         \
            sequence,                                         \
            outer_count,                                                      \
            heads,                                                         \
            batch,                                                         \
            io_dim,                                                        \
            scale);                                                        \
  } while (false)
  const bool outer_eq = outer_count == sequence;
  if (tile_major_grid) {
    if (exact_tiles) {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD(true, true, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD(true, true, false);
    } else {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD(true, false, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD(true, false, false);
    }
  } else {
    if (exact_tiles) {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD(false, true, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD(false, true, false);
    } else {
      if (outer_eq) LAUNCH_TURBOPAIRFORMER_FORWARD(false, false, true);
      else LAUNCH_TURBOPAIRFORMER_FORWARD(false, false, false);
    }
  }
#undef LAUNCH_TURBOPAIRFORMER_FORWARD
}

std::vector<torch::Tensor> turbopairformer_pack_qkv(
    const torch::Tensor& q_source,
    const torch::Tensor& k_source,
    const torch::Tensor& v_source,
    int64_t batch_outer,
    int64_t heads64,
    int64_t sequence64,
    int64_t head_dim64) {
  TORCH_CHECK(
      batch_outer > 0 && heads64 > 0 && heads64 <= 65535 &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          head_dim64 > 0 && head_dim64 <= 64 && head_dim64 % 2 == 0,
      "TurboPairFormer pack requires positive batch_outer/N, H<=65535, and an "
      "even D<=64");
  TORCH_CHECK(
      batch_outer <=
          std::numeric_limits<int64_t>::max() / heads64 / sequence64 /
              head_dim64,
      "TurboPairFormer pack element count overflows int64");
  const int64_t elements =
      batch_outer * heads64 * sequence64 * head_dim64;
  const auto check_source = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q_source.device() &&
            tensor.dtype() == torch::kBFloat16 && tensor.is_contiguous() &&
            tensor.numel() == elements,
        "TurboPairFormer pack sources must be contiguous BF16 tensors with the "
        "declared element count on one CUDA device");
  };
  check_source(q_source);
  check_source(k_source);
  check_source(v_source);

  const int64_t rows = batch_outer * heads64 * sequence64;
  const int64_t blocks = (rows + 7) / 8;
  TORCH_CHECK(
      blocks <= 2147483647LL,
      "TurboPairFormer pack exceeds the CUDA grid-x contract");
  const int heads = static_cast<int>(heads64);
  const int sequence = static_cast<int>(sequence64);
  const int head_dim = static_cast<int>(head_dim64);
  const c10::cuda::CUDAGuard device_guard(q_source.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto options = q_source.options();
  auto out_q = torch::empty(
      {batch_outer, heads64, sequence64, head_dim64}, options);
  auto out_k = torch::empty_like(out_q);
  auto out_v = torch::empty_like(out_q);
  const auto aligned = [](const torch::Tensor& tensor, int bytes) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) &
            static_cast<std::uintptr_t>(bytes - 1)) == 0;
  };
  const int row_bytes = head_dim * static_cast<int>(sizeof(__nv_bfloat16));
  const auto safe_width = [&](int bytes) {
    return row_bytes % bytes == 0 &&
        aligned(q_source, bytes) && aligned(k_source, bytes) &&
        aligned(v_source, bytes) && aligned(out_q, bytes) &&
        aligned(out_k, bytes) && aligned(out_v, bytes);
  };
  // These are hardware copy widths, not a list of supported problem shapes.
  // BF16 tensors guarantee the two-byte scalar alignment used by fallback.
  const int copy_bytes = safe_width(16) ? 16
      : (safe_width(8) ? 8 : (safe_width(4) ? 4 : 2));
  const int64_t vectors_per_row = row_bytes / copy_bytes;
  const int64_t total_vectors = rows * vectors_per_row;
  constexpr int kCopyThreads = 128;
  const int64_t copy_blocks = std::min<int64_t>(
      (total_vectors + kCopyThreads - 1) / kCopyThreads, 2147483647LL);
  const dim3 block(kCopyThreads);
  const dim3 grid(static_cast<unsigned int>(copy_blocks));
#define LAUNCH_TURBOPAIRFORMER_PACK_COPY(CopyType)                                  \
  turbopairformer_pack_qkv_kernel<CopyType><<<grid, block, 0, stream>>>(              \
      reinterpret_cast<const CopyType*>(q_source.data_ptr()),                 \
      reinterpret_cast<const CopyType*>(k_source.data_ptr()),                 \
      reinterpret_cast<const CopyType*>(v_source.data_ptr()),                 \
      reinterpret_cast<CopyType*>(out_q.data_ptr()),                           \
      reinterpret_cast<CopyType*>(out_k.data_ptr()),                           \
      reinterpret_cast<CopyType*>(out_v.data_ptr()),                           \
      total_vectors, heads, sequence, vectors_per_row)
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
  TORCH_CHECK(
      cudaGetLastError() == cudaSuccess,
      "TurboPairFormer pack QKV launch failed");
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
          (sequence64 + kBlockM - 1) / kBlockM <= 65535,
      "batch/outer/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int outer_count = static_cast<int>(outer64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const auto check_q_like = [&](const torch::Tensor& tensor) {
    TORCH_CHECK(
        tensor.is_cuda() && tensor.device() == q.device() &&
            tensor.dtype() == torch::kBFloat16 &&
            tensor.sizes() == q.sizes() && tensor.is_contiguous(),
        "k/v/out must be contiguous BF16 tensors matching q on one device");
  };
  check_q_like(k);
  check_q_like(v);
  TORCH_CHECK(
      out.is_cuda() && out.device() == q.device() &&
          out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
          out.sizes() ==
              torch::IntArrayRef(
                  {batch, outer_count, sequence, heads, head_dim}),
      "out must be contiguous BF16 [B,O,N,H,D] on q.device");
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(
      mask.is_cuda() && mask.device() == q.device() &&
          mask.dtype() == torch::kFloat32 && mask.is_contiguous() &&
          mask.sizes() ==
              torch::IntArrayRef({batch, outer_count, 1, 1, sequence}),
      "mask must be contiguous FP32 [B,O,1,1,N] on q.device");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() ==
              torch::IntArrayRef({batch, 1, heads, sequence, sequence}),
      "bias must be contiguous BF16 [B,1,H,N,N] on q.device");
  TORCH_CHECK(
      lse.is_cuda() && lse.device() == q.device() &&
          lse.dtype() == torch::kFloat32 && lse.is_contiguous() &&
          lse.sizes() ==
              torch::IntArrayRef({batch, outer_count, heads, sequence}),
      "lse must be contiguous FP32 [B,O,H,N] on q.device");

  // TMA rejects misaligned K/V addresses; the legacy cp.async lane also
  // requires 16-byte addresses. Reject before either launch rather than
  // falling through from a failed TMA eligibility check to unsafe cp.async.
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(k.data_ptr()) % 16 == 0 &&
          reinterpret_cast<std::uintptr_t>(v.data_ptr()) % 16 == 0,
      "TurboPairFormer forward K/V storage must be 16-byte aligned");
  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  // TMA lanes need io_dim == storage lane AND the flattened (batch, outer,
  // head) plane coordinate inside the tensor map's int32 contract; the
  // legacy lane zero-extends any io_dim % 8 == 0 head dim to its storage
  // lane at staging time and covers every remaining shape.
  const bool tma_planes_ok = batch64 * outer64 * heads64 <= 2147483647LL;
  TORCH_CHECK(
      head_dim % 8 == 0 && head_dim <= 128,
      "head_dim must be a multiple of 8 and <= 128, got ", head_dim);
  const int storage = head_dim <= 16
      ? 16
      : (head_dim <= 32 ? 32 : (head_dim <= 64 ? 64 : 128));
  switch (storage) {
    case 16:
      if (tma_planes_ok && head_dim == 16) {
        launch_forward_tma<16>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, shadow_residual, stream);
      } else {
        launch_forward<16>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, head_dim, shadow_residual, stream);
      }
      break;
    case 32:
      if (tma_planes_ok && head_dim == 32) {
        launch_forward_tma<32>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, shadow_residual, stream);
      } else {
        launch_forward<32>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, head_dim, shadow_residual, stream);
      }
      break;
    case 64:
      if (tma_planes_ok && head_dim == 64) {
        launch_forward_tma<64>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, shadow_residual, stream);
      } else {
        launch_forward<64>(
            q, k, v, mask, bias, out, lse, sequence, outer_count, heads,
            batch, head_dim, shadow_residual, stream);
      }
      break;
    default:
      launch_forward<128>(
          q, k, v, mask, bias, out, lse, sequence, outer_count, heads, batch,
          head_dim, shadow_residual, stream);
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

// Shadow variant: additionally emits the BF16 rounding residual of the
// output (same [B,O,N,H,D] layout) -- the compressed state that lets the
// backward evaluate a probability-consistent delta without a second
// reconstruction.
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
      shadow_residual.is_cuda() &&
          shadow_residual.device() == out.device() &&
          shadow_residual.dtype() == torch::kBFloat16 &&
          shadow_residual.is_contiguous() &&
          shadow_residual.sizes() == out.sizes(),
      "shadow_residual must be contiguous BF16 matching out");
  turbopairformer_triangle_forward_impl(
      q, k, v, mask, bias, out, lse,
      reinterpret_cast<__nv_bfloat16*>(shadow_residual.data_ptr()));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "turbopairformer_pack_qkv",
      &turbopairformer_pack_qkv,
      "TurboPairFormer fused physical [N,H,D] to logical [H,N,D] QKV pack");
  module.def(
      "turbopairformer_triangle_forward_out",
      &turbopairformer_triangle_forward_out,
      "TurboPairFormer Triangle Attention forward (online softmax)");
  module.def(
      "turbopairformer_triangle_forward_shadow_out",
      &turbopairformer_triangle_forward_shadow_out,
      "Forward + BF16 rounding-residual shadow for the SR delta");
}
