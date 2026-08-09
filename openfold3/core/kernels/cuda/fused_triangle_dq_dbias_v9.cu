// SPDX-License-Identifier: Apache-2.0
// Streaming-line fused dQ + dBias, bounded-wave v9.
//
// Modern rebuild of the v7 leaf under the streaming contract (own fixed
// reduction tree, tolerance vs PyTorch, replay-bitwise vs self):
//   * raw mma.sync + ldmatrix fragments (v7 used WMMA with fp32 shared
//     round-trips and CTA-wide barriers per tile);
//   * dS = P * (dP - delta) is pure elementwise on identical accumulator
//     layouts, so the derivative never leaves registers;
//   * one CTA owns one (query-tile, head, 4-outer leaf). K/V depend on the
//     outer, so each warp stages its own K/V tile privately (warp-sync
//     only); CTA barriers exist solely around the per-tile cross-warp
//     dBias combine;
//   * the CTA's dBias plane slice has exactly one writer -- zero cross-CTA
//     coordination, zero atomics, one leaf launch (v7 needed six);
//   * a bounded [leaf,batch,head,query-slab,key] workspace is reused. Query
//     slabs and batch chunks partition independent outputs; leaf waves retain
//     one global fixed left-associated chain in FP32.
//
// Fixed dBias tree: plane[leaf] = (w0 + w1) + (w2 + w3);
// dbias = ((plane0 + plane1) + plane2) + ... in global leaf order. Wave
// boundaries only spill/reload the FP32 root and do not change arithmetic.
//
// Layouts match streaming_triangle_forward_v1: q/k/v/dO [B,N,H,N,D] BF16,
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

template <
    int HeadDim,
    int StaticSequence,
    bool FullQueryTiles,
    bool TaskInGridZ>
__global__ __launch_bounds__(kThreads) void fused_dq_dbias_v9_leaf_kernel(
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
    int heads,
    int batch_base,
    int active_batch,
    int batch_capacity,
    int query_base,
    int query_rows,
    int query_capacity,
    int leaf_base,
    float scale) {
  const int sequence =
      StaticSequence == 0 ? runtime_sequence : StaticSequence;
  constexpr int outer_leaf = kOuterLeaf;
  constexpr int passes = kPasses;
  // Official static lanes divide every tile exactly; their guards are
  // compile-time true and vanish, restoring the ungated codegen.
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
    const int outer = leaf * outer_leaf + pass * kWarpsPerBlock + warp;
    const bool outer_valid = kSequenceExact || outer < sequence;
    const int outer_load = outer_valid ? outer : sequence - 1;
    const int64_t tensor_base =
        ((static_cast<int64_t>(batch_index) * sequence + outer_load) * heads +
         head) *
        static_cast<int64_t>(sequence) * HeadDim;
    const int64_t row_stats_base =
        ((static_cast<int64_t>(batch_index) * sequence + outer_load) * heads +
         head) * sequence;
    const int64_t mask_base =
        (static_cast<int64_t>(batch_index) * sequence + outer_load) * sequence;

    for (int element = lane; element < kQueryBlock * HeadDim;
         element += kWarpSize) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int gq = FullQueryTiles || query_start + row < query_end
          ? query_start + row
          : sequence - 1;
      const int64_t src = tensor_base +
          static_cast<int64_t>(gq) * HeadDim + col;
      my_q[row * kQStride + col] = q[src];
      my_do[row * kQStride + col] = d_output[src];
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

#pragma unroll 1
    for (int key_start = 0; key_start < sequence; key_start += kKeyTile) {
      // Private K/V tile stage for this warp's outer (warp-sync only).
      for (int element = lane; element < kKeyTile * HeadDim;
           element += kWarpSize) {
        const int row = element / HeadDim;
        const int col = element - row * HeadDim;
        const int gk = kSequenceExact || key_start + row < sequence
            ? key_start + row
            : sequence - 1;
        const int64_t src = tensor_base +
            static_cast<int64_t>(gk) * HeadDim + col;
        my_k[row * kKVStride + col] = k[src];
        my_v[row * kKVStride + col] = v[src];
      }
      __syncwarp();

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
              if ((kSequenceExact || (outer_valid && key < sequence)) &&
                  (FullQueryTiles || qrow < query_end)) {
                const float value = score[column].v[index] * scale +
                    mask[mask_base + key] +
                    __bfloat162float(
                        bias[bias_base +
                             static_cast<int64_t>(
                                 sub * kQueryTile + group + half * 8) *
                                 sequence +
                             key]);
                const float probability =
                    lse == -std::numeric_limits<float>::infinity()
                    ? 0.0F
                    : __expf(value - lse);
                d_score = probability * (dp[column].v[index] - dlt);
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
      __syncthreads();
    }

    // dQ epilogue for this warp's outer: scale once, store BF16.
    const int64_t dq_row_base = tensor_base +
        static_cast<int64_t>(query_start) * HeadDim;
    if (kSequenceExact || outer_valid) {
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
            out_dq[dq_row_base + static_cast<int64_t>(row) * HeadDim +
                   column] =
                __float2bfloat16_rn(dq_acc[sub][tile].v[half * 2] * scale);
            out_dq[dq_row_base + static_cast<int64_t>(row) * HeadDim +
                   column + 1] =
                __float2bfloat16_rn(
                    dq_acc[sub][tile].v[half * 2 + 1] * scale);
          }
        }
      }
    }
  }
}

// dbias[h,q,k] = ((plane0 + plane1) + plane2) + ... in leaf order.
__global__ void fused_dq_dbias_v9_finalize_kernel(
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
__global__ void reduce_broadcast_dbias_v9_kernel(
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
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    cudaStream_t stream) {
  constexpr int kQStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  constexpr int kSharedBytes =
      (2 * kWarpsPerBlock * kQueryBlock * kQStride +
       kWarpsPerBlock * 2 * kKeyTile * kKVStride) *
          static_cast<int>(sizeof(__nv_bfloat16)) +
      kWarpsPerBlock * kQueryBlock * kKeyTile *
          static_cast<int>(sizeof(float));
  const float scale = 1.0F / std::sqrt(static_cast<float>(HeadDim));
  const int leaves = (sequence + kOuterLeaf - 1) / kOuterLeaf;
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
        fused_dq_dbias_v9_leaf_kernel<                                    \
            HeadDim, StaticSequence, FullTiles, TaskInZ>,                  \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
        kSharedBytes);                                                     \
    TORCH_CHECK(                                                           \
        status == cudaSuccess,                                             \
        "failed to opt in streaming dQ/dBias shared memory: ",            \
        cudaGetErrorString(status));                                       \
  } while (false)
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(true, true);
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(true, false);
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(false, true);
  SET_DQ_DBIAS_SMEM_ATTRIBUTE(false, false);
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
#define LAUNCH_DQ_DBIAS_LEAF(FullTiles, TaskInZ)                           \
  fused_dq_dbias_v9_leaf_kernel<                                          \
      HeadDim, StaticSequence, FullTiles, TaskInZ>                         \
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
          heads,                                                          \
          batch_base,                                                     \
          active_batch,                                                   \
          batch_chunk,                                                    \
          query_base,                                                     \
          query_rows,                                                     \
          query_span,                                                     \
          leaf_base,                                                      \
          scale)
        if (query_rows % kQueryBlock == 0) {
          if (task_in_grid_z) {
            LAUNCH_DQ_DBIAS_LEAF(true, true);
          } else {
            LAUNCH_DQ_DBIAS_LEAF(true, false);
          }
        } else {
          if (task_in_grid_z) {
            LAUNCH_DQ_DBIAS_LEAF(false, true);
          } else {
            LAUNCH_DQ_DBIAS_LEAF(false, false);
          }
        }
#undef LAUNCH_DQ_DBIAS_LEAF
        TORCH_CHECK(
            cudaGetLastError() == cudaSuccess, "v9 slab leaf launch failed");
        fused_dq_dbias_v9_finalize_kernel
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
            "v9 slab finalizer launch failed");
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
    int heads,
    int batch,
    int query_span,
    int resident_leaves,
    int batch_chunk,
    cudaStream_t stream) {
  // The three official training crops only remove guards and strength-reduce
  // address arithmetic. Bounded waves, ownership, tiles, and the fixed dBias
  // tree are identical to the runtime-N fallback used by every other length.
  switch (sequence) {
    case 384:
      launch<HeadDim, 384>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dq, out_dbias, workspace, sequence, heads,
                           batch, query_span, resident_leaves, batch_chunk,
                           stream);
      break;
    case 640:
      launch<HeadDim, 640>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dq, out_dbias, workspace, sequence, heads,
                           batch, query_span, resident_leaves, batch_chunk,
                           stream);
      break;
    case 768:
      launch<HeadDim, 768>(q, k, v, d_output, logsumexp, delta, mask, bias,
                           out_dq, out_dbias, workspace, sequence, heads,
                           batch, query_span, resident_leaves, batch_chunk,
                           stream);
      break;
    default:
      launch<HeadDim, 0>(q, k, v, d_output, logsumexp, delta, mask, bias,
                         out_dq, out_dbias, workspace, sequence, heads, batch,
                         query_span, resident_leaves, batch_chunk, stream);
      break;
  }
}

void fused_triangle_dq_dbias_v9_out(
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
  const int64_t sequence64 = q.size(1);
  const int64_t heads64 = q.size(2);
  const int64_t head_dim64 = q.size(4);
  TORCH_CHECK(
      batch64 > 0 && batch64 <= std::numeric_limits<int>::max() &&
          sequence64 > 0 && sequence64 <= std::numeric_limits<int>::max() &&
          heads64 > 0 && heads64 <= 65535 && head_dim64 > 0 &&
          head_dim64 <= std::numeric_limits<int>::max() &&
          batch64 * sequence64 <= 2147483647LL &&
          (sequence64 + kQueryBlock - 1) / kQueryBlock <= 65535,
      "batch/sequence/head count exceeds the CUDA grid contract");
  const int batch = static_cast<int>(batch64);
  const int sequence = static_cast<int>(sequence64);
  const int heads = static_cast<int>(heads64);
  const int head_dim = static_cast<int>(head_dim64);
  const int64_t leaves64 =
      (sequence64 + kOuterLeaf - 1) / kOuterLeaf;
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
  TORCH_CHECK(q.size(3) == sequence && q.is_contiguous());
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
  check_q_like(d_output);
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
              torch::IntArrayRef({batch, sequence, 1, 1, sequence}));
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
  switch (head_dim) {
    case 16:
      dispatch_sequence<16>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, batch, query_span,
                 resident_leaves, batch_chunk, stream);
      break;
    case 32:
      dispatch_sequence<32>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, batch, query_span,
                 resident_leaves, batch_chunk, stream);
      break;
    case 64:
      dispatch_sequence<64>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, batch, query_span,
                 resident_leaves, batch_chunk, stream);
      break;
    case 128:
      dispatch_sequence<128>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, batch, query_span,
                 resident_leaves, batch_chunk, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported head_dim for v9: ", head_dim);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v9 launch failed");
}

void reduce_broadcast_dbias_v9_out(
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
  reduce_broadcast_dbias_v9_kernel<<<blocks, threads, 0, stream>>>(
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
      "fused_triangle_dq_dbias_v9_out",
      &fused_triangle_dq_dbias_v9_out,
      "Streaming fused dQ+dBias v9 (query-slab deterministic waves)");
  module.def(
      "reduce_broadcast_dbias_v9_out",
      &reduce_broadcast_dbias_v9_out,
      "Fixed-order broadcast dBias reduction");
}
