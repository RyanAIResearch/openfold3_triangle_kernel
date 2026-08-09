// SPDX-License-Identifier: Apache-2.0
// Streaming-line fused dQ + dBias, v9.
//
// Modern rebuild of the v7 leaf under the streaming contract (own fixed
// reduction tree, tolerance vs PyTorch, replay-bitwise vs self):
//   * raw mma.sync + ldmatrix fragments (v7 used WMMA with fp32 shared
//     round-trips and CTA-wide barriers per tile);
//   * dS = P * (dP - delta) is pure elementwise on identical accumulator
//     layouts, so the derivative never leaves registers;
//   * one CTA owns one (query-tile, head, 16-outer leaf). K/V depend on the
//     outer, so each warp stages its own K/V tile privately (warp-sync
//     only); CTA barriers exist solely around the per-tile cross-warp
//     dBias combine;
//   * the CTA's dBias plane slice has exactly one writer and the four
//     passes accumulate serially inside the CTA -- zero cross-CTA
//     coordination, zero atomics, one leaf launch (v7 needed six);
//   * finalize folds the leaf planes with one fixed left-associated chain.
//
// Fixed dBias tree: plane[leaf] = sum over passes (serial) of
// ((w0 + w1) + (w2 + w3)); dbias = ((plane0 + plane1) + plane2) + ...
//
// Layouts match streaming_triangle_forward_v1: q/k/v/dO [1,N,4,N,32] BF16,
// mask [1,N,1,1,N] FP32 additive, bias [1,1,4,N,N] BF16, lse/delta
// [1,N,4,N] FP32, out_dq [1,N,4,N,32] BF16, out_dbias [1,1,4,N,N] FP32.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

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

template <int HeadDim, int StaticSequence>
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
    int runtime_outer_leaf,
    float scale) {
  const int sequence =
      StaticSequence == 0 ? runtime_sequence : StaticSequence;
  // Adaptive dBias plane policy: official static lanes keep the frozen
  // leaf of 4 outers (bitwise-preserving); the runtime lane grows the
  // leaf with N so the workspace stays O(N^2) instead of O(N^3).
  const int outer_leaf =
      StaticSequence == 0 ? runtime_outer_leaf : kOuterLeaf;
  const int passes = outer_leaf / kWarpsPerBlock;
  // Official static lanes divide every tile exactly; their guards are
  // compile-time true and vanish, restoring the ungated codegen.
  constexpr bool kExact = StaticSequence != 0;
  constexpr int kQStride = HeadDim + 8;
  constexpr int kKVStride = HeadDim + 8;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int query_start = blockIdx.x * kQueryBlock;
  const int head = blockIdx.y;
  const int leaf = blockIdx.z;
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
      ((static_cast<int64_t>(leaf) * heads + head) *
           static_cast<int64_t>(sequence) +
       query_start) *
      sequence;
  const int64_t bias_base =
      (static_cast<int64_t>(head) * sequence + query_start) *
      static_cast<int64_t>(sequence);

#pragma unroll 1
  for (int pass = 0; pass < passes; ++pass) {
    const int outer = leaf * outer_leaf + pass * kWarpsPerBlock + warp;
    const bool outer_valid = kExact || outer < sequence;
    const int outer_load = outer_valid ? outer : sequence - 1;
    const int64_t tensor_base =
        (static_cast<int64_t>(outer_load) * heads + head) *
        static_cast<int64_t>(sequence) * HeadDim;
    const int64_t row_stats_base =
        (static_cast<int64_t>(outer_load) * heads + head) * sequence;
    const int64_t mask_base = static_cast<int64_t>(outer_load) * sequence;

    for (int element = lane; element < kQueryBlock * HeadDim;
         element += kWarpSize) {
      const int row = element / HeadDim;
      const int col = element - row * HeadDim;
      const int gq = kExact || query_start + row < sequence
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
            (kExact || qrow < sequence ? qrow : sequence - 1);
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
        const int gk = kExact || key_start + row < sequence
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
              if (kExact ||
                  (outer_valid && key < sequence && qrow < sequence)) {
                const float value = score[column].v[index] * scale +
                    mask[mask_base + key] +
                    __bfloat162float(
                        bias[bias_base +
                             static_cast<int64_t>(
                                 sub * kQueryTile + group + half * 8) *
                                 sequence +
                             key]);
                const float probability = __expf(value - lse);
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
        if (!kExact &&
            (query_start + row >= sequence || key_start + key >= sequence)) {
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
    if (kExact || outer_valid) {
#pragma unroll
      for (int sub = 0; sub < kQuerySubTiles; ++sub) {
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = sub * kQueryTile + group + half * 8;
            if (!kExact && query_start + row >= sequence) {
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
    int64_t plane_elements,
    int leaves) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= plane_elements) {
    return;
  }
  float root = __bfloat162float(workspace[linear]);
#pragma unroll 1
  for (int leaf = 1; leaf < leaves; ++leaf) {
    root = __fadd_rn(
        root,
        __bfloat162float(
            workspace[static_cast<int64_t>(leaf) * plane_elements + linear]));
  }
  out_dbias[linear] = root;
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
    int outer_leaf,
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
  const int leaves = (sequence + outer_leaf - 1) / outer_leaf;
  const dim3 grid(
      (sequence + kQueryBlock - 1) / kQueryBlock,
      static_cast<unsigned int>(heads),
      leaves);
  const cudaError_t attribute_status = cudaFuncSetAttribute(
      fused_dq_dbias_v9_leaf_kernel<HeadDim, StaticSequence>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSharedBytes);
  TORCH_CHECK(
      attribute_status == cudaSuccess,
      "failed to opt in streaming dQ/dBias shared memory: ",
      cudaGetErrorString(attribute_status));
  fused_dq_dbias_v9_leaf_kernel<HeadDim, StaticSequence>
      <<<grid, kThreads, kSharedBytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(d_output.data_ptr()),
          logsumexp.data_ptr<float>(),
          delta.data_ptr<float>(),
          mask.data_ptr<float>(),
          reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out_dq.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr()),
          sequence,
          heads,
          outer_leaf,
          scale);
  const int64_t plane_elements =
      static_cast<int64_t>(heads) * sequence * sequence;
  const int finalize_threads = 256;
  const int64_t finalize_blocks =
      (plane_elements + finalize_threads - 1) / finalize_threads;
  fused_dq_dbias_v9_finalize_kernel
      <<<static_cast<unsigned int>(finalize_blocks),
         finalize_threads,
         0,
         stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr()),
          out_dbias.data_ptr<float>(),
          plane_elements,
          leaves);
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
    int outer_leaf,
    cudaStream_t stream) {
  // Official lengths get compile-time-sequence fast lanes (full unrolling
  // and strength reduction); every other length runs the identical source
  // with a runtime bound. One kernel, compiler-specialized lanes.
  switch (sequence) {
    case 384:
      launch<HeadDim, 384>(q, k, v, d_output, logsumexp, delta, mask,
                           bias, out_dq, out_dbias, workspace, sequence,
                           heads, kOuterLeaf, stream);
      break;
    case 640:
      launch<HeadDim, 640>(q, k, v, d_output, logsumexp, delta, mask,
                           bias, out_dq, out_dbias, workspace, sequence,
                           heads, kOuterLeaf, stream);
      break;
    case 768:
      launch<HeadDim, 768>(q, k, v, d_output, logsumexp, delta, mask,
                           bias, out_dq, out_dbias, workspace, sequence,
                           heads, kOuterLeaf, stream);
      break;
    default:
      launch<HeadDim, 0>(q, k, v, d_output, logsumexp, delta, mask, bias,
                         out_dq, out_dbias, workspace, sequence, heads,
                         outer_leaf, stream);
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
    torch::Tensor& workspace) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.dim() == 5 && q.size(2) >= 1);
  const int sequence = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(4));
  TORCH_CHECK(
      sequence > 0 && sequence <= 65535 && heads <= 65535,
      "sequence/head count exceeds the CUDA y/z grid contract");
  TORCH_CHECK(q.size(0) == 1 && q.size(3) == sequence && q.is_contiguous());
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
          mask.sizes() == torch::IntArrayRef({1, sequence, 1, 1, sequence}));
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == q.device() &&
          bias.dtype() == torch::kBFloat16 && bias.is_contiguous() &&
          bias.sizes() == torch::IntArrayRef({1, 1, heads, sequence, sequence}));
  TORCH_CHECK(
      out_dbias.is_cuda() && out_dbias.device() == q.device() &&
          out_dbias.dtype() == torch::kFloat32 && out_dbias.is_contiguous() &&
          out_dbias.sizes() ==
              torch::IntArrayRef({1, 1, heads, sequence, sequence}));
  const int outer_leaf =
      kOuterLeaf * ((sequence + 767) / 768);  // adaptive plane policy
  const int64_t leaves = (sequence + outer_leaf - 1) / outer_leaf;
  TORCH_CHECK(
      workspace.is_cuda() && workspace.device() == q.device() &&
      workspace.dtype() == torch::kBFloat16 && workspace.is_contiguous() &&
      workspace.numel() >=
          leaves * static_cast<int64_t>(heads) * sequence * sequence);

  const c10::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (head_dim) {
    case 16:
      dispatch_sequence<16>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, outer_leaf, stream);
      break;
    case 32:
      dispatch_sequence<32>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, outer_leaf, stream);
      break;
    case 64:
      dispatch_sequence<64>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, outer_leaf, stream);
      break;
    case 128:
      dispatch_sequence<128>(q, k, v, d_output, logsumexp, delta, mask, bias, out_dq,
                 out_dbias, workspace, sequence, heads, outer_leaf, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported head_dim for v9: ", head_dim);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v9 launch failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_triangle_dq_dbias_v9_out",
      &fused_triangle_dq_dbias_v9_out,
      "Streaming fused dQ+dBias v8 (register dS, single-writer planes)");
}
