# TurboPairFormer CUDA Kernels: Architecture & Developer Tutorial

This document serves as the comprehensive architectural reference and developer guide for the **TurboPairFormer** CUDA kernels in OpenFold3 (`openfold3/core/kernels/cuda/`).

TurboPairFormer provides high-throughput, memory-fused, and replay-deterministic implementations of the two core geometric operations in the AlphaFold3 / OpenFold3 Pairformer stack:
1. **Triangle Attention** (with Starting / Ending node variants and Pair Bias)
2. **Triangle Multiplicative Update (TriMul)** (Incoming and Outgoing edge updates)

---

## 1. Directory Structure & Kernel Overview

```
openfold3/core/kernels/cuda/
├── turbopairformer_triangle_attention.py   # Public PyTorch Autograd facade (entry point)
├── turbopairformer_triangle_forward.cu     # Forward: online softmax, cp.async & TMA+WGMMA
├── turbopairformer_triangle_forward.py     # C++ extension loader for forward
├── turbopairformer_triangle_delta.cu       # Backward: row-correction delta precompute
├── turbopairformer_triangle_delta.py       # C++ extension loader for delta
├── turbopairformer_triangle_dq_dbias.cu    # Backward: fused dQ + dBias (and full SR backward)
├── turbopairformer_triangle_dq_dbias.py    # C++ extension loader for dQ/dBias
├── turbopairformer_triangle_dkdv.cu        # Backward: split dK + dV kernels
├── turbopairformer_triangle_dkdv.py        # C++ extension loader for dK/dV
├── turbopairformer_trimul_cuda.cu          # TriMul: In/Out matrix multiplications & gating
├── turbopairformer_trimul.py               # TriMul PyTorch wrapper
├── _aot_runtime.py                         # AOT & JIT compilation / symbol loader
└── tutorial.md                             # Architecture & developer tutorial (this file)
```

---

## 2. Triangle Attention Architecture

### 2.1 Mathematical Formulation

Triangle Attention applies scaled dot-product attention along edges of a residue pair graph, augmented with a pair-representation bias. Fix one batch item, outer plane, and head; their indices are omitted below. Then $Q$, $K$, and $V$ are $N \times D$ matrices:

$$S = \frac{QK^T}{\sqrt{D}} + \text{mask} + \text{bias}$$

$$P = \text{softmax}(S) \qquad \text{(row-wise)}$$

$$O = PV$$

The mask is a length-$N$ row vector broadcast across query rows; bias has shape $N \times N$. Here $O$ denotes the output matrix; in tensor shapes below, $O$ denotes the outer dimension. Full batch and head dimensions are shown once in the layout table rather than repeated in every equation.

### 2.2 The 5D Tensor Shape Contract

In standard Multi-Head Attention, tensors are 4D: $[B, H, N, D]$. In OpenFold3 Triangle Attention, tensors are **5D**: $[B, O, H, N, D]$.

| Dimension | Symbol | Meaning | Broadcasting Behavior |
| :--- | :--- | :--- | :--- |
| **Batch** | $B$ | Batch size | Standard batching |
| **Outer Planes** | $O$ | Independent node/edge planes | In Triangle Attention, $O$ represents the orthogonal dimension of the pair matrix (e.g., rows when attending over columns). $O$ is not required to equal $N$. |
| **Heads** | $H$ | Attention heads | Head parallel axis |
| **Sequence Length** | $N$ | Number of tokens / residues | $Q$ and $K$ sequence length |
| **Head Dimension** | $D$ | Channels per head | $D \in \{16, 32, 64\}$ (public facade) |

#### Memory Layouts at Kernel Boundary:
- **Inputs ($Q, K, V$)**: `[B, O, H, N, D]` BF16, contiguous.
- **Mask**: `[B, O, 1, 1, N]` FP32 additive mask (shared across heads and query rows within each $(B, O)$ plane).
- **Bias**: `[B, 1, H, N, N]` BF16 additive pair bias (shared across outer planes $O$).
- **Output ($O$)**: `[B, O, N, H, D]` BF16 (transposed for downstream linear projections).
- **LSE ($\log \sum \exp$)**: `[B, O, H, N]` FP32 log-sum-exp statistics for backward.
- **Shadow Residual**: `[B, O, N, H, D]` BF16 output rounding residual ($O_{\text{fp32}} - O_{\text{bf16}}$).

---

## 3. Forward Pass: Dual-Kernel Engine ([turbopairformer_triangle_forward.cu](file:///Users/yideran/Desktop/research_product/openfold_product/openfold3/core/kernels/cuda/turbopairformer_triangle_forward.cu))

The forward pass keeps attention weights entirely on-chip using **Online Softmax** (FlashAttention-2 recurrence).

```
   +---------------------------------------------------------------+
   |             Input Validation & Dispatch (Host)                |
   |              turbopairformer_triangle_forward_impl            |
   +---------------------------------------------------------------+
                               |
               +---------------+---------------+
               |                               |
       D in {16, 32, 64} &&              All other shapes /
       Planes <= int32 max               Fallback path
               v                               v
   +-----------------------+       +-----------------------+
   |   Path 2: TMA + WGMMA |       |  Path 1: cp.async     |
   |   (Hopper SM90a Spec) |       |  + mma.sync (SM80+)   |
   +-----------------------+       +-----------------------+
```

### 3.1 Online Softmax Recurrence

For one query row, key tiles of size $kBlockN = 64$ are streamed sequentially. In each iteration, $s$ is the current tile's score vector and $V$ its value matrix. The scalars $m$ and $l$ hold the running maximum and weight sum; $A$ is the length-$D$ unnormalized output vector. Initially $m=-\infty$, $l=0$, and $A=0$. Lowercase $p$ denotes the current tile's unnormalized weights, and $m'$ the updated maximum:

1. **Tile Maximum**:
   $$m' = \max(m, \max(s))$$
2. **Rescale Factor**:
   $$\alpha = \exp(m - m') \quad (\alpha = 0 \text{ if } m = -\infty)$$
3. **Tile Unnormalized Probabilities**:
   $$p = \exp(s - m') \quad (p = 0 \text{ for entries with } s = -\infty)$$
4. **Update Denominator & Accumulator**:
   $$l = \alpha l + \sum p$$
   $$A = \alpha A + \text{BF16}(p)V$$
   $$m = m'$$
5. **Final Epilogue**:
   $$O = \text{BF16}(A / l), \quad \text{LSE} = m + \ln(l)$$

These short symbols are local to the equations. In the code, $m$, $l$, and $A$ correspond to `running_row_max`, `running_weight_sum`, and the row's distributed `output_accumulators`. The running state and weight sum use FP32; the matrix product uses BF16-rounded $p$. If all scores are true $-\infty$ and $V$ is finite, the epilogue writes zero output and $\text{LSE}=-\infty$.

### 3.2 The Standardized Six Stages of Execution

Both the `cp.async` and `TMA` kernels strictly adhere to the same 6-stage lifecycle:

```
  +-------------------------------------------------------------------------+
  | Stage 1: Tile Acquire & Prefetch (Double buffering K/V/Mask)            |
  +-------------------------------------------------------------------------+
                                    |
  +-------------------------------------------------------------------------+
  | Stage 2: QK^T Matrix Multiply (mma_m16n8k16 or wgmma_m64n64k16)         |
  +-------------------------------------------------------------------------+
                                    |
  +-------------------------------------------------------------------------+
  | Stage 3: Apply Scale, Mask, Bias & Compute Tile-Max (quad_max)          |
  +-------------------------------------------------------------------------+
                                    |
  +-------------------------------------------------------------------------+
  | Stage 4: Rescale Softmax State (m_state, l_state, o_acc)                |
  +-------------------------------------------------------------------------+
                                    |
  +-------------------------------------------------------------------------+
  | Stage 5: Compute Probabilities (exp) & PV Accumulation (mma_m16n8k16)   |
  +-------------------------------------------------------------------------+
                                    |
  +-------------------------------------------------------------------------+
  | Stage 6: Epilogue (Normalize O, write LSE and Shadow Residual)          |
  +-------------------------------------------------------------------------+
```

### 3.3 Hardware Fragments & Thread-Coordinate Mapping

Each Thread Block owns **64 query rows** and **64 key columns**. A block contains 4 warps (128 threads), so **1 warp owns 16 query rows**.

#### The `struct Accumulator { float values[4]; }` Mapping:
In an `m16n8` tensor core accumulator:
- 32 threads in a warp are split into 8 groups of 4 threads:
  - $\text{group} = \text{lane} / 4 \in [0, 7]$
  - $\text{thread\_in\_group} = \text{lane} \pmod 4 \in [0, 3]$
- Each group of 4 threads handles two rows: $r = \text{group}$ and $r + 8$:
  - `values[0]`, `values[1]` map to $(r, 2 \cdot \text{thread\_in\_group})$ and $(r, 2 \cdot \text{thread\_in\_group} + 1)$
  - `values[2]`, `values[3]` map to $(r + 8, 2 \cdot \text{thread\_in\_group})$ and $(r + 8, 2 \cdot \text{thread\_in\_group} + 1)$

#### Intra-Warp Reductions (`quad_max` and `quad_sum`):
Because a single query row is shared by the 4 threads in each group, reducing across that row requires an XOR shuffle only across delta 1 and 2:
```cpp
__device__ __forceinline__ float quad_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 1, kWarpSize));
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, 2, kWarpSize));
    return value;
}
```

### 3.4 Shared Memory Layout & Swizzling

#### Path 1: `cp.async` Kernel
- Shared memory buffer:
  `Q [64 x kKVStride] | K0 [64 x kKVStride] | V0 [64 x kKVStride] | K1 [64 x kKVStride] | V1 [64 x kKVStride]`
- `kKVStride = HeadDim + 8` elements to avoid bank conflicts across consecutive rows.

#### Path 2: `TMA + WGMMA` Kernel (Hopper SM90a)
- Utilizes hardware TMA (Tensor Memory Accelerator) with 128-byte bank swizzling (`CU_TENSOR_MAP_SWIZZLE_128B`).
- Double-buffered barriers via `mbarrier` with alternating phase parity (`(key_tile >> 1) & 1`).

---

## 4. Backward Pass & Single-Reconstruction Architecture

Classic FlashAttention backward computes $dK$ and $dV$ in one pass, then $dQ$ in another pass, requiring recomputing the full probability matrix $P$ twice.

### 4.1 Single-Reconstruction (SR)

TurboPairFormer uses **Single-Reconstruction**:
- Each tile of $P$ is reconstructed **exactly once** in chip memory.
- All four gradients ($dQ$, $dK$, $dV$, $dBias$) are accumulated concurrently.
- Reconstructed $P$ and $dS = P \cdot (dP - \Delta)$ are kept in registers and consumed immediately without saving to HBM.

### 4.2 Shadow Residual Delta Precomputation ([turbopairformer_triangle_delta.cu](file:///Users/yideran/Desktop/research_product/openfold_product/openfold3/core/kernels/cuda/turbopairformer_triangle_delta.cu))

For one query row, attention backward requires a dot product over its head channels. Write $g$ for the upstream output-gradient vector and $O$ for the stored output vector:
$$\Delta = \langle g, O \rangle$$

When $O$ is written to HBM in BF16, quantization error introduces noise into $\Delta$.
- Let $F$ be the normalized FP32 output before storing $O=\text{BF16}(F)$. Forward saves a BF16 approximation of the rounding residual:
  $$r = \text{BF16}(F - \text{float}(O))$$
- In [`turbopairformer_triangle_delta.cu`](file:///Users/yideran/Desktop/research_product/openfold_product/openfold3/core/kernels/cuda/turbopairformer_triangle_delta.cu), the shadow variant computes:
  $$\Delta = \langle \text{float}(g),\ \text{float}(O) + \text{float}(r) \rangle$$
- This reduces the error from storing the output in BF16 without re-traversing probabilities. The residual itself is rounded, so it does not exactly recover $F$.

### 4.3 Deterministic dBias Reduction Tree ([turbopairformer_triangle_dq_dbias.cu](file:///Users/yideran/Desktop/research_product/openfold_product/openfold3/core/kernels/cuda/turbopairformer_triangle_dq_dbias.cu))

In AlphaFold3, $dBias$ has shape $[B, 1, H, N, N]$ and sums contributions over all outer planes $O$. Naive implementations use atomic adds, causing non-deterministic floating-point reordering.

TurboPairFormer guarantees **bitwise reproducible gradients**:
- **Fixed Reduction Tree**:
  $$\text{plane}_{\text{leaf}} = (w_0 + w_1) + (w_2 + w_3)$$
  $$dBias = ((\text{plane}_0 + \text{plane}_1) + \text{plane}_2) + \dots$$
- **Zero cross-CTA atomics**: Each leaf plane is written to a designated workspace slice and folded deterministically in FP32.

---

## 5. Triangle Multiplicative Update ([turbopairformer_trimul_cuda.cu](file:///Users/yideran/Desktop/research_product/openfold_product/openfold3/core/kernels/cuda/turbopairformer_trimul_cuda.cu))

Triangle Multiplicative Update (TriMul) models 3-way residue interactions. For one batch item and hidden channel, $A$ and $B$ are $N \times N$ matrices and $Z$ is their contraction result:
- **Outgoing**: $Z = AB^T$
- **Incoming**: $Z = A^T B$

The matrix product sums over the shared residue index. The symbols here are local to TriMul; $A$ is a projected pair matrix, not the attention accumulator above.

The CUDA kernel fuses:
1. Two input linear projections and gating equations.
2. Contraction over the common index $k$.
3. Output gating and layer norm preparation.

---

## 6. Numerical Contract & Validation

1. **Replay Determinism**:
   - Repeated runs with the exact same inputs, binary, and GPU model produce **identical bitwise outputs** in both forward and backward.
2. **PyTorch Agreement Contract**:
   - Bitwise agreement with PyTorch native eager ops is **not** claimed or required, due to fused FMA/WGMMA contraction ordering differences.
   - Numerical accuracy is verified against a **double-precision (FP64) reference implementation**; error ratios must remain strictly bounded:
     $$\frac{\|O_{\text{kernel}} - O_{\text{fp64}}\|_2}{\|O_{\text{pytorch}} - O_{\text{fp64}}\|_2} \le 1.0 + \epsilon$$
3. **Mask Arithmetic Contract**:
   - Additive mask entries equal to true $-\infty$ force attention weights to zero.
   - Fully masked rows produce exact zero output vectors and $\text{LSE} = -\infty$.

---

## 7. Developer Quick Reference

### Launching Forward via Python:
```python
from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (
    turbopairformer_triangle_attention,
)

# Inputs: q, k, v are [B, O, H, N, D], mask is [B, O, 1, 1, N], bias is [B, 1, H, N, N]
out = turbopairformer_triangle_attention(
    q=q,
    k=k,
    v=v,
    bias=bias,
    mask=mask,
    enable_shadow_residual=True,  # Default: True for high-precision backward
)
# Output out is [B, O, N, H, D]
```

### Adding or Modifying Kernels:
1. Follow the **Six-Stage Execution Template** (Stages 1–6) to keep maintenance predictable.
2. Maintain the **No Atomics Rule** for reduction operations to preserve replay determinism.
3. Validate changes against FP64 reference scripts in `tests/` before committing.
