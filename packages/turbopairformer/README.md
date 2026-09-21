# TurboPairFormer

TurboPairFormer provides optimized, autograd-supported GPU kernels for
**Triangle Attention** and **Triangle Multiplicative Update** (TriMul) to
accelerate protein folding model training.

The project brings together researchers from Stevens Institute of Technology,
Lambda, and OpenFold.

## Installation

### Prerequisites

The 0.1.1 prebuilt wheel requires:

- **Python 3.14** (validated with Python 3.14.7).
- **PyTorch 2.10.0+cu128**, using the CUDA 12.8 runtime.
- **Linux x86-64**, with glibc ≥2.28.
- An **NVIDIA Hopper GPU**, compute capability 9.0, and a compatible NVIDIA
  driver. H100 is validated; H200 is not yet validated.

Installing the prebuilt wheel does not require a local CUDA toolkit or compiler.

### Install with pip

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer
python -m pip install turbopairformer==0.1.1
```

## Quick Start

### Triangle Attention

Q, K and V use `[B, O, H, N, D]`: `B` is the batch size, `O` indexes outer
rows, `H` is the number of attention heads, `N` is the attended sequence length,
and `D` is the head dimension. Pass raw, unscaled Q; the operator applies
`1/sqrt(D)` internally. The output uses `[B, O, N, H, D]`, swapping the sequence
and head axes relative to the inputs.

```python
import torch
from turbopairformer import turbo_attention

B, O, H, N, D = 1, 128, 4, 128, 32
options = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = [torch.randn(B, O, H, N, D, **options) for _ in range(3)]
pair_bias = torch.randn(B, 1, H, N, N, **options)

output = turbo_attention(q, k, v, pair_bias=pair_bias)
output.float().square().mean().backward()

print(output.shape)  # torch.Size([1, 128, 128, 4, 32])
```

`pair_bias` is required; pass an explicit zero tensor for a model without bias.
The optional `res_mask` is a floating **additive** mask shaped `[B, O, 1, 1, N]`:
zero keeps keys and negative infinity excludes them. `res_mask=None` keeps all
keys. Boolean masks are not accepted for attention.

### Triangle Multiplicative Update (TriMul)

TriMul takes a BF16 pair tensor `[B, N, N, C]` and returns the same shape and
dtype. Both `"outgoing"` and `"incoming"` directions support autograd.

```python
import torch
from turbopairformer import turbo_trimul

z = torch.randn(
    1, 128, 128, 128,
    device="cuda", dtype=torch.bfloat16, requires_grad=True,
)

output = turbo_trimul(z, direction="outgoing")  # or "incoming"
output.float().square().mean().backward()

print(output.shape)  # torch.Size([1, 128, 128, 128])
```

This is a demonstration: omitted projection and gate matrices are randomly
initialized **on every call**; omitted normalization weights and biases become
ones and zeros. These temporary tensors are not trainable model state. For
training or repeated evaluation, supply and reuse all **eight persistent FP32
parameters**: `norm_in_weight`, `norm_in_bias`, `p_in_weight`, `g_in_weight`,
`norm_out_weight`, `norm_out_bias`, `p_out_weight`, and `g_out_weight`. The
[complete training example](https://github.com/RyanAIResearch/openfold3_triangle_kernel/blob/feat/turbopairformer-pip-package/packages/turbopairformer/turbopairformer/API_GUIDE.md#trimul---triangle-multiplicative-update)
shows their shapes and optimizer setup.

TriMul's optional `mask` is a **multiplicative** pair mask broadcastable to
`[B, N, N]`: one keeps pairs and zero excludes them. It accepts bool, BF16 or
FP32 tensors; `mask=None` keeps all pairs. Neither operator differentiates masks.

## Performance & Hardware Support

TurboPairFormer provides CUDA implementations of Triangle Attention and Triangle
Multiplicative Update with first-order PyTorch autograd. The 0.1.1 wheel has been
validated on **NVIDIA H100** with the software stack listed above. **H200 has not
yet been validated**, and other GPU architectures are outside this wheel's
supported runtime.

Both operators use BF16 activations. Attention supports head dimensions
`1 <= D <= 64`; TriMul supports `N > 100` and channel sizes `32`, `64`, `96`, or
`128`. Additional shape, layout and launch limits are documented in
`turbopairformer api support`. Performance depends on the input shapes and host
model; benchmark your workload on the target hardware.

## Command-line API guide

Version 0.1.1 adds an offline command-line guide. The
previously published 0.1.0 wheel predates this CLI.

```bash
turbopairformer --help
turbopairformer api turbo_attention
turbopairformer api turbo_trimul
turbopairformer api support
turbopairformer api distributed
turbopairformer api environment
turbopairformer api internals
turbopairformer api --all
```

Use `turbopairformer api` for the full API index, or look up an internal symbol
with `turbopairformer api triangle_dq_dbias.workspace_plan`. Help includes
parameter contracts, shapes, masks, autograd examples and source-derived
signatures. `python -m turbopairformer` accepts the same arguments. Viewing help
does not import Torch or load CUDA. The
[offline API guide](https://github.com/RyanAIResearch/openfold3_triangle_kernel/blob/feat/turbopairformer-pip-package/packages/turbopairformer/turbopairformer/API_GUIDE.md)
is also readable directly in the repository.

## Authors & Acknowledgments

- **Yide Ran** ¹
- **Chelsea Lowman** ²
- **Jan Domanski** ³
- **David Hartmann** ²
- **Jenke Scheen** ³
- **Jennifer Wei** ³
- **Chuan Li** ²
- **Jianwen Xie** ² ⁺
- **Zhaozhuo Xu** ¹ ⁺

## Affiliations

- ¹ Stevens Institute of Technology
- ² Lambda
- ³ OpenFold
- ⁺ Equal advising

## License

Apache-2.0
