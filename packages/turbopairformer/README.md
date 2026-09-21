# TurboPairFormer

TurboPairFormer provides optimized GPU kernels for Triangle Attention and
Triangle Multiplicative Update, with PyTorch autograd support for protein folding
model training.

The project brings together researchers from Stevens Institute of Technology,
Lambda, and OpenFold.

## Installation

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

pip install turbopairformer
```

Requires **Python 3.14**, **Linux x86-64** (glibc ≥2.28),
**PyTorch 2.10.0 / CUDA 12.8**, and an **NVIDIA Hopper GPU**.
H100 is tested; H200 is not yet validated. No CUDA toolkit needed.

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
signatures. `python -m turbopairformer` is equivalent. Viewing help does not
import Torch or load CUDA. The [offline API guide](turbopairformer/API_GUIDE.md)
is also readable directly in the repository.

## Quick Start

### Triangle Attention

```python
import torch
from turbopairformer import turbo_attention, turbo_trimul

options = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = [torch.randn(1, 128, 4, 128, 32, **options) for _ in range(3)]
pair_bias = torch.randn(1, 1, 4, 128, 128, **options)

output = turbo_attention(q, k, v, pair_bias=pair_bias)
output.float().square().mean().backward()

print(output.shape)  # torch.Size([1, 128, 128, 4, 32])
```

### Triangle Multiplicative Update

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

Omitted TriMul weights are initialized on each call for this example. For model
training, pass your persistent projection, gate, and normalization weights.

Masks are optional and default to all positions valid. Pass a mask for padded
inputs. `pair_bias` is required.

## Project Team

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
