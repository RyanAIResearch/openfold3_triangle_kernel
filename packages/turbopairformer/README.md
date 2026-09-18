# TurboPairFormer

Fast CUDA kernels for Triangle Attention and Triangle Multiplicative Update
in protein folding models, with PyTorch autograd support.

## Installation

Use a fresh Python 3.14 environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install turbopairformer
```

Requires Linux x86-64 (glibc ≥2.28), PyTorch 2.10.0 with CUDA 12.8, and a Hopper
GPU. H100 is tested; H200 execution is not yet validated. The wheel includes
precompiled kernels; no CUDA toolkit is needed.

## Quick start

```python
import torch
from turbopairformer import turbo_attention, turbo_trimul

options = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = [torch.randn(1, 128, 4, 128, 32, **options) for _ in range(3)]
pair_bias = torch.randn(1, 1, 4, 128, 128, **options)

output = turbo_attention(q, k, v, pair_bias=pair_bias)
output.float().square().mean().backward()
```

For TriMul, use your model's compatible triangle-multiplication module and BF16
pair tensor `z`:

```python
output = turbo_trimul(module, z)
```

Masks are optional and default to all positions valid. Pass a mask for padded
inputs. `pair_bias` is required.

## Contributors

Yide Ran¹, Chelsea Lowman², Jan Domanski³, David Hartmann², Jenke Scheen³,
Jennifer Wei³, Chuan Li², Jianwen Xie²⁺, Zhaozhuo Xu¹⁺.

¹ Stevens Institute of Technology · ² Lambda · ³ OpenFold

⁺ Equal advising.

## License

Apache-2.0. Derived from [AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3).
See [development documentation](https://github.com/RyanAIResearch/openfold3_triangle_kernel/blob/feat/turbopairformer-pip-package/packages/turbopairformer/DEVELOPMENT.md)
for supported layouts and build details.
