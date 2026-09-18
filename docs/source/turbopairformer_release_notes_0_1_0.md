# TurboPairFormer 0.1.0

TurboPairFormer is a Python package providing two CUDA operators for AlphaFold3-style models:

- **TurboTriangleAttention** — Triangle Attention with autograd support.
- **TurboTriangleMultiplicative** — Triangle Multiplicative Update with autograd support.

The package name is `turbopairformer`. Its public entry points are:

```python
import torch
from turbopairformer import turbo_attention, turbo_trimul

B, N, H, D = 1, 128, 4, 32
# q/k/v: [batch, outer, heads, sequence, head_dim]; here outer = N.
q, k, v = [
    torch.randn(B, N, H, N, D, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
    for _ in range(3)
]
# Pair bias is shared across the outer dimension.
pair_bias = torch.randn(B, 1, H, N, N, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)

attention_output = turbo_attention(q, k, v, pair_bias=pair_bias)
attention_output.float().square().mean().backward()
print(attention_output.shape)  # torch.Size([1, 128, 128, 4, 32])
```

All positions in this synthetic example are valid, so the mask is omitted.
Single-reconstruction backward and the saved forward rounding residual are
enabled by default; no environment switches are needed for this example.
For TriMul, pass your existing triangle-multiplication module and pair tensor:

```python
trimul_output = turbo_trimul(module, z)
```

Both masks are optional and default to `None` (all positions valid, noncausal).
`pair_bias` remains required. Explicit masks are supported for padded inputs.

The standalone package can be used without installing OpenFold3. It includes precompiled Hopper kernels, so users do not need a CUDA compiler or a runtime JIT build.

## Developers

TurboPairFormer contributors, including contributors from Lambda.
Individual developer names and credit order will be added after confirmation.

## Installation

**pip** — in a fresh Python 3.14 virtual environment:

```bash
pip install turbopairformer
```

**Pixi / OpenFold3** — for users with repository access, from a separate checkout:

```bash
pixi install -e openfold3-cuda12
```

[Environment setup and detailed instructions](https://github.com/RyanAIResearch/openfold3_triangle_kernel/blob/feat/turbopairformer-pip-package/docs/source/turbopairformer_pip_quickstart.md).

## Supported configuration

- Linux x86-64 with glibc 2.28 or newer.
- CPython 3.14.
- PyPI PyTorch 2.10.0 with CUDA runtime 12.8.
- Hopper compute capability 9.0 (`sm_90a`): H100 is tested; H200 hardware execution remains unverified.
- OpenFold3 integration supports the documented single-GPU launch path. Full multi-GPU host training is not yet qualified.

## Validation

The attached release wheel passed **215 tests in an H100 environment with zero skips**. The default Attention call was verified to generate and consume its rounding residual and execute the single-reconstruction backward kernel. Default and explicitly enabled execution produced bitwise-identical outputs and gradients. All five installed CUDA extensions were checked, including forward/backward execution, numerical checks and repeatability checks. Wheel portability and package metadata checks passed. A fresh isolated Pixi environment passed installation, installed-wheel identity checks, doctor, and public-operator GPU smoke tests with this same wheel. A separate single-H100 crop-384 training check with single-reconstruction backward and rounding residual explicitly enabled completed all four configured backends (five warmup plus 20 measured steps each), with all 100 recorded losses finite and TF32 enabled consistently. These short runs validate integration; their backend routing differs and they do not establish convergence or reproduce a historical performance claim.

The attached wheel, `release-validation.json`, `installed-aot.json` and `standalone-gpu.xml` identify the exact tested artifact. These checks do not establish a new performance result or qualify H200 and full multi-GPU host training.

## License and upstream acknowledgments

TurboPairFormer is distributed under Apache-2.0. It integrates with the AQLab OpenFold3 project; upstream copyright and license notices are retained in the source.
