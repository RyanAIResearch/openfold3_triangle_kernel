# StableFold

Standalone CUDA Triangle Attention and Triangle Multiplicative Update (TriMul)
kernels for AlphaFold3-style pair stacks. Importing and using this package does
not require OpenFold3. The package name remains `stablefold`; a later project
rename is outside this release work.

## Supported build

The first wheel targets Linux x86-64, CPython 3.14, PyTorch 2.10.0 with its CUDA
12.8 runtime, and NVIDIA H100 or H200 GPUs (Hopper, compute capability 9.0).
Both use the same `sm_90a` wheel. The build uses CUDA toolkit 12.9.
The toolkit is needed only to build the wheel, not to run an installed wheel.
Other Python, Torch, CUDA runtime, GPU and platform combinations are not covered
by this wheel. The loader verifies the installed ABI and binary hashes before
loading an extension and reports mismatches instead of silently compiling.

GPU eligibility uses compute capability 9.0; it does not require the device
name to contain H100. NVIDIA lists both H100 and H200 as
[compute capability 9.0](https://developer.nvidia.com/cuda/gpus).
Validation receipts identify the actual GPU used: H100 measurements are not
H200 measurements, and fixed-configuration replay does not promise identical
bits across different GPU models.

## Install

The intended public installation command, once a release has been published, is:

```bash
python -m pip install stablefold
```

This branch does not itself publish a PyPI release. For local validation, install
the wheel produced by the build command below into the supported environment:

```bash
python -m pip install /absolute/path/to/stablefold-0.1.0-cp314-cp314-linux_x86_64.whl
```

A locally installable Linux wheel still needs portability auditing and a
manylinux release artifact before PyPI upload. Installing from the repository
without enabling AOT building is rejected; it does not create an apparently
working package with missing CUDA binaries.

## Triangle Attention

```python
import os
os.environ["STABLEFOLD_SR_BACKWARD"] = "1"
os.environ["STABLEFOLD_SR_SHADOW"] = "1"

import torch
import stablefold

# q/k/v: [batch, outer, heads, sequence, head_dim], raw unscaled BF16.
shape = (1, 8, 4, 128, 32)
q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
mask = torch.zeros((1, 8, 1, 1, 128), device="cuda", dtype=torch.bfloat16)
bias = torch.zeros((1, 1, 4, 128, 128), device="cuda", dtype=torch.bfloat16,
                   requires_grad=True)
supported, reason = stablefold.is_triangle_attention_supported(q, k, v, mask, bias)
assert supported, reason
out = stablefold.triangle_attention(q, k, v, mask, bias)
# Output: [batch, outer, sequence, heads, head_dim].
out.float().square().mean().backward()
```

The mask is additive: zero keeps a key and negative infinity masks it. The facade
checks supported shapes, dtypes and layouts; unsupported inputs raise. A host
model can call the support check first and choose its own fallback.

## Triangle Multiplicative Update

```python
support = stablefold.check_trimul_support(module, z, mask)
assert support, support.reason
out = stablefold.run_trimul(module, z, mask)
out.float().square().mean().backward()
```

`module` supplies the existing model weights: `c_z`, `c_hidden`, `_outgoing`, six
bias-free FP32 linear layers (`linear_a_p`, `linear_a_g`, `linear_b_p`,
`linear_b_g`, `linear_g`, `linear_z`), and FP32 `layer_norm_in` / `layer_norm_out`
with epsilon 1e-5. `z` is BF16 `[batch..., N, N, C]`. The supported fast path has
`N > 100`, `C == c_hidden` in `{32, 64, 96, 128}`. Incoming and outgoing updates
and first-order gradients are supported. See `tools/smoke_installed.py` for a
complete host-independent module and forward/backward example.

## Build and verify

In a Linux CPython 3.14 environment with `torch==2.10.0` installed from PyPI:

```bash
python -m pip install setuptools wheel patchelf
CUDA_HOME=/absolute/path/to/cuda-12.9 bash tools/build_sm90_wheel.sh
python -m pip install dist/stablefold-0.1.0-cp314-cp314-linux_x86_64.whl
cd /tmp
python /absolute/path/to/packages/stablefold/tools/smoke_installed.py
```

The builder copies clean source files to an isolated directory, builds all five
extensions, removes build-host RPATH/RUNPATH entries with `patchelf` before
hashing the binaries, validates the wheel, and checks imports from a pip target install
with no compiler/JIT available. Source placeholders remain unchanged. To build
uncommitted development changes, explicitly set `STABLEFOLD_ALLOW_DIRTY_BUILD=1`;
such wheels are marked dirty and are not release artifacts.

Run `pytest /absolute/path/to/packages/stablefold/tests` against the installed
wheel on H100 or H200. The optional test dependencies are `pytest` and
`pytorch-lightning`. `tests/host_openfold3/` additionally requires OpenFold3 and
is skipped when it is absent. The standalone suite and the installed-wheel
smoke serve different purposes: the smoke verifies loading and autograd, while
the numerical tests retain the reference error bounds.

Source-only development requires the explicit `STABLEFOLD_SOURCE_ONLY=1` build
opt-in and `STABLEFOLD_ALLOW_JIT=1` runtime opt-in. Editable AOT builds write an
ignored `_build_manifest.local.json`; wheels embed `_build_manifest.json`.

## Host integration and scope

OpenFold3 integration lives under `integration/openfold3/` and the host
repository's `scripts/stablefold/`. Installing this package alone does not change
a host model's backend. The reviewed fork defaults to its in-tree implementation;
package use is explicit with `OPENFOLD3_TRIANGLE_PROVIDER=stablefold`.

The optional `stablefold.distributed.StableFoldKernelDDPCallback` has a separate
wheel/rank contract. Full multi-GPU OpenFold3 integration of this new pip wheel
has not been qualified; use the supported single-GPU package path for this work.

Triangle Attention outputs and gradients are bitwise reproducible against
themselves for the tested fixed configurations. The FP64-oracle test bounds
relative error to at most 1.1 times its stock BF16 reference's relative error
plus `1e-6`. This FP64-oracle test does not cover TriMul or establish end-to-end
model accuracy. Historical training speedups from the host
repository are not automatically results for this standalone wheel.

## License and attribution

Derived from [RyanAIResearch/openfold3_triangle_kernel](https://github.com/RyanAIResearch/openfold3_triangle_kernel)
and [AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3).
Apache-2.0; retain `LICENSE` and `NOTICE`.
