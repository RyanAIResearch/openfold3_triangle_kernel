# TurboPairFormer

Standalone CUDA Triangle Attention and Triangle Multiplicative Update (TriMul)
kernels for AlphaFold3-style pair stacks. Importing and using this package does
not require OpenFold3. The distribution and Python import are both
`turbopairformer`, with two direct operator calls:

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
For TriMul, a standalone example needs only a BF16 pair tensor:

```python
z = torch.randn(
    1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
trimul_output = turbo_trimul(z, direction="outgoing")
```

Omitted TriMul parameters are initialized afresh on each call for demonstration.
For training, pass persistent parameter tensors as described below.

Both masks default to `None`: every position is valid, with no causal mask.
`pair_bias` remains required; pass it by keyword when omitting the attention
mask. For padded inputs, pass explicit masks:

```python
attention_output = turbo_attention(q, k, v, attention_mask, pair_bias)
trimul_output = turbo_trimul(z, direction="outgoing", mask=pair_mask)
```

## Offline command-line reference

Version 0.1.1 adds `turbopairformer --help` and
`python -m turbopairformer --help`. The published 0.1.0 wheel does not include
this entry point; use a build from this revision. To preview the guide from
the repository without installing Torch or a CUDA wheel:

```bash
PYTHONPATH=packages/turbopairformer python -m turbopairformer --help
PYTHONPATH=packages/turbopairformer python -m turbopairformer api --all
```

Run those commands from the repository root. Help describes only the two
operators: their calls, parameters, input/output shapes and gradients.
`turbopairformer api attention` and `turbopairformer api trimul` show one
operator; `api` and `api --all` show both. The descriptions live in
[API_GUIDE.md](turbopairformer/API_GUIDE.md) and print without importing Torch
or loading CUDA.

## Supported build

The first wheel targets Linux x86-64, CPython 3.14, PyTorch 2.10.0 with its CUDA
12.8 runtime, and NVIDIA H100 or H200 GPUs (Hopper, compute capability 9.0).
Both use the same `sm_90a` wheel. The build uses CUDA toolkit 12.9.
The toolkit is needed only to build the wheel, not to run an installed wheel.
Other Python, Torch, CUDA runtime, GPU and platform combinations are not covered
by this wheel. The loader verifies the installed ABI and binary hashes before
loading an extension and reports mismatches instead of silently compiling.

Each public wheel is built from a clean commit and released with its own H100
installation and numerical-test receipts. Historical wheels under other names
do not certify TurboPairFormer artifacts. H200 execution remains unverified.

GPU eligibility uses compute capability 9.0; it does not require the device
name to contain H100. NVIDIA lists both H100 and H200 as
[compute capability 9.0](https://developer.nvidia.com/cuda/gpus).
Validation receipts identify the actual GPU used: H100 measurements are not
H200 measurements, and fixed-configuration replay does not promise identical
bits across different GPU models.

## Install

Use a fresh Python 3.14 environment on Linux x86-64 (glibc 2.28 or newer):

```bash
python3.14 -m venv .venv-turbopairformer
source .venv-turbopairformer/bin/activate
python -m pip install turbopairformer
```

The Python import is `turbopairformer`; PyPI package names are case-insensitive.
Pip installs the PyPI Torch 2.10.0 CUDA 12.8 runtime dependencies. An existing
Conda Torch build or a Torch wheel from a different CUDA index is not compatible;
use a fresh environment instead of mixing these distributions.
Pip resolves dependencies in the active environment and can replace an existing
Torch version. The dedicated venv above keeps other environments unchanged. The
OpenFold3 integration also supports a project-local Pixi environment for users
with access to the repository; from the packaging branch run
`pixi install -e openfold3-cuda12` and use `pixi run -e openfold3-cuda12`
for its commands. Do not install into a shared training or system Python prefix.

To install a wheel produced by the build command:

```bash
python -m pip install /absolute/path/to/turbopairformer-0.1.1-cp314-cp314-manylinux_2_28_x86_64.whl
```

Public wheels contain all five precompiled CUDA extensions. No local CUDA toolkit
or JIT compilation is needed. Installing from the repository without enabling AOT
building is rejected because it would omit these binaries.

## Triangle Attention

The complete random-input example above includes forward and backward execution.
The output layout is `[batch, outer, sequence, heads, head_dim]`.

The attention mask is additive: zero keeps a key and negative infinity masks it.
Omitting it (or passing `None`) means no masked keys and no causal restriction.
Use an explicit mask to exclude padded or invalid positions. `pair_bias` must
still be supplied, including an explicit zero tensor if the model has no bias. The facade
checks supported shapes, dtypes and layouts; unsupported inputs raise. A host
model can call `turbopairformer.is_triangle_attention_supported` first and choose
its own fallback.

## Triangle Multiplicative Update

```python
from turbopairformer import turbo_trimul

out = turbo_trimul(z, direction="outgoing", mask=mask, **weights)
out.float().square().mean().backward()
```

The functional API accepts eight optional FP32 parameter tensors on the same
device as BF16 `z`. To use trained parameters, keep these tensors in your model
or optimizer and pass them by keyword on every call:

| Keyword | Shape |
| --- | --- |
| `norm_in_weight`, `norm_in_bias` | `[C]` each |
| `p_in_weight`, `g_in_weight` | `[2*C, C]` each; a-branch rows, then b-branch rows |
| `norm_out_weight`, `norm_out_bias` | `[C]` each |
| `p_out_weight`, `g_out_weight` | `[C, C]` each |

All supplied tensors must be contiguous. Gradients propagate to caller-owned
parameters, including both halves of the packed input projection/gate matrices.
`direction` is `"outgoing"` or `"incoming"`; `eps` is fixed at `1e-5`.
Linear projection/gate biases are not supported by this CUDA implementation.

Missing normalization weights/biases become ones/zeros. Missing matrices are
sampled independently with Torch RNG from a normal distribution with standard
deviation `1/sqrt(C)`, in FP32 on `z.device`, without `requires_grad`.
These defaults are newly allocated each call and are not persistent model
parameters. Set the Torch seed again to reproduce a demonstration; pass and
reuse all eight tensors for training or repeated evaluation.

Existing `turbo_trimul(module, z, mask=None)` integrations remain compatible.
The legacy `module` supplies `c_z`, `c_hidden`, `_outgoing`, six
bias-free FP32 linear layers (`linear_a_p`, `linear_a_g`, `linear_b_p`,
`linear_b_g`, `linear_g`, `linear_z`), and FP32 `layer_norm_in` / `layer_norm_out`
with epsilon 1e-5. `z` is BF16 `[batch..., N, N, C]`. The supported fast path has
`N > 100`, `C == c_hidden` in `{32, 64, 96, 128}`. Incoming and outgoing updates
and first-order gradients are supported. See `tools/smoke_installed.py` for a
complete host-independent module and forward/backward example.

## Build and verify

The public builder requires Docker with GPU access on Linux x86-64. It uses a pinned manylinux
2.28 base, PyPI Torch 2.10.0 / CUDA 12.8 and CUDA toolkit 12.9:

```bash
bash tools/build_manylinux_wheel.sh /absolute/path/to/empty-wheel-output
```

The build runs `auditwheel repair`, verifies the allowed external Torch/CUDA
runtime dependencies, and records the final extension hashes after repair.
The release workflow also requires installed-wheel H100 execution and numerical
test evidence for the exact wheel digest. See the repository's
`scripts/turbopairformer/` release instructions.

For development builds without the manylinux container:

In a Linux CPython 3.14 environment with `torch==2.10.0` installed from PyPI:

```bash
python -m pip install setuptools wheel patchelf
CUDA_HOME=/absolute/path/to/cuda-12.9 bash tools/build_sm90_wheel.sh
python -m pip install dist/turbopairformer-0.1.1-cp314-cp314-linux_x86_64.whl
cd /tmp
python /absolute/path/to/packages/turbopairformer/tools/smoke_installed.py
```

The builder copies clean source files to an isolated directory, builds all five
extensions, removes build-host RPATH/RUNPATH entries with `patchelf` before
hashing the binaries, validates the wheel, and checks imports from a pip target install
with no compiler/JIT available. Source placeholders remain unchanged. To build
uncommitted development changes, explicitly set `TURBOPAIRFORMER_ALLOW_DIRTY_BUILD=1`;
such wheels are marked dirty and are not release artifacts.

Run `pytest /absolute/path/to/packages/turbopairformer/tests` against the installed
wheel on H100 or H200. The optional test dependencies are `pytest` and
`pytorch-lightning`. `tests/host_openfold3/` additionally requires OpenFold3 and
is skipped when it is absent. The standalone suite and the installed-wheel
smoke serve different purposes: the smoke verifies loading and autograd, while
the numerical tests retain the reference error bounds.

Source-only development requires the explicit `TURBOPAIRFORMER_SOURCE_ONLY=1` build
opt-in and `TURBOPAIRFORMER_ALLOW_JIT=1` runtime opt-in. Editable AOT builds write an
ignored `_build_manifest.local.json`; wheels embed `_build_manifest.json`.

The defaults are `TURBOPAIRFORMER_SR_BACKWARD=1` (single-reconstruction backward),
`TURBOPAIRFORMER_SR_SHADOW=1` (save and use the forward rounding residual), and
`TURBOPAIRFORMER_ALLOW_JIT=0`. Advanced users may explicitly override them.
The corresponding `OPENFOLD3_*` aliases remain accepted; canonical
`TURBOPAIRFORMER_*` values take precedence, including explicit `0` values.

Version 0.1.1 uses `turbopairformer_*` C++ exports, Python wrappers, AOT module
keys and extension names. Public Torch custom operations use the
`turbopairformer` namespace. Precompiled extensions must come from the same
wheel as the Python package.

## Host integration and scope

OpenFold3 integration lives under `integration/openfold3/` and the host
repository's `scripts/turbopairformer/`. Installing this package alone does not change
a host model's backend. The reviewed fork defaults to its in-tree implementation;
package use is explicit with `OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer`.

The optional `turbopairformer.distributed.TurboPairFormerKernelDDPCallback` has a separate
wheel/rank contract. Full multi-GPU OpenFold3 integration of this new pip wheel
has not been qualified; use the supported single-GPU package path for this work.

The release tests check fixed-configuration Triangle Attention bitwise replay.
The retained FP64-oracle test bounds
relative error to at most 1.1 times its stock BF16 reference's relative error
plus `1e-6`. This FP64-oracle test does not cover TriMul or establish end-to-end
model accuracy. Historical training speedups from the host
repository are not automatically results for this standalone wheel.

## Contributors

Yide Ran¹, Chelsea Lowman², Jan Domanski³, David Hartmann², Jenke Scheen³,
Jennifer Wei³, Chuan Li², Jianwen Xie²⁺, Zhaozhuo Xu¹⁺.

¹ Stevens Institute of Technology · ² Lambda · ³ OpenFold

⁺ Equal advising.

## License and attribution

Derived from [RyanAIResearch/openfold3_triangle_kernel](https://github.com/RyanAIResearch/openfold3_triangle_kernel)
and [AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3).
Apache-2.0; retain `LICENSE` and `NOTICE`.
