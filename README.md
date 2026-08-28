# StableFold for OpenFold3

StableFold is a Hopper-optimized Triangle Attention and Triangle
Multiplicative Update backend integrated directly into OpenFold3. It keeps the
OpenFold3 training entry points, datasets, checkpoints, and configuration
structure unchanged; users select the backend through the model configuration.

The central optimization is a single-reconstruction Triangle Attention
backward. A flash-style two-pass implementation reconstructs the attention
probabilities and score gradients once for `dK/dV` and again for `dQ/dBias`.
StableFold reconstructs each tile once and immediately uses it for all four
gradients. A fixed-order `dBias` reduction replaces floating-point atomics, and
an optional forward rounding residual improves the row-scalar used by the
backward without changing the public BF16 output.

> **Status: internal preview / production candidate, not GA.** Use a feature
> flag and retain the stock or cuEquivariance route as a rollback. The validated
> scope is a single NVIDIA H100, BF16 training, local batch size 1, and the
> OpenFold3 shapes described below. Multi-GPU/DDP, H200, FSDP, `torch.compile`,
> and local batch sizes greater than 1 are not part of this release claim.

## Validated result

On one H100 at crop 768, the full StableFold configuration (StableFold Triangle
Attention plus StableFold Triangle Multiplicative Update) reduced synchronized
training-step time by **6.26%** and step energy by **7.76%** relative to the full
cuEquivariance configuration over a 1,200-update continuation from the same
optimizer-bearing checkpoint. Validation and checkpoint I/O were excluded from
the timing. This is a measured OpenFold3/H100 result, not a guarantee for other
models, hardware, software ABIs, or shapes.

## Requirements

- Linux x86-64
- NVIDIA H100 80 GB, compute capability 9.0 (`sm_90a`)
- CPython 3.14
- PyTorch 2.10.0 built for CUDA 12.9
- CUDA toolkit/NVCC 12.9 when building from source or enabling the current
  StableFold Triangle Multiplicative Update implementation
- BF16 Q/K/V; BF16 or FP32 pair bias; FP16, BF16, or FP32 additive mask

The prebuilt Triangle Attention extensions contain Hopper-specific SASS and no
PTX fallback. An ABI or device mismatch is an error, not a reason to load the
binary anyway.

## 1. Clone the canonical source

This is a private repository. Ask the maintainer to grant access to your GitHub
account and authenticate with GitHub before cloning. The repository intentionally
has one maintained branch. Clone it into a fresh environment rather than
installing it over a locally modified OpenFold3 tree:

```bash
git clone https://github.com/RyanAIResearch/openfold3_triangle_kernel.git
cd openfold3_triangle_kernel
git rev-parse HEAD
```

For a reproducible run, record the printed commit and do not pull while a
campaign is active.

## 2. Install

### Option A: install an attested internal H100 wheel

Binary artifacts are not stored in this Git repository. If the maintainer sent
you an internal release bundle, it must contain:

```text
openfold3-*.whl
openfold3-*.whl.sha256
openfold3-*.whl.attestation.json
aot_clean_install_smoke.json
```

Verify and install it in an environment that already contains the complete
OpenFold3 dependency set and the exact Python/Torch/CUDA ABI listed above:

```bash
sha256sum --check openfold3-*.whl.sha256
python -m pip install --no-deps --force-reinstall openfold3-*.whl
```

The wheel owns four AOT Triangle Attention extensions: forward, delta, `dK/dV`,
and `dQ/dBias`. Production Triangle Attention does not invoke NVCC or silently
fall back to JIT. If you received only the GitHub link and no binary bundle, use
the source-build route below.

### Option B: build the H100 wheel from this repository

Only build from a clean committed checkout in the exact Python/Torch/CUDA
environment above:

```bash
export CUDA_HOME=/path/to/cuda-12.9
export CUDACXX="$CUDA_HOME/bin/nvcc"
tools/build_sm90_wheel.sh
python -m pip install --no-deps --force-reinstall dist/openfold3-*.whl
```

The builder clones the exact commit into a temporary directory, produces the
four `sm_90a` extensions, attests their symbols and SASS, and runs a
compiler-free forward/backward smoke test.

### Current full-backend packaging note

The Triangle Multiplicative Update source is included, but it currently uses
`torch.utils.cpp_extension.load_inline` on first use. Therefore:

- Triangle Attention alone can run compiler-free from the AOT wheel;
- the full StableFold backend currently requires NVCC 12.9 once to build TMU;
- use a persistent absolute `TORCH_EXTENSIONS_DIR` so the TMU cache is not
  rebuilt between runs;
- do not advertise the full backend as a compiler-free wheel until TMU is
  included as an additional attested AOT extension.

For an internal full-backend trial:

```bash
export CUDA_HOME=/path/to/cuda-12.9
export TORCH_EXTENSIONS_DIR=/absolute/persistent/path/stablefold_torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
```

## 3. Enable StableFold

Set the SR-R backward contract before starting Python:

```bash
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
unset OPENFOLD3_TRIANGLE_ALLOW_JIT
```

`OPENFOLD3_SR_SHADOW=1` stores a BF16 rounding residual in the forward and uses
it only to improve the backward row scalar. It does not change the model-visible
forward output.

In a training YAML, select the backend under the train memory settings:

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_stable_triangle_attention: true
          use_stable_triangle_multiplicative: true
          use_cueq_triangle_kernels: false
          use_triton_triangle_kernels: false
          use_deepspeed_evo_attention: false
          use_lma: false
          chunk_size: null
```

Keep the existing OpenFold3 architecture defaults when using StableFold
Triangle Multiplicative Update: every relevant MSA, Pairformer, and template
stack must retain `fuse_projection_weights: false`. Everything else in the
existing OpenFold3 training command can remain unchanged.

For an Attention-only compiler-free canary, set:

replace the corresponding values in the same `memory.train` block:

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_stable_triangle_attention: true
          use_stable_triangle_multiplicative: false
          use_cueq_triangle_kernels: false
          use_triton_triangle_kernels: false
          use_deepspeed_evo_attention: false
          use_lma: false
          chunk_size: null
```

## 4. Verify before training

Run the packaging and routing tests:

```bash
python -m unittest \
  openfold3.tests.core.kernels.test_stablefold_triangle_aot_packaging \
  openfold3.tests.core.kernels.test_stablefold_triangle_packaging \
  openfold3.tests.core.kernels.test_stablefold_triangle_workspace \
  openfold3.tests.core.model.test_attention_stable_routing
```

Run one public-facade forward/backward smoke:

```bash
python scripts/benchmarks/stablefold_triangle_sanitizer_smoke.py \
  --sequence 65 --heads 4 --head-dim 32 --prefix 1
```

Before trusting performance, run a short paired canary with the same checkpoint,
data order, precision, crop, and logging configuration as the comparator. Check:

- finite loss and gradients;
- expected StableFold route counts and no silent fallback;
- no JIT/rebuild of the four Triangle Attention extensions;
- peak memory and OOM headroom on the largest real sample;
- synchronized training-step time after warmup;
- checkpoint save/resume and validation.

## 5. Roll back

No code change is required to return to cuEquivariance:

```yaml
use_stable_triangle_attention: false
use_stable_triangle_multiplicative: false
use_cueq_triangle_kernels: true
```

```bash
unset OPENFOLD3_SR_BACKWARD
unset OPENFOLD3_SR_SHADOW
```

Unsupported StableFold Attention shapes are explicitly routed to the stock
OpenFold3 implementation; the public StableFold facade itself fails closed.
Monitor fallback counts rather than assuming every call used the custom kernel.

## Supported scope and claim boundaries

- Main measured crops: 384, 640, and 768
- Main measured head dimensions: 16, 24, 32, and 48 across real OpenFold3 sites
- Runtime Triangle Attention contract: positive prefixes, runtime sequence
  length/head count, and `1 <= D <= 64`, subject to CUDA grid and memory limits
- Determinism means bitwise self-replay with the same backend, inputs, software
  ABI, workspace plan, and hardware; it does not mean cross-backend or
  cross-hardware bit equality
- The rounding residual corrects the backward row-scalar channel only; it does
  not eliminate every BF16 error
- The speedup comes from single reconstruction; the fixed reduction tree and
  rounding residual each have a cost
- Do not infer convergence superiority or memory savings from the timing result

The detailed kernel contract is in
[`PRODUCTION_MANIFEST.md`](openfold3/core/kernels/cuda/PRODUCTION_MANIFEST.md).
Experimental protocols and receipts are documented under
[`scripts/benchmarks/sr/`](scripts/benchmarks/sr/) and [`paper/`](paper/).

## Direct operator entry point

Users integrating below the OpenFold3 model router can call:

```python
from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
    stablefold_triangle_attention,
)

output = stablefold_triangle_attention(q, k, v, mask, pair_bias)
```

Use the public facade rather than importing private extension symbols. The facade
validates the tensor/device contract and supplies PyTorch autograd integration.

## License and attribution

This repository is derived from
[AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3) and is distributed
under Apache-2.0. Retain [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
[`UPSTREAM.md`](UPSTREAM.md), and [`CITATION.cff`](CITATION.cff) when sharing an
internal source or binary bundle.
