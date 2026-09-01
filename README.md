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
> flag and retain the stock or cuEquivariance route as a rollback. Single-H100
> BF16 training with local batch size 1 is validated; a kernel-only DDP
> preflight is available, but real 2-GPU and 2-node hardware qualification is
> still pending. DDP, multi-node launch, EMA, gradients, checkpoints, losses,
> and logging remain the pinned AQLab implementation. H200, FSDP,
> `torch.compile`, Lightning
> DeepSpeedStrategy/ZeRO, and local batch sizes greater than 1 are not part of
> this release claim.

The wheel checksum and attestation are reproducibility and integrity records,
not an external signature or Sigstore provenance chain; this preview assumes
the GitHub repository, build host, and shared artifact storage are trusted.

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
- CUDA toolkit/NVCC 12.9 only when building the five AOT extensions from source
- BF16 Q/K/V; BF16 or FP32 pair bias; FP16, BF16, or FP32 additive mask

The prebuilt Triangle Attention extensions contain Hopper-specific SASS and no
PTX fallback. An ABI or device mismatch is an error, not a reason to load the
binary anyway.

## Pixi production workflow

Clone the maintained `main` branch into a clean checkout:

```bash
mkdir -p /mnt/shared/openfold3
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git \
  /mnt/shared/openfold3/src
cd /mnt/shared/openfold3/src
git rev-parse HEAD
test -z "$(git status --porcelain=v1)"
```

Create the locked, non-editable H100 environment and let the checked-in Pixi
tasks prepare the published PDB portion and build all five AOT extensions;
provision the other datasets named by AQLab's official recipe before training:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi install --locked -e openfold3-h100
pixi run --locked -e openfold3-h100 prepare-training-data
pixi run --locked -e openfold3-h100 doctor-h100
pixi run --locked -e openfold3-h100 build-install-aot
pixi run --as-is -e openfold3-h100 verify-aot
```

Copy and explicitly configure AQLab's official runner YAML, choose one backend,
and start with one H100:

```bash
export BACKEND=stablefold  # aqlab | cueq | stablefold | deepspeed-evo
export OF3_RUN_CWD=/mnt/shared/openfold3/run
mkdir -p "$OF3_RUN_CWD"
cp examples/training_yamls/initial_training.yml "$OF3_RUN_CWD/base.yml"
${EDITOR:-vi} "$OF3_RUN_CWD/base.yml"
pixi run --as-is -e openfold3-h100 select-training-backend \
  "$OF3_RUN_CWD/base.yml" --backend "$BACKEND" \
  --output "$OF3_RUN_CWD/train.${BACKEND}.yml"
pixi run --as-is -e openfold3-h100 train-single-gpu \
  train --runner-yaml "$OF3_RUN_CWD/train.${BACKEND}.yml" --seed 42
```

The production Pixi environment deliberately contains no editable OpenFold3
dependency. The build task emits and installs one attested wheel, while every
training task checks that the installed package, Git commit, ABI, embedded
manifest, and five extension hashes still match before launching outside the
source checkout. The task then directly executes AQLab `run_openfold` or
standard `torchrun`; it does not replace any training, strategy, checkpoint,
logging, or loss behavior. StableFold runtime is compiler-free and keeps JIT
disabled, while the selector changes only the training kernel-route flags.

For an end-to-end bare-metal path covering clone, data, compilation, backend
selection, single-GPU training, single-node DDP, and multi-node launch—with
three plain-English sentences per step—see
[Bare-metal training quickstart](docs/source/bare_metal_distributed_quickstart.md).

## Enable StableFold manually

Set the SR-R backward contract before starting Python:

```bash
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
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

For an Attention-only compiler-free canary, replace the corresponding values
in the same `memory.train` block:

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
- no JIT/rebuild of the four Triangle Attention extensions or the fifth TriMul
  extension;
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

StableFold TriMul uses a fixed, non-configurable short-sequence boundary:
valid TriMul calls with `N <= 100` are metadata-unsupported by the AOT path and
the model selector routes them directly to stock PyTorch. The H100 BF16 AOT
path starts at `N = 101` for `C_z = C_hidden` in `{32, 64, 96, 128}`. The
boundary value matches cuEquivariance 0.10's
`CUEQ_TRIMUL_FALLBACK_THRESHOLD=100`, but StableFold does not read that or any
other environment variable. Unsupported direct-facade calls remain
fail-closed; only the model selector owns the stock fallback.

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
