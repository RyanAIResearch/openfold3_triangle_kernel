# TurboPairFormer for OpenFold3

TurboPairFormer is a Hopper-optimized Triangle Attention and Triangle
Multiplicative Update backend integrated directly into OpenFold3. It keeps the
OpenFold3 training entry points, datasets, checkpoints, and configuration
structure unchanged; users select the backend through the model configuration.

The two operators are **TurboTriangleAttention** (Triangle Attention) and
**TurboTriangleMultiplicative** (Triangle Multiplicative Update).

The integrated backend is selected with `--backend turbopairformer`. Current
module names, extension identities, artifact paths and kernel receipts use the
TurboPairFormer name.

This `main` branch provides the integrated OpenFold3 production workflow. The
standalone **TurboPairFormer** Python package is being prepared on
[`feat/turbopairformer-pip-package`](https://github.com/RyanAIResearch/openfold3_triangle_kernel/tree/feat/turbopairformer-pip-package)
and has not yet been published to PyPI.

The central optimization is a single-reconstruction Triangle Attention
backward. A flash-style two-pass implementation reconstructs the attention
probabilities and score gradients once for `dK/dV` and again for `dQ/dBias`.
TurboPairFormer reconstructs each tile once and immediately uses it for all four
gradients. A fixed-order `dBias` reduction replaces floating-point atomics, and
an optional forward rounding residual improves the row-scalar used by the
backward without changing the public BF16 output.

## Parameters available in OpenFold3

![OpenFold3 parameter sets and the OpenFold3 versions they are compatible with](docs/imgs/openfold3_parameter_table.png)

Each parameter set is compatible with a specific range of OpenFold3 releases. See [OpenFold3 Parameters](https://openfold-3.readthedocs.io/en/latest/parameters_reference.html) in our documentation for checkpoint file names and details on how parameters are resolved at inference time.

### OpenBind-0
The default parameters shipped with OpenFold3 `>=0.5.0` are [OpenBind-0](https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/), which is trained using the forthcoming OpenFold3 architecture with a June 2025 training cutoff. See the [announcement blog post](https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/) for benchmarks and details. We reccomend using the OpenBind-0 model over OpenFold3-preview2 for everyday usage.

If you want to keep using the OpenFold3-preview2 weights, install an older release of OpenFold3 - see [how do I use an older set of parameters](https://openfold-3.readthedocs.io/en/latest/parameters_reference.html#how-do-i-use-an-older-set-of-parameters) in our documentation.

## Technical report
A technical description of the OpenFold3-preview2 model version is available [here](https://portal.openfold.omsf.io/reports/of3p2_technical_report.pdf). Note that the OpenFold3-preview2 weights require an [older release](https://openfold-3.readthedocs.io/en/latest/parameters_reference.html#how-do-i-use-an-older-set-of-parameters) of OpenFold3.

## Requirements

- Linux x86-64
- NVIDIA H100 80 GB, compute capability 9.0 (`sm_90a`)
- CPython 3.14
- PyTorch 2.10.0 built for CUDA 12.9
- NVIDIA driver compatible with the locked CUDA 12.9 runtime; users do not need NVCC
- BF16 Q/K/V; BF16 or FP32 pair bias; FP16, BF16, or FP32 additive mask

The prebuilt Triangle Attention extensions contain Hopper-specific SASS and no
PTX fallback. An ABI or device mismatch is an error, not a reason to load the
binary anyway.

## Upstream synchronization

The source tree includes AQLab `main` through
[`7de748b7`](https://github.com/aqlaboratory/openfold-3/commit/7de748b7bc93adb5af0a4032d5e9f208b6e1325f)
(2026-09-18), including template-server reliability, inference intermediate
management, and memory-efficient metrics updates. See [`UPSTREAM.md`](UPSTREAM.md).
The production environment uses a separately built, pinned binary. Source
updates and namespace changes require their own artifact verification; the
qualification of an earlier binary does not transfer to a renamed build.

## Pixi production workflow

Clone the current `main` branch into a clean checkout:

```bash
mkdir -p /mnt/shared/openfold3
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git \
  /mnt/shared/openfold3/src
cd /mnt/shared/openfold3/src
git rev-parse HEAD
test -z "$(git status --porcelain=v1)"
```

Create the locked, non-editable H100 environment and install its pinned wheel;
users do not need a local kernel build command:

Run the two installation commands once for either one GPU or multiple GPUs on
one host. For multiple hosts, run them once on node 0 only when the checkout
and `.pixi` prefix are on the same shared filesystem; with node-local storage,
run them independently on every host because Pixi does not synchronize nodes.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 doctor-h100
pixi run -e openfold3-cuda12 verify-aot
```

The production Pixi environment deliberately contains no editable OpenFold3
dependency. `pixi install` installs the repository-pinned wheel directly from
the committed lockfile without compiling any TurboPairFormer source, while every
training task checks
that the installed package, artifact commit, ABI, embedded manifest, wheel
RECORD, and five extension hashes still match before launching outside the
source checkout. Dependencies may probe the Pixi-provided compiler ABI and
CUDA version, but the task does not invoke a TurboPairFormer kernel build, CMake,
Ninja, or JIT. It then directly executes AQLab `run_openfold` or standard
`torchrun` without replacing training, strategy, checkpoint, logging, or loss
behavior, and the selector changes only the training kernel-route flags.

The integration checkout and binary artifact have separate identities. The
installed wheel records its build source and extension hashes; the installer
verifies them before launch. Current release artifacts are stored under
`release_artifacts/turbopairformer-h100-20260918/`.

The renamed binary requires its own H100 execution validation. Consult the
artifact's accompanying validation report for completed checks. Full training,
checkpoint/resume and multi-GPU DDP qualification of this renamed artifact are
pending; historical results are not certification of the current binary.

For a copy-and-run bare-metal path covering clone, data, prebuilt installation,
backend selection, single-GPU training, single-node DDP, and multi-node launch,
see the
[Bare-metal training quickstart](docs/source/bare_metal_distributed_quickstart.md).

## Enable TurboPairFormer manually

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

Keep the existing OpenFold3 architecture defaults when using TurboPairFormer
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
  openfold3.tests.core.kernels.test_turbopairformer_triangle_aot_packaging \
  openfold3.tests.core.kernels.test_turbopairformer_triangle_packaging \
  openfold3.tests.core.kernels.test_turbopairformer_triangle_workspace \
  openfold3.tests.core.model.test_attention_stable_routing
```

Run one public-facade forward/backward smoke:

```bash
python scripts/benchmarks/turbopairformer_triangle_sanitizer_smoke.py \
  --sequence 65 --heads 4 --head-dim 32 --prefix 1
```

Before trusting performance, run a short paired canary with the same checkpoint,
data order, precision, crop, and logging configuration as the comparator. Check:

- finite loss and gradients;
- expected TurboPairFormer route counts and no silent fallback;
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

Unsupported TurboPairFormer Attention shapes are explicitly routed to the stock
OpenFold3 implementation; the public TurboPairFormer facade itself fails closed.
Monitor fallback counts rather than assuming every call used the custom kernel.

TurboPairFormer TriMul uses a fixed, non-configurable short-sequence boundary:
valid TriMul calls with `N <= 100` are metadata-unsupported by the AOT path and
the model selector routes them directly to stock PyTorch. The H100 BF16 AOT
path starts at `N = 101` for `C_z = C_hidden` in `{32, 64, 96, 128}`. The
boundary value matches cuEquivariance 0.10's
`CUEQ_TRIMUL_FALLBACK_THRESHOLD=100`, but TurboPairFormer does not read that or any
other environment variable. Unsupported direct-facade calls remain
fail-closed; only the model selector owns the stock fallback.

## Supported scope and claim boundaries

- Runtime Triangle Attention contract: positive prefixes, runtime sequence
  length/head count, and `1 <= D <= 64`, subject to CUDA grid and memory limits
- Determinism means bitwise self-replay with the same backend, inputs, software
  ABI, workspace plan, and hardware; it does not mean cross-backend or
  cross-hardware bit equality
- The rounding residual corrects the backward row-scalar channel only; it does
  not eliminate every BF16 error
- Single reconstruction reduces repeated attention reconstruction work; the
  fixed reduction tree and rounding residual also have execution costs
- Do not infer convergence superiority or memory savings from the timing result

The detailed kernel contract is in
[`PRODUCTION_MANIFEST.md`](openfold3/core/kernels/cuda/PRODUCTION_MANIFEST.md).
Current experiment tooling is under [`scripts/benchmarks/sr/`](scripts/benchmarks/sr/).
The [methodology overview](paper/README.md) explains the evidence boundaries.
Earlier manuscripts, raw receipts and result figures remain in the
[historical source snapshot](https://github.com/RyanAIResearch/openfold3_triangle_kernel/tree/a522470b500d040e5649e1deed1a7a9581438c62); their version-specific results have not
been relabeled as measurements of this renamed release.

## Direct operator entry point

Users integrating below the OpenFold3 model router can call:

```python
from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (
    turbopairformer_triangle_attention,
)

output = turbopairformer_triangle_attention(q, k, v, mask, pair_bias)
```

Use the public facade rather than importing private extension symbols. The facade
validates the tensor/device contract and supplies PyTorch autograd integration.

## License and attribution

This repository is derived from
[AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3) and is distributed
under Apache-2.0. Retain [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
[`UPSTREAM.md`](UPSTREAM.md), and [`CITATION.cff`](CITATION.cff) when sharing an
internal source or binary bundle. `CITATION.cff` records the upstream OpenFold3
reference; TurboPairFormer developer credits and its own citation are pending
confirmation.
