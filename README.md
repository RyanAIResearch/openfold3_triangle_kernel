# Trionce for OpenFold3

Trionce is a Hopper-optimized Triangle Attention and Triangle Multiplicative
Update (TriMul) backend for OpenFold3, shipped as a standalone pip package. It
keeps OpenFold3's training entry points, datasets, checkpoints, and
configuration structure unchanged; you select the backend through one
configuration flag per operator.

The central optimization is a single-reconstruction Triangle Attention
backward. A flash-style two-pass implementation reconstructs the attention
probabilities and score gradients once for `dK/dV` and again for `dQ/dBias`.
Trionce reconstructs each tile once and immediately uses it for all four
gradients. A fixed-order `dBias` reduction replaces floating-point atomics, and
an optional forward rounding residual improves the row-scalar used by the
backward without changing the public BF16 output.

Measured on one H100 80 GB against OpenFold3's stock attention, forward plus
backward at the pair-stack layouts a training run issues:

| Layout | Trionce | Stock | Speedup | FP64-oracle error (fwd) |
| --- | --- | --- | --- | --- |
| `trunk_n384_d32` | 1.87 ms | 6.41 ms | **3.43x** | 1.83e-03 against 3.11e-03 |
| `trunk_n640_d32` | 6.97 ms | 28.97 ms | **4.16x** | 1.83e-03 against 3.20e-03 |
| `trunk_n768_d32` | 11.97 ms | 49.35 ms | **4.12x** | — |

Forward-only reaches 5.50x at crop 768. Outputs and gradients replay bitwise
across identical calls. Reproduce both tables with the commands in
[Verifying an install](#verifying-an-install).

> **Naming.** This backend was previously called StableFold. The kernels are
> unchanged; the package, its public API, and its execution flags are now named
> `trionce`. The `STABLEFOLD_*` and `OPENFOLD3_*` spellings of the execution
> flags are still honoured, so an existing launcher keeps working.

## OpenBind-0

The default parameters shipped with OpenFold3 `>=0.5.0` are
[OpenBind-0](https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/),
trained using the forthcoming OpenFold3 architecture with a June 2025 training
cutoff. See the announcement blog post for benchmarks and details. We recommend
OpenBind-0 over OpenFold3-preview2 for everyday usage.

To keep using the OpenFold3-preview2 weights, install an older release of
OpenFold3 — see
[how do I use an older set of parameters](https://openfold-3.readthedocs.io/en/latest/parameters_reference.html#how-do-i-use-an-older-set-of-parameters).

## Technical report

A technical description of the OpenFold3-preview2 model version is available
[here](https://portal.openfold.omsf.io/reports/of3p2_technical_report.pdf).

## Requirements

- Linux x86-64
- NVIDIA H100 80 GB, compute capability 9.0 (`sm_90a`)
- CPython 3.14, PyTorch 2.10.0 built for CUDA 12.9
- BF16 Q/K/V; BF16 or FP32 pair bias; FP16, BF16, or FP32 additive mask
- `nvcc` 12.9 to build the package; not needed to run a prebuilt wheel

The extensions contain Hopper-specific SASS and no PTX fallback. An ABI or
device mismatch is an error, not a reason to load the binary anyway.

---

# 1. Install and usage

Two ways to get the kernels. Pick one.

### A. The `trionce` pip package (recommended)

The package lives in [`packages/trionce`](packages/trionce) and installs
independently of OpenFold3.

```bash
git clone https://github.com/RyanAIResearch/openfold3_triangle_kernel.git
cd openfold3_triangle_kernel

curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi install -e openfold3-cuda12-dev

pixi run -e openfold3-cuda12-dev install-trionce
```

`install-trionce` builds the five `sm_90a` extensions ahead of time and
editable-installs the package, then runs the preflight. Budget about fifteen
minutes for the compile. It is a task rather than a Pixi dependency because the
build needs `TRIONCE_BUILD_AOT=1`, `TORCH_CUDA_ARCH_LIST=9.0a`, and
`--no-build-isolation`, none of which a declarative dependency can express;
declaring one would install a source-only package whose loader refuses to run.

To install a prebuilt wheel instead of compiling:

```bash
uv pip install packages/trionce/release_artifacts/trionce-0.1.0-cp314-cp314-linux_x86_64.whl
```

### B. The prebuilt OpenFold3 wheel

The `openfold3-cuda12` environment installs a repository-pinned OpenFold3 wheel
with the kernels already compiled in, and needs no build step:

```bash
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 doctor-h100
pixi run -e openfold3-cuda12 verify-aot
```

Use this for a locked production deployment. Use A when you want the kernels as
a dependency you can install into another environment, or when you are working
on the kernels themselves.

## Verifying an install

```bash
pixi run -e openfold3-cuda12-dev doctor-trionce
```

```
trionce doctor: PASS
  gpu       NVIDIA H100 80GB HBM3 (sm_90a)
  torch     2.10.0 / CUDA 12.9
  package   trionce 0.1.0
  aot       5 extensions @ <commit>
  provider  trionce
```

The doctor checks the GPU is `sm_90a`, that `trionce` imports from a real
install, that all five AOT extensions load and match their recorded hashes, and
that the provider resolves as intended. It exits nonzero on the first failure
with the remedy in the message.

Reproduce the performance and accuracy tables above:

```bash
python tools/benchmark_trionce_provider.py --iters 20   # CUDA-event timing
python tools/benchmark_trionce_accuracy.py              # FP64-oracle error
```

## Usage from Python

```python
import trionce

# q/k/v: [*, O, H, N, D] BF16, raw and unscaled -- the kernel applies 1/sqrt(D)
# mask:  [*, O, 1, 1, N] additive
# bias:  [*, 1, H, N, N] BF16 or FP32
supported, reason = trionce.is_triangle_attention_supported(q, k, v, mask, bias)
out = trionce.triangle_attention(q, k, v, mask, bias)   # -> [*, O, N, H, D]
```

The facade fails closed: an unsupported shape raises rather than quietly
computing something else. Model-level routing with a stock fallback belongs to
the host model, which OpenFold3 already implements.

Execution flags, read from the environment:

| Flag | Default | Meaning |
| --- | --- | --- |
| `TRIONCE_SR_BACKWARD=1` | recommended | single-reconstruction backward lane |
| `TRIONCE_SR_SHADOW=1` | recommended | store a BF16 rounding residual in the forward to sharpen the backward row scalar |
| `TRIONCE_ALLOW_JIT=1` | off | source-tree development only: compile the `.cu` files with nvcc on first use |

`TRIONCE_SR_SHADOW=1` does not change the model-visible forward output.

## Which kernel build serves OpenFold3

OpenFold3 reaches the kernels through
[`openfold3/core/kernels/cuda/provider.py`](openfold3/core/kernels/cuda/provider.py),
which resolves one of two providers:

| Provider | Serves from | Selected by |
| --- | --- | --- |
| `trionce` | the installed `trionce` package | `OPENFOLD3_TRIANGLE_PROVIDER=trionce`, or unset with the package installed |
| `intree` | `openfold3/core/kernels/cuda/stablefold_*` | `OPENFOLD3_TRIANGLE_PROVIDER=intree` |

Both build byte-identical `.cu` sources. `openfold3/tests/test_trionce_provider_equivalence.py`
requires their forward outputs and every gradient to match **bitwise**, so the
choice is a packaging decision, not a numerical one. Selection is memoized, so a
run cannot change build midway; requesting `trionce` without the package
installed raises at startup rather than quietly serving the other build. With
the package absent and the variable unset, OpenFold3 behaves exactly as before.

---

# 2. Running a single-GPU OpenFold3 experiment

This path has been run end to end on an H100 80 GB.

### Prepare data

Follow the
[bare-metal training quickstart](docs/source/bare_metal_distributed_quickstart.md)
to generate a runnable PDB subset. It writes `datasets/train_pdb_subset.yaml`.

### Select the backend

```bash
pixi run -e openfold3-cuda12-dev python -m tools.select_training_backend \
  datasets/train_pdb_subset.yaml \
  --backend trionce \
  --output datasets/train_pdb_subset.trionce.yaml
```

Backend keys: `aqlab` (stock PyTorch), `triton`, `cueq`, `deepspeed-evo`,
`stablefold` (in-tree build), `trionce` (pip package). The last two write the
same model flags and differ only in which build serves them.

Add `--single-gpu-smoke` for a one-step, validation-free YAML — the right first
run after an install.

### Train

```bash
export OF3_RUN_CWD="$PWD/../openfold3-run"
export CUDA_VISIBLE_DEVICES=0

scripts/trionce/run_single_gpu.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.trionce.yaml" \
  --seed 42
```

The launcher refuses a `PYTHONPATH`/`PYTHONHOME`/`LD_PRELOAD` that could shadow
the environment, pins the provider and execution flags, requires the run
directory to sit outside the checkout, runs the doctor, validates the runner
YAML, and only then starts training.

For a pure 400-step run with no validation, set in the YAML:

```yaml
data_module_args:
  batch_size: 1
  num_workers: 10
  epoch_len: 400
pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: -1
  max_steps: 400
  limit_val_batches: 0
  num_sanity_val_steps: 0
```

### Compare against the in-tree build

Any launcher takes `TRIONCE_PROVIDER=intree` to run the identical YAML on the
vendored build. If a failure persists under both, it is not a packaging problem.

```bash
TRIONCE_PROVIDER=intree scripts/trionce/run_single_gpu.sh --runner-yaml ... --seed 42
```

---

# 3. Running on several GPUs or several nodes

> **Not yet verified.** The development machine has a single GPU, so these
> paths have never been executed. The scripts are written against this
> repository's existing distributed quickstart and pass shell syntax checks,
> but treat them as unproven. `scripts/trionce/README.md` carries a staged
> bring-up checklist — two GPUs before eight, two nodes at one GPU each before
> eight each — so each step fails cheaply before the next costs an allocation.

### Several GPUs, one node

Topology goes in the YAML. At batch size one,
`epoch_len = max_steps * devices * num_nodes`:

```yaml
data_module_args:
  num_workers: 10
  epoch_len: 3200
pl_trainer_args:
  devices: 8
  num_nodes: 1
  max_steps: 400
```

```bash
export OF3_RUN_CWD="$PWD/../openfold3-run"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

scripts/trionce/run_single_node.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.trionce.yaml" \
  --seed 42
```

PyTorch Lightning starts the local DDP workers, so **do not wrap this in
`torchrun`**. The script refuses to start if the YAML's `devices` disagrees with
`CUDA_VISIBLE_DEVICES`.

### Several nodes

Run on every allocated node, changing only `NODE_RANK`:

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export OF3_RUN_CWD=/shared/openfold3-run
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

scripts/trionce/run_multinode.sh \
  --runner-yaml /shared/openfold3_triangle_kernel/datasets/train_pdb_subset.trionce.yaml \
  --seed 42
```

Under Slurm, `scripts/trionce/sbatch_multinode.sh` allocates the nodes and
starts one `torchrun` per node.

Two hard requirements, inherited from the repository's existing distributed
contract:

1. The repository, Pixi environment, runner YAML, dataset, and run directory
   must be at the **same absolute paths on every node**.
2. `OF3_RUN_CWD` must be a **genuinely shared read-write filesystem**. Rank 0
   writes the kernel receipt there and every other rank reads it back; a path
   that merely has the same name on each node will hang, not report a mismatch.

### What a healthy distributed run shows

- Rank 0 writes `<OF3_RUN_CWD>/stablefold_kernel_ddp.json` with
  `"status": "PASS"` and **one entry per rank**. A missing rank means that rank
  could not read the receipt back, i.e. the run directory is not really shared.
- No `stablefold fallback to stock` at the trunk pair-stack shapes.
- Every node's doctor reports the same `aot ... @ <commit>` line. A differing
  commit means the nodes are not running the same build.

Full operator's manual, including a troubleshooting table for the problems hit
bringing this up: [`scripts/trionce/README.md`](scripts/trionce/README.md).

---

## Configuring the backend by hand

If you are not using `select_training_backend`, set the flags directly:

```bash
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
```

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

Keep the OpenFold3 architecture defaults: every relevant MSA, Pairformer, and
template stack must retain `fuse_projection_weights: false` for the TriMul fast
path to apply. For an attention-only canary, set
`use_stable_triangle_multiplicative: false` and leave the rest.

## Rolling back

No code change is required to return to cuEquivariance:

```yaml
use_stable_triangle_attention: false
use_stable_triangle_multiplicative: false
use_cueq_triangle_kernels: true
```

```bash
unset OPENFOLD3_SR_BACKWARD OPENFOLD3_SR_SHADOW
```

Unsupported Attention shapes are routed to the stock OpenFold3 implementation
by the model selector; the facade itself fails closed. Monitor fallback counts
rather than assuming every call used the custom kernel.

TriMul uses a fixed, non-configurable short-sequence boundary: valid calls with
`N <= 100` are metadata-unsupported by the AOT path and the selector routes them
to stock PyTorch. The H100 BF16 AOT path starts at `N = 101` for
`C_z = C_hidden` in `{32, 64, 96, 128}`. The boundary matches cuEquivariance
0.10's `CUEQ_TRIMUL_FALLBACK_THRESHOLD=100`, but Trionce does not read that or
any other environment variable for it.

## Tests

```bash
# the pip package against the in-tree build: bitwise on every output and gradient
pytest openfold3/tests/test_trionce_provider_equivalence.py --import-mode=importlib

# floors under the speedup and the accuracy, so a regression fails loudly
pytest openfold3/tests/test_trionce_performance_regression.py --import-mode=importlib

# the package's own suite; run it from the project directory, where its
# pyproject testpaths apply and the two-process tests can re-import `tests`
cd packages/trionce && pytest tests
```

## Supported scope and claim boundaries

- Main measured crops: 384, 640, and 768
- Main measured head dimensions: 16, 24, 32, and 48 across real OpenFold3 sites
- Runtime Triangle Attention contract: positive prefixes, runtime sequence
  length and head count, and `1 <= D <= 64`, subject to CUDA grid and memory
  limits
- Determinism means bitwise self-replay with the same backend, inputs, software
  ABI, workspace plan, and hardware; it does not mean cross-backend or
  cross-hardware bit equality
- The rounding residual corrects the backward row-scalar channel only; it does
  not eliminate every BF16 error
- The speedup comes from single reconstruction; the fixed reduction tree and
  rounding residual each have a cost
- Do not infer convergence superiority or memory savings from the timing result
- OpenFold3 training is not bit-deterministic across processes: two runs of the
  same YAML with the same seed produce different weights, because model
  initialization is not fully pinned. End-to-end checkpoint comparison therefore
  cannot isolate a kernel difference; the bitwise kernel-level equivalence tests
  are the instrument that can

The detailed kernel contract is in
[`PRODUCTION_MANIFEST.md`](openfold3/core/kernels/cuda/PRODUCTION_MANIFEST.md).
Experimental protocols and receipts are under
[`scripts/benchmarks/sr/`](scripts/benchmarks/sr/) and [`paper/`](paper/).

## License and attribution

This repository is derived from
[AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3) and is distributed
under Apache-2.0. Retain [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
[`UPSTREAM.md`](UPSTREAM.md), and [`CITATION.cff`](CITATION.cff) when sharing an
internal source or binary bundle.
