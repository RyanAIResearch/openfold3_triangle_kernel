# Trionce as a pip package

Trionce is the deterministic Hopper (`sm_90a`) kernel pair behind OpenFold3's
stable triangle routes: Triangle Attention and the Triangle Multiplicative
Update. This page covers installing it as a standalone pip package and running
OpenFold3 against it on one GPU, several GPUs of one node, and several nodes.

The package lives in [`trionce/`](../../trionce) and is installed separately
from OpenFold3, the same way a host model adopts cuEquivariance: one pip
dependency, one backend key, and a stock fallback for unsupported shapes.

## What changes, and what does not

OpenFold3 reaches the kernels through
`openfold3/core/kernels/cuda/provider.py`, which resolves one of two providers:

| Provider | Serves from | Selected by |
| --- | --- | --- |
| `trionce` | the installed `trionce` package | `OPENFOLD3_TRIANGLE_PROVIDER=trionce`, or unset with the package installed |
| `intree` | `openfold3/core/kernels/cuda/stablefold_*` | `OPENFOLD3_TRIANGLE_PROVIDER=intree` |

Both providers compile byte-identical `.cu` sources, so the choice is a
packaging decision, not a numerical one.
`openfold3/tests/test_trionce_provider_equivalence.py` holds them to that:
forward outputs and every gradient must match **bitwise**, and both must
report the same support verdicts so they fall back to stock math at exactly
the same shapes.

Selection is memoized on first use, so a run cannot change provider midway.
Requesting `trionce` without the package installed raises at startup rather
than quietly serving the other build. With the package absent and the variable
unset, OpenFold3 behaves exactly as it did before.

## Requirements

- Linux x86-64 with an NVIDIA H100 80 GB (compute capability 9.0, `sm_90a`)
- The pinned Pixi toolchain: CPython 3.14, PyTorch 2.10.0, CUDA 12.9
- `nvcc` 12.9, which the Pixi environment already provides

## Install

```bash
pixi install -e openfold3-cuda12-dev

ENVP="$PWD/.pixi/envs/openfold3-cuda12-dev"
CUDA_HOME="$ENVP" \
TORCH_CUDA_ARCH_LIST=9.0a \
TRIONCE_BUILD_AOT=1 \
uv pip install --python "$ENVP/bin/python" --no-build-isolation -e trionce
```

`TRIONCE_BUILD_AOT=1` is what compiles the five `sm_90a` extensions ahead of
time and writes `trionce/_build_manifest.json`, the receipt the loader checks
at import. Expect the build to take tens of minutes; the WGMMA/TMA kernels are
large. Without that variable `pip install` produces a source-only package whose
loader deliberately refuses to run.

`TORCH_CUDA_ARCH_LIST=9.0a` is required exactly as written: the kernels embed
architecture-specific SASS and no PTX, so they neither build for nor run on any
other architecture.

Verify the install before trusting a run:

```bash
cd /tmp && "$PWD/../<repo>/.pixi/envs/openfold3-cuda12-dev/bin/python" \
  <repo>/tools/trionce_doctor.py
```

Run it by path from outside the checkout. An interpreter started in the
repository root puts the root on `sys.path`, where the package's own `trionce/`
project folder shadows the install as an empty namespace package — setuptools
appends its editable finder *after* the standard path finder, so the namespace
wins and the doctor reports a broken install that is in fact fine.

The doctor checks the GPU is `sm_90a`, the package imports from where you think
it does, all five AOT extensions load and match their recorded hashes, and the
provider resolves as intended. It exits nonzero on the first failure. Pass
`--json` for a machine-readable report, or `--expect-provider trionce` to fail
unless that provider is the one serving.

## Select the backend

The backend key writes the model flags into the runner YAML:

```bash
.pixi/envs/openfold3-cuda12-dev/bin/python -m tools.select_training_backend \
  datasets/train_pdb_subset.yaml \
  --backend trionce \
  --output datasets/train_pdb_subset.trionce.yaml
```

`trionce` and `stablefold` write the same model flags
(`use_stable_triangle_attention`, `use_stable_triangle_multiplicative`); they
differ only in which build serves them, which the launch scripts pin through
`OPENFOLD3_TRIANGLE_PROVIDER`. Every other backend key
(`aqlab`, `triton`, `cueq`, `deepspeed-evo`) is unchanged.

Add `--single-gpu-smoke` to generate a one-step, validation-free runner YAML
for an installation check.

## Run

`scripts/trionce/README.md` carries the full operator's manual for these
launchers, including a troubleshooting table and the staged checklist for
bringing up multi-GPU and multi-node runs, which have not been executed here.


The three launchers share `scripts/trionce/trionce_env.sh`, which pins the
provider, exports the execution flags, checks the run directory is outside the
checkout, and runs the doctor preflight before the job starts.

`OPENFOLD3_SR_BACKWARD=1` and `OPENFOLD3_SR_SHADOW=1` are exported by default;
both kernel builds honour those spellings, which keeps the two providers
directly comparable.

### One GPU

```bash
export OF3_RUN_CWD="$PWD/../openfold3-run"
export CUDA_VISIBLE_DEVICES=0

scripts/trionce/run_single_gpu.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.trionce.yaml" \
  --seed 42
```

### Several GPUs, one node

```bash
export OF3_RUN_CWD="$PWD/../openfold3-run"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

scripts/trionce/run_single_node.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.trionce.yaml" \
  --seed 42
```

PyTorch Lightning starts the local DDP workers, so this must not be wrapped in
`torchrun`. The script refuses to start if the YAML's `devices` disagrees with
`CUDA_VISIBLE_DEVICES`. Configure the topology in the YAML:

```yaml
data_module_args:
  num_workers: 10
  epoch_len: 3200          # max_steps * devices * num_nodes, at batch size 1
pl_trainer_args:
  devices: 8
  num_nodes: 1
  max_steps: 400
```

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

`scripts/trionce/sbatch_multinode.sh` submits the same thing through Slurm with
one `torchrun` per node.

The repository, Pixi environment, runner YAML, dataset, and run directory must
be at the same paths on every node, and `OF3_RUN_CWD` must be a genuinely
shared read-write filesystem: rank 0 writes the kernel receipt there and every
other rank reads it back, so a path that merely has the same name on each node
is not enough. Set `epoch_len` to `max_steps * devices * num_nodes`.

## Confirm the migration changed nothing

```bash
.pixi/envs/openfold3-cuda12-dev/bin/python -m pytest \
  openfold3/tests/test_trionce_provider_equivalence.py -v
```

The suite runs both providers in one process -- they register their
`torch.library` ops in different namespaces (`trionce::` and
`openfold3_stablefold::`), so they coexist -- and requires bitwise-identical
forward outputs and gradients from Triangle Attention at four pair-stack
shapes and from the fused TriMul, including every parameter gradient. It also
checks each provider reproduces itself across reruns, without which the
cross-provider comparison would prove nothing.

To compare a real training step, run the same runner YAML twice, changing only
the provider:

```bash
TRIONCE_PROVIDER=intree  scripts/trionce/run_single_gpu.sh --runner-yaml ... --seed 42
TRIONCE_PROVIDER=trionce scripts/trionce/run_single_gpu.sh --runner-yaml ... --seed 42
```

Both runs must reach `max_steps` without a `stablefold fallback to stock`
warning at the trunk shapes. Note that the single-GPU smoke preset disables
W&B and checkpointing and OpenFold3 ships no CSV logger, so no loss is written
to disk for a numeric comparison; the bitwise kernel equivalence above is the
claim that carries the weight, and the benchmarks below cover speed and
accuracy.

## Speed and accuracy

Packaging must not cost throughput or numerical quality:

```bash
python tools/benchmark_trionce_provider.py --iters 20   # CUDA-event timing
python tools/benchmark_trionce_accuracy.py              # FP64-oracle error
```

Measured on one H100 80 GB, forward+backward at `trunk_n640_d32`: 6.97 ms for
`trionce` against 6.96 ms for `intree` (packaging cost 1.00x) and 28.97 ms for
stock attention (4.16x speedup). L2 relative error against an FP64 oracle on the
forward output: `1.831e-03` for both kernel providers against `3.201e-03` for
stock, every provider replaying bitwise.

## Package tests

The package ships its own suite, which needs an H100:

```bash
.pixi/envs/openfold3-cuda12-dev/bin/python -m pytest trionce/tests -v
```

`trionce/tests/host_openfold3/` additionally needs an OpenFold3 installation
and is skipped otherwise.

## Execution flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `OPENFOLD3_TRIANGLE_PROVIDER` | `trionce` when installed | which kernel build serves the stable routes |
| `TRIONCE_SR_BACKWARD=1` | recommended | single-reconstruction backward lane |
| `TRIONCE_SR_SHADOW=1` | recommended | store a BF16 rounding residual in the forward to sharpen the backward row scalar |
| `TRIONCE_ALLOW_JIT=1` | off | source-tree development only: compile the `.cu` files with nvcc on first use |

The `STABLEFOLD_*` and `OPENFOLD3_*` spellings of the last three are accepted as
aliases, so an existing launcher keeps working; the canonical name wins when
more than one is set.
