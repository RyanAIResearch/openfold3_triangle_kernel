# OpenFold3 + StableFold Training Quick Start

This guide follows the Lambda OpenFold3 quick-start flow: set up the
environment, prepare a small PDB dataset, remove samples affected by unavailable
files, and launch training on one or more GPUs.

The current prebuilt StableFold package supports full NVIDIA H100 80 GB GPUs on
Linux x86-64. It does not require a system CUDA Toolkit or local kernel build.

## Setup

### Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
```

Restart the shell if the `pixi` command is not immediately available.

### Clone the Repository

```bash
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git
cd openfold3_triangle_kernel
```

### Set Up OpenFold3 and StableFold

```bash
pixi install -e openfold3-cuda12
```

This one command creates the environment and installs the repository-pinned
OpenFold3 wheel with the prebuilt StableFold kernels. No separate setup command
or local kernel build is required.

To optionally check the GPU, environment, and all five AOT kernel modules, run:

```bash
pixi run -e openfold3-cuda12 doctor-h100
pixi run -e openfold3-cuda12 verify-aot
```

If verification reports `StableFold AOT artifacts require a clean Git checkout`, first
inspect the checkout:

```bash
git status --short
```

When the only output is `M pixi.lock`, the checkout predates the current
lockfile. Update it without reinstalling the repository:

```bash
git restore --source=HEAD -- pixi.lock
git pull --ff-only origin main
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 verify-aot
```

Do not restore `pixi.lock` when you intentionally edited dependencies, and do
not discard any other paths reported by `git status`.

## Prepare Training Data

### Prepare a Runnable PDB Subset

For installation checks and backend comparisons on up to eight GPUs, generate a
reproducible subset with 512 training structures. OpenFold3's distributed
sampler samples with replacement, so the same subset can provide the 3,200
draws used by an eight-GPU, 400-step timing run. This compact subset is intended
for functional and performance validation, not model-quality or convergence
claims:

```bash
pixi run -e openfold3-cuda12 prepare-pdb-subset -- \
  --train-size 512 \
  --seed 42 \
  --workers 24
```

This is the recommended data command. It performs the complete sequence:

1. generates the reproducible train cache and pinned validation cache;
2. downloads target structures, alignment arrays, templates, and reference
   molecules;
3. performs a second resumable download pass for transient failures;
4. removes samples affected by persistent `404` or `NoSuchKey` responses; and
5. validates every directory and cache path in the resulting runner YAML.

The command succeeds only after writing a runner YAML whose configured data
paths exist:

```text
datasets/train_pdb_subset.yaml
datasets/pdb_subset_purification_report.json
```

Downloads are resumable. If the command is interrupted, paste the same command
again; existing files are skipped. Persistent missing S3 objects are recorded
in the JSON report and only dependent samples are removed. If the download
directory is absent, purification removes every training sample, or a runner
path is invalid, the command exits nonzero and does not claim the data is ready.

To check the prepared YAML without starting training, run:

```bash
pixi run -e openfold3-cuda12 validate-training-data \
  datasets/train_pdb_subset.yaml
```

The one-step single-GPU smoke test does not consume all 512 structures. The same
prepared subset can be reused for the full eight-GPU backend comparison.

The original sampled caches are never modified. For debugging, the individual
`generate-pdb-subset`, `download-pdb-subset`, and `purify-pdb-subset` tasks
remain available, but the all-in-one command above is the supported quick-start
path.

### Prepare Full-Crop Benchmark Data

Backend timing at crop sizes 384, 640, and 768 must not be diluted by samples
that are padded up to the requested token dimension. Build one conservative
filtered cache that is safe for all three sizes:

```bash
mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"

pixi run -e openfold3-cuda12 prepare-crop-benchmark-data -- \
  "$PWD/datasets/train_pdb_subset.yaml" \
  "$CROP_DATA_ROOT" \
  --crop-sizes 384 640 768 \
  --num-workers 10

for CROP in 384 640 768; do
  pixi run -e openfold3-cuda12 validate-training-data \
    "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml"
done
```

The preparation command never changes the source cache and refuses to overwrite
an existing output directory. It retains only structures whose preprocessed NPZ
annotations have at least 768 unique token IDs, at least 768 resolved token
centers, and no more chains than the configured chain-crop limit. It writes the
selected structure IDs, per-structure counts, input and output SHA256 digests,
and the three runner-YAML digests to `provenance.json`.

This is a conservative static prefilter. Every benchmark command below also
passes `--require-full-crop`, which checks the collated CPU `token_mask` for
every consumed batch and fails before training if any requested token position
is padding. This guarantee concerns the model's token dimension; normal
MSA/template/atom batching and padding remain part of the unchanged OpenFold3
data pipeline. It is a no-padding check, not a per-step fingerprint proving that
independent processes selected identical token, MSA, and template features.

### Choose a Training Backend

Use StableFold by default:

```bash
pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend stablefold \
  --output datasets/train_pdb_subset.stablefold.yaml
```

The command can be pasted directly into Bash.

To select another implementation, change `BACKEND` and use the generic form:

```bash
export BACKEND=cueq

pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend "$BACKEND" \
  --output "datasets/train_pdb_subset.${BACKEND}.yaml"
```

| `BACKEND` | Selected operator route |
| --- | --- |
| `aqlab` | Stock OpenFold3 PyTorch operators |
| `triton` | OpenFold3 Triton triangle attention with the stock PyTorch training TMU path |
| `cueq` | NVIDIA cuEquivariance triangle kernels |
| `stablefold` | StableFold triangle attention and triangle multiplicative kernels |
| `deepspeed-evo` | DeepSpeed Evo attention operator; this does not enable ZeRO |

The selector first disables every mutually exclusive route and then enables the
requested backend, so rerunning it cannot leave mixed backend flags. Diffusion
stays on OpenFold3's original path. OpenFold3 continues to own the model, data
pipeline, loss, optimizer, EMA, checkpointing, and distributed training.

The production comparison below uses the first four keys exactly as written:
`aqlab`, `triton`, `cueq`, and `stablefold`. These are configuration keys, not
display-name aliases. In particular, selecting `triton` enables OpenFold3's
Triton triangle-attention route; training TMU remains on its stock PyTorch path
because OpenFold3 training does not use the in-place inference path.

## Training

### First: Run a One-Step Single-GPU Smoke Test

Always run this test before requesting multiple GPUs or starting a long job. It
creates a separate runner YAML for one BF16 training step with validation, W&B,
and checkpoint output disabled:

```bash
export BACKEND=stablefold

pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend "$BACKEND" \
  --single-gpu-smoke \
  --output "datasets/single_gpu_smoke.${BACKEND}.yaml"
```

The `--single-gpu-smoke` preset sets these values automatically; no manual YAML
edit is required:

```yaml
experiment_settings:
  restart_checkpoint_path: null

data_module_args:
  batch_size: 1
  num_workers: 0
  num_workers_validation: 0
  epoch_len: 1

checkpoint_config:
  save_top_k: 0
  save_last: false

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: 1
  max_steps: 1
  limit_val_batches: 0
  num_sanity_val_steps: 0
```

The official launcher runs outside the source checkout. Create its working
directory and expose one H100:

```bash
mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-gpu \
  train \
  --runner-yaml "$PWD/datasets/single_gpu_smoke.${BACKEND}.yaml" \
  --seed 42
```

A successful smoke test completes exactly one forward pass, backward pass, and
optimizer step through the selected backend. To smoke-test another backend,
change only `BACKEND`, regenerate its runner YAML, and rerun the same launch
command.

This exact `stablefold` smoke path has been verified to reach `max_steps=1` on
a Lambda H100 80 GB GPU. The first launch can spend about a minute in CPU-heavy
model and data setup before the training progress bar appears. Warnings that no
logger is configured are expected because the smoke preset deliberately
disables W&B; they are not training failures.

### Compare Backend Training Time and Loss

Run every backend from scratch with the same full-crop data, initialization
seed, data seed, BF16 precision, and GPU topology. The benchmark disables
validation, W&B, and checkpoint output. The following optional single-H100
recipe covers crop sizes 384, 640, and 768. Each arm excludes five warmup steps
and then measures 20 optimizer training steps:

```bash
set -euo pipefail
mkdir -p ../openfold3-run/backend-comparison-1gpu
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
export CUDA_VISIBLE_DEVICES=0
BACKENDS=(aqlab triton cueq stablefold)
MATRIX_FAILED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

for CROP in 384 640 768; do
  export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-1gpu/crop${CROP}"
  if [[ -e "$RESULTS_ROOT" ]]; then
    echo "refusing non-fresh result directory: $RESULTS_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT/logs"
  FAILED_BACKENDS=()
  for BACKEND in "${BACKENDS[@]}"; do
    if ! pixi run -e openfold3-cuda12 benchmark-single-gpu \
        --runner-yaml "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml" \
        --backend "$BACKEND" \
        --output-root "$RESULTS_ROOT" \
        --warmup-steps 5 \
        --measure-steps 20 \
        --devices 1 \
        --num-workers 10 \
        --require-full-crop \
        --use-tf32 \
        --seed 42 2>&1 | tee "$RESULTS_ROOT/logs/${BACKEND}.log"; then
      FAILED_BACKENDS+=("$BACKEND")
      continue
    fi
    if [[ ! -f "$RESULTS_ROOT/$BACKEND/summary.json" ]]; then
      FAILED_BACKENDS+=("$BACKEND")
    fi
  done

  if (( ${#FAILED_BACKENDS[@]} )); then
    echo "crop $CROP is incomplete: ${FAILED_BACKENDS[*]}; not plotting" >&2
    MATRIX_FAILED=1
    continue
  fi
  pixi run -e openfold3-cuda12 plot-backend-comparison \
    "$RESULTS_ROOT" \
    --baseline triton
done
if (( MATRIX_FAILED )); then
  exit 1
fi
```

The plot uses `triton` as the performance baseline because that is the explicit
OpenFold3 Triton route under test; the `aqlab` PyTorch route remains a separate
row. `deepspeed-evo` is supported by the selector but is deliberately outside
this four-route matrix. Run all arms on the same otherwise-idle machine; do not
compare results collected on different GPU models or while another process is
using a device.

The model remains in `bf16-mixed` precision. `--use-tf32` enables TF32 only for
the remaining FP32 matrix multiplications and is passed to every backend so the
comparison is matched. The repository pins all three cuEquivariance
distributions to `0.10.0`; its full-training triangle-attention backward sees
an FP32 bias and requires TF32. Keeping this setting explicit prevents a backend
arm from silently selecting a different FP32 matmul policy. It is a benchmark
protocol choice, not the current AQLab training CLI default.

For the matched two-H100 comparison, change both the visible-device list and
`--devices`, and keep its results separate from the one-GPU run:

```bash
set -euo pipefail
mkdir -p ../openfold3-run/backend-comparison-2gpu
export DEVICES=2
export CUDA_VISIBLE_DEVICES=0,1
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
BACKENDS=(aqlab triton cueq stablefold)
MATRIX_FAILED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

for CROP in 384 640 768; do
  export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-2gpu/crop${CROP}"
  if [[ -e "$RESULTS_ROOT" ]]; then
    echo "refusing non-fresh result directory: $RESULTS_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT/logs"
  FAILED_BACKENDS=()
  for BACKEND in "${BACKENDS[@]}"; do
    if ! pixi run -e openfold3-cuda12 benchmark-single-node \
        --runner-yaml "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml" \
        --backend "$BACKEND" \
        --output-root "$RESULTS_ROOT" \
        --warmup-steps 5 \
        --measure-steps 20 \
        --devices "$DEVICES" \
        --num-workers 10 \
        --require-full-crop \
        --use-tf32 \
        --seed 42 2>&1 | tee "$RESULTS_ROOT/logs/${BACKEND}.log"; then
      FAILED_BACKENDS+=("$BACKEND")
      continue
    fi
    if [[ ! -f "$RESULTS_ROOT/$BACKEND/summary.json" ]]; then
      FAILED_BACKENDS+=("$BACKEND")
    fi
  done

  if (( ${#FAILED_BACKENDS[@]} )); then
    echo "crop $CROP is incomplete: ${FAILED_BACKENDS[*]}; not plotting" >&2
    MATRIX_FAILED=1
    continue
  fi
  pixi run -e openfold3-cuda12 plot-backend-comparison \
    "$RESULTS_ROOT" \
    --baseline triton
done
if (( MATRIX_FAILED )); then
  exit 1
fi
```

For an eight-H100 run, use the following copy-paste block on the eight-GPU
machine. It refuses a non-fresh per-crop result directory and retains the same
four routes, three crops, workers per rank, and timing protocol:

```bash
set -euo pipefail
mkdir -p ../openfold3-run/backend-comparison-8gpu
export DEVICES=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
BACKENDS=(aqlab triton cueq stablefold)
MATRIX_FAILED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

for CROP in 384 640 768; do
  export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-8gpu/crop${CROP}"
  if [[ -e "$RESULTS_ROOT" ]]; then
    echo "refusing non-fresh result directory: $RESULTS_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT/logs"
  FAILED_BACKENDS=()
  for BACKEND in "${BACKENDS[@]}"; do
    if ! pixi run -e openfold3-cuda12 benchmark-single-node \
        --runner-yaml "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml" \
        --backend "$BACKEND" \
        --output-root "$RESULTS_ROOT" \
        --warmup-steps 5 \
        --measure-steps 20 \
        --devices "$DEVICES" \
        --num-workers 10 \
        --require-full-crop \
        --use-tf32 \
        --seed 42 2>&1 | tee "$RESULTS_ROOT/logs/${BACKEND}.log"; then
      FAILED_BACKENDS+=("$BACKEND")
      continue
    fi
    if [[ ! -f "$RESULTS_ROOT/$BACKEND/summary.json" ]]; then
      FAILED_BACKENDS+=("$BACKEND")
    fi
  done

  if (( ${#FAILED_BACKENDS[@]} )); then
    echo "crop $CROP is incomplete: ${FAILED_BACKENDS[*]}; not plotting" >&2
    MATRIX_FAILED=1
    continue
  fi
  pixi run -e openfold3-cuda12 plot-backend-comparison \
    "$RESULTS_ROOT" \
    --baseline triton
done
if (( MATRIX_FAILED )); then
  exit 1
fi
```

Do not wrap `benchmark-single-node` in `torchrun`: PyTorch Lightning launches
the local DDP workers. The benchmark sets per-GPU batch size to one, global
batch size to `DEVICES`, and
`epoch_len = (warmup_steps + measure_steps) * DEVICES`; `max_steps` remains the
same optimizer-step count for 1, 2, 4, and 8 GPUs.

The production timing protocol uses 10 DataLoader workers per rank. That means
10 workers on one GPU, 20 on two GPUs, and 80 on eight GPUs. Use the same value
for every backend in one comparison and reserve enough CPU cores. The recorded
`steady_step_wall_seconds` includes data waiting, while `gpu_step_mean_ms` in
`summary.json` does not, so their difference shows how much time the data
pipeline costs. The one-step installation smoke test above intentionally keeps
`num_workers: 0`; it is not a performance measurement.

Ten is the `DataModuleArgs` code default when a runner YAML omits the field; it
is not a universal value in every upstream recipe. For example, the bundled
phase-training YAMLs explicitly request 16. This benchmark pins 10 rather than
relying on either implicit defaults or a phase-specific override, and records
the effective value in every successful summary.

Do not combine these results with older runs that used a different worker
count. The outer sampler sequence is fixed by `data_seed`, but each worker owns
a seeded random stream used by stochastic crop, MSA, and template
featurization. Changing the worker count can therefore change the concrete
random features assigned to later samples even though it does not change the
training objective or sampling distribution. Matched backend arms must use the
same worker count, seed, topology, and fresh-process launch protocol.

Compare backends only within the same GPU count. Changing the GPU count changes
the global batch and therefore can change the loss trajectory even when the
seed is held fixed. Use a separate `RESULTS_ROOT` and generate a separate
comparison for each of 1, 2, 4, and 8 GPUs.

Each successful backend directory contains:

```text
runner.yaml
steps.csv
summary.json
```

`steps.csv` records the training loss and CUDA-event step time. On DDP, loss is
the mean across ranks at each optimizer step and GPU step time is the maximum
across ranks, representing the step's critical path. `summary.json` records the
topology, device/world-size counts, global batch size, total fit wall time,
steady-state wall seconds per training step, global samples per second, and
mean, median, p95, minimum, and maximum critical-path GPU step time. It also
records the source and effective runner hashes, checkout identity, package
versions, workers per rank, crop size, TF32 state, all mutually exclusive route
flags, and the full-crop gate outcome. Warmup is excluded from every
steady-state timing statistic. CUDA events are recorded asynchronously; device
synchronization occurs only at the measurement boundaries, and rank aggregation
occurs after the measured interval, so the benchmark does not add a
synchronization to every measured step.

The comparison command writes:

```text
comparison/backend_comparison.csv
comparison/backend_comparison.json
comparison/training_time.png
comparison/loss_curves.png
```

`training_time.png` shows seconds per optimizer training step and speedup over
the selected baseline for the recorded GPU count. `loss_curves.png` overlays
matched-seed training loss by optimizer step; for multi-GPU runs it plots the
global mean loss. The CSV and JSON files are the source of truth for descriptive
tables and plots; paper claims require repeated complete matched blocks as
described below. The comparison command refuses to combine summaries whose seed,
workers, TF32 setting, warmup/measurement counts, crop, runner hash, checkout,
or full-crop status differ.

The copy-paste loops also retain one log per requested arm and verify that all
four `summary.json` files exist before plotting. A failed arm may leave only its
log and `runner.yaml`; the benchmark does not fabricate `steps.csv`,
`summary.json`, or structured OOM evidence. Preserve the log, record the failure
separately, and do not present a plot from the remaining arms as a complete
four-route matrix.

### Reproduced Two-H100 Results

These results were recorded on September 4, 2026. The table below is a
descriptive production screen, not a convergence or statistical-significance
claim. Every requested arm was launched on the same otherwise-idle two-H100
host. The benchmark harness and configuration tools came from clean checkout
`5d4c79748fa5c76ce6a69478c6f4d38395d81b2c`; that checkout hash does **not**
identify the imported model implementation. The installed OpenFold3 runtime
wheel is
`0.5.1.dev5860+g87ebe22da` (CUDA 12.9, PyTorch 2.10.0, Triton 3.6.0, and
cuEquivariance 0.10.0). Each arm used batch size one per GPU, BF16 mixed
precision, TF32 enabled for the common FP32 matmuls, seed/data seed 42, 10 data
workers per rank, five excluded warmup steps, and 20 measured optimizer steps.
The standalone `stablefold-0.1.0` artifact under `stablefold/release_artifacts`
was not installed by this Pixi environment and is not the binary measured in
this table.

The source purified cache SHA256 is
`f7772e0b2d41bc1b1b521409eead4b056a9be660fbbff2971ef48a8b81ffd35f`.
The filtered-cache SHA256 is
`6416bbadcf034bbe2974f7eb378df1dd25abe5c24380880b27c2c42a5bc2dac8`.
It contains 19 structures; the smallest has 835 tokens and 826 resolved token
centers, and the largest chain count is 16. Every successful run reports
`full_crop_validation: passed`; any failed arm is retained and identified in
the table rather than silently removed or replaced.

`Speedup vs Triton` is Triton wall seconds per step divided by the row's wall
seconds per step within the same crop. DDP GPU time is reduced by the maximum
rank at every step before the 20-step mean is taken. Global samples/s uses the
two-sample global batch. Results from other GPU counts must be reported as
separate blocks because their global batches and loss trajectories differ.

| Crop | Backend route | Wall s/step | Global samples/s | Critical-path GPU ms/step | Speedup vs Triton | Status |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 384 | AQLab PyTorch (`aqlab`) | 6.900 | 0.2899 | 6658 | 0.785x | passed |
| 384 | AQLab Triton (`triton`) | 5.413 | 0.3695 | 5173 | 1.000x | passed |
| 384 | cuEquivariance (`cueq`) | 4.550 | 0.4396 | 4309 | 1.190x | passed |
| 384 | StableFold (`stablefold`) | **4.286** | **0.4666** | **4047** | **1.263x** | passed |
| 640 | AQLab PyTorch (`aqlab`) | 20.845 | 0.09594 | 20489 | 0.680x | passed |
| 640 | AQLab Triton (`triton`) | 14.176 | 0.1411 | 13840 | 1.000x | passed |
| 640 | cuEquivariance (`cueq`) | 9.706 | 0.2061 | 9373 | 1.460x | passed |
| 640 | StableFold (`stablefold`) | **9.153** | **0.2185** | **8822** | **1.549x** | passed |
| 768 | AQLab PyTorch (`aqlab`) | -- | -- | -- | -- | OOM in first backward pass |
| 768 | AQLab Triton (`triton`) | 21.695 | 0.09219 | 21301 | 1.000x | passed |
| 768 | cuEquivariance (`cueq`) | 13.707 | 0.1459 | 13310 | 1.583x | passed |
| 768 | StableFold (`stablefold`) | **12.730** | **0.1571** | **12335** | **1.704x** | passed |

The PyTorch crop-768 failure is a measured result, not a missing cell: both DDP
ranks requested another 6.75 GiB during the first backward pass while each
process was using 73.4--73.7 GiB of its 79.18-GiB GPU. PyTorch reported
60.39--61.30 GiB allocated and 10.49--11.13 GiB reserved but unallocated. No
timing or speedup is imputed for the failed arm. The sanitized machine-readable
receipt is kept under
`release_artifacts/backend-matrix-h100-2gpu-20260904/`; it intentionally omits
hostnames, user names, absolute paths, raw logs, and training data.

```{figure} ../imgs/backend_loss_2gpu_crops_384_640_768.png
:alt: Two-H100 OpenFold3 loss curves for four backends at crop sizes 384, 640, and 768, with differences from the Triton route.
:width: 100%

Two-H100 training loss for the same measured runs. The shaded region contains
the five timing-warmup steps; lower panels show each route minus Triton at the
same optimizer step. AQLab PyTorch has no crop-768 curve because that arm ran
out of memory.
```

StableFold and Triton remain closely aligned over this short screen. The maximum
absolute loss differences were 0.00171 at crop 384 and 0.00402 at crop 640. At
crop 768, StableFold has three isolated deviations at one-based plotted steps
15, 21, and 24 (zero-based CSV indices 14, 20, and 23): -0.0320, +0.0713, and
-0.0559, before finishing 0.00323 below Triton. They do
not form a sustained divergence in these 25 steps, but this run did not record
per-batch feature or permutation fingerprints, so it cannot prove that the
routes consumed bit-identical stochastic features or identify the cause of
those points. These curves support short-run numerical stability only; they do
not establish long-horizon convergence or unchanged model accuracy.

Only the complete two-H100 block is published here. A partial single-H100
campaign was deliberately stopped and is excluded rather than presented as a
complete matrix.

The published receipt does not retain launch order, so this block cannot be used
to quantify warm-machine or position bias. There is also only one seed/process
block per cell, so the 20 temporally adjacent steps are not 20 independent
replicates. For paper claims, repeat complete matched blocks with independent
seeds, rotate or randomize backend launch order, record that order, and report
confidence intervals. The eight-H100 commands above intentionally contain no
estimated numbers; add that table only from a completed eight-GPU run.

### Scheme 1: Single-GPU Training

The generated YAML defaults to epoch-based training. For a pure 400-training-step
run with no validation, update the existing entries in
`datasets/train_pdb_subset.${BACKEND}.yaml`:

```yaml
experiment_settings:
  output_dir: ./openfold3_output
  restart_checkpoint_path: null

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

`max_steps` is the stopping condition. Validation and validation sanity checks
are disabled, so the result covers exactly 400 optimizer training steps.

Use these topology values:

```yaml
data_module_args:
  num_workers: 10
  epoch_len: 400

pl_trainer_args:
  devices: 1
  num_nodes: 1
  max_steps: 400
```

Launch training on one H100:

```bash
mkdir -p ../openfold3-run
export BACKEND=stablefold
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-gpu \
  train \
  --runner-yaml "$PWD/datasets/train_pdb_subset.${BACKEND}.yaml" \
  --seed 42
```

### Scheme 2: Multi-GPU Training

#### Multiple GPUs on One Node

For eight H100s on one node, configure:

```yaml
data_module_args:
  num_workers: 10
  epoch_len: 3200

pl_trainer_args:
  devices: 8
  num_nodes: 1
  max_steps: 400
```

Then launch OpenFold3 once:

```bash
mkdir -p ../openfold3-run
export BACKEND=stablefold
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-node \
  train \
  --runner-yaml "$PWD/datasets/train_pdb_subset.${BACKEND}.yaml" \
  --seed 42
```

Do not wrap this single-node command in `torchrun`. PyTorch Lightning starts
the local DDP workers.

For batch size one, use:

```text
epoch_len = max_steps * devices * num_nodes
```

This gives every topology the same number of global optimizer steps.

#### Multiple Nodes

For two nodes with eight H100s per node, configure:

```yaml
data_module_args:
  num_workers: 10
  epoch_len: 6400

pl_trainer_args:
  devices: 8
  num_nodes: 2
  max_steps: 400
```

The repository, Pixi environment, prepared runner YAML, dataset, and output
directory must use the same paths on every node. The run directory must be a
real shared, read-write filesystem: StableFold rank 0 writes the kernel receipt
there and every other rank reads it back. A path that merely has the same name
on each node is not sufficient.

Install the pinned environment on every node from the shared repository. Then
create and verify the shared run directory once before launching workers:

```bash
cd /shared/openfold3_triangle_kernel
pixi install -e openfold3-cuda12

mkdir -p /shared/openfold3-run
test -r /shared/openfold3-run && test -w /shared/openfold3-run
test -r "$PWD/datasets/train_pdb_subset.stablefold.yaml"
```

Run the following command on both allocated nodes from
`/shared/openfold3_triangle_kernel`, changing `NODE_RANK` to `0` or `1`:

```bash
export BACKEND=stablefold
export OF3_RUN_CWD=/shared/openfold3-run
export RUNNER_YAML="$PWD/datasets/train_pdb_subset.${BACKEND}.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-multinode \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --module openfold3.run_openfold \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42
```

Use Slurm or the site's scheduler to allocate the nodes and start the per-node
commands concurrently. The next section provides ready-to-submit Slurm batch
scripts.

#### Multiple Nodes with Slurm

Slurm can start every worker for you, on one node or on several. Submit the
batch scripts below with `sbatch`; the shared-path requirements from the
previous section still apply.

Rules that the pinned Lightning 2.6.1 enforces:

- `--nodes` must equal `num_nodes`, and `--ntasks-per-node` must equal
  `devices`. Never use bare `--ntasks`.
- Set `devices` and `num_nodes` in the runner YAML before submitting. If both
  are still `1`, Lightning does not read the Slurm environment at all: every
  task silently trains alone on its own GPU and writes into the same output
  directory. The benchmark task avoids this because `--devices` and
  `--num-nodes` set both values.
- Request GPUs with `--gres=gpu:N`. Do not use `--gpus-per-task` or
  `--gpu-bind`; every task must see all GPUs on its node.
- Use `sbatch`, not an `salloc` shell. Lightning ignores Slurm when the job
  name is `bash` or `interactive`.
- Run `pixi install -e openfold3-cuda12` on the shared checkout before
  submitting.

Lightning reads ranks and world size from Slurm and uses the first node of the
allocation as the rendezvous address, so no `MASTER_ADDR` or `NODE_RANK` is
needed. Every `srun` below carries `--kill-on-bad-exit=1`. It is not an early
abort: it tells Slurm to terminate the whole job step as soon as any single
rank exits with an error. Without it, one crashed rank leaves the other
fifteen blocked inside NCCL until the job's time limit expires. Each `srun`
also repeats `--cpus-per-task`: since Slurm 22.05, `srun` no longer inherits
that value from `sbatch`, and without it every task is confined to one CPU
core, which starves the data loader workers.

##### Compare Backends on Two Nodes

This is the multi-node counterpart of the single-node comparison loop above.
The `benchmark-multinode` task runs the same benchmark tool; `--num-nodes`
makes it set `epoch_len = (warmup_steps + measure_steps) * devices * num_nodes`
and a global batch of `devices * num_nodes`, so the optimizer-step count stays
the same as on one node. Under `srun` the tool starts once per task, Lightning
joins the tasks into one DDP group, and only global rank 0 writes results.

Save the following as `/shared/openfold3-run/compare_2x8.sbatch` and submit it
with `sbatch /shared/openfold3-run/compare_2x8.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=of3-compare-2x8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=11
#SBATCH --time=24:00:00
#SBATCH --output=/shared/openfold3-run/slurm-%j.out

set -euo pipefail

export NNODES=2
export GPUS_PER_NODE=8
export CROP=768
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_RUN_CWD=/shared/openfold3-run
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-${NNODES}x${GPUS_PER_NODE}gpu-crop${CROP}"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

cd /shared/openfold3_triangle_kernel

for BACKEND in aqlab triton cueq stablefold; do
  srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    pixi run -e openfold3-cuda12 benchmark-multinode \
    --runner-yaml "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml" \
    --backend "$BACKEND" \
    --output-root "$RESULTS_ROOT" \
    --warmup-steps 10 \
    --measure-steps 400 \
    --devices "$GPUS_PER_NODE" \
    --num-nodes "$NNODES" \
    --num-workers 10 \
    --require-full-crop \
    --use-tf32 \
    --seed 42
done

pixi run -e openfold3-cuda12 plot-backend-comparison \
  "$RESULTS_ROOT" \
  --baseline triton
```

The four backends run one after another inside the same allocation, on the
same nodes, and the plot is produced on the batch host once the loop finishes.
`--num-workers 10` lets each rank featurize the next sample while the GPU works
on the current one; without it the GPU idles during featurization and every
step waits for the slowest rank's data. Keep the value identical for every
backend and for any single-node run you compare against, and keep
`--cpus-per-task` at least one larger than it.
The shown request consumes 88 schedulable CPU cores per node (eight training
processes plus 80 workers). Adjust the request and site-specific CPU binding if
the node exposes fewer allocatable cores; do not silently lower workers for only
one backend.

The result layout matches the single-node comparison, with two additions:
`summary.json` records `"topology": "multi-node-ddp"` and `"num_nodes": 2`,
and the plot titles report the world size, here 16 GPUs. Compare backends only
within the same node and GPU count; a rerun into an existing `RESULTS_ROOT`
stops with `benchmark output already exists`, so rename or delete the directory
first.

To benchmark a single node under Slurm, submit the same script with
`--nodes=1`, `NNODES=1`, and `benchmark-single-node`.

##### Train on Two Nodes

Configure `datasets/train_pdb_subset.${BACKEND}.yaml` as in the previous
section (`epoch_len: 6400`, `devices: 8`, `num_nodes: 2`, `max_steps: 400`).
Save this as `/shared/openfold3-run/train_2x8.sbatch` and run
`sbatch /shared/openfold3-run/train_2x8.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=of3-stablefold-2x8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=11
#SBATCH --time=12:00:00
#SBATCH --output=/shared/openfold3-run/slurm-%j.out

set -euo pipefail

export BACKEND=stablefold
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_RUN_CWD=/shared/openfold3-run
unset PYTHONPATH PYTHONHOME LD_PRELOAD
# export NCCL_SOCKET_IFNAME=eth0   # only if NCCL picks the wrong interface

cd /shared/openfold3_triangle_kernel
export RUNNER_YAML="$PWD/datasets/train_pdb_subset.${BACKEND}.yaml"

srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  pixi run -e openfold3-cuda12 train-single-node \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42
```

`train-single-node` is the plain `run_openfold` launcher. Under `srun` it
starts once per task, and Lightning joins the sixteen tasks into one DDP group.
`--cpus-per-task=11` covers `num_workers: 10` plus the training process.

##### Smoke Test Before the Full Run

First check that the StableFold receipt round-trips across nodes with one GPU
per node. Create a one-step YAML:

```bash
cd /shared/openfold3_triangle_kernel
export BACKEND=stablefold

pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend "$BACKEND" \
  --single-gpu-smoke \
  --output "datasets/two_node_smoke.${BACKEND}.yaml"
```

Set `num_nodes: 2` and `epoch_len: 2` in that YAML. Copy `train_2x8.sbatch`,
change it to `--ntasks-per-node=1`, `--gres=gpu:1`, and
`RUNNER_YAML="$PWD/datasets/two_node_smoke.${BACKEND}.yaml"`, and submit it.

##### Verify

A run passes when the job exits zero and the receipt written by rank 0 reports
`PASS` with one hostname per rank:

```bash
grep '"status"' /shared/openfold3-run/<output_dir>/logs/stablefold_kernel_ddp.json
grep -c '"hostname"' /shared/openfold3-run/<output_dir>/logs/stablefold_kernel_ddp.json
```

The hostname count equals the world size: `16` for the full run and `2` for
the smoke test, whose `output_dir` is `openfold3_smoke_stablefold`. For a
benchmark, the receipt sits under `$RESULTS_ROOT/<backend>/logs/`.

##### Alternative: One `torchrun` per Node

To keep the `train-multinode` task, run one task per node and let Slurm supply
the node rank and master address:

```bash
#!/bin/bash
#SBATCH --job-name=of3-stablefold-2x8-torchrun
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=/shared/openfold3-run/slurm-%j.out

set -euo pipefail

export BACKEND=stablefold
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_RUN_CWD=/shared/openfold3-run
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29500
unset PYTHONPATH PYTHONHOME LD_PRELOAD

cd /shared/openfold3_triangle_kernel
export RUNNER_YAML="$PWD/datasets/train_pdb_subset.${BACKEND}.yaml"

srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
  pixi run -e openfold3-cuda12 train-multinode \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module openfold3.run_openfold \
    train \
    --runner-yaml "$RUNNER_YAML" \
    --seed 42
'
```

Keep the single quotes: `SLURM_NODEID` differs per task and must expand inside
the task shell. The benchmark tool accepts this launcher too; rank 0 is taken
from the `RANK` that `torchrun` exports.

##### Troubleshooting

| Symptom | Fix |
| --- | --- |
| Lightning: `devices` or `num_nodes` `does not match` | Align `--ntasks-per-node` and `--nodes` with the YAML or with `--devices` and `--num-nodes`. |
| `multiple DDP ranks are bound to the same physical CUDA device` | Remove `--gpus-per-task` and `--gpu-bind`. |
| Every task prints its own progress bar and no receipt appears | The runner YAML still has `devices: 1` and `num_nodes: 1`, so each task trains alone. Set both values. |
| `benchmark output already exists` | Rename or delete `$RESULTS_ROOT/<backend>` before rerunning. |
| `StableFold kernel receipt readback hash mismatch` | The run directory is not consistent across nodes. Mount NFS with `actimeo=0` or use Lustre or GPFS. |
| NCCL hangs at startup | Set `NCCL_SOCKET_IFNAME` to the interface that reaches the other node. |

## Fine-tuning

To initialize training from a checkpoint, keep a new output directory and set
the checkpoint path in the runner YAML:

```yaml
experiment_settings:
  mode: train
  output_dir: ./finetune_output
  seed: 42
  restart_checkpoint_path: /path/to/pretrained.ckpt
```

Then use the same single-GPU or distributed launch command.

## Checkpointing and Resuming

Checkpoints are saved under `{output_dir}/checkpoints/`. The checkpoint policy
can be configured in the runner YAML:

```yaml
checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: -1
```

To resume the same output directory from its most recent checkpoint, set:

```yaml
experiment_settings:
  restart_checkpoint_path: last
```

To start from scratch, set `restart_checkpoint_path: null`.
