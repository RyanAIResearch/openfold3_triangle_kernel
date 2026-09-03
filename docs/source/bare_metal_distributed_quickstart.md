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

For the standard 400-step comparison on up to eight GPUs, generate a
reproducible subset with 4,000 training structures. Eight GPUs consume 3,200
sample draws over 400 optimizer steps at per-GPU batch size one, so 4,000
provides coverage plus room for samples removed during purification:

```bash
pixi run -e openfold3-cuda12 prepare-pdb-subset -- \
  --train-size 4000 \
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

The one-step single-GPU smoke test does not consume all 4,000 structures; the
larger subset is recommended so the same prepared data can also be reused for
the full eight-GPU comparison. A smaller subset remains valid for installation
and functional smoke testing.

The original sampled caches are never modified. For debugging, the individual
`generate-pdb-subset`, `download-pdb-subset`, and `purify-pdb-subset` tasks
remain available, but the all-in-one command above is the supported quick-start
path.

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
| `cueq` | NVIDIA cuEquivariance triangle kernels |
| `stablefold` | StableFold triangle attention and triangle multiplicative kernels |
| `deepspeed-evo` | DeepSpeed Evo attention operator; this does not enable ZeRO |

The selector first disables every mutually exclusive route and then enables the
requested backend, so rerunning it cannot leave mixed backend flags. Diffusion
stays on OpenFold3's original path. OpenFold3 continues to own the model, data
pipeline, loss, optimizer, EMA, checkpointing, and distributed training.

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

Run every backend from scratch with the same prepared data, initialization
seed, data seed, BF16 precision, and GPU topology. The benchmark disables
validation, W&B, and checkpoint output. The following single-H100 smoke
comparison excludes five warmup steps, then measures 20 optimizer training
steps:

```bash
mkdir -p ../openfold3-run/backend-comparison
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

for BACKEND in aqlab cueq stablefold; do
  pixi run -e openfold3-cuda12 benchmark-single-gpu \
    --runner-yaml "$PWD/datasets/train_pdb_subset.yaml" \
    --backend "$BACKEND" \
    --output-root "$RESULTS_ROOT" \
    --warmup-steps 5 \
    --measure-steps 20 \
    --devices 1 \
    --seed 42
done

pixi run -e openfold3-cuda12 plot-backend-comparison \
  "$RESULTS_ROOT" \
  --baseline aqlab
```

Add `deepspeed-evo` to the loop to compare that operator route as well. For a
paper-quality 400-step comparison, use the same command with
`--warmup-steps 10 --measure-steps 400` for every backend. Run the backends on
the same otherwise-idle GPU; do not compare results collected on different GPU
models or while another process is using the device.

For a single-node multi-GPU comparison, change `DEVICES` and
`CUDA_VISIBLE_DEVICES` together. This example benchmarks all backends on eight
GPUs and keeps its results separate from the one-GPU run:

```bash
export DEVICES=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-${DEVICES}gpu"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

for BACKEND in aqlab cueq stablefold; do
  pixi run -e openfold3-cuda12 benchmark-single-node \
    --runner-yaml "$PWD/datasets/train_pdb_subset.yaml" \
    --backend "$BACKEND" \
    --output-root "$RESULTS_ROOT" \
    --warmup-steps 10 \
    --measure-steps 400 \
    --devices "$DEVICES" \
    --seed 42
done

pixi run -e openfold3-cuda12 plot-backend-comparison \
  "$RESULTS_ROOT" \
  --baseline aqlab
```

Use the same command for two or four GPUs by setting `DEVICES=2` with
`CUDA_VISIBLE_DEVICES=0,1`, or `DEVICES=4` with
`CUDA_VISIBLE_DEVICES=0,1,2,3`. Do not wrap `benchmark-single-node` in
`torchrun`: PyTorch Lightning launches the local DDP workers. The benchmark
sets per-GPU batch size to one, global batch size to `DEVICES`, and
`epoch_len = (warmup_steps + measure_steps) * DEVICES`; `max_steps` remains the
same optimizer-step count for 1, 2, 4, and 8 GPUs.

Compare backends only within the same GPU count. Changing the GPU count changes
the global batch and therefore can change the loss trajectory even when the
seed is held fixed. Use a separate `RESULTS_ROOT` and generate a separate
comparison for each of 1, 2, 4, and 8 GPUs.

Each backend directory contains:

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
mean, median, p95, minimum, and maximum critical-path GPU step time. Warmup is
excluded from every steady-state timing statistic. CUDA events are recorded
asynchronously; device synchronization occurs only at the measurement
boundaries, and rank aggregation occurs after the measured interval, so the
benchmark does not add a synchronization to every measured step.

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
global mean loss. The CSV and JSON files are the source of truth for tables and
paper plots.

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
  epoch_len: 400

pl_trainer_args:
  devices: 1
  num_nodes: 1
  max_steps: 400
```

Launch training on one H100:

```bash
export BACKEND=stablefold
export CUDA_VISIBLE_DEVICES=0

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
  epoch_len: 3200

pl_trainer_args:
  devices: 8
  num_nodes: 1
  max_steps: 400
```

Then launch OpenFold3 once:

```bash
export BACKEND=stablefold
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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
commands concurrently.

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
