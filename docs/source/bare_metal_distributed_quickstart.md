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
pixi install --frozen -e openfold3-cuda12
pixi run --frozen -e openfold3-cuda12 setup-stablefold-h100
```

Use `--frozen` on every production Pixi command in this guide. Pixi otherwise
may rewrite the committed `pixi.lock` before starting a task, while the
StableFold artifact verifier intentionally requires a clean checkout.

The setup command installs the prebuilt StableFold kernels. To check the GPU,
environment, and all five AOT kernel modules, run:

```bash
pixi run --frozen -e openfold3-cuda12 doctor-h100
pixi run --frozen -e openfold3-cuda12 verify-aot
```

If setup reports `StableFold AOT artifacts require a clean Git checkout`, first
inspect the checkout:

```bash
git status --short
```

When the only output is `M pixi.lock` immediately after a plain `pixi install`,
the change is a generated lockfile rewrite. Recover without reinstalling the
repository:

```bash
git restore --source=HEAD -- pixi.lock
pixi install --frozen -e openfold3-cuda12
pixi run --frozen -e openfold3-cuda12 setup-stablefold-h100
```

Do not restore `pixi.lock` when you intentionally edited dependencies, and do
not discard any other paths reported by `git status`.

## Prepare Training Data

### Generate a PDB Subset

For a quick training experiment, generate a reproducible subset with 1,000
training structures:

```bash
pixi run --frozen -e openfold3-cuda12 generate-pdb-subset -- \
  --train-size 1000 --seed 42
```

### Download the Subset

```bash
pixi run --frozen -e openfold3-cuda12 download-pdb-subset -- \
  --train-size 1000 --workers 24
```

The command downloads target structures, alignment arrays, templates, and
reference molecules. It also generates:

```text
datasets/train_pdb_subset.yaml
```

Downloads are resumable. If the command is interrupted, run it again; existing
files are skipped.

### Purify Samples Affected by 404s

Rerun the download once before purification to rule out a temporary network
failure. If an object still reports `404`, `NoSuchKey`, or remains missing, run:

```bash
pixi run --frozen -e openfold3-cuda12 purify-pdb-subset -- \
  --train-size 1000
```

The purifier checks every sample's target structure, alignments, templates, and
reference molecules. Samples that depend on a missing artifact or unreadable
target structure are removed from a new cache; the original cache is unchanged.
It writes:

```text
datasets/train_pdb_subset.purified.yaml
datasets/pdb_subset_purification_report.json
```

The JSON report records every removed PDB ID and the files that were missing.
Run the purifier even when no 404 is reported so the following commands always
use the same runner YAML path.

### Choose a Training Backend

Use StableFold by default:

```bash
pixi run --frozen -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.purified.yaml \
  --backend stablefold \
  --output datasets/train_pdb_subset.purified.stablefold.yaml
```

The command can be pasted directly into Bash.

To select another implementation, change `BACKEND` and use the generic form:

```bash
export BACKEND=cueq

pixi run --frozen -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.purified.yaml \
  --backend "$BACKEND" \
  --output "datasets/train_pdb_subset.purified.${BACKEND}.yaml"
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

pixi run --frozen -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.purified.yaml \
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

pixi run --frozen -e openfold3-cuda12 train-single-gpu \
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

### Scheme 1: Single-GPU Training

The generated YAML defaults to epoch-based training. For a pure 400-training-step
run with no validation, update the existing entries in
`datasets/train_pdb_subset.purified.${BACKEND}.yaml`:

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

pixi run --frozen -e openfold3-cuda12 train-single-gpu \
  train \
  --runner-yaml "$PWD/datasets/train_pdb_subset.purified.${BACKEND}.yaml" \
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

pixi run --frozen -e openfold3-cuda12 train-single-node \
  train \
  --runner-yaml "$PWD/datasets/train_pdb_subset.purified.${BACKEND}.yaml" \
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

The repository, Pixi environment, purified runner YAML, dataset, and output
directory must use the same paths on every node. Run the following command on
both allocated nodes, changing `NODE_RANK` to `0` or `1`:

```bash
export BACKEND=stablefold
export OF3_RUN_CWD=/shared/openfold3-run
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run --frozen -e openfold3-cuda12 train-multinode \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --module openfold3.run_openfold \
  train \
  --runner-yaml "/shared/train_pdb_subset.purified.${BACKEND}.yaml" \
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
