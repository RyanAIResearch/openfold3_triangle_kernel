# OpenFold3 H100 + StableFold reproduction guide (no kernel build)

This workflow preserves AQLab OpenFold3 data, training entry points, Lightning DDP, EMA, gradients, losses, checkpoints, and loggers while changing only the selected operator backend.
All five StableFold `sm_90a` AOT extensions are shipped in a qualified wheel inside this repository, so users do not need a system CUDA Toolkit, user-invoked NVCC, CMake, or runtime JIT.
Complete steps 1–4, choose exactly one launch topology from steps 5–7, and use the same Git revision, Pixi lock, runner YAML, and shared paths on every node.

| Topology | YAML `devices` | YAML `num_nodes` | Launch method |
| --- | ---: | ---: | --- |
| One GPU | 1 | 1 | Invoke `run_openfold` once |
| Multiple GPUs on one host | Visible GPUs per host | 1 | Invoke `run_openfold` once; Lightning spawns local DDP |
| Multiple hosts | GPUs per host | Host count | Invoke standard `torchrun` once on each host |

## 1. Bootstrap every H100 host

Use Linux x86-64 with full NVIDIA H100 80 GB GPUs, a compatible NVIDIA driver, `git`, `curl`, `tmux`, and a filesystem mounted at `/mnt/shared` on every node; a system CUDA Toolkit and NVCC are not required.
Install Pixi on every host, then clone the `main` branch once on node 0 into the shared filesystem after configuring GitHub read access if the repository is private.
Confirm that every host reads the same clean checkout and that `git rev-parse HEAD` returns the same value everywhere.

```bash
# Every host.
sudo apt-get update
sudo apt-get install -y git curl ca-certificates tmux
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"

# Node 0 only; /mnt/shared must be mounted on every node.
export OF3_SRC=/mnt/shared/openfold3/src
mkdir -p /mnt/shared/openfold3
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git "$OF3_SRC"
cd "$OF3_SRC"
git rev-parse HEAD
test -z "$(git status --porcelain=v1)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

## 2. Install Pixi and the prebuilt StableFold wheel

The non-editable `openfold3-h100` environment pins Python 3.14, PyTorch 2.10, CUDA runtime 12.9, and Lightning 2.6.1.
`setup-stablefold-h100` validates and installs the wheel stored in the checkout without compiling StableFold source or downloading a kernel binary; dependencies may still probe the Pixi-provided compiler ABI and CUDA version.
Install once into the shared prefix on node 0, then run the read-only doctor and AOT verifier on every host; success reports artifact commit `df1957e623d1283ba2a572cb730a6b130fbf1b0c`, wheel SHA-256 `85fc7b48…377ac5`, and H100 capability `[9, 0]`.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

# Node 0 only when /mnt/shared/openfold3/src is the same shared filesystem on every host.
# Pixi does not synchronize node-local prefixes; for local storage, run both commands on every host.
pixi install -e openfold3-h100
pixi run -e openfold3-h100 setup-stablefold-h100

# Every host: check the environment.
pixi run -e openfold3-h100 doctor-h100
pixi run -e openfold3-h100 verify-aot
```

## 3. Run a one-step StableFold smoke test

Use AQLab's subset tools to run one BF16 training step that jointly checks data loading, CUDA, StableFold routing, backward, and the official `run_openfold train` entry point.
The commands write all generated data, configuration, and output outside the checkout and use an empty validation cache so the downloader verifies exactly the artifacts consumed by this validation-disabled smoke.
This canary deliberately disables validation and checkpoint output for speed, so use AQLab's full training test when checkpoint and validation coverage is required.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export SUBSET_ROOT="$OF3_WORK/subset"
mkdir -p "$SUBSET_ROOT" "$OF3_WORK/configs" "$OF3_WORK/outputs/smoke"
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-h100 generate-pdb-subset -- \
  --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --runner-yaml "$SUBSET_ROOT/train_pdb_subset.yaml"
pixi run -e openfold3-h100 python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SUBSET_ROOT"]) / "validation_cache_with_templates_subset_0.json"
path.write_text(json.dumps({
    "_type": "ValidationDatasetCache",
    "name": "Validation-disabled-0",
    "reference_molecule_data": {},
    "structure_data": {},
}))
print(path)
PY
pixi run -e openfold3-h100 download-pdb-subset -- \
  --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --val-size 0
pixi run -e openfold3-h100 download-pdb-subset -- \
  --verify --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --val-size 0

export SMOKE_BASE="$OF3_WORK/configs/smoke.base.yml"
export SMOKE_YAML="$OF3_WORK/configs/smoke.stablefold.yml"
pixi run -e openfold3-h100 python - <<'PY'
import os
from pathlib import Path
import yaml

src = Path(os.environ["SUBSET_ROOT"]) / "train_pdb_subset.yaml"
dst = Path(os.environ["SMOKE_BASE"])
cfg = yaml.safe_load(src.read_text())
cfg["experiment_settings"].update(
    output_dir=str(Path(os.environ["OF3_WORK"]) / "outputs" / "smoke"),
    restart_checkpoint_path=None,
)
cfg["data_module_args"].update(
    batch_size=1, num_workers=0, num_workers_validation=0, epoch_len=1
)
cfg["checkpoint_config"].update(save_top_k=0, save_last=False)
cfg["logging_config"]["wandb_config"] = None
cfg["pl_trainer_args"].update(
    devices=1,
    num_nodes=1,
    precision="bf16-mixed",
    max_epochs=1,
    max_steps=1,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    log_every_n_steps=1,
)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

pixi run -e openfold3-h100 select-training-backend \
  "$SMOKE_BASE" --backend stablefold --output "$SMOKE_YAML"
export OF3_RUN_CWD="$OF3_WORK"
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
pixi run -e openfold3-h100 train-single-gpu \
  train --runner-yaml "$SMOKE_YAML" --seed 42 \
  2>&1 | tee "$OF3_WORK/outputs/smoke/train.log"
```

## 4. Prepare a 4,096-example canary and select a backend

For a longer public-data canary, use AQLab's subset tools to sample a deterministic 4,096-entry candidate pool and rerun the downloader after any network interruption because existing files are skipped.
Set `BACKEND` to `aqlab` for stock PyTorch, `cueq` for cuEquivariance, `stablefold` for StableFold Attention plus TriMul, or `deepspeed-evo` for the DeepSpeed Evo attention operator rather than ZeRO; the selector changes only six mutually exclusive training operator flags and leaves optimizer, EMA, loss, checkpoint, and DDP behavior untouched.
The example below creates a crop-384, local-batch-one, BF16, 400-step PDB canary with validation disabled; production training must instead provision every dataset referenced by AQLab's official `initial_training.yml`.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export SUBSET_ROOT="$OF3_WORK/subset"
export BACKEND=${BACKEND:-stablefold}
mkdir -p "$OF3_WORK/configs" "$OF3_WORK/outputs" "$OF3_WORK/logs"
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-h100 generate-pdb-subset -- \
  --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --runner-yaml "$SUBSET_ROOT/train_pdb_subset_4096.yaml" \
  --train-size 4096 --seed 42
pixi run -e openfold3-h100 download-pdb-subset -- \
  --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --train-size 4096 --val-size 0 --workers 24 \
  2>&1 | tee "$OF3_WORK/logs/download_subset_4096.log"
pixi run -e openfold3-h100 download-pdb-subset -- \
  --verify --target-dir "$SUBSET_ROOT" \
  --output-dir "$SUBSET_ROOT/pdb_training_set" \
  --train-size 4096 --val-size 0 \
  2>&1 | tee "$OF3_WORK/logs/verify_subset_4096.log"

export TRAIN_BASE="$OF3_WORK/configs/train_4096_crop384.base.yml"
export TRAIN_BACKEND="$OF3_WORK/configs/train_4096_crop384.${BACKEND}.yml"
pixi run -e openfold3-h100 python - <<'PY'
import os
from pathlib import Path
import yaml

steps = 400
src = Path(os.environ["SUBSET_ROOT"]) / "train_pdb_subset_4096.yaml"
dst = Path(os.environ["TRAIN_BASE"])
cfg = yaml.safe_load(src.read_text())
cfg["experiment_settings"].update(restart_checkpoint_path=None)
cfg["data_module_args"].update(
    batch_size=1, num_workers=4, num_workers_validation=0, epoch_len=steps
)
cfg["checkpoint_config"].update(save_top_k=0, save_last=True)
cfg["logging_config"]["wandb_config"] = None
cfg["pl_trainer_args"].update(
    precision="bf16-mixed",
    max_epochs=1,
    max_steps=steps,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    log_every_n_steps=1,
)
train = cfg["dataset_configs"]["train"]["weighted-pdb"]
train["weight"] = 1.0
train["config"]["crop"]["token_crop"]["token_budget"] = 384
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

pixi run -e openfold3-h100 select-training-backend \
  "$TRAIN_BASE" --backend "$BACKEND" --output "$TRAIN_BACKEND"
```

## 5. Launch on one H100

Set `devices: 1` and `num_nodes: 1`, expose exactly one H100, and invoke the official `run_openfold` entry point once.
The small configuration block below writes a topology-specific runner and output directory without changing training code.
The Pixi task verifies the prebuilt wheel, moves outside the checkout, and then directly executes AQLab's launcher.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export INPUT_YAML="$OF3_WORK/configs/train_4096_crop384.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/configs/single_gpu.${BACKEND}.yml"
export OUTPUT_DIR="$OF3_WORK/outputs/single_gpu.${BACKEND}"
export OF3_RUN_CWD="$OF3_WORK"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUTPUT_DIR"
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-h100 python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["INPUT_YAML"]).read_text())
cfg["experiment_settings"]["output_dir"] = os.environ["OUTPUT_DIR"]
cfg["data_module_args"]["epoch_len"] = 400
cfg["pl_trainer_args"].update(devices=1, num_nodes=1)
Path(os.environ["RUNNER_YAML"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

set -o pipefail
pixi run -e openfold3-h100 train-single-gpu \
  train --runner-yaml "$RUNNER_YAML" --seed 42 \
  2>&1 | tee "$OUTPUT_DIR/train.log"
```

## 6. Launch multiple H100s on one host

For eight local GPUs, set `devices: 8` and `num_nodes: 1`, expose exactly eight H100s, and invoke `run_openfold` once without wrapping it in `torchrun`.
Lightning preserves AQLab's original DDP behavior while each StableFold kernel executes only on its rank-local GPU and current CUDA stream.
Keep local batch size one and make `epoch_len` divisible by world size; 3,200 examples below represent 400 eight-GPU optimizer steps.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export INPUT_YAML="$OF3_WORK/configs/train_4096_crop384.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/configs/one_node_8gpu.${BACKEND}.yml"
export OUTPUT_DIR="$OF3_WORK/outputs/one_node_8gpu.${BACKEND}"
export OF3_RUN_CWD="$OF3_WORK"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$OUTPUT_DIR"
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-h100 python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["INPUT_YAML"]).read_text())
cfg["experiment_settings"]["output_dir"] = os.environ["OUTPUT_DIR"]
cfg["data_module_args"]["epoch_len"] = 400 * 8
cfg["pl_trainer_args"].update(devices=8, num_nodes=1)
Path(os.environ["RUNNER_YAML"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

set -o pipefail
pixi run -e openfold3-h100 train-single-node \
  train --runner-yaml "$RUNNER_YAML" --seed 42 \
  2>&1 | tee "$OUTPUT_DIR/train.log"
```

## 7. Launch multiple H100 hosts

For two eight-GPU hosts, generate one shared runner on node 0 and run the same `torchrun` command concurrently on both hosts with `NODE_RANK=0` and `1`.
Set `MASTER_ADDR` to node 0's reachable address, mount data, configuration, output, and logs as the same shared read/write paths, and let Slurm or the site supervisor provide allocation and global failure handling.
StableFold adds only AOT, ABI, device, cross-rank identity, and strict-route checks; AQLab and standard PyTorch/Lightning continue to own sampling, DDP, gradient synchronization, and training state.

```bash
# Node 0 only: write the shared runner once.
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export INPUT_YAML="$OF3_WORK/configs/train_4096_crop384.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/configs/two_node_8gpu.${BACKEND}.yml"
export OUTPUT_DIR="$OF3_WORK/outputs/two_node_8gpu.${BACKEND}"
mkdir -p "$OUTPUT_DIR"
cd "$OF3_SRC"

pixi run -e openfold3-h100 python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["INPUT_YAML"]).read_text())
cfg["experiment_settings"]["output_dir"] = os.environ["OUTPUT_DIR"]
cfg["data_module_args"]["epoch_len"] = 400 * 8 * 2
cfg["pl_trainer_args"].update(devices=8, num_nodes=2)
Path(os.environ["RUNNER_YAML"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

# Every node: repeat all exports and change NODE_RANK only.
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export RUNNER_YAML="$OF3_WORK/configs/two_node_8gpu.${BACKEND}.yml"
export OUTPUT_DIR="$OF3_WORK/outputs/two_node_8gpu.${BACKEND}"
export OF3_RUN_CWD="$OF3_WORK"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export GPUS_PER_NODE=8
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export NODE_RANK=0
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

set -o pipefail
pixi run -e openfold3-h100 train-multinode \
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" --module openfold3.run_openfold \
  train --runner-yaml "$RUNNER_YAML" --seed 42 \
  2>&1 | tee "$OUTPUT_DIR/train.node${NODE_RANK}.log"
```

## 8. Verify the run and understand the current boundary

Run `verify-aot` before each launch and continue to use AQLab's existing console, W&B, checkpoints, and cluster monitoring for loss, OOM, and failure diagnosis.
Prebuilt installation, five-module forward/backward, routing, and the training entry point are qualified on one H100, but real 2+ GPU NCCL and two-node H100 soak gates remain pending, so this is a wired production candidate rather than multi-node GA.
The wheel supports full H100 `sm_90a` GPUs only and does not claim H200, B200, MIG, FSDP, ZeRO, `torch.compile`, or local batch sizes greater than one.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
cd "$OF3_SRC"
unset PYTHONPATH PYTHONHOME LD_PRELOAD
pixi run -e openfold3-h100 verify-aot

grep -RIE 'Traceback|CUDA error|out of memory|OOM|NCCL.*timeout' \
  /mnt/shared/openfold3/outputs --include='*.log' || true
find /mnt/shared/openfold3/outputs -path '*/checkpoints/*' -type f -print
```
