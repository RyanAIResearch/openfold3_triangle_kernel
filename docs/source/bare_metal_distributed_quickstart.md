# OpenFold3 + StableFold Training Commands

## Install

```bash
set -euo pipefail

curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"

git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git
cd openfold3_triangle_kernel

git pull --ff-only origin main
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain)"
git rev-parse HEAD

pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 doctor-h100
pixi run -e openfold3-cuda12 verify-aot
```

## Prepare Data

```bash
set -euo pipefail

mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain)"
git rev-parse HEAD > "$OF3_RUN_CWD/source_commit.txt"

pixi run -e openfold3-cuda12 prepare-pdb-subset -- \
  --train-size 512 \
  --seed 42 \
  --workers 24

pixi run -e openfold3-cuda12 validate-training-data \
  datasets/train_pdb_subset.yaml

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

## One-GPU Smoke Test

```bash
set -euo pipefail

export BACKEND=stablefold

pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend "$BACKEND" \
  --single-gpu-smoke \
  --output "datasets/single_gpu_smoke.${BACKEND}.yaml"

mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-gpu \
  train \
  --runner-yaml "$PWD/datasets/single_gpu_smoke.${BACKEND}.yaml" \
  --seed 42
```

## Compare AQLab PyTorch, AQLab Triton, cuEq, and StableFold

```bash
set -euo pipefail

# Set DEVICES to 1, 2, or 8.
export DEVICES=1
export WARMUP_STEPS=5
export MEASURE_STEPS=20

case "$DEVICES" in
  1)
    export CUDA_VISIBLE_DEVICES=0
    export BENCHMARK_TASK=benchmark-single-gpu
    ;;
  2)
    export CUDA_VISIBLE_DEVICES=0,1
    export BENCHMARK_TASK=benchmark-single-node
    ;;
  8)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export BENCHMARK_TASK=benchmark-single-node
    ;;
  *)
    echo "DEVICES must be 1, 2, or 8" >&2
    exit 2
    ;;
esac

mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(<"$OF3_RUN_CWD/source_commit.txt")"
git rev-parse HEAD

export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
export RESULTS_BASE="$OF3_RUN_CWD/backend-comparison-${DEVICES}gpu"
export MATRIX_FAILED=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

BACKENDS=(aqlab triton cueq stablefold)
CROPS=(384 640 768)

for CROP in "${CROPS[@]}"; do
  export RESULTS_ROOT="$RESULTS_BASE/crop${CROP}"
  if [[ -e "$RESULTS_ROOT" ]]; then
    echo "refusing non-fresh result directory: $RESULTS_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT/logs"
  FAILED_BACKENDS=()

  for BACKEND in "${BACKENDS[@]}"; do
    if ! pixi run -e openfold3-cuda12 "$BENCHMARK_TASK" \
        --runner-yaml "$CROP_DATA_ROOT/runner_crop_${CROP}.yaml" \
        --backend "$BACKEND" \
        --output-root "$RESULTS_ROOT" \
        --warmup-steps "$WARMUP_STEPS" \
        --measure-steps "$MEASURE_STEPS" \
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
    echo "crop $CROP is incomplete: ${FAILED_BACKENDS[*]}" >&2
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

## Configure a Training Runner

```bash
set -euo pipefail

export SOURCE_RUNNER="$PWD/datasets/train_pdb_subset.yaml"
export BACKEND=stablefold
export MAX_STEPS=400
mkdir -p ../openfold3-run/training
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export RUN_GROUP="$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"

for TOPOLOGY in 1x1 1x2 1x8 2x8; do
  IFS=x read -r NUM_NODES DEVICES <<< "$TOPOLOGY"
  export NUM_NODES DEVICES
  export RUNNER_YAML="$PWD/datasets/train.${BACKEND}.${TOPOLOGY}.yaml"
  export OUTPUT_DIR="$OF3_RUN_CWD/training/${BACKEND}-${TOPOLOGY}-${RUN_GROUP}"
  test ! -e "$OUTPUT_DIR"

  pixi run -e openfold3-cuda12 select-training-backend \
    "$SOURCE_RUNNER" \
    --backend "$BACKEND" \
    --output "$RUNNER_YAML"

  pixi run -e openfold3-cuda12 python - <<'PY'
import os
from pathlib import Path

import yaml

path = Path(os.environ["RUNNER_YAML"])
devices = int(os.environ["DEVICES"])
num_nodes = int(os.environ["NUM_NODES"])
max_steps = int(os.environ["MAX_STEPS"])
config = yaml.safe_load(path.read_text())

config.setdefault("experiment_settings", {}).update(
    output_dir=os.environ["OUTPUT_DIR"],
    restart_checkpoint_path=None,
)
config.setdefault("data_module_args", {}).update(
    batch_size=1,
    num_workers=10,
    epoch_len=max_steps * devices * num_nodes,
)
config.setdefault("pl_trainer_args", {}).update(
    devices=devices,
    num_nodes=num_nodes,
    precision="bf16-mixed",
    max_epochs=-1,
    max_steps=max_steps,
)

path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

  pixi run -e openfold3-cuda12 validate-training-data "$RUNNER_YAML"
done
```

## Train on One Node

```bash
set -euo pipefail

export BACKEND=stablefold
export NUM_NODES=1
export DEVICES=1
export RUNNER_YAML="$PWD/datasets/train.${BACKEND}.${NUM_NODES}x${DEVICES}.yaml"

case "$DEVICES" in
  1)
    export CUDA_VISIBLE_DEVICES=0
    export TRAIN_TASK=train-single-gpu
    ;;
  2)
    export CUDA_VISIBLE_DEVICES=0,1
    export TRAIN_TASK=train-single-node
    ;;
  8)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export TRAIN_TASK=train-single-node
    ;;
  *)
    echo "DEVICES must be 1, 2, or 8" >&2
    exit 2
    ;;
esac

mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 "$TRAIN_TASK" \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42
```

## Train on Two Nodes with Eight GPUs per Node

```bash
# Run once before launching either node.
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
cd /shared/openfold3_triangle_kernel
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain)"
git rev-parse HEAD
pixi install -e openfold3-cuda12

mkdir -p /shared/openfold3-run
test -r /shared/openfold3-run
test -w /shared/openfold3-run
test -r "$PWD/datasets/train.stablefold.2x8.yaml"
```

```bash
# Run on both nodes.
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
cd /shared/openfold3_triangle_kernel

export OF3_RUN_CWD=/shared/openfold3-run
export RUNNER_YAML="$PWD/datasets/train.stablefold.2x8.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES=2
export GPUS_PER_NODE=8
: "${NODE_RANK:?export NODE_RANK=0 on node 0 or NODE_RANK=1 on node 1}"
: "${MASTER_ADDR:?export MASTER_ADDR to the node-0 reachable address}"
export NODE_RANK MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
case "$NODE_RANK" in
  0|1) ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac
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
