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
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 doctor-h100
pixi run -e openfold3-cuda12 verify-aot
```

## 1 GPU

```bash
set -euo pipefail

export SOURCE_RUNNER=/absolute/path/to/config.yaml
export BACKEND=stablefold
export RUN_ID="${BACKEND}-1gpu-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
test -r "$SOURCE_RUNNER"

mkdir -p ../openfold3-run
export RUN_ROOT="$(cd ../openfold3-run && pwd)"
export OF3_RUN_CWD="$RUN_ROOT/$RUN_ID"
test ! -e "$OF3_RUN_CWD"
mkdir -p "$OF3_RUN_CWD"
export RUNNER_YAML="$OF3_RUN_CWD/config.yaml"

sed -E \
  -e 's/^  devices: [0-9]+$/  devices: 1/' \
  -e 's/^  num_nodes: [0-9]+$/  num_nodes: 1/' \
  -e "s/^    id:.*$/    id: $RUN_ID/" \
  "$SOURCE_RUNNER" > "$RUNNER_YAML"

grep -qx '  devices: 1' "$RUNNER_YAML"
grep -qx '  num_nodes: 1' "$RUNNER_YAML"
grep -qx "    id: $RUN_ID" "$RUNNER_YAML"

pixi run -e openfold3-cuda12 select-training-backend \
  "$RUNNER_YAML" \
  --backend "$BACKEND"

export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-gpu \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42 \
  --data-seed 42
```

## 2 GPUs

```bash
set -euo pipefail

export SOURCE_RUNNER=/absolute/path/to/config.yaml
export BACKEND=stablefold
export RUN_ID="${BACKEND}-2gpu-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
test -r "$SOURCE_RUNNER"

mkdir -p ../openfold3-run
export RUN_ROOT="$(cd ../openfold3-run && pwd)"
export OF3_RUN_CWD="$RUN_ROOT/$RUN_ID"
test ! -e "$OF3_RUN_CWD"
mkdir -p "$OF3_RUN_CWD"
export RUNNER_YAML="$OF3_RUN_CWD/config.yaml"

sed -E \
  -e 's/^  devices: [0-9]+$/  devices: 2/' \
  -e 's/^  num_nodes: [0-9]+$/  num_nodes: 1/' \
  -e "s/^    id:.*$/    id: $RUN_ID/" \
  "$SOURCE_RUNNER" > "$RUNNER_YAML"

grep -qx '  devices: 2' "$RUNNER_YAML"
grep -qx '  num_nodes: 1' "$RUNNER_YAML"
grep -qx "    id: $RUN_ID" "$RUNNER_YAML"

pixi run -e openfold3-cuda12 select-training-backend \
  "$RUNNER_YAML" \
  --backend "$BACKEND"

export CUDA_VISIBLE_DEVICES=0,1
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-node \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42 \
  --data-seed 42
```

## 8 GPUs

```bash
set -euo pipefail

export SOURCE_RUNNER=/absolute/path/to/config.yaml
export BACKEND=stablefold
export RUN_ID="${BACKEND}-8gpu-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}"
test -r "$SOURCE_RUNNER"

mkdir -p ../openfold3-run
export RUN_ROOT="$(cd ../openfold3-run && pwd)"
export OF3_RUN_CWD="$RUN_ROOT/$RUN_ID"
test ! -e "$OF3_RUN_CWD"
mkdir -p "$OF3_RUN_CWD"
export RUNNER_YAML="$OF3_RUN_CWD/config.yaml"

sed -E \
  -e 's/^  devices: [0-9]+$/  devices: 8/' \
  -e 's/^  num_nodes: [0-9]+$/  num_nodes: 1/' \
  -e "s/^    id:.*$/    id: $RUN_ID/" \
  "$SOURCE_RUNNER" > "$RUNNER_YAML"

grep -qx '  devices: 8' "$RUNNER_YAML"
grep -qx '  num_nodes: 1' "$RUNNER_YAML"
grep -qx "    id: $RUN_ID" "$RUNNER_YAML"

pixi run -e openfold3-cuda12 select-training-backend \
  "$RUNNER_YAML" \
  --backend "$BACKEND"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH PYTHONHOME LD_PRELOAD

pixi run -e openfold3-cuda12 train-single-node \
  train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42 \
  --data-seed 42
```

## Multi-node Setup

```bash
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
export SOURCE_ROOT=/shared/openfold3_triangle_kernel

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  git clone --branch main --single-branch \
    https://github.com/RyanAIResearch/openfold3_triangle_kernel.git \
    "$SOURCE_ROOT"
fi

cd "$SOURCE_ROOT"
git pull --ff-only origin main
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 verify-aot
mkdir -p /shared/openfold3-run
```

## Multi-node (`submit.sbatch`)

```bash
#!/usr/bin/env bash
#SBATCH --job-name=stablefold-train
#SBATCH --partition=h100full
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=11
#SBATCH --output=/shared/openfold3-run/slurm-%j.log
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append

set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
export SOURCE_ROOT=/shared/openfold3_triangle_kernel
export SOURCE_RUNNER=/absolute/path/to/config.yaml
export BACKEND=stablefold
export RUN_ID="${BACKEND}-2x8-${SLURM_JOB_ID:?}"
export OF3_RUN_CWD="/shared/openfold3-run/$RUN_ID"
test -r "$SOURCE_RUNNER"

unset PYTHONPATH PYTHONHOME LD_PRELOAD
cd "$SOURCE_ROOT"
eval "$(pixi shell-hook --manifest-path "$SOURCE_ROOT/pixi.toml" \
  -e openfold3-cuda12)"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

RUN_OPENFOLD="$(command -v run_openfold)"
case "$RUN_OPENFOLD" in
  "$CONDA_PREFIX/bin/run_openfold") ;;
  *) echo "run_openfold is outside the Pixi environment" >&2; exit 2 ;;
esac

mkdir -p "$OF3_RUN_CWD"
export RUNNER_YAML="$OF3_RUN_CWD/config.yaml"

sed -E \
  -e 's/^  devices: [0-9]+$/  devices: 8/' \
  -e 's/^  num_nodes: [0-9]+$/  num_nodes: 2/' \
  -e "s/^    id:.*$/    id: $RUN_ID/" \
  "$SOURCE_RUNNER" > "$RUNNER_YAML"

grep -qx '  devices: 8' "$RUNNER_YAML"
grep -qx '  num_nodes: 2' "$RUNNER_YAML"
grep -qx "    id: $RUN_ID" "$RUNNER_YAML"

python -m tools.select_training_backend \
  "$RUNNER_YAML" \
  --backend "$BACKEND"
python tools/validate_training_data.py \
  "$RUNNER_YAML" \
  --base-dir "$OF3_RUN_CWD"
python -m tools.manage_attested_sm90_wheel verify-prebuilt

export TRITON_CACHE_DIR="/tmp/$USER/triton"
export TORCH_EXTENSIONS_DIR="/tmp/$USER/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

cd "$OF3_RUN_CWD"
srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$RUN_OPENFOLD" train \
  --runner-yaml "$RUNNER_YAML" \
  --seed 42 \
  --data-seed 42
```

```bash
sbatch submit.sbatch
```
