# Bare-metal training quickstart

This guide takes a new user from an empty H100 host to a real OpenFold3 training process.
Complete steps 1 through 4, choose exactly one launch topology from steps 5 through 7, and then use step 8 to verify the run.
The StableFold multi-GPU and multi-node path is an enforced production candidate rather than a GA claim.
Its fail-closed software contract and two-process CPU/Gloo tests are complete, while real 2-GPU NCCL and real 2-node H100 qualification remain pending.

| Launch choice | YAML `devices` | YAML `num_nodes` | Launcher | World size |
| --- | ---: | ---: | --- | ---: |
| One H100 | 1 | 1 | Run `run_openfold` once | 1 |
| One host with eight H100s | 8 | 1 | Run `run_openfold` once and let Lightning spawn | 8 |
| Two hosts with eight H100s each | 8 | 2 | Run `torchrun` on both hosts | 16 |

## 1. Verify the host and clone the repository

Use a Linux x86-64 host with a full NVIDIA H100 80 GB GPU, an NVIDIA driver compatible with CUDA 12.9, Git, curl, and enough disk space for the training set.
For multi-node training, mount the same writable shared filesystem at `/mnt/shared` on every node, allow all nodes to reach node 0 over the rendezvous port, and install Pixi on every node.
Run the first labeled section on every node, then authenticate with GitHub if required and run the second labeled section only on node 0 to clone `main`, record its commit, and create the shared working directories.

```bash
# Every node
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"

# Node 0 only
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
mkdir -p "$OF3_WORK"
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git "$OF3_SRC"
git -C "$OF3_SRC" rev-parse HEAD
test -z "$(git -C "$OF3_SRC" status --porcelain=v1)"
mkdir -p "$OF3_WORK"/{artifacts,run,runs}
```

## 2. Create the environment and prepare data

Install the committed `openfold3-cuda12` Pixi environment before touching the data so every command uses the repository's locked Python, PyTorch, CUDA, and preprocessing dependencies.
The shortest reproducible path downloads the published preprocessed PDB training set, while a custom corpus must instead follow the {doc}`data pipeline reference <data_pipeline_reference>` from raw structures, alignments, templates, and caches.
Run these commands on node 0 and do not continue until both dataset-cache checks succeed.

```bash
cd "$OF3_SRC"
pixi install --locked -e openfold3-cuda12
pixi run -e openfold3-cuda12 aws s3 sync \
  s3://openfold3-data/pdb_training_set/ \
  "$OF3_WORK/pdb_training_set/" --no-sign-request
cp "$OF3_SRC/examples/training_yamls/stablefold_distributed_pdb.yml" \
  "$OF3_WORK/run/base.yml"
test -f "$OF3_WORK/pdb_training_set/dataset_caches/training_cache_with_templates.json"
test -f "$OF3_WORK/pdb_training_set/dataset_caches/validation_cache_with_templates.json"
```

## 3. Build and install the five kernels

Build only from the clean committed checkout because the attestation binds the wheel, five `sm_90a` AOT extensions, CUDA sources, and Git commit into one artifact.
The builder produces Triangle Attention forward, delta, `dK/dV`, and `dQ/dBias` extensions plus the Triangle Multiplicative Update extension, so the StableFold backend never needs runtime NVCC or JIT compilation.
Run these commands once in the shared immutable Pixi prefix that every node will use.

```bash
cd "$OF3_SRC"
pixi run -e openfold3-cuda12 bash -lc \
  'export CUDA_HOME="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc"; tools/build_sm90_wheel.sh'
(cd dist && sha256sum --check openfold3-*.whl.sha256)
cp dist/openfold3-*.whl* dist/aot_clean_install_smoke.json "$OF3_WORK/artifacts/"
"$OF3_SRC/.pixi/envs/openfold3-cuda12/bin/python" -m pip \
  install --no-deps --force-reinstall "$OF3_WORK"/artifacts/openfold3-*.whl
test "$(find "$OF3_WORK/artifacts" -maxdepth 1 -name 'openfold3-*.whl' | wc -l)" -eq 1
```

## 4. Choose one optimization backend

Set `BACKEND` to `aqlab` for native AQLab/OpenFold3 stock PyTorch, `cueq` for cuEquivariance, `stablefold` for StableFold Attention plus TriMul, or `deepspeed-evo` for the DeepSpeed Evo attention operator.
The backend choice sets mutually exclusive operator flags, and `deepspeed-evo` does not enable Lightning DeepSpeedStrategy or ZeRO.
Run these commands in every shell that will launch a worker, using the same `BACKEND` value on every node and `--as-is` so Pixi does not replace the attested wheel with an editable install.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export OF3_ENV="$OF3_SRC/.pixi/envs/openfold3-cuda12"
export BACKEND=stablefold  # aqlab | cueq | stablefold | deepspeed-evo
eval "$(pixi shell-hook --manifest-path "$OF3_SRC/pixi.toml" \
  --environment openfold3-cuda12 --as-is)"
cd "$OF3_WORK/run"
"$OF3_ENV/bin/python" - <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import openfold3

module = Path(openfold3.__file__).resolve()
prefix = Path(sys.prefix).resolve()
if not module.is_relative_to(prefix):
    raise SystemExit(f"OpenFold3 was imported outside the wheel prefix: {module}")
distribution = importlib.metadata.distribution("openfold3")
for member in distribution.files or ():
    if Path(str(member)).name != "direct_url.json":
        continue
    direct_url = json.loads(Path(distribution.locate_file(member)).read_text())
    if direct_url.get("dir_info", {}).get("editable", False):
        raise SystemExit("Editable OpenFold3 installs are not allowed")
print("Verified wheel runtime:", module)
PY
```

## 5. Start a single-GPU training run

Start here when validating a new machine because this path exercises the real data loader, model, loss, optimizer, and selected kernels without distributed communication.
The selector writes a dedicated one-GPU YAML and output directory without modifying the checked-in base configuration.
Run these commands from the shared run directory and keep the three StableFold variables identical when comparing backends.

```bash
export RUN_DIR="$OF3_WORK/runs/single_gpu_${BACKEND}"
test ! -e "$RUN_DIR"
"$OF3_ENV/bin/python" "$OF3_SRC/tools/select_training_backend.py" \
  "$OF3_WORK/run/base.yml" --backend "$BACKEND" \
  --output "$OF3_WORK/run/single_gpu_${BACKEND}.yml" \
  --output-dir "$RUN_DIR" \
  --devices-per-node 1 --num-nodes 1 --epoch-len 128000 --fresh
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
"$OF3_ENV/bin/run_openfold" train \
  --runner-yaml "$OF3_WORK/run/single_gpu_${BACKEND}.yml" --seed 42
```

## 6. Start a single-node multi-GPU training run

Use this alternative only after step 5 succeeds, and set `GPUS_PER_NODE` to the number of full H100 GPUs visible on the host.
The selector requires `epoch_len` to be divisible by the world size and StableFold distributed v1 requires local batch size one, per-sample clipping, DDP, and one process per GPU.
Run `run_openfold` once with these commands and let Lightning create one local worker for each visible GPU.

```bash
export GPUS_PER_NODE=8
export RUN_DIR="$OF3_WORK/runs/one_node_${GPUS_PER_NODE}gpu_${BACKEND}"
test ! -e "$RUN_DIR"
"$OF3_ENV/bin/python" "$OF3_SRC/tools/select_training_backend.py" \
  "$OF3_WORK/run/base.yml" --backend "$BACKEND" \
  --output "$OF3_WORK/run/one_node_${GPUS_PER_NODE}gpu_${BACKEND}.yml" \
  --output-dir "$RUN_DIR" \
  --devices-per-node "$GPUS_PER_NODE" --num-nodes 1 \
  --epoch-len 128000 --fresh
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$OF3_ENV/bin/run_openfold" train \
  --runner-yaml "$OF3_WORK/run/one_node_${GPUS_PER_NODE}gpu_${BACKEND}.yml" \
  --seed 42
```

## 7. Start a multi-node multi-GPU training run

Use this alternative only when every node can read the same dataset, YAML, output directory, and exact wheel and when every node has the same number and model of full H100 GPUs.
Generate the YAML once on node 0, then set the same `NNODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` on every node while assigning each node a unique zero-based `NODE_RANK`.
Run the preparation block once on node 0 and run the launch block concurrently on every node, because setting `num_nodes` in YAML does not create remote workers.

```bash
# Node 0 only
export NNODES=2
export GPUS_PER_NODE=8
export RUN_DIR="$OF3_WORK/runs/${NNODES}node_${GPUS_PER_NODE}gpu_${BACKEND}"
test ! -e "$RUN_DIR"
"$OF3_ENV/bin/python" "$OF3_SRC/tools/select_training_backend.py" \
  "$OF3_WORK/run/base.yml" --backend "$BACKEND" \
  --output "$OF3_WORK/run/${NNODES}node_${GPUS_PER_NODE}gpu_${BACKEND}.yml" \
  --output-dir "$RUN_DIR" \
  --devices-per-node "$GPUS_PER_NODE" --num-nodes "$NNODES" \
  --epoch-len 128000 --fresh
sha256sum "$OF3_WORK/run/${NNODES}node_${GPUS_PER_NODE}gpu_${BACKEND}.yml"

# Every node, with a unique NODE_RANK
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=10.0.0.10  # replace with node 0's routable address
export MASTER_PORT=29500
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
export OPENFOLD3_SR_BACKWARD=1
export OPENFOLD3_SR_SHADOW=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$OF3_ENV/bin/torchrun" --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --no-python \
  "$OF3_ENV/bin/run_openfold" train \
  --runner-yaml "$OF3_WORK/run/${NNODES}node_${GPUS_PER_NODE}gpu_${BACKEND}.yml" \
  --seed 42
```

## 8. Verify that training is healthy

A run is healthy only after optimizer steps advance, the total, diffusion, and distogram losses remain finite, and the log contains no traceback, CUDA error, NCCL timeout, or out-of-memory marker.
StableFold distributed runs must also emit `distributed_contract.json` and its SHA-256 sidecar before the first training step, proving that all ranks agreed on the wheel, AOT binaries, runtime, configuration, checkpoint, and physical GPU bindings.
Use these commands from another shell, setting `EXPECT_DISTRIBUTED=1` after launching step 6 or 7, to inspect progress and verify every required distributed receipt.

```bash
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=stablefold
export RUN_DIR="$OF3_WORK/runs/single_gpu_${BACKEND}"  # set the run chosen above
export EXPECT_DISTRIBUTED=0  # use 1 only for StableFold step 6 or 7
test -d "$RUN_DIR"
find "$RUN_DIR" -type f \
  \( -name 'metrics.csv' -o -name 'console_logs.log' \
     -o -name 'console_logs.rank*.log' \
     -o -name 'distributed_contract.json' \) -print
find "$RUN_DIR" -name 'console_logs*.log' -exec tail -n 20 {} \;
if grep -ERinI 'Traceback|CUDA error|NCCL.*(timeout|error)|out of memory' "$RUN_DIR"; then
  echo "Fatal marker found in $RUN_DIR" >&2
  exit 1
fi
if [[ "$BACKEND" == stablefold && "$EXPECT_DISTRIBUTED" == 1 ]]; then
  test -f "$RUN_DIR/logs/distributed_contract.json"
  test -f "$RUN_DIR/logs/distributed_contract.json.sha256"
  (cd "$RUN_DIR/logs" && sha256sum --check distributed_contract.json.sha256)
fi
```
