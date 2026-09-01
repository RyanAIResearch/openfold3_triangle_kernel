# Pixi-first bare-metal training quickstart

This guide adds a locked H100 environment, attested StableFold AOT kernels, and a backend-flag selector around the unchanged AQLab OpenFold3 training path.
It does not replace AQLab's runner YAML, Lightning strategy, optimizer, EMA, gradient handling, checkpointing, logging, or loss semantics.
Complete steps 1 through 4, choose one launch topology from steps 5 through 7, and use step 8 for the kernel preflight while following the upstream training guide for run health.

| Launch choice | YAML `devices` | YAML `num_nodes` | Pixi task | Launcher |
| --- | ---: | ---: | --- | --- |
| One H100 | 1 | 1 | `train-single-gpu` | AQLab `run_openfold` |
| One host with multiple H100s | GPUs on the host | 1 | `train-single-node` | AQLab `run_openfold` and Lightning |
| Multiple H100 hosts | GPUs per host | hosts | `train-multinode` | Standard `torchrun` |

## 1. Bootstrap every host

Use Linux x86-64 with full NVIDIA H100 80 GB GPUs, a CUDA 12.9-compatible driver, Git, curl, and a filesystem mounted at `/mnt/shared` on every node.
Clone the maintained branch once into that shared filesystem, and create only the work and runner directories because the AOT publisher creates its artifact directory atomically.
Keep the checkout committed and clean because the attested builder rejects modified or untracked source files.

```bash
# Every host.
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"

# Node 0 only when the checkout and work root are shared.
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
mkdir -p "$OF3_WORK/run" "$OF3_WORK/runs"
git clone --branch main --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git "$OF3_SRC"
git -C "$OF3_SRC" rev-parse HEAD
test -z "$(git -C "$OF3_SRC" status --porcelain=v1)"
```

## 2. Create the environment and prepare data

Install the locked `openfold3-h100` environment once in the shared checkout, which contains no editable OpenFold3 package and pins the ABI used by the StableFold wheel.
The data task downloads and verifies only the published PDB portion; reproduce the complete upstream recipe by provisioning every dataset named in AQLab's `initial_training.yml` and the upstream data documentation.
Copy the official YAML outside the checkout, then explicitly set its dataset paths, output directory, restart choice, W&B values, and later its topology without changing any training semantics in code.

```bash
# Node 0 only for the shared Pixi prefix.
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
cd "$OF3_SRC"
pixi install --locked -e openfold3-h100

# Node 0 only.
pixi run --locked -e openfold3-h100 prepare-training-data \
  --work-root "$OF3_WORK"
cp examples/training_yamls/initial_training.yml \
  "$OF3_WORK/run/initial_training.yml"
${EDITOR:-vi} "$OF3_WORK/run/initial_training.yml"
```

## 3. Build, install, and verify the five AOT kernels

Run the H100 doctor and canonical builder once from the clean checkout to produce and install the five `sm_90a` extensions.
The build binds the source commit, wheel, ABI, embedded manifest, binary hashes, and compiler-free smoke result into one artifact bundle at `/mnt/shared/openfold3/artifacts`.
Run the non-mutating verification from every host before training so each host proves that it can execute the shared immutable prefix.

```bash
# Node 0 only when the Pixi prefix is shared.
export OF3_SRC=/mnt/shared/openfold3/src
cd "$OF3_SRC"
pixi run --locked -e openfold3-h100 doctor-h100
pixi run --locked -e openfold3-h100 build-install-aot

# Every host using that shared prefix.
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 verify-aot

```

## 4. Select one training-kernel backend

Choose `aqlab` for stock PyTorch, `cueq` for cuEquivariance, `stablefold` for StableFold Triangle Attention plus TriMul, or `deepspeed-evo` for the DeepSpeed Evo attention operator.
The selector writes only the mutually exclusive `model_update.custom.settings.memory.train` route flags and preserves topology, data, restart, checkpoint, logging, and every other YAML value.
Write a separate output YAML so that the official upstream recipe remains an auditable input to every topology-specific copy.

```bash
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export BACKEND_YAML="$OF3_WORK/run/initial_training.${BACKEND}.yml"
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 select-training-backend \
  "$OF3_WORK/run/initial_training.yml" \
  --backend "$BACKEND" --output "$BACKEND_YAML"
```

## 5. Start one H100

Copy the backend YAML, explicitly set `pl_trainer_args.devices: 1`, `pl_trainer_args.num_nodes: 1`, and a unique `experiment_settings.output_dir`, and make any restart choice exactly as documented upstream.
The Pixi task changes only to the external run directory and then executes AQLab's `run_openfold` with the arguments shown below.
EMA, gradients, losses, checkpoints, loggers, seed handling, and the Lightning strategy therefore remain the pinned AQLab implementation.

```bash
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export BACKEND_YAML="$OF3_WORK/run/initial_training.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/run/single_gpu.${BACKEND}.yml"
export OF3_RUN_CWD="$OF3_WORK/run"
cp "$BACKEND_YAML" "$RUNNER_YAML"
${EDITOR:-vi} "$RUNNER_YAML"
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 train-single-gpu \
  train --runner-yaml "$RUNNER_YAML" --seed 42
```

## 6. Start multiple H100s on one host

Copy the backend YAML, explicitly set `devices` to the visible GPU count, set `num_nodes: 1`, and choose a new upstream output directory.
Invoke `run_openfold` once and let the pinned AQLab Lightning configuration create its local DDP workers.
Do not wrap this case in `torchrun`, and retain all upstream accumulation, gradient, EMA, checkpoint, and logging behavior.

```bash
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export GPUS_PER_NODE=8
export BACKEND_YAML="$OF3_WORK/run/initial_training.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/run/one_node_${GPUS_PER_NODE}gpu.${BACKEND}.yml"
export OF3_RUN_CWD="$OF3_WORK/run"
cp "$BACKEND_YAML" "$RUNNER_YAML"
${EDITOR:-vi} "$RUNNER_YAML"
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 train-single-node \
  train --runner-yaml "$RUNNER_YAML" --seed 42
```

## 7. Start multiple H100s on multiple hosts

Copy the backend YAML once, explicitly set `devices` to GPUs per host, set `num_nodes` to the host count, and use paths visible at the same locations on every host.
Run the standard `torchrun` command concurrently on every host with a unique zero-based `NODE_RANK`; the Pixi task adds no manifest, claims, rendezvous policy, retry policy, process capture, or status files.
Use Slurm or the site's normal supervisor for allocation and global failure handling while AQLab continues to own the training implementation.

```bash
# Prepare the shared YAML once, then edit the three explicit upstream fields.
export OF3_WORK=/mnt/shared/openfold3
export BACKEND=${BACKEND:-stablefold}
export NNODES=2
export GPUS_PER_NODE=8
export BACKEND_YAML="$OF3_WORK/run/initial_training.${BACKEND}.yml"
export RUNNER_YAML="$OF3_WORK/run/${NNODES}node_${GPUS_PER_NODE}gpu.${BACKEND}.yml"
cp "$BACKEND_YAML" "$RUNNER_YAML"
${EDITOR:-vi} "$RUNNER_YAML"

# Run on every host with that host's NODE_RANK.
export PATH="$HOME/.pixi/bin:$PATH"
export OF3_SRC=/mnt/shared/openfold3/src
export OF3_WORK=/mnt/shared/openfold3
export NNODES=2
export GPUS_PER_NODE=8
export BACKEND=${BACKEND:-stablefold}
export RUNNER_YAML="$OF3_WORK/run/${NNODES}node_${GPUS_PER_NODE}gpu.${BACKEND}.yml"
export OF3_RUN_CWD="$OF3_WORK/run"
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export NODE_RANK=0
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 train-multinode \
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" --module openfold3.run_openfold \
  train --runner-yaml "$RUNNER_YAML" --seed 42
```

## 8. Verify the kernel preflight and use upstream run health

Run `verify-aot` immediately before a launch to prove that the external working directory imports the installed, attested wheel and that all five AOT identities still match.
Use the unchanged AQLab W&B logger, console output, Lightning checkpoints, and cluster monitoring to judge training progress and failures.
There is deliberately no StableFold CSV loss parser, workflow-status receipt, checkpoint policy, or replacement health model in this Pixi layer.

```bash
export OF3_SRC=/mnt/shared/openfold3/src
cd "$OF3_SRC"
pixi run --as-is -e openfold3-h100 verify-aot
```
