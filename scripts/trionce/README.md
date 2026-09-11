# Running OpenFold3 on Trionce kernels

Launch scripts for training OpenFold3 with the `trionce` pip package serving the
stable triangle routes, on one GPU, on several GPUs of one node, and across
nodes.

> **Validation status.** The single-GPU path has been run end to end on an H100
> 80 GB: install, preflight, and a one-step training run on both kernel
> providers. **The multi-GPU and multi-node paths have not been executed** —
> the development machine has a single GPU. They are written against the
> repository's existing distributed quickstart and pass shell syntax checks, but
> the first person to run them should treat them as unverified and work through
> the checklist in [Handing off multi-GPU and multi-node](#handing-off-multi-gpu-and-multi-node).

## What these scripts do

All three launchers source `trionce_env.sh`, which performs the same preflight
before any training starts:

| Step | Why it is there |
| --- | --- |
| Refuse a set `PYTHONPATH` / `PYTHONHOME` / `LD_PRELOAD` | These silently shadow the Pixi environment, and a run then reports one kernel build while executing another |
| Pin `OPENFOLD3_TRIANGLE_PROVIDER` | Makes the serving kernel build explicit instead of letting it be inferred |
| Export `OPENFOLD3_SR_BACKWARD` / `OPENFOLD3_SR_SHADOW` | Both kernel builds read these spellings, so the two providers stay directly comparable |
| Set `OPENFOLD3_TRIANGLE_ALLOW_JIT` from the provider | The pip package must serve its attested AOT binaries; the in-tree copy has no compiled artifacts in a source checkout and needs its development JIT lane |
| Redirect `TORCH_EXTENSIONS_DIR` when JIT is active | `~/.cache/torch_extensions` is root-owned on some hosts and the compile dies minutes into setup |
| Require `OF3_RUN_CWD` outside the checkout | OpenFold3 writes checkpoints and logs relative to the working directory |
| Rewrite `--runner-yaml` to an absolute path | The launchers `cd` into the run directory, where a relative path resolves elsewhere |
| Run `tools/trionce_doctor.py` | Fails before the job starts if the GPU, the install, the AOT binaries, or the provider is not what the run assumes |

## Prerequisites

```bash
cd <repo>
pixi install -e openfold3-cuda12-dev

ENVP="$PWD/.pixi/envs/openfold3-cuda12-dev"
CUDA_HOME="$ENVP" TORCH_CUDA_ARCH_LIST=9.0a TRIONCE_BUILD_AOT=1 \
  uv pip install --python "$ENVP/bin/python" --no-build-isolation -e trionce
```

The build compiles five `sm_90a` extensions and takes roughly fifteen minutes.
`TORCH_CUDA_ARCH_LIST=9.0a` is required exactly as written: the kernels embed
architecture-specific SASS and no PTX.

Verify before trusting any run — **from outside the repository root**:

```bash
cd /tmp && "$ENVP/bin/python" <repo>/tools/trionce_doctor.py --expect-provider trionce
```

Expected:

```
trionce doctor: PASS
  gpu       NVIDIA H100 80GB HBM3 (sm_90a)
  torch     2.10.0 / CUDA 12.9
  package   trionce 0.1.0
  aot       5 extensions @ <commit>
  provider  trionce
```

## Prepare a runner YAML

```bash
"$ENVP/bin/python" -m tools.select_training_backend \
  datasets/train_pdb_subset.yaml \
  --backend trionce \
  --output datasets/train_pdb_subset.trionce.yaml
```

`--backend trionce` and `--backend stablefold` write the same model flags; they
differ only in which build serves them, which the launcher pins. Add
`--single-gpu-smoke` for a one-step, validation-free YAML.

## One GPU

```bash
export OF3_RUN_CWD="$PWD/../openfold3-run"
export CUDA_VISIBLE_DEVICES=0

scripts/trionce/run_single_gpu.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.trionce.yaml" \
  --seed 42
```

## Several GPUs, one node

Set the topology in the runner YAML first. At batch size one:

```text
epoch_len = max_steps * devices * num_nodes
```

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
`torchrun`**. The script refuses to start if `pl_trainer_args.devices` disagrees
with `CUDA_VISIBLE_DEVICES`.

## Several nodes

```yaml
data_module_args:
  epoch_len: 6400
pl_trainer_args:
  devices: 8
  num_nodes: 2
  max_steps: 400
```

Run on every allocated node, changing only `NODE_RANK`:

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
export OF3_RUN_CWD=/shared/openfold3-run
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

scripts/trionce/run_multinode.sh \
  --runner-yaml /shared/<repo>/datasets/train_pdb_subset.trionce.yaml \
  --seed 42
```

Under Slurm, `sbatch_multinode.sh` allocates the nodes and starts one `torchrun`
per node. Edit `SHARED_REPO`, `OF3_RUN_CWD`, and `RUNNER_YAML` for the site:

```bash
sbatch scripts/trionce/sbatch_multinode.sh
```

**Hard requirements for multi-node**, both inherited from the repository's
existing distributed contract:

1. The repository, Pixi environment, runner YAML, dataset, and run directory
   must be at the **same absolute paths on every node**.
2. `OF3_RUN_CWD` must be a **genuinely shared read-write filesystem**. Rank 0
   writes the kernel receipt there and every other rank reads it back; a path
   that merely has the same name on each node will hang or fail the contract,
   not merely disagree.

## Handing off multi-GPU and multi-node

These paths are unexecuted. Work through this in order; each step fails fast and
cheaply before the next one costs an allocation.

1. **Doctor on every node.** From outside the checkout, on each node:
   `python <repo>/tools/trionce_doctor.py --expect-provider trionce`. All must
   report the same `aot ... @ <commit>` line. A differing commit means the nodes
   are not running the same build.
2. **Single GPU on one node.** Run `run_single_gpu.sh` with a
   `--single-gpu-smoke` YAML. This has been validated here, so a failure is
   environmental, not a script defect.
3. **Two GPUs on one node** before eight. Set `devices: 2`, `epoch_len:
   max_steps * 2`. Confirms DDP starts and the kernel preflight agrees across
   local ranks.
4. **Full node.** Then `devices: 8`.
5. **Two nodes, one GPU each**, before eight each. This isolates the
   rendezvous and shared-filesystem requirements from the GPU count.
6. **Full multi-node.**

What a healthy distributed run shows:

- Rank 0 writes `<OF3_RUN_CWD>/stablefold_kernel_ddp.json` with
  `"status": "PASS"` and one entry per rank. Every rank must appear; a missing
  rank means that rank could not read the receipt back, i.e. the run directory
  is not genuinely shared.
- The training log contains no `stablefold fallback to stock` for pair-stack
  shapes. An occasional fallback at an unusual shape is by design, but it should
  not appear at the trunk sites.
- Loss at step *n* is comparable to the single-GPU run at the same global step
  count, allowing for the different effective batch size.

If something goes wrong, `TRIONCE_PROVIDER=intree` runs the identical YAML on
the in-tree kernel build. If the failure persists under both providers it is not
a packaging problem.

## Troubleshooting

Problems hit while bringing this up, and what they look like:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `trionce doctor: FAIL: 'trionce' resolved to a namespace package` | The interpreter started in the repository root, whose `trionce/` project folder shadows the install — setuptools appends its editable finder *after* the standard path finder, so the empty namespace wins | Run from outside the checkout, or `pytest --import-mode=importlib` |
| `AOTExtensionError: the installed package contains a source-tree manifest` | The AOT build wrote its receipt to a temporary build directory instead of next to the binaries | Rebuild with `TRIONCE_BUILD_AOT=1`; the fix is in `trionce/setup.py` |
| 100 compile errors in `type_traits`, `__is_array is undefined` | `nvcc` drove the system GCC 11.4 against the environment's conda GCC 14.3 headers | Put `$ENVP/bin` on `PATH` and set `CC`/`CXX` to the environment's compilers (`trionce_env.sh` does this) |
| `PermissionError: .../torch_extensions/py314_cu129` | `~/.cache/torch_extensions` is root-owned | Set `TORCH_EXTENSIONS_DIR` somewhere writable (`trionce_env.sh` does this) |
| Process alive, GPU at 0%, no compiler running, no progress | A killed compile left torch's `FileBaton` lock behind; the next run waits on it forever | Delete the stale `lock` file under `TORCH_EXTENSIONS_DIR` |
| Every run recompiles all five kernels | Scripts disagree on `nvcc` flags; torch bakes the command line into `build.ninja` | Keep the JIT flags in one place and source it everywhere |
| Two runs interfere, both stall | Concurrent processes racing on the same ninja build directory | Serialise; do not run two training or test jobs against the same `TORCH_EXTENSIONS_DIR` |

## Verifying kernels after a change

```bash
# bitwise equivalence, pip package against the in-tree build
python -m pytest openfold3/tests/test_trionce_provider_equivalence.py \
  --import-mode=importlib -v

# speed: packaging cost, and the speedup over stock attention
python tools/benchmark_trionce_provider.py --iters 20

# accuracy against an FP64 oracle, and run-to-run determinism
python tools/benchmark_trionce_accuracy.py
```

Reference numbers measured on one H100 80 GB, forward+backward:

| layout | trionce | intree | stock | packaging cost | speedup |
| --- | --- | --- | --- | --- | --- |
| trunk_n640_d32 | 6.97 ms | 6.96 ms | 28.97 ms | 1.00x | 4.16x |
| trunk_n768_d32 | 11.97 ms | 11.92 ms | 49.35 ms | 1.00x | 4.12x |

Accuracy at `trunk_n640_d32`, L2 relative error against an FP64 oracle: forward
output `1.831e-03` for both kernel providers against `3.201e-03` for stock, with
every provider replaying bitwise.
