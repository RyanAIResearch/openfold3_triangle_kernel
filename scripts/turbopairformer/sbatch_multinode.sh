#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=of3-turbopairformer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=/shared/openfold3-run/slurm-%j.out
#
# Submit a multi-node TurboPairFormer training run:
#
#   sbatch scripts/turbopairformer/sbatch_multinode.sh
#
# One task per node starts one torchrun, which forks the eight local ranks.
# Edit SHARED_REPO, OF3_RUN_CWD and RUNNER_YAML for the site, and keep every
# path identical on every node -- the DDP kernel preflight has rank 0 write a
# receipt that the other ranks read back, so the run directory must be a real
# shared filesystem, not merely the same name on each node.

set -euo pipefail
if [[ ${TURBOPAIRFORMER_PROVIDER:-turbopairformer} == turbopairformer ]]; then
  echo "TurboPairFormer pip provider DDP integration is not yet validated; use the single-GPU launcher." >&2
  exit 2
fi

SHARED_REPO=${SHARED_REPO:-/shared/openfold3_triangle_kernel}
export OF3_RUN_CWD=${OF3_RUN_CWD:-/shared/openfold3-run}
RUNNER_YAML=${RUNNER_YAML:-$SHARED_REPO/datasets/train_pdb_subset.turbopairformer.yaml}

export TURBOPAIRFORMER_PROVIDER=${TURBOPAIRFORMER_PROVIDER:-turbopairformer}
export NNODES=${SLURM_NNODES:?not running under Slurm}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# NCCL picks the wrong interface on some fabrics; set this only if it does.
# export NCCL_SOCKET_IFNAME=eth0

unset PYTHONPATH PYTHONHOME LD_PRELOAD

mkdir -p "$OF3_RUN_CWD"
test -r "$RUNNER_YAML"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  export NODE_RANK=$SLURM_NODEID
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
  exec '"$SHARED_REPO"'/scripts/turbopairformer/run_multinode.sh \
    --runner-yaml '"$RUNNER_YAML"' \
    --seed 42
'
