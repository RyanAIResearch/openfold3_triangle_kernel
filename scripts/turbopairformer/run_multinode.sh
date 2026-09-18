#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Train OpenFold3 across several nodes with the TurboPairFormer kernels, one torchrun
# per node.  Run this on every allocated node, changing only NODE_RANK.
#
# Usage:
#   export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0
#   export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
#   export OF3_RUN_CWD=/shared/openfold3-run
#   scripts/turbopairformer/run_multinode.sh --runner-yaml <path> [--seed 42]
#
# Requirements that a multi-node run adds on top of the single-node one:
#
#   * the repository, Pixi environment, runner YAML, dataset, and run
#     directory are at the same paths on every node;
#   * OF3_RUN_CWD is a genuinely shared read-write filesystem -- the DDP kernel
#     preflight has rank 0 write a receipt there that every other rank reads
#     back, so same-name-different-disk fails the contract;
#   * every rank runs the same kernel provider, which this script pins.
#
# The runner YAML carries the topology:
#   pl_trainer_args: { devices: 8, num_nodes: 2, max_steps: 400 }
#   data_module_args: { epoch_len: 6400 }   # max_steps * devices * num_nodes

set -euo pipefail
if [[ ${TURBOPAIRFORMER_PROVIDER:-turbopairformer} == turbopairformer ]]; then
  echo "TurboPairFormer pip provider currently supports single-GPU OpenFold3 launches only; its host DDP integration is not yet validated." >&2
  echo "Use run_single_gpu.sh, or use the separately certified intree distributed workflow." >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/turbopairformer_env.sh"

turbopairformer_rewrite_args "$@"

: "${NNODES:?set NNODES to the number of nodes}"
: "${GPUS_PER_NODE:?set GPUS_PER_NODE to the GPUs each node contributes}"
: "${NODE_RANK:?set NODE_RANK to the 0-based rank of this node}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 address}"
MASTER_PORT=${MASTER_PORT:-29500}

if ((NODE_RANK < 0 || NODE_RANK >= NNODES)); then
  echo "NODE_RANK=$NODE_RANK is outside 0..$((NNODES - 1))" >&2
  exit 2
fi

# Rank 0 writes the kernel receipt into the run directory and the other ranks
# read it back, so a private per-node directory deadlocks the preflight rather
# than reporting a mismatch.  Writability is the cheap half of that check.
if [[ ! -w "$OF3_RUN_CWD" ]]; then
  echo "OF3_RUN_CWD is not writable on this node: $OF3_RUN_CWD" >&2
  exit 2
fi

turbopairformer_report
echo " nodes        $NNODES x $GPUS_PER_NODE GPU (this node rank $NODE_RANK)"
echo " rendezvous   $MASTER_ADDR:$MASTER_PORT"
echo "=============================================================="

turbopairformer_preflight

"$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/validate_training_data.py" \
  "$RUNNER_YAML" --base-dir "$OF3_RUN_CWD"

cd "$OF3_RUN_CWD"
exec "$PIXI_PREFIX/bin/torchrun" \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  --module openfold3.run_openfold \
  train "${TURBOPAIRFORMER_ARGS[@]}"
