#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Train OpenFold3 on one GPU with the StableFold triangle kernels.
#
# Usage:
#   scripts/stablefold/run_single_gpu.sh --runner-yaml <path> [--seed 42] [extra args]
#
# The runner YAML must already select a stable-kernel backend, which
# tools/select_training_backend.py writes:
#
#   .pixi/envs/openfold3-cuda12-dev/bin/python -m tools.select_training_backend \
#     datasets/train_pdb_subset.yaml --backend stablefold \
#     --output datasets/train_pdb_subset.stablefold.yaml
#
# Set STABLEFOLD_PROVIDER=intree to run the same YAML on the in-tree kernel build,
# which is how the two are compared.

source "$(dirname "${BASH_SOURCE[0]}")/stablefold_env.sh"

stablefold_rewrite_args "$@"

# One GPU: pin the first visible device unless the caller chose another.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "single-GPU launch expects one device, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

"$PIXI_PREFIX/bin/python" - "$RUNNER_YAML" <<'PYTHON'
import sys

import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle) or {}
trainer = config.get("pl_trainer_args") or {}
if trainer.get("devices", 1) != 1 or trainer.get("num_nodes", 1) != 1:
    raise SystemExit(
        "single-GPU launch requires pl_trainer_args.devices=1 and num_nodes=1; "
        "prepare the YAML with --single-gpu-smoke or set those fields explicitly"
    )
PYTHON

stablefold_report
stablefold_preflight

"$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/validate_training_data.py" \
  "$RUNNER_YAML" --base-dir "$OF3_RUN_CWD"

cd "$OF3_RUN_CWD"
exec "$PIXI_PREFIX/bin/run_openfold" train "${STABLEFOLD_ARGS[@]}"
