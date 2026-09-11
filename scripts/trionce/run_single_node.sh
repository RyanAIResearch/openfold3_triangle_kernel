#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Train OpenFold3 on several GPUs of one node with the Trionce kernels.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   scripts/trionce/run_single_node.sh --runner-yaml <path> [--seed 42]
#
# PyTorch Lightning starts the local DDP workers itself, so this launch must
# NOT be wrapped in torchrun.  The runner YAML carries the topology:
#
#   pl_trainer_args: { devices: 8, num_nodes: 1, max_steps: 400 }
#   data_module_args: { epoch_len: 3200 }   # max_steps * devices * num_nodes
#
# The device count in the YAML must match CUDA_VISIBLE_DEVICES; a mismatch is
# reported here rather than after model setup.

source "$(dirname "${BASH_SOURCE[0]}")/trionce_env.sh"

trionce_rewrite_args "$@"

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "set CUDA_VISIBLE_DEVICES to the devices this node should use" >&2
  exit 2
fi

visible_count=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
configured_devices=$(
  "$PIXI_PREFIX/bin/python" - "$RUNNER_YAML" <<'PYTHON'
import sys

import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle) or {}
print((config.get("pl_trainer_args") or {}).get("devices", ""))
PYTHON
)
if [[ -n "$configured_devices" && "$configured_devices" != "$visible_count" ]]; then
  echo "runner YAML requests devices=$configured_devices but " \
       "CUDA_VISIBLE_DEVICES exposes $visible_count" >&2
  exit 2
fi

trionce_report
trionce_preflight

"$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/validate_training_data.py" \
  "$RUNNER_YAML" --base-dir "$OF3_RUN_CWD"

cd "$OF3_RUN_CWD"
exec "$PIXI_PREFIX/bin/run_openfold" train "${TRIONCE_ARGS[@]}"
