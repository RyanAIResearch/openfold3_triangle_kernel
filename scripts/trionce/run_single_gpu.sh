#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Train OpenFold3 on one GPU with the Trionce triangle kernels.
#
# Usage:
#   scripts/trionce/run_single_gpu.sh --runner-yaml <path> [--seed 42] [extra args]
#
# The runner YAML must already select a stable-kernel backend, which
# tools/select_training_backend.py writes:
#
#   .pixi/envs/openfold3-cuda12-dev/bin/python -m tools.select_training_backend \
#     datasets/train_pdb_subset.yaml --backend trionce \
#     --output datasets/train_pdb_subset.trionce.yaml
#
# Set TRIONCE_PROVIDER=intree to run the same YAML on the in-tree kernel build,
# which is how the two are compared.

source "$(dirname "${BASH_SOURCE[0]}")/trionce_env.sh"

trionce_rewrite_args "$@"

# One GPU: pin the first visible device unless the caller chose another.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "single-GPU launch expects one device, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

trionce_report
trionce_preflight

"$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/validate_training_data.py" \
  "$RUNNER_YAML" --base-dir "$OF3_RUN_CWD"

cd "$OF3_RUN_CWD"
exec "$PIXI_PREFIX/bin/run_openfold" train "${TRIONCE_ARGS[@]}"
