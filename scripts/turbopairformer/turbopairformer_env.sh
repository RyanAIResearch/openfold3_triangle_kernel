#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shared launch environment for the TurboPairFormer training scripts.  Source it; do
# not execute it.  It resolves the Pixi prefix, pins the kernel provider and
# execution flags, checks the run directory, and preflights the install.
#
# Inputs (all optional, with the defaults shown):
#   TURBOPAIRFORMER_PYTHON=<venv>/bin/python   optional explicit environment
#   TURBOPAIRFORMER_PIXI_ENV=openfold3-cuda12-dev  Pixi environment to run inside
#   TURBOPAIRFORMER_PROVIDER=turbopairformer               turbopairformer (pip package) or intree
#   OF3_RUN_CWD=<repo>/../openfold3-run    working directory for the run
#   TURBOPAIRFORMER_SKIP_DOCTOR=0                  set to 1 to skip the preflight

set -euo pipefail

# The scripts live in <repo>/scripts/turbopairformer.
TURBOPAIRFORMER_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$TURBOPAIRFORMER_SCRIPT_DIR/../.." && pwd -P)

if [[ ${TURBOPAIRFORMER_PROVIDER:-turbopairformer} == turbopairformer && -z ${TURBOPAIRFORMER_PYTHON:-} ]]; then
  echo "Set TURBOPAIRFORMER_PYTHON to a dedicated compatible environment containing TurboPairFormer and this OpenFold3 host revision." >&2
  exit 2
fi
source "$TURBOPAIRFORMER_SCRIPT_DIR/_resolve_env.sh"
PIXI_PREFIX="$PREFIX"
TURBOPAIRFORMER_PIXI_ENV=${TURBOPAIRFORMER_PIXI_ENV:-openfold3-cuda12-dev}

# A stray interpreter path silently shadows the environment's packages, which
# is exactly how a run ends up on a different kernel build than it reports.
for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "unset $name before launching; it can shadow the Pixi environment" >&2
    exit 2
  fi
done

TURBOPAIRFORMER_PROVIDER=${TURBOPAIRFORMER_PROVIDER:-turbopairformer}
case "$TURBOPAIRFORMER_PROVIDER" in
  turbopairformer|intree) ;;
  *)
    echo "TURBOPAIRFORMER_PROVIDER must be 'turbopairformer' or 'intree', got: $TURBOPAIRFORMER_PROVIDER" >&2
    exit 2
    ;;
esac
export OPENFOLD3_TRIANGLE_PROVIDER="$TURBOPAIRFORMER_PROVIDER"

# Canonical package settings take precedence. Mirror them into the host's
# OPENFOLD3_* spellings so both providers run under the same execution policy.
export OPENFOLD3_SR_BACKWARD=${TURBOPAIRFORMER_SR_BACKWARD:-${STABLEFOLD_SR_BACKWARD:-${OPENFOLD3_SR_BACKWARD:-1}}}
export OPENFOLD3_SR_SHADOW=${TURBOPAIRFORMER_SR_SHADOW:-${STABLEFOLD_SR_SHADOW:-${OPENFOLD3_SR_SHADOW:-1}}}
export PYTHONNOUSERSITE=1

# Launchers require an AOT install for either provider. Source development JIT
# is deliberately separate from these launch commands and their preflight.
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
export TURBOPAIRFORMER_ALLOW_JIT=0
if [[ "$TURBOPAIRFORMER_PROVIDER" == turbopairformer ]]; then
  # Keep the installed package and host settings aligned.
  export TURBOPAIRFORMER_SR_BACKWARD=${TURBOPAIRFORMER_SR_BACKWARD:-$OPENFOLD3_SR_BACKWARD}
  export TURBOPAIRFORMER_SR_SHADOW=${TURBOPAIRFORMER_SR_SHADOW:-$OPENFOLD3_SR_SHADOW}
fi

# OpenFold3 writes checkpoints and logs relative to the working directory, so
# the run happens outside the checkout to keep the source tree clean.
OF3_RUN_CWD=${OF3_RUN_CWD:-$(cd "$REPO_ROOT/.." && pwd -P)/openfold3-run}
mkdir -p "$OF3_RUN_CWD"
OF3_RUN_CWD=$(cd "$OF3_RUN_CWD" && pwd -P)
case "$OF3_RUN_CWD/" in
  "$REPO_ROOT/"*)
    echo "OF3_RUN_CWD must be outside the source checkout: $OF3_RUN_CWD" >&2
    exit 2
    ;;
esac
export OF3_RUN_CWD

turbopairformer_preflight() {
  if [[ ${TURBOPAIRFORMER_SKIP_DOCTOR:-0} == 1 ]]; then
    return 0
  fi
  # Check installed modules from outside the checkout, where source directories
  # cannot accidentally satisfy imports intended to verify installed packages.
  (
    cd "$OF3_RUN_CWD"
    "$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/turbopairformer_doctor.py" \
      --expect-provider "$TURBOPAIRFORMER_PROVIDER"
  )
}

# Parse the caller's arguments, resolve --runner-yaml against the directory the
# caller is in, and rebuild the argument list with the absolute path.  The
# launchers cd into OF3_RUN_CWD, where a relative path would resolve elsewhere
# or not at all.  Sets RUNNER_YAML and TURBOPAIRFORMER_ARGS.
turbopairformer_rewrite_args() {
  local -a arguments=("$@")
  local index skip_next=0 candidate
  RUNNER_YAML=""
  TURBOPAIRFORMER_ARGS=()

  for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[$index]}" in
      --runner-yaml) RUNNER_YAML=${arguments[$((index + 1))]:-} ;;
      --runner-yaml=*) RUNNER_YAML=${arguments[$index]#--runner-yaml=} ;;
    esac
  done
  if [[ -z "$RUNNER_YAML" ]]; then
    echo "a --runner-yaml <path> argument is required" >&2
    exit 2
  fi

  candidate=$RUNNER_YAML
  case "$candidate" in
    /*) ;;
    *) candidate="$(pwd -P)/$candidate" ;;
  esac
  if [[ ! -r "$candidate" ]]; then
    echo "runner YAML is not readable: $candidate" >&2
    exit 2
  fi
  RUNNER_YAML=$candidate

  for ((index = 0; index < ${#arguments[@]}; index++)); do
    if ((skip_next)); then
      skip_next=0
      continue
    fi
    case "${arguments[$index]}" in
      --runner-yaml)
        TURBOPAIRFORMER_ARGS+=(--runner-yaml "$RUNNER_YAML")
        skip_next=1
        ;;
      --runner-yaml=*) TURBOPAIRFORMER_ARGS+=("--runner-yaml=$RUNNER_YAML") ;;
      *) TURBOPAIRFORMER_ARGS+=("${arguments[$index]}") ;;
    esac
  done
}

turbopairformer_report() {
  echo "=============================================================="
  echo " repository   $REPO_ROOT"
  echo " python       $PIXI_PREFIX/bin/python"
  echo " provider     $OPENFOLD3_TRIANGLE_PROVIDER"
  echo " run cwd      $OF3_RUN_CWD"
  echo " runner yaml  ${RUNNER_YAML:-<none>}"
  echo " devices      ${CUDA_VISIBLE_DEVICES:-<all visible>}"
  echo "=============================================================="
}
