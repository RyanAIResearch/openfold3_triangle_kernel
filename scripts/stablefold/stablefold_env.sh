#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shared launch environment for the StableFold training scripts.  Source it; do
# not execute it.  It resolves the Pixi prefix, pins the kernel provider and
# execution flags, checks the run directory, and preflights the install.
#
# Inputs (all optional, with the defaults shown):
#   STABLEFOLD_PYTHON=<venv>/bin/python   optional explicit environment
#   STABLEFOLD_PIXI_ENV=openfold3-cuda12-dev  Pixi environment to run inside
#   STABLEFOLD_PROVIDER=stablefold               stablefold (pip package) or intree
#   OF3_RUN_CWD=<repo>/../openfold3-run    working directory for the run
#   STABLEFOLD_SKIP_DOCTOR=0                  set to 1 to skip the preflight

set -euo pipefail

# The scripts live in <repo>/scripts/stablefold.
STABLEFOLD_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$STABLEFOLD_SCRIPT_DIR/../.." && pwd -P)

if [[ ${STABLEFOLD_PROVIDER:-stablefold} == stablefold && -z ${STABLEFOLD_PYTHON:-} ]]; then
  echo "Set STABLEFOLD_PYTHON to a dedicated compatible environment containing StableFold and this OpenFold3 host revision." >&2
  exit 2
fi
source "$STABLEFOLD_SCRIPT_DIR/_resolve_env.sh"
PIXI_PREFIX="$PREFIX"
STABLEFOLD_PIXI_ENV=${STABLEFOLD_PIXI_ENV:-openfold3-cuda12-dev}

# A stray interpreter path silently shadows the environment's packages, which
# is exactly how a run ends up on a different kernel build than it reports.
for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "unset $name before launching; it can shadow the Pixi environment" >&2
    exit 2
  fi
done

STABLEFOLD_PROVIDER=${STABLEFOLD_PROVIDER:-stablefold}
case "$STABLEFOLD_PROVIDER" in
  stablefold|intree) ;;
  *)
    echo "STABLEFOLD_PROVIDER must be 'stablefold' or 'intree', got: $STABLEFOLD_PROVIDER" >&2
    exit 2
    ;;
esac
export OPENFOLD3_TRIANGLE_PROVIDER="$STABLEFOLD_PROVIDER"

# Execution flags.  Both kernel builds read the OPENFOLD3_* spellings, so the
# two providers run under identical policy and stay comparable.
export OPENFOLD3_SR_BACKWARD=${OPENFOLD3_SR_BACKWARD:-${STABLEFOLD_SR_BACKWARD:-1}}
export OPENFOLD3_SR_SHADOW=${OPENFOLD3_SR_SHADOW:-${STABLEFOLD_SR_SHADOW:-1}}
export PYTHONNOUSERSITE=1

# Launchers require an AOT install for either provider. Source development JIT
# is deliberately separate from these launch commands and their preflight.
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
export STABLEFOLD_ALLOW_JIT=0
if [[ "$STABLEFOLD_PROVIDER" == stablefold ]]; then
  # Preserve explicit values; the package policy rejects conflicting aliases.
  export STABLEFOLD_SR_BACKWARD=${STABLEFOLD_SR_BACKWARD:-$OPENFOLD3_SR_BACKWARD}
  export STABLEFOLD_SR_SHADOW=${STABLEFOLD_SR_SHADOW:-$OPENFOLD3_SR_SHADOW}
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

stablefold_preflight() {
  if [[ ${STABLEFOLD_SKIP_DOCTOR:-0} == 1 ]]; then
    return 0
  fi
  # Check installed modules from outside the checkout, where source directories
  # cannot accidentally satisfy imports intended to verify installed packages.
  (
    cd "$OF3_RUN_CWD"
    "$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/stablefold_doctor.py" \
      --expect-provider "$STABLEFOLD_PROVIDER"
  )
}

# Parse the caller's arguments, resolve --runner-yaml against the directory the
# caller is in, and rebuild the argument list with the absolute path.  The
# launchers cd into OF3_RUN_CWD, where a relative path would resolve elsewhere
# or not at all.  Sets RUNNER_YAML and STABLEFOLD_ARGS.
stablefold_rewrite_args() {
  local -a arguments=("$@")
  local index skip_next=0 candidate
  RUNNER_YAML=""
  STABLEFOLD_ARGS=()

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
        STABLEFOLD_ARGS+=(--runner-yaml "$RUNNER_YAML")
        skip_next=1
        ;;
      --runner-yaml=*) STABLEFOLD_ARGS+=("--runner-yaml=$RUNNER_YAML") ;;
      *) STABLEFOLD_ARGS+=("${arguments[$index]}") ;;
    esac
  done
}

stablefold_report() {
  echo "=============================================================="
  echo " repository   $REPO_ROOT"
  echo " python       $PIXI_PREFIX/bin/python"
  echo " provider     $OPENFOLD3_TRIANGLE_PROVIDER"
  echo " run cwd      $OF3_RUN_CWD"
  echo " runner yaml  ${RUNNER_YAML:-<none>}"
  echo " devices      ${CUDA_VISIBLE_DEVICES:-<all visible>}"
  echo "=============================================================="
}
