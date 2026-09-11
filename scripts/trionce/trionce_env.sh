#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shared launch environment for the Trionce training scripts.  Source it; do
# not execute it.  It resolves the Pixi prefix, pins the kernel provider and
# execution flags, checks the run directory, and preflights the install.
#
# Inputs (all optional, with the defaults shown):
#   TRIONCE_PIXI_ENV=openfold3-cuda12-dev  Pixi environment to run inside
#   TRIONCE_PROVIDER=trionce               trionce (pip package) or intree
#   OF3_RUN_CWD=<repo>/../openfold3-run    working directory for the run
#   TRIONCE_SKIP_DOCTOR=0                  set to 1 to skip the preflight

set -euo pipefail

# The scripts live in <repo>/scripts/trionce.
TRIONCE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "$TRIONCE_SCRIPT_DIR/../.." && pwd -P)

TRIONCE_PIXI_ENV=${TRIONCE_PIXI_ENV:-openfold3-cuda12-dev}
PIXI_PREFIX="$REPO_ROOT/.pixi/envs/$TRIONCE_PIXI_ENV"
if [[ ! -x "$PIXI_PREFIX/bin/python" ]]; then
  echo "no Pixi environment at $PIXI_PREFIX" >&2
  echo "create it with: pixi install -e $TRIONCE_PIXI_ENV" >&2
  exit 2
fi

# A stray interpreter path silently shadows the environment's packages, which
# is exactly how a run ends up on a different kernel build than it reports.
for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "unset $name before launching; it can shadow the Pixi environment" >&2
    exit 2
  fi
done

TRIONCE_PROVIDER=${TRIONCE_PROVIDER:-trionce}
case "$TRIONCE_PROVIDER" in
  trionce|intree) ;;
  *)
    echo "TRIONCE_PROVIDER must be 'trionce' or 'intree', got: $TRIONCE_PROVIDER" >&2
    exit 2
    ;;
esac
export OPENFOLD3_TRIANGLE_PROVIDER="$TRIONCE_PROVIDER"

# Execution flags.  Both kernel builds read the OPENFOLD3_* spellings, so the
# two providers run under identical policy and stay comparable.
export OPENFOLD3_SR_BACKWARD=${OPENFOLD3_SR_BACKWARD:-1}
export OPENFOLD3_SR_SHADOW=${OPENFOLD3_SR_SHADOW:-1}
export PYTHONNOUSERSITE=1

# The pip package ships compiled extensions, so JIT there would mean the AOT
# artifacts failed to load and a run would silently continue on a build nobody
# attested.  The in-tree copy has no compiled artifacts in a source checkout,
# so its development JIT lane is what serves it.
if [[ -z ${OPENFOLD3_TRIANGLE_ALLOW_JIT:-} ]]; then
  if [[ "$TRIONCE_PROVIDER" == intree ]]; then
    OPENFOLD3_TRIANGLE_ALLOW_JIT=1
  else
    OPENFOLD3_TRIANGLE_ALLOW_JIT=0
  fi
fi
export OPENFOLD3_TRIANGLE_ALLOW_JIT

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

# torch writes JIT builds under ~/.cache/torch_extensions, which is root-owned
# on some hosts.  Keep the cache beside the run directory so a compile does not
# die on a permission error minutes into setup.
if [[ "$OPENFOLD3_TRIANGLE_ALLOW_JIT" == 1 && -z ${TORCH_EXTENSIONS_DIR:-} ]]; then
  TORCH_EXTENSIONS_DIR="$OF3_RUN_CWD/.torch_extensions"
  mkdir -p "$TORCH_EXTENSIONS_DIR"
  export TORCH_EXTENSIONS_DIR
fi

trionce_preflight() {
  if [[ ${TRIONCE_SKIP_DOCTOR:-0} == 1 ]]; then
    return 0
  fi
  # Run the doctor by path, from outside the checkout.  An interpreter started
  # in the repository root puts the root on sys.path, where the package's own
  # project folder shadows the install as an empty namespace package -- the
  # doctor would then diagnose a broken install that is in fact fine.
  (
    cd "$OF3_RUN_CWD"
    "$PIXI_PREFIX/bin/python" "$REPO_ROOT/tools/trionce_doctor.py" \
      --expect-provider "$TRIONCE_PROVIDER"
  )
}

# Parse the caller's arguments, resolve --runner-yaml against the directory the
# caller is in, and rebuild the argument list with the absolute path.  The
# launchers cd into OF3_RUN_CWD, where a relative path would resolve elsewhere
# or not at all.  Sets RUNNER_YAML and TRIONCE_ARGS.
trionce_rewrite_args() {
  local -a arguments=("$@")
  local index skip_next=0 candidate
  RUNNER_YAML=""
  TRIONCE_ARGS=()

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
        TRIONCE_ARGS+=(--runner-yaml "$RUNNER_YAML")
        skip_next=1
        ;;
      --runner-yaml=*) TRIONCE_ARGS+=("--runner-yaml=$RUNNER_YAML") ;;
      *) TRIONCE_ARGS+=("${arguments[$index]}") ;;
    esac
  done
}

trionce_report() {
  echo "=============================================================="
  echo " repository   $REPO_ROOT"
  echo " pixi env     $TRIONCE_PIXI_ENV"
  echo " provider     $OPENFOLD3_TRIANGLE_PROVIDER"
  echo " run cwd      $OF3_RUN_CWD"
  echo " runner yaml  ${RUNNER_YAML:-<none>}"
  echo " devices      ${CUDA_VISIBLE_DEVICES:-<all visible>}"
  echo "=============================================================="
}
