#!/usr/bin/env bash
set -euo pipefail

for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "backend benchmark forbids $name" >&2
    exit 2
  fi
done

prefix=$(cd "${CONDA_PREFIX:?CONDA_PREFIX is not set}" && pwd -P)
source_root=$(cd "${PIXI_PROJECT_ROOT:?PIXI_PROJECT_ROOT is not set}" && pwd -P)
run_cwd=${OF3_RUN_CWD:-/mnt/shared/openfold3/run}
if [[ ! -d "$run_cwd" ]]; then
  echo "OF3_RUN_CWD is not a directory: $run_cwd" >&2
  exit 2
fi
run_cwd=$(cd "$run_cwd" && pwd -P)
case "$run_cwd/" in
  "$source_root/"*)
    echo "OF3_RUN_CWD must be outside the source checkout: $run_cwd" >&2
    exit 2
    ;;
esac

(
  cd "$source_root"
  "$prefix/bin/python" -m tools.manage_attested_sm90_wheel verify-prebuilt
)

cd "$run_cwd"
exec "$prefix/bin/python" "$source_root/tools/benchmark_training_backend.py" "$@"
