#!/usr/bin/env bash
set -euo pipefail

launcher=${1:?expected run_openfold or torchrun}
shift

case "$launcher" in
  run_openfold|torchrun) ;;
  *)
    echo "unsupported official training launcher: $launcher" >&2
    exit 2
    ;;
esac

for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "official wheel launch forbids $name" >&2
    exit 2
  fi
done

prefix=$(cd "${CONDA_PREFIX:?CONDA_PREFIX is not set}" && pwd -P)
launcher_path=$(command -v "$launcher")
case "$launcher_path" in
  "$prefix/bin/"*) ;;
  *)
    echo "$launcher does not resolve inside the Pixi prefix: $launcher_path" >&2
    exit 2
    ;;
esac

run_cwd=${OF3_RUN_CWD:-/mnt/shared/openfold3/run}
if [[ ! -d "$run_cwd" ]]; then
  echo "OF3_RUN_CWD is not a directory: $run_cwd" >&2
  exit 2
fi

run_cwd=$(cd "$run_cwd" && pwd -P)
source_root=$(cd "${PIXI_PROJECT_ROOT:?PIXI_PROJECT_ROOT is not set}" && pwd -P)
case "$run_cwd/" in
  "$source_root/"*)
    echo "OF3_RUN_CWD must be outside the source checkout: $run_cwd" >&2
    exit 2
    ;;
esac

artifact_dir=${OF3_ARTIFACT_DIR:-/mnt/shared/openfold3/artifacts}
(
  cd "$source_root"
  "$prefix/bin/python" -m tools.manage_attested_sm90_wheel verify \
    --artifact-dir "$artifact_dir"
)

cd "$run_cwd"
exec "$launcher_path" "$@"
