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
caller_cwd=$(pwd -P)
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

runner_yaml=
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --runner-yaml)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "--runner-yaml requires a path" >&2
        exit 2
      fi
      runner_yaml=${arguments[$((index + 1))]}
      ;;
    --runner-yaml=*) runner_yaml=${arguments[$index]#--runner-yaml=} ;;
  esac
done
if [[ -n "$runner_yaml" ]]; then
  case "$runner_yaml" in
    /*) ;;
    *) runner_yaml="$caller_cwd/$runner_yaml" ;;
  esac
  "$prefix/bin/python" "$source_root/tools/validate_training_data.py" \
    "$runner_yaml" --base-dir "$run_cwd"
fi

(
  cd "$source_root"
  "$prefix/bin/python" -m tools.manage_attested_sm90_wheel verify-prebuilt
)

cd "$run_cwd"
exec "$launcher_path" "$@"
