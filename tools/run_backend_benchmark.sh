#!/usr/bin/env bash
set -euo pipefail

for name in PYTHONPATH PYTHONHOME LD_PRELOAD; do
  if [[ -n ${!name:-} ]]; then
    echo "backend benchmark forbids $name" >&2
    exit 2
  fi
done

prefix=$(cd "${CONDA_PREFIX:?CONDA_PREFIX is not set}" && pwd -P)
caller_cwd=$(pwd -P)
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

runner_yaml=
devices=1
num_nodes=1
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
    --devices|--num-nodes)
      if ((index + 1 >= ${#arguments[@]})); then
        echo "${arguments[$index]} requires a count" >&2
        exit 2
      fi
      if [[ ${arguments[$index]} == --devices ]]; then
        devices=${arguments[$((index + 1))]}
      else
        num_nodes=${arguments[$((index + 1))]}
      fi
      ;;
    --devices=*) devices=${arguments[$index]#--devices=} ;;
    --num-nodes=*) num_nodes=${arguments[$index]#--num-nodes=} ;;
  esac
done
if [[ ${OPENFOLD3_TRIANGLE_PROVIDER:-intree} == turbopairformer ]] && \
   [[ "$devices" != 1 || "$num_nodes" != 1 ]]; then
  echo "standalone TurboPairFormer benchmarks currently require --devices 1 --num-nodes 1; multi-GPU validation is pending" >&2
  exit 2
fi
if [[ -n "$runner_yaml" ]]; then
  case "$runner_yaml" in
    /*) ;;
    *) runner_yaml="$caller_cwd/$runner_yaml" ;;
  esac
  "$prefix/bin/python" "$source_root/tools/validate_training_data.py" \
    "$runner_yaml" --base-dir "$run_cwd"
fi

case "${OPENFOLD3_TRIANGLE_PROVIDER:-intree}" in
  turbopairformer)
    (
      cd "$run_cwd"
      "$prefix/bin/python" "$source_root/tools/turbopairformer_doctor.py" \
        --expect-provider turbopairformer
    )
    ;;
  intree)
    (
      cd "$source_root"
      "$prefix/bin/python" -m tools.manage_attested_sm90_wheel verify-prebuilt
    )
    ;;
  *)
    echo "unknown OPENFOLD3_TRIANGLE_PROVIDER: $OPENFOLD3_TRIANGLE_PROVIDER" >&2
    exit 2
    ;;
esac

cd "$run_cwd"
exec "$prefix/bin/python" "$source_root/tools/benchmark_training_backend.py" "$@"
