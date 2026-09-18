#!/usr/bin/env bash
# Validate the final repaired wheel in a separate image without the CUDA toolkit.
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 /absolute/path/to/wheel.whl /absolute/path/to/empty-evidence-dir" >&2
  exit 2
fi
SOURCE_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
WHEEL_DIR="$(cd "$(dirname "$1")" && pwd)"
WHEEL_NAME="$(basename "$1")"
mkdir -p "$2"
OUTPUT_ROOT="$(cd "$2" && pwd)"
if [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "use an empty evidence directory" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "validate from the clean release source" >&2
  exit 2
fi
docker run --rm --gpus all --network none \
  --mount "type=bind,src=$SOURCE_ROOT,dst=/source,readonly" \
  --mount "type=bind,src=$WHEEL_DIR,dst=/wheel,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/evidence" \
  --env "WHEEL_NAME=$WHEEL_NAME" \
  --env "SOURCE_COMMIT=$SOURCE_COMMIT" \
  "turbopairformer-manylinux-runtime:$SOURCE_COMMIT" bash -ceu '
    cd /tmp
    unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
    export TURBOPAIRFORMER_ALLOW_JIT=0 OPENFOLD3_TRIANGLE_ALLOW_JIT=0 STABLEFOLD_ALLOW_JIT=0
    export OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer
    export CUDA_HOME=/nonexistent TORCH_EXTENSIONS_DIR=/tmp/no-jit-cache
    python -m venv --system-site-packages /tmp/validation
    PYTHON=/tmp/validation/bin/python
    "$PYTHON" -m pip install --no-deps --no-compile "/wheel/$WHEEL_NAME" > /evidence/install.log 2>&1
    "$PYTHON" -m pip check >> /evidence/install.log 2>&1
    "$PYTHON" -c '\''import importlib.util, pathlib, turbopairformer; p=pathlib.Path(turbopairformer.__file__).resolve(); assert str(p).startswith("/tmp/validation/lib/python3.14/site-packages/"); assert importlib.util.find_spec("openfold3") is None; print(p)'\'' > /evidence/installed-package.txt
    chmod -R a-w /tmp/validation/lib/python3.14/site-packages/turbopairformer
    env PATH=/nonexistent "$PYTHON" /source/packages/turbopairformer/tools/smoke_installed.py > /evidence/installed-smoke.json
    env PATH=/nonexistent "$PYTHON" /source/tools/turbopairformer_doctor.py --expect-provider turbopairformer --json > /evidence/installed-aot.json
    TURBOPAIRFORMER_TEST_WHEEL="/wheel/$WHEEL_NAME" "$PYTHON" -m pytest \
      /source/packages/turbopairformer/tests \
      --ignore=/source/packages/turbopairformer/tests/host_openfold3 \
      --import-mode=prepend -o cache_dir=/tmp/pytest-cache \
      --junitxml=/evidence/standalone-gpu.xml -ra > /evidence/standalone-gpu.log 2>&1
    "$PYTHON" -m twine check --strict "/wheel/$WHEEL_NAME" > /evidence/twine.log 2>&1
    "$PYTHON" -m pip freeze --all > /evidence/runtime-pip-freeze.txt
    cp "/wheel/$WHEEL_NAME" /evidence/
  '
printf 'Installed-wheel GPU evidence: %s\n' "$OUTPUT_ROOT"
