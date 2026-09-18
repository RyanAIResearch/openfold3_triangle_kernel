#!/usr/bin/env bash
# Build the standalone TurboPairFormer wheel in a clean temporary source directory.
# Requires Linux x86_64, CPython 3.14, PyPI Torch 2.10.0+cu128, CUDA toolkit 12.9.
# Output is a local linux_x86_64 wheel. PyPI requires a separate manylinux audit.
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_ROOT"
GIT_STATUS="$(git status --porcelain --untracked-files=all 2>/dev/null)"
if [[ "${TURBOPAIRFORMER_ALLOW_DIRTY_BUILD:-0}" != "1" ]] && [[ -n "$GIT_STATUS" ]]; then
  echo "refusing an uncommitted AOT build; commit first or set TURBOPAIRFORMER_ALLOW_DIRTY_BUILD=1" >&2
  exit 2
fi

export TORCH_CUDA_ARCH_LIST=9.0a
export PYTHONHASHSEED=0
export LC_ALL=C
export TZ=UTC
export TURBOPAIRFORMER_BUILD_AOT=1
export TURBOPAIRFORMER_AOT_GIT_COMMIT="$(git rev-parse HEAD)"
export TURBOPAIRFORMER_AOT_GIT_DIRTY=0
if [[ -n "$GIT_STATUS" ]]; then export TURBOPAIRFORMER_AOT_GIT_DIRTY=1; fi
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME must name the pinned CUDA 12.9 build toolkit" >&2
  exit 4
fi
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.9'
PYTHON_BIN="$(command -v python)"
if ! command -v patchelf >/dev/null 2>&1; then
  echo "install build tools first: python -m pip install setuptools wheel patchelf" >&2
  exit 4
fi
"$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch

assert sys.platform == "linux" and platform.machine() == "x86_64"
assert sys.version_info[:2] == (3, 14), sys.version
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
PY

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/turbopairformer-build.XXXXXX")"
trap 'chmod -R u+w "$BUILD_ROOT" 2>/dev/null; rm -rf "$BUILD_ROOT"' EXIT
"$PYTHON_BIN" - "$SOURCE_ROOT" "$BUILD_ROOT/source" <<'PY'
import shutil
import sys

shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns(
    "build", "dist", "*.egg-info", "*.so", "*.local.json", "__pycache__",
    ".pytest_cache", ".ruff_cache", "release_artifacts", ".git",
))
PY
cd "$BUILD_ROOT/source"
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$BUILD_ROOT/wheels" .
mapfile -t WHEELS < <(find "$BUILD_ROOT/wheels" -maxdepth 1 -type f -name 'turbopairformer-*.whl')
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "expected one wheel, found ${#WHEELS[@]}" >&2
  exit 3
fi
WHEEL="${WHEELS[0]}"
VALIDATOR_ARGS=()
if [[ "$TURBOPAIRFORMER_AOT_GIT_DIRTY" == "1" ]]; then VALIDATOR_ARGS+=(--allow-dirty); fi
"$PYTHON_BIN" tools/validate_wheel.py "$WHEEL" "${VALIDATOR_ARGS[@]}" > "$BUILD_ROOT/validation.json"

# Clean target install, imported outside the checkout with no compiler or JIT.
"$PYTHON_BIN" -m pip install --no-deps --no-compile --target "$BUILD_ROOT/site" "$WHEEL"
chmod -R a-w "$BUILD_ROOT/site"
cd "$BUILD_ROOT"
env -u TURBOPAIRFORMER_ALLOW_JIT -u OPENFOLD3_TRIANGLE_ALLOW_JIT -u STABLEFOLD_ALLOW_JIT -u TRIONCE_ALLOW_JIT \
  PATH="$BUILD_ROOT" PYTHONPATH="$BUILD_ROOT/site" PYTHONDONTWRITEBYTECODE=1 \
  CUDA_HOME=/nonexistent TORCH_EXTENSIONS_DIR="$BUILD_ROOT/no-jit-cache" \
  "$PYTHON_BIN" -P - <<'PY'
import importlib
import json
import turbopairformer

modules = ("triangle_forward", "triangle_delta", "triangle_dkdv",
           "triangle_dq_dbias", "trimul_cuda")
for name in modules:
    importlib.import_module(f"turbopairformer.{name}").load_extension()
print(json.dumps({"version": turbopairformer.__version__, "loaded": modules}, indent=2))
PY
mkdir -p "$SOURCE_ROOT/dist"
cp "$WHEEL" "$SOURCE_ROOT/dist/"
cp "$BUILD_ROOT/validation.json" "$SOURCE_ROOT/dist/$(basename "$WHEEL").validation.json"
cd "$SOURCE_ROOT/dist"
sha256sum "$(basename "$WHEEL")" > "$(basename "$WHEEL").sha256"
echo "$SOURCE_ROOT/dist/$(basename "$WHEEL")"
