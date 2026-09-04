#!/usr/bin/env bash
# Build the stablefold sm_90a wheel from a clean checkout.
#
# Required environment: the pinned Python 3.14 / Torch 2.10.0 / CUDA 12.9
# interpreter first on PATH, and CUDA_HOME naming a CUDA 12.9 toolkit
# (for a Pixi environment: CUDA_HOME=<checkout>/.pixi/envs/<env>).
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_ROOT"

if [[ "${STABLEFOLD_ALLOW_DIRTY_BUILD:-0}" != "1" ]] && \
   [[ -n "$(git status --porcelain --untracked-files=all 2>/dev/null)" ]]; then
  echo "refusing an uncommitted AOT build; commit first or set STABLEFOLD_ALLOW_DIRTY_BUILD=1" >&2
  exit 2
fi

export TORCH_CUDA_ARCH_LIST=9.0a
export PYTHONHASHSEED=0
export LC_ALL=C
export TZ=UTC
export STABLEFOLD_BUILD_AOT=1
export STABLEFOLD_AOT_GIT_COMMIT="${STABLEFOLD_AOT_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export STABLEFOLD_AOT_GIT_DIRTY="${STABLEFOLD_AOT_GIT_DIRTY:-$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo 1 || echo 0)}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD 2>/dev/null || date +%s)}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME must name the pinned CUDA 12.9 build environment" >&2
  exit 4
fi
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.9'
PYTHON_BIN="$(command -v python)"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 14), sys.version
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
PY

rm -rf build dist stablefold.egg-info
"$PYTHON_BIN" setup.py --quiet bdist_wheel

mapfile -t WHEELS < <(find dist -maxdepth 1 -type f -name 'stablefold-*.whl')
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "expected one wheel, found ${#WHEELS[@]}" >&2
  exit 3
fi
WHEEL="${WHEELS[0]}"
(
  cd dist
  sha256sum "$(basename "$WHEEL")" > "$(basename "$WHEEL").sha256"
)

# Clean-install smoke: import the wheel from a read-only directory with no
# compiler on PATH and no JIT permission, so a missing or ABI-mismatched
# binary fails here rather than on a training node.
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/stablefold-smoke.XXXXXX")"
trap 'chmod -R u+w "$SMOKE_ROOT" 2>/dev/null; rm -rf "$SMOKE_ROOT"' EXIT
"$PYTHON_BIN" -m zipfile -e "$WHEEL" "$SMOKE_ROOT/site"
chmod -R a-w "$SMOKE_ROOT/site"
# Run from the smoke root with -P so the checkout's own ``stablefold/`` source
# tree (cwd) cannot shadow the extracted wheel on sys.path.
cd "$SMOKE_ROOT"
env -u STABLEFOLD_ALLOW_JIT -u OPENFOLD3_TRIANGLE_ALLOW_JIT \
  PATH="$SMOKE_ROOT" PYTHONPATH="$SMOKE_ROOT/site" PYTHONDONTWRITEBYTECODE=1 \
  CUDA_HOME=/nonexistent TORCH_EXTENSIONS_DIR="$SMOKE_ROOT/no-jit-cache" \
  "$PYTHON_BIN" -P - <<'PY'
import json
import stablefold
from stablefold._aot_runtime import distributed_aot_identity
identity = distributed_aot_identity(
    ("stablefold_forward", "stablefold_delta", "stablefold_dkdv",
     "stablefold_dq_dbias", "stablefold_trimul")
)
print(json.dumps({"version": stablefold.__version__, "git": identity["git"],
                  "extensions": sorted(identity["extensions"])}, indent=2))
PY
cd "$SOURCE_ROOT"
echo "$SOURCE_ROOT/$WHEEL"
