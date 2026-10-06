#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_ROOT"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "refusing an uncommitted AOT build; commit the complete release first" >&2
  exit 2
fi

export TORCH_CUDA_ARCH_LIST=9.0a
export PYTHONHASHSEED=0
export LC_ALL=C
export TZ=UTC
export ZERO_AR_DATE=1
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
export OPENFOLD3_AOT_GIT_COMMIT="$(git rev-parse HEAD)"
export OPENFOLD3_AOT_GIT_DIRTY=0
export OPENFOLD3_BUILD_AOT=1
export MAX_JOBS="${MAX_JOBS:-8}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME must name the pinned CUDA 12.9 build environment" >&2
  exit 4
fi
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
"$CUDACXX" --version | grep -q 'release 12.9'
export CXX="${CXX:-$(command -v c++)}"
"$CXX" --version >/dev/null
PYTHON_BIN="$(command -v python)"

python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 14), sys.version
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
PY

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/openfold3-aot-sm90a-build.XXXXXX")"
cleanup() {
  # The clean-install gate deliberately makes its target read-only. Restore
  # ownership permissions on this exact mktemp path before removing it.
  chmod -R u+w -- "$BUILD_ROOT" 2>/dev/null || true
  rm -r -- "$BUILD_ROOT"
}
trap cleanup EXIT
FRESH_SOURCE="$BUILD_ROOT/source"
BUILD_OUTPUT="$BUILD_ROOT/dist"
git clone --quiet --no-local --no-checkout "$SOURCE_ROOT" "$FRESH_SOURCE"
git -C "$FRESH_SOURCE" checkout --quiet --detach "$OPENFOLD3_AOT_GIT_COMMIT"
if [[ -n "$(git -C "$FRESH_SOURCE" status --porcelain --untracked-files=all)" ]]; then
  echo "fresh AOT build clone is unexpectedly dirty" >&2
  exit 5
fi

mkdir -p "$BUILD_OUTPUT"
cd "$FRESH_SOURCE"
"$PYTHON_BIN" -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "$BUILD_OUTPUT"

mapfile -t WHEELS < <(find "$BUILD_OUTPUT" -maxdepth 1 -type f -name '*.whl')
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "expected one wheel, found ${#WHEELS[@]}" >&2
  exit 3
fi
WHEEL="${WHEELS[0]}"
"$PYTHON_BIN" tools/attest_sm90_wheel.py \
  --require-binary-tools \
  "$WHEEL"

RUNTIME_ROOT="$BUILD_ROOT/runtime"
EMPTY_PATH="$BUILD_ROOT/no-build-tools"
mkdir -p "$RUNTIME_ROOT" "$EMPTY_PATH"
"$PYTHON_BIN" -m pip install \
  --no-deps \
  --target "$RUNTIME_ROOT" \
  "$WHEEL"
chmod -R a-w "$RUNTIME_ROOT"
# DeepSpeed is an optional dependency inherited from the build interpreter.
# Its import probes nvcc even though this smoke exercises only the prebuilt
# TurboPairFormer extensions. Select DeepSpeed's documented CPU
# accelerator for this child process so CUDA_HOME=/nonexistent continues to
# prove that the kernel wheel itself has no compiler dependency. This setting
# is deliberately scoped to the smoke and is never exported to training.
env -u OPENFOLD3_TRIANGLE_ALLOW_JIT \
  DS_ACCELERATOR=cpu \
  PATH="$EMPTY_PATH" \
  PYTHONPATH="$RUNTIME_ROOT" \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_HOME=/nonexistent \
  TORCH_EXTENSIONS_DIR="$RUNTIME_ROOT/no-jit-cache" \
  "$PYTHON_BIN" "$FRESH_SOURCE/tools/aot_clean_install_smoke.py" \
  > "$BUILD_OUTPUT/aot_clean_install_smoke.json"

mkdir -p "$SOURCE_ROOT/dist"
for artifact in \
  "$WHEEL" \
  "$WHEEL.sha256" \
  "$WHEEL.attestation.json" \
  "$BUILD_OUTPUT/aot_clean_install_smoke.json"; do
  cp "$artifact" "$SOURCE_ROOT/dist/"
done
echo "$SOURCE_ROOT/dist/$(basename "$WHEEL")"
