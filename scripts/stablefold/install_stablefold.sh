#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build an AOT wheel, install it non-editably into this checkout's Pixi env,
# and verify the installed package from outside the checkout.
set -euo pipefail

if [[ -z ${STABLEFOLD_PYTHON:-} ]]; then
  cat >&2 <<'MESSAGE'
Set STABLEFOLD_PYTHON to a dedicated CPython 3.14 environment with Torch 2.10.0
and CUDA runtime 12.8. The older shared Pixi CUDA 12.9 environment is not used.
For example:
  python3.14 -m venv /absolute/path/to/stablefold-venv
  /absolute/path/to/stablefold-venv/bin/python -m pip install torch==2.10.0 setuptools wheel ninja patchelf
  export STABLEFOLD_PYTHON=/absolute/path/to/stablefold-venv/bin/python
  export CUDA_HOME=/absolute/path/to/cuda-12.9
Then rerun this script.
MESSAGE
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/_resolve_env.sh"
"$PREFIX/bin/python" - <<'PYTHON'
import sys

import torch

if (
    sys.version_info[:2] != (3, 14)
    or torch.__version__.split("+")[0] != "2.10.0"
    or torch.version.cuda != "12.8"
):
    raise SystemExit(
        "StableFold requires a dedicated CPython 3.14 / Torch 2.10.0 / "
        "CUDA runtime 12.8 build environment. Install torch==2.10.0 from PyPI "
        "there; do not replace packages in a shared training environment."
    )
PYTHON


BUILD_CUDA_HOME=${CUDA_HOME:-$PREFIX}
if [[ ! -x "$BUILD_CUDA_HOME/bin/nvcc" ]]; then
  echo "no nvcc in $BUILD_CUDA_HOME; set CUDA_HOME to the CUDA 12.9 toolkit" >&2
  exit 2
fi
if [[ -x "$PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="$PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
export PATH="$PREFIX/bin:$PATH"
export CUDA_HOME="$BUILD_CUDA_HOME"
export TORCH_CUDA_ARCH_LIST=9.0a
export STABLEFOLD_BUILD_AOT=1
export MAX_JOBS=${MAX_JOBS:-16}
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT
cd "$BUILD_DIR"
echo "building a StableFold wheel with five sm_90a extensions"
if command -v uv > /dev/null 2>&1; then
  uv build --python "$PREFIX/bin/python" --no-build-isolation --wheel \
    --out-dir "$BUILD_DIR/wheels" "$REPO_ROOT/packages/stablefold"
else
  "$PREFIX/bin/python" -m pip wheel --no-build-isolation --no-deps \
    --wheel-dir "$BUILD_DIR/wheels" "$REPO_ROOT/packages/stablefold"
fi
shopt -s nullglob
WHEELS=("$BUILD_DIR"/wheels/stablefold-*.whl)
if [[ ${#WHEELS[@]} != 1 ]]; then
  echo "expected exactly one StableFold wheel in $BUILD_DIR/wheels" >&2
  exit 2
fi
if command -v uv > /dev/null 2>&1; then
  uv pip install --python "$PREFIX/bin/python" --no-deps --reinstall "${WHEELS[0]}"
else
  "$PREFIX/bin/python" -m pip install --no-deps --force-reinstall "${WHEELS[0]}"
fi
export OPENFOLD3_TRIANGLE_PROVIDER=stablefold
export STABLEFOLD_ALLOW_JIT=0
export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
"$PREFIX/bin/python" "$REPO_ROOT/tools/stablefold_doctor.py" --expect-provider stablefold
