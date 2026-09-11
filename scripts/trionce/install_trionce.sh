#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# AOT-build and editable-install the Trionce sm_90a kernels into the Pixi
# environment that is running this task.
#
#   pixi run -e openfold3-cuda12-dev install-trionce
#
# This exists as a task rather than a pypi path dependency because the build
# needs three things a declarative dependency cannot express: the AOT opt-in,
# the exact architecture list, and no build isolation (the extension links the
# runtime Torch ABI).  Declared as a dependency, the package would install
# source-only and its loader would refuse to run -- worse than absent, because
# it would look installed.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_resolve_env.sh"
cd "$REPO_ROOT"

if [[ ! -x "$PREFIX/bin/nvcc" ]]; then
  echo "no nvcc in $PREFIX; the kernels need the CUDA 12.9 toolchain" >&2
  exit 2
fi

# nvcc drives a host compiler from PATH.  A system GCC older than the
# environment's conda toolchain cannot parse its libstdc++ headers.
if [[ -x "$PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="$PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$PREFIX/bin/x86_64-conda-linux-gnu-c++"
fi
export PATH="$PREFIX/bin:$PATH"
export CUDA_HOME="$PREFIX"
export TORCH_CUDA_ARCH_LIST=9.0a
export TRIONCE_BUILD_AOT=1
export MAX_JOBS=${MAX_JOBS:-16}

echo "building the sm_90a extensions; this takes roughly fifteen minutes"
if command -v uv > /dev/null 2>&1; then
  uv pip install --python "$PREFIX/bin/python" --no-build-isolation -e packages/trionce
else
  "$PREFIX/bin/python" -m pip install --no-build-isolation -e packages/trionce
fi

echo ""
exec bash "$REPO_ROOT/scripts/trionce/doctor_trionce.sh"
