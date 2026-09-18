# SPDX-License-Identifier: Apache-2.0
#
# Resolve the explicit Python environment or this checkout's Pixi prefix.  Sourced, not executed.
#
# Under `pixi run` CONDA_PREFIX already names the environment, but a stray
# CONDA_PREFIX from an ambient conda shell would otherwise be taken at face
# value and the tool would run against an interpreter with no torch.  Accept it
# only when it really is one of this checkout's environments.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
PIXI_ENVS="$REPO_ROOT/.pixi/envs"

PREFIX=""
if [[ -n ${TURBOPAIRFORMER_PYTHON:-} ]]; then
  if [[ ! -x "$TURBOPAIRFORMER_PYTHON" ]]; then
    echo "TURBOPAIRFORMER_PYTHON must name an executable Python interpreter" >&2
    exit 2
  fi
  PREFIX=$("$TURBOPAIRFORMER_PYTHON" -c 'import sys; print(sys.prefix)')
elif [[ -n ${CONDA_PREFIX:-} ]] && [[ "$(cd "$CONDA_PREFIX" 2>/dev/null && pwd -P)" == "$PIXI_ENVS"/* ]]; then
  PREFIX=$(cd "$CONDA_PREFIX" && pwd -P)
else
  PREFIX="$PIXI_ENVS/${TURBOPAIRFORMER_PIXI_ENV:-openfold3-cuda12-dev}"
fi

if [[ ! -x "$PREFIX/bin/python" ]]; then
  echo "no Python environment at $PREFIX" >&2
  echo "set TURBOPAIRFORMER_PYTHON to your virtualenv Python, or run pixi install -e ${TURBOPAIRFORMER_PIXI_ENV:-openfold3-cuda12-dev}" >&2
  exit 2
fi
