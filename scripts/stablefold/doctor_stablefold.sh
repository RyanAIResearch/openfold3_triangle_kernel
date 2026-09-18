#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Preflight the StableFold install:
#
#   pixi run -e openfold3-cuda12-dev doctor-stablefold
#
# Also works when invoked directly; see _resolve_env.sh for how the interpreter
# is chosen.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_resolve_env.sh"

# Run from outside the checkout and by path, so nothing on sys.path can shadow
# the installed package.
DOCTOR_CWD=$(mktemp -d)
trap 'rm -rf "$DOCTOR_CWD"' EXIT
cd "$DOCTOR_CWD"
export OPENFOLD3_TRIANGLE_PROVIDER=${OPENFOLD3_TRIANGLE_PROVIDER:-${STABLEFOLD_PROVIDER:-stablefold}}
"$PREFIX/bin/python" "$REPO_ROOT/tools/stablefold_doctor.py" "$@"
