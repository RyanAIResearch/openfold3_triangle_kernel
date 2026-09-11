#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Preflight the Trionce install:
#
#   pixi run -e openfold3-cuda12-dev doctor-trionce
#
# Also works when invoked directly; see _resolve_env.sh for how the interpreter
# is chosen.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_resolve_env.sh"

# Run from outside the checkout and by path, so nothing on sys.path can shadow
# the installed package.
cd "$(mktemp -d)"
exec "$PREFIX/bin/python" "$REPO_ROOT/tools/trionce_doctor.py" "$@"
