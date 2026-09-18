# SPDX-License-Identifier: Apache-2.0

"""The short public names are callable regardless of internal import order."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "preload",
    (
        "",
        "import turbopairformer._trimul",
        "import turbopairformer._triangle_attention",
        "from turbopairformer import run_trimul, triangle_attention",
        "from turbopairformer import TurboTriangleMultiplicative, TurboTriangleAttention",
    ),
)
def test_public_functions_survive_internal_import_order(preload):
    script = f"""
import importlib
import sys
import turbopairformer
assert 'turbopairformer._trimul' not in sys.modules
assert 'turbopairformer._triangle_attention' not in sys.modules
{preload}
from turbopairformer import turbo_attention, turbo_trimul
from turbopairformer import TurboTriangleAttention, TurboTriangleMultiplicative
from turbopairformer import triangle_attention, run_trimul
assert callable(turbo_attention) and callable(turbo_trimul)
assert turbo_attention is triangle_attention is TurboTriangleAttention
assert turbo_trimul is run_trimul is TurboTriangleMultiplicative
assert turbo_attention.__name__ == 'turbo_attention'
assert turbo_trimul.__name__ == 'turbo_trimul'
assert not hasattr(turbopairformer, 'attention')
assert not hasattr(turbopairformer, 'trimul')
importlib.import_module('turbopairformer._trimul')
importlib.import_module('turbopairformer._triangle_attention')
assert turbopairformer.turbo_attention is turbo_attention
assert turbopairformer.turbo_trimul is turbo_trimul
assert not any(name.startswith('turbopairformer._ext.') for name in sys.modules)
assert 'turbopairformer._aot_runtime' not in sys.modules
assert 'openfold3' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_functions_reject_cpu_inputs_without_loading_extensions():
    script = """
import sys
import torch
from turbopairformer import turbo_attention, turbo_trimul
q = torch.zeros((1, 1, 1, 8, 32), dtype=torch.bfloat16)
mask = torch.zeros((1, 1, 1, 1, 8), dtype=torch.float32)
bias = torch.zeros((1, 1, 1, 8, 8), dtype=torch.bfloat16)
try:
    turbo_attention(q, q, q, mask, bias)
except ValueError as error:
    assert 'unsupported TurboPairFormer' in str(error)
else:
    raise AssertionError('turbo_attention accepted CPU input')
z = torch.zeros((1, 8, 8, 32), dtype=torch.bfloat16)
try:
    turbo_trimul(object(), z)
except ValueError as error:
    assert 'unsupported [z_device]' in str(error)
else:
    raise AssertionError('turbo_trimul accepted CPU input')
assert not any(name.startswith('turbopairformer._ext.') for name in sys.modules)
assert 'turbopairformer._aot_runtime' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
