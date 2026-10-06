# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the optional provider's layout and runtime boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import openfold3.core.kernels.protenix as provider
from openfold3.core.kernels.protenix._layout import _normalize_inputs


def _inputs(shape=(3, 2, 5, 16)):
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16) for _ in range(3))
    mask = torch.zeros((3, 1, 1, 5), dtype=torch.bfloat16)
    bias = torch.randn((1, 2, 5, 5), dtype=torch.bfloat16)
    return q, k, v, mask, bias


def test_provider_import_does_not_import_torch_triton_or_global_protenix():
    # Isolate this provider from the unrelated optional DeepSpeed import in
    # openfold3.__init__, which may itself load Torch and Triton.
    kernel_root = Path(provider.__file__).parents[1]
    script = f"""
import sys
import types
from pathlib import Path
kernel_root = Path({str(kernel_root)!r})
for name, path in [
    ("openfold3", kernel_root.parents[1]),
    ("openfold3.core", kernel_root.parent),
    ("openfold3.core.kernels", kernel_root),
]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
import openfold3.core.kernels.protenix
assert "torch" not in sys.modules
assert "triton" not in sys.modules
assert not any(n == "protenix" or n.startswith("protenix.") for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("separate_bias", [False, True])
def test_4d_layout_keeps_chunk_row_bias(separate_bias):
    q, k, v, mask, bias = _inputs()
    if separate_bias:
        bias = torch.arange(3 * 2 * 5 * 5).reshape(3, 2, 5, 5)
    nq, _, _, nm, nb, original = _normalize_inputs(q, k, v, mask, bias)
    assert original == tuple(q.shape)
    assert tuple(nq.shape) == ((3, 1, 2, 5, 16) if separate_bias else (1, 3, 2, 5, 16))
    assert torch.equal(nq.reshape(q.shape), q)
    assert nm.shape == ((3, 1, 1, 1, 5) if separate_bias else (1, 3, 1, 1, 5))
    assert torch.equal(nb[:, 0], bias) if separate_bias else torch.equal(nb[0], bias)


def test_5d_batch_broadcast_reduces_pair_bias_gradients():
    q = torch.randn(2, 3, 2, 5, 16)
    mask = torch.zeros(1, 3, 1, 1, 5)
    bias = torch.randn(1, 1, 2, 5, 5, requires_grad=True)
    _, _, _, nm, nb, _ = _normalize_inputs(q, q, q, mask, bias)
    assert nm.shape == (2, 3, 1, 1, 5)
    nb.sum().backward()
    assert torch.equal(bias.grad, torch.full_like(bias, 2))


def test_6d_template_broadcast_preserves_distinct_batch_bias():
    q = torch.randn(2, 3, 4, 2, 5, 16)
    mask = torch.zeros(2, 1, 4, 1, 1, 5)
    bias = torch.arange(2 * 2 * 5 * 5, dtype=torch.float32).reshape(2, 1, 1, 2, 5, 5)
    bias.requires_grad_()
    nq, _, _, nm, nb, _ = _normalize_inputs(q, q, q, mask, bias)
    assert nq.shape == (6, 4, 2, 5, 16)
    assert nm.shape == (6, 4, 1, 1, 5)
    assert torch.equal(nb[:3], bias[0].expand(3, 1, 2, 5, 5))
    assert torch.equal(nb[3:], bias[1].expand(3, 1, 2, 5, 5))
    nb.sum().backward()
    assert torch.equal(bias.grad, torch.full_like(bias, 3))


@pytest.mark.parametrize(
    ("field", "transform", "message"),
    [
        (0, lambda t: t[..., :-1, :], "identical shapes"),
        (0, lambda t: t.float(), "bfloat16"),
        (4, lambda t: t.float(), "bfloat16"),
        (3, lambda t: t.bool(), "additive mask must"),
        (3, lambda t: t.requires_grad_(), "trainable additive mask"),
        (2, lambda t: t.to("meta"), "same CUDA device"),
    ],
)
def test_validation_rejects_invalid_inputs_before_loading_kernel(
    field, transform, message
):
    inputs = list(_inputs())
    inputs[field] = transform(inputs[field])
    with patch.object(provider, "_native_function") as native:
        with pytest.raises((ValueError, TypeError), match=message):
            provider.attention_scaled(*inputs)
        native.assert_not_called()


def test_cpu_execution_is_explicitly_rejected():
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        provider.attention_scaled(*_inputs())


@pytest.mark.parametrize("dimension", [8, 24])
def test_invalid_head_dimension_is_rejected(dimension):
    with pytest.raises(ValueError, match="power of two"):
        provider.attention_scaled(*_inputs((3, 2, 5, dimension)))


def test_adapter_preserves_scaled_q_and_makes_dout_contiguous():
    observed = {}

    class Echo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, mask, bias, deterministic):
            observed["q"] = q.detach().clone()
            observed["deterministic"] = deterministic
            return q.contiguous().clone()

        @staticmethod
        def backward(ctx, dout):
            observed["contiguous"] = dout.is_contiguous()
            return dout, None, None, None, None, None

    inputs = list(_inputs())
    inputs[0].requires_grad_()
    with (
        patch.object(provider, "_validate_inputs"),
        patch.object(provider, "_native_function", return_value=Echo),
    ):
        output = provider.attention_scaled(*inputs)
    assert output.shape == (3, 5, 2, 16)
    assert torch.equal(observed["q"], inputs[0].unsqueeze(0).transpose(-2, -3))
    assert observed["deterministic"] is False
    dout = torch.randn(3, 5, 2, 32, dtype=torch.bfloat16)[..., ::2]
    assert not dout.is_contiguous()
    output.backward(dout)
    assert observed["contiguous"]
    assert torch.equal(inputs[0].grad, dout.transpose(-2, -3))


def test_runtime_failure_propagates():
    inputs = _inputs()
    native = SimpleNamespace(
        apply=lambda *_: (_ for _ in ()).throw(RuntimeError("kernel failed"))
    )
    with (
        patch.object(provider, "_validate_inputs"),
        patch.object(provider, "_native_function", return_value=native),
        pytest.raises(RuntimeError, match="kernel failed"),
    ):
        provider.attention_scaled(*inputs)


def test_missing_triton_has_actionable_error():
    provider._native_function.cache_clear()
    with (
        patch.object(
            provider, "import_module", side_effect=ModuleNotFoundError(name="triton")
        ),
        pytest.raises(ImportError, match="GPU dependencies"),
    ):
        provider._native_function()
    provider._native_function.cache_clear()


def test_vendored_file_integrity_matches_provenance():
    root = Path(provider.__file__).parent
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    assert manifest["commit"] == "4c355be4553512f72453ecbfb65e69f4c35d1413"
    assert manifest["strict_upstream"] is True
    assert manifest["algorithm_changes"] is False
    for row in manifest["files"]:
        assert (
            hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
            == row["vendored_sha256"]
        )

        upstream_bytes = (root / row["path"]).read_bytes()
        for original, vendored in manifest["import_rewrites"].items():
            upstream_bytes = upstream_bytes.replace(
                vendored.encode(), original.encode()
            )
        assert hashlib.sha256(upstream_bytes).hexdigest() == row["upstream_sha256"]
