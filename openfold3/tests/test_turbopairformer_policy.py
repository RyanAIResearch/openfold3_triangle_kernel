"""CPU checks for the optional package boundary and stock attention fallback."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

import openfold3.core.model.primitives.attention as attention
from openfold3.core.kernels.turbopairformer_utils import (
    validate_turbopairformer_attention_policy,
)


def install_policy(monkeypatch, validate=None):
    module = ModuleType("turbopairformer.policy")
    if validate is not None:
        module.validate_attention_policy = validate
    monkeypatch.setitem(sys.modules, "turbopairformer.policy", module)


def test_historical_package_fails_with_source_install_instruction(monkeypatch):
    install_policy(monkeypatch)
    with pytest.raises(RuntimeError, match="run-current-turbopairformer"):
        validate_turbopairformer_attention_policy()


def test_package_policy_errors_are_preserved(monkeypatch):
    validate = Mock(side_effect=RuntimeError("single reconstruction requires SR=1"))
    install_policy(monkeypatch, validate)
    with pytest.raises(RuntimeError, match="single reconstruction requires SR=1"):
        validate_turbopairformer_attention_policy()
    validate.assert_called_once_with()


def test_invalid_policy_is_rejected_before_support_or_fallback(monkeypatch):
    install_policy(monkeypatch, Mock(side_effect=RuntimeError("SR=0 is disabled")))
    support = Mock(return_value=(False, "unsupported"))
    fallback = Mock()
    monkeypatch.setattr(
        attention, "is_triangle_attention_supported", support, raising=False
    )
    monkeypatch.setattr(attention, "_attention", fallback)
    with pytest.raises(RuntimeError, match="SR=0 is disabled"):
        attention._turbopairformer_triangle_attn(None, None, None, [None, None])
    support.assert_not_called()
    fallback.assert_not_called()


def test_supported_route_validates_then_preserves_public_arguments(monkeypatch):
    events = []
    install_policy(monkeypatch, lambda: events.append("policy"))

    def support(*args):
        events.append("support")
        return True, "supported"

    kernel = Mock(return_value="kernel result")
    monkeypatch.setattr(
        attention, "is_triangle_attention_supported", support, raising=False
    )
    monkeypatch.setattr(attention, "turbo_attention", kernel, raising=False)
    assert attention._turbopairformer_triangle_attn(1, 2, 3, [4, 5]) == "kernel result"
    assert events == ["policy", "support"]
    kernel.assert_called_once_with(1, 2, 3, 4, 5)


def test_unsupported_fp32_fallback_retains_scaling_layout_and_gradients(monkeypatch):
    validate = Mock()
    install_policy(monkeypatch, validate)
    monkeypatch.setattr(
        attention,
        "is_triangle_attention_supported",
        lambda *args: (False, "FP32 uses the host fallback"),
        raising=False,
    )
    generator = torch.Generator().manual_seed(12)
    q, k, v = [
        torch.randn(1, 2, 3, 5, 4, generator=generator, requires_grad=True)
        for _ in range(3)
    ]
    biases = [torch.zeros(1, 2, 1, 1, 5), torch.zeros(1, 1, 3, 5, 5)]
    actual = attention._turbopairformer_triangle_attn(q, k, v, biases)
    expected = attention._attention(q * 0.5, k, v, biases).transpose(-2, -3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    validate.assert_called_once_with()


def test_stock_attention_does_not_require_the_optional_policy(monkeypatch):
    validate = Mock(side_effect=RuntimeError("optional package unavailable"))
    monkeypatch.setattr(
        attention, "validate_turbopairformer_attention_policy", validate
    )
    layer = attention.Attention(8, 8, 8, 4, 2)
    inputs = torch.randn(1, 5, 8, requires_grad=True)
    output = layer(inputs, inputs)
    output.sum().backward()
    assert output.shape == inputs.shape
    assert inputs.grad is not None
    validate.assert_not_called()
