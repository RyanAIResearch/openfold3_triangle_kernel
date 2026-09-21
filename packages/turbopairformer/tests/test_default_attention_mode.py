# SPDX-License-Identifier: Apache-2.0

"""The default public call executes the installed SR and residual kernels."""

from __future__ import annotations

from collections import Counter
from functools import wraps

import pytest
import torch

from turbopairformer import _triangle_attention as facade
from turbopairformer import triangle_dkdv, triangle_dq_dbias, turbo_attention

requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="installed Hopper sm_90a CUDA kernels are required",
)


def _clear_sr_environment(monkeypatch):
    for prefix in ("TURBOPAIRFORMER", "OPENFOLD3"):
        for suffix in ("BACKWARD", "SHADOW"):
            monkeypatch.delenv(f"{prefix}_SR_{suffix}", raising=False)
    for name in (
        "TURBOPAIRFORMER_ALLOW_JIT",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT",
    ):
        monkeypatch.setenv(name, "0")


def _assert_bits_equal(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _inputs_and_gradient():
    generator = torch.Generator(device="cuda").manual_seed(20260918)
    shape = (1, 4, 2, 128, 32)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    inputs.append(
        torch.randn(
            (1, 1, 2, 128, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    gradient = torch.randn(
        (1, 4, 128, 2, 32),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return inputs, gradient


def _run(inputs, gradient):
    q, k, v, bias = [value.detach().clone().requires_grad_() for value in inputs]
    output = turbo_attention(q, k, v, pair_bias=bias)
    gradients = torch.autograd.grad(output, (q, k, v, bias), gradient)
    torch.cuda.synchronize(q.device)
    result = (output, *gradients)
    for value in result:
        assert torch.isfinite(value).all()
    return tuple(value.detach().clone() for value in result)


def _spy_aot_calls(monkeypatch):
    """Observe real extension calls, preserving their arguments and results."""
    calls = Counter()
    residuals = []
    forward = facade._forward_extension()
    delta = facade._delta_extension()
    dq_dbias = triangle_dq_dbias.load_extension()
    dkdv = triangle_dkdv.load_extension()

    def instrument(extension, name):
        original = getattr(extension, name)

        @wraps(original)
        def observed(*args, **kwargs):
            # The original installed CUDA extension performs the operation.
            result = original(*args, **kwargs)
            calls[name] += 1
            if name == "turbopairformer_triangle_forward_shadow_out":
                output, shadow = args[5], args[7]
                assert shadow.shape == output.shape
                assert shadow.dtype == output.dtype == torch.bfloat16
                assert shadow.device == output.device
                assert shadow.data_ptr() != output.data_ptr()
                assert torch.isfinite(shadow).all()
                assert torch.count_nonzero(shadow).item() > 0
                residuals.append(
                    {
                        "pointer": shadow.data_ptr(),
                        "value": shadow.detach().clone(),
                        "consumed": False,
                    }
                )
            elif name == "turbopairformer_triangle_delta_shadow_out":
                shadow = args[2]
                assert len(residuals) == 1
                residual = residuals[0]
                assert shadow.data_ptr() == residual["pointer"]
                _assert_bits_equal(shadow, residual["value"])
                residual["consumed"] = True
            return result

        monkeypatch.setattr(extension, name, observed)

    for extension, names in (
        (
            forward,
            (
                "turbopairformer_triangle_forward_out",
                "turbopairformer_triangle_forward_shadow_out",
            ),
        ),
        (
            delta,
            (
                "turbopairformer_triangle_delta_out",
                "turbopairformer_triangle_delta_shadow_out",
            ),
        ),
        (
            dq_dbias,
            (
                "turbopairformer_triangle_dq_dbias_out",
                "turbopairformer_triangle_sr_backward_out",
            ),
        ),
        (
            dkdv,
            (
                "turbopairformer_triangle_dkdv_out",
                "turbopairformer_triangle_dkdv_delta_true_out",
            ),
        ),
    ):
        for name in names:
            instrument(extension, name)
    return calls, residuals


@requires_hopper
def test_default_attention_executes_shadow_and_sr_like_explicit_flags(monkeypatch):
    _clear_sr_environment(monkeypatch)
    inputs, gradient = _inputs_and_gradient()
    calls, residuals = _spy_aot_calls(monkeypatch)
    expected_calls = {
        "turbopairformer_triangle_forward_shadow_out": 1,
        "turbopairformer_triangle_delta_shadow_out": 1,
        "turbopairformer_triangle_sr_backward_out": 1,
    }

    default = _run(inputs, gradient)
    assert calls == expected_calls
    assert len(residuals) == 1 and residuals[0]["consumed"]

    calls.clear()
    residuals.clear()
    monkeypatch.setenv("TURBOPAIRFORMER_SR_BACKWARD", "1")
    monkeypatch.setenv("TURBOPAIRFORMER_SR_SHADOW", "1")
    explicit = _run(inputs, gradient)
    assert calls == expected_calls
    assert len(residuals) == 1 and residuals[0]["consumed"]
    for actual, reference in zip(default, explicit, strict=True):
        _assert_bits_equal(actual, reference)


@requires_hopper
def test_explicit_classic_attention_bypasses_shadow_and_sr(monkeypatch):
    _clear_sr_environment(monkeypatch)
    monkeypatch.setenv("TURBOPAIRFORMER_SR_BACKWARD", "0")
    monkeypatch.setenv("TURBOPAIRFORMER_SR_SHADOW", "0")
    inputs, gradient = _inputs_and_gradient()
    calls, residuals = _spy_aot_calls(monkeypatch)

    _run(inputs, gradient)

    assert calls == {
        "turbopairformer_triangle_forward_out": 1,
        "turbopairformer_triangle_delta_out": 1,
        "turbopairformer_triangle_dkdv_delta_true_out": 1,
        "turbopairformer_triangle_dq_dbias_out": 1,
    }
    assert not residuals
