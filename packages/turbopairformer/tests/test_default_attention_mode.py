# SPDX-License-Identifier: Apache-2.0

"""Public SR routing, with CPU extension doubles and optional Hopper checks."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace

import pytest
import torch

from turbopairformer import _triangle_attention as facade
from turbopairformer import policy, triangle_dkdv, triangle_dq_dbias, turbo_attention

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


@pytest.fixture(autouse=True)
def isolated_attention_policy(monkeypatch):
    _clear_sr_environment(monkeypatch)
    policy._set_distributed_strict(False)
    yield
    policy._set_distributed_strict(False)


@pytest.fixture
def cpu_extensions(monkeypatch):
    """Exercise the real facade/autograd/helper with small CPU extension doubles.

    These operations only expose routing and saved-state use; they do not
    approximate or validate the CUDA attention arithmetic.
    """
    calls = Counter()
    state = {}

    def forward(q, k, v, mask, bias, output, lse):
        calls["forward"] += 1
        output.copy_(q.transpose(-2, -3))
        lse.zero_()

    def forward_shadow(q, k, v, mask, bias, output, lse, shadow):
        calls["forward_shadow"] += 1
        output.copy_(q.transpose(-2, -3))
        lse.zero_()
        shadow.fill_(0.5)
        state["shadow_pointer"] = shadow.data_ptr()

    def delta(gradient, output, result):
        calls["delta"] += 1
        result.copy_((gradient.float() * output.float()).sum(-1).transpose(-2, -1))

    def delta_shadow(gradient, output, shadow, result):
        calls["delta_shadow"] += 1
        assert shadow.data_ptr() == state["shadow_pointer"]
        result.copy_(
            (gradient.float() * (output.float() + shadow.float()))
            .sum(-1)
            .transpose(-2, -1)
        )

    def sr_backward(*args):
        calls["sr_backward"] += 1
        state["delta"] = args[5].clone()
        for value, output in enumerate(args[8:12], start=1):
            output.fill_(value)

    def forbidden_split(*args, **kwargs):
        pytest.fail("The public attention call reached split backward")

    forward_extension = SimpleNamespace(
        turbopairformer_triangle_forward_out=forward,
        turbopairformer_triangle_forward_shadow_out=forward_shadow,
    )
    delta_extension = SimpleNamespace(
        turbopairformer_triangle_delta_out=delta,
        turbopairformer_triangle_delta_shadow_out=delta_shadow,
    )
    backward_extension = SimpleNamespace(
        turbopairformer_triangle_sr_backward_out=sr_backward,
        turbopairformer_triangle_dq_dbias_out=forbidden_split,
    )
    monkeypatch.setattr(
        facade, "is_triangle_attention_supported", lambda *a: (True, "")
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(facade, "_forward_extension", lambda: forward_extension)
    monkeypatch.setattr(facade, "_delta_extension", lambda: delta_extension)
    monkeypatch.setattr(triangle_dq_dbias, "load_extension", lambda: backward_extension)
    monkeypatch.setattr(triangle_dkdv, "load_extension", forbidden_split)
    return calls, state


def _cpu_inputs():
    shape = (1, 2, 2, 64, 8)
    return (
        *(
            torch.full(shape, value, dtype=torch.bfloat16, requires_grad=True)
            for value in (2.0, 3.0, 4.0)
        ),
        torch.zeros((1, 1, 2, 64, 64), dtype=torch.bfloat16, requires_grad=True),
    )


def _cpu_backward(output, inputs):
    gradients = torch.autograd.grad(output, inputs, torch.ones_like(output))
    for value, (gradient, source) in enumerate(zip(gradients, inputs, strict=True), 1):
        assert gradient.shape == source.shape
        assert gradient.dtype == source.dtype
        assert torch.equal(gradient, torch.full_like(source, value))
    return gradients


@pytest.mark.parametrize("prefix", ("TURBOPAIRFORMER", "OPENFOLD3"))
def test_cpu_default_and_explicit_one_use_same_sr_routes(
    cpu_extensions, monkeypatch, prefix
):
    calls, state = cpu_extensions
    expected = {"forward_shadow": 1, "delta_shadow": 1, "sr_backward": 1}
    inputs = _cpu_inputs()
    default = turbo_attention(*inputs[:3], pair_bias=inputs[3])
    default_gradients = _cpu_backward(default, inputs)
    assert calls == expected
    assert torch.equal(state["delta"], torch.full_like(state["delta"], 20))

    calls.clear()
    for suffix in ("BACKWARD", "SHADOW"):
        monkeypatch.setenv(f"{prefix}_SR_{suffix}", "1")
    explicit_inputs = _cpu_inputs()
    explicit = turbo_attention(*explicit_inputs[:3], pair_bias=explicit_inputs[3])
    explicit_gradients = _cpu_backward(explicit, explicit_inputs)
    assert calls == expected
    assert torch.equal(default, explicit)
    for actual, reference in zip(explicit_gradients, default_gradients, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("name", (policy.SR_BACKWARD, "OPENFOLD3_SR_BACKWARD"))
def test_disabled_sr_rejects_before_support_check_or_extension(name, monkeypatch):
    monkeypatch.setenv(name, "0")

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled SR must be rejected before tensor validation or CUDA")

    monkeypatch.setattr(facade, "is_triangle_attention_supported", forbidden)
    monkeypatch.setattr(facade, "_forward_extension", forbidden)
    with pytest.raises(RuntimeError, match="single-reconstruction"):
        turbo_attention(None, None, None)


@pytest.mark.parametrize("name", (policy.SR_SHADOW, "OPENFOLD3_SR_SHADOW"))
def test_cpu_shadow_zero_still_uses_sr(cpu_extensions, monkeypatch, name):
    calls, state = cpu_extensions
    monkeypatch.setenv(name, "0")
    inputs = _cpu_inputs()
    output = turbo_attention(*inputs[:3], pair_bias=inputs[3])
    _cpu_backward(output, inputs)

    assert calls == {"forward": 1, "delta": 1, "sr_backward": 1}
    assert "shadow_pointer" not in state
    assert torch.equal(state["delta"], torch.full_like(state["delta"], 16))


@pytest.mark.parametrize("shadow", ("0", "1"))
def test_cpu_backward_uses_saved_shadow_after_environment_change(
    cpu_extensions, monkeypatch, shadow
):
    calls, state = cpu_extensions
    monkeypatch.setenv(policy.SR_SHADOW, shadow)
    inputs = _cpu_inputs()
    output = turbo_attention(*inputs[:3], pair_bias=inputs[3])
    monkeypatch.setenv(policy.SR_BACKWARD, "0")
    monkeypatch.setenv(policy.SR_SHADOW, "1" if shadow == "0" else "0")

    def forbidden_flag_read(*args):
        pytest.fail("Backward must use the saved forward state, not read flags again")

    monkeypatch.setattr(facade, "distributed_execution_flag", forbidden_flag_read)
    _cpu_backward(output, inputs)

    suffix = "_shadow" if shadow == "1" else ""
    assert calls == {f"forward{suffix}": 1, f"delta{suffix}": 1, "sr_backward": 1}
    assert torch.equal(
        state["delta"], torch.full_like(state["delta"], 20 if shadow == "1" else 16)
    )


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
def test_explicit_shadow_ablation_retains_sr(monkeypatch):
    _clear_sr_environment(monkeypatch)
    monkeypatch.setenv("TURBOPAIRFORMER_SR_SHADOW", "0")
    inputs, gradient = _inputs_and_gradient()
    calls, residuals = _spy_aot_calls(monkeypatch)

    _run(inputs, gradient)

    assert calls == {
        "turbopairformer_triangle_forward_out": 1,
        "turbopairformer_triangle_delta_out": 1,
        "turbopairformer_triangle_sr_backward_out": 1,
    }
    assert not residuals
