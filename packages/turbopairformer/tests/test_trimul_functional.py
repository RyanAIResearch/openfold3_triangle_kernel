# SPDX-License-Identifier: Apache-2.0
"""Tensor API validation, packed-weight autograd, and installed-Hopper parity."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from turbopairformer import _trimul, turbo_trimul
from turbopairformer.trimul_support import (
    SupportResult,
    check_trimul_tensor_support,
)

WEIGHT_SHAPES = {
    "norm_in_weight": lambda c: (c,),
    "norm_in_bias": lambda c: (c,),
    "p_in_weight": lambda c: (2 * c, c),
    "g_in_weight": lambda c: (2 * c, c),
    "norm_out_weight": lambda c: (c,),
    "norm_out_bias": lambda c: (c,),
    "p_out_weight": lambda c: (c, c),
    "g_out_weight": lambda c: (c, c),
}


def _weights(channels, device="cpu"):
    return {
        name: (torch.randn(shape(channels), device=device) / channels**0.5)
        .detach()
        .requires_grad_()
        for name, shape in WEIGHT_SHAPES.items()
    }


def _reference_op(
    z, mask, ap, ag, bp, bg, gate, proj, niw, nib, now, nob, outgoing, save_state
):
    """CPU stand-in at the existing custom-op boundary, with its actual ABI."""
    z_ln = F.layer_norm(z.float(), (z.shape[-1],), niw, nib, 1e-5)
    a = F.linear(z_ln, ap) * F.linear(z_ln, ag).sigmoid() * mask[..., None]
    b = F.linear(z_ln, bp) * F.linear(z_ln, bg).sigmoid() * mask[..., None]
    equation = "...ikc,...jkc->...ijc" if outgoing else "...kic,...kjc->...ijc"
    pair = torch.einsum(equation, a, b)
    normalized = F.layer_norm(pair, (pair.shape[-1],), now, nob, 1e-5)
    return ((F.linear(normalized, proj) * F.linear(z_ln, gate).sigmoid()).to(z.dtype),)


def _packed_reference(z, mask, direction, weights):
    """Independent reference using packed projection/gate matrices directly."""
    c = z.shape[-1]
    normalized = F.layer_norm(
        z.float(), (c,), weights["norm_in_weight"], weights["norm_in_bias"], 1e-5
    )
    ab = F.linear(normalized, weights["p_in_weight"])
    ab = ab * F.linear(normalized, weights["g_in_weight"]).sigmoid()
    a, b = (ab * mask[..., None]).chunk(2, dim=-1)
    a, b = a.movedim(-1, -3), b.movedim(-1, -3)
    if direction == "outgoing":
        pair = a @ b.transpose(-1, -2)
    else:
        pair = a.transpose(-1, -2) @ b
    pair = F.layer_norm(
        pair.movedim(-3, -1),
        (c,),
        weights["norm_out_weight"],
        weights["norm_out_bias"],
        1e-5,
    )
    return (
        F.linear(pair, weights["p_out_weight"])
        * F.linear(normalized, weights["g_out_weight"]).sigmoid()
    ).to(z.dtype)


@pytest.mark.parametrize("direction", ("outgoing", "incoming"))
@pytest.mark.parametrize("input_grad", (True, False))
def test_tensor_api_preserves_packed_weight_gradients_and_batch_prefix(
    direction, input_grad
):
    torch.manual_seed(773)
    z = torch.randn(2, 1, 5, 5, 32, dtype=torch.bfloat16, requires_grad=input_grad)
    weights = _weights(32)
    mask = torch.ones(5, 5)
    mask[:, -1] = 0
    with (
        patch(
            "turbopairformer.trimul_support.check_trimul_tensor_support",
            return_value=SupportResult(True, "supported", "CPU boundary test"),
        ),
        patch.object(_trimul, "_trimul_fwd_v1", side_effect=_reference_op),
    ):
        actual = turbo_trimul(z, direction=direction, mask=mask, **weights)
        with torch.no_grad():
            inference = turbo_trimul(z, direction=direction, mask=mask, **weights)
        assert not inference.requires_grad
        torch.testing.assert_close(inference, actual.detach(), rtol=0, atol=0)
    expected = _packed_reference(z, mask, direction, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dout = torch.randn_like(actual)
    targets = (z, *weights.values()) if input_grad else tuple(weights.values())
    actual_gradients = torch.autograd.grad(actual, targets, dout)
    expected_gradients = torch.autograd.grad(expected, targets, dout)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-5, atol=2e-5
        )
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum() > 0


def test_demo_weights_follow_torch_rng_without_hidden_persistent_parameters():
    z = torch.randn(1, 5, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    calls = []

    def record(*args):
        calls.append(args)
        return _reference_op(*args)

    with (
        patch(
            "turbopairformer.trimul_support.check_trimul_tensor_support",
            return_value=SupportResult(True, "supported", "CPU boundary test"),
        ),
        patch.object(_trimul, "_trimul_fwd_v1", side_effect=record),
    ):
        torch.manual_seed(61)
        first = turbo_trimul(z)
        second = turbo_trimul(z=z)
        torch.manual_seed(61)
        replay = turbo_trimul(z, "outgoing")
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert not torch.equal(first, second)
    assert first.requires_grad
    first.sum().backward()
    assert torch.isfinite(z.grad).all()
    for tensor in calls[0][2:12]:
        assert tensor.dtype == torch.float32
        assert not tensor.requires_grad
    assert torch.equal(calls[0][8], torch.ones(32))
    assert torch.equal(calls[0][9], torch.zeros(32))
    assert torch.equal(calls[0][10], torch.ones(32))
    assert torch.equal(calls[0][11], torch.zeros(32))


def test_invalid_cpu_tensor_does_not_allocate_demo_weights_or_load_cuda():
    z = torch.zeros(1, 128, 128, 32, dtype=torch.bfloat16)
    before = torch.random.get_rng_state().clone()
    with (
        patch.object(_trimul, "_get_ext", side_effect=AssertionError("loaded CUDA")),
        pytest.raises(ValueError, match="unsupported.*z_device"),
    ):
        turbo_trimul(z)
    assert torch.equal(before, torch.random.get_rng_state())


def test_partial_defaults_preserve_supplied_parameters_and_gradients():
    z = torch.randn(1, 5, 5, 32, dtype=torch.bfloat16)
    supplied = _weights(32)
    del supplied["g_in_weight"]
    del supplied["norm_out_bias"]
    original = {name: value.detach().clone() for name, value in supplied.items()}
    calls = []

    def record(*args):
        calls.append(args)
        return _reference_op(*args)

    with (
        patch(
            "turbopairformer.trimul_support.check_trimul_tensor_support",
            return_value=SupportResult(True, "supported", "CPU boundary test"),
        ),
        patch.object(_trimul, "_trimul_fwd_v1", side_effect=record),
    ):
        values = []
        for _ in range(2):
            torch.manual_seed(413)
            output = turbo_trimul(z, g_in_weight=None, **supplied)
            gradients = torch.autograd.grad(
                output.float().sum(), tuple(supplied.values())
            )
            values.append((output, *gradients))
    assert not z.requires_grad
    for actual, replay in zip(*values, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, replay, rtol=0, atol=0)
    assert calls[0][6] is supplied["g_out_weight"]
    assert calls[0][7] is supplied["p_out_weight"]
    assert not calls[0][3].requires_grad  # Generated a-gate matrix slice.
    assert not calls[0][5].requires_grad  # Generated b-gate matrix slice.
    assert torch.equal(calls[0][11], torch.zeros(32))  # Omitted output norm bias.
    assert set(supplied) == set(original)
    for name, parameter in supplied.items():
        assert parameter.is_leaf and parameter.requires_grad
        torch.testing.assert_close(parameter.detach(), original[name], rtol=0, atol=0)


@pytest.fixture
def fake_hopper():
    mode = FakeTensorMode()

    def tensor(shape, dtype=torch.bfloat16, requires_grad=False):
        return FakeTensor(
            mode,
            torch.empty(shape, device="meta", dtype=dtype, requires_grad=requires_grad),
            torch.device("cuda", 0),
        )

    with (
        patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        patch("torch.cuda.get_device_name", return_value="NVIDIA H100"),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=132),
        ),
    ):
        yield tensor


def test_metadata_accepts_missing_weights_without_allocating_them(fake_hopper):
    z = fake_hopper((1, 128, 128, 32))
    with (
        patch("torch.empty", side_effect=AssertionError("allocated a parameter")),
        patch("torch.randn", side_effect=AssertionError("allocated a parameter")),
        patch("torch.ones", side_effect=AssertionError("allocated a parameter")),
        patch("torch.zeros", side_effect=AssertionError("allocated a parameter")),
    ):
        result = check_trimul_tensor_support(z)
    assert result, result.reason


@pytest.mark.parametrize(
    "invalid",
    (
        "direction",
        "eps",
        "shape",
        "dtype",
        "mask_grad",
        "short_sequence",
    ),
)
def test_metadata_rejects_unsupported_functional_inputs(fake_hopper, invalid):
    z = fake_hopper((1, 128, 128, 32))
    kwargs = {}
    if invalid == "direction":
        kwargs["direction"] = "sideways"
    elif invalid == "eps":
        kwargs["eps"] = 1e-4
    elif invalid == "shape":
        kwargs["p_in_weight"] = fake_hopper((32, 32), torch.float32)
    elif invalid == "dtype":
        kwargs["p_in_weight"] = fake_hopper((64, 32), torch.bfloat16)
    elif invalid == "mask_grad":
        kwargs["mask"] = fake_hopper((128, 128), torch.float32, True)
    else:
        z = fake_hopper((1, 100, 100, 32))
    result = check_trimul_tensor_support(z, **kwargs)
    assert not result
    with (
        patch.object(
            _trimul, "_trimul_fwd_v1", side_effect=AssertionError("launched CUDA")
        ),
        pytest.raises((ValueError, TypeError)),
    ):
        turbo_trimul(z, **kwargs)


def _assert_rejected_without_side_effects(z, kwargs, code):
    result = check_trimul_tensor_support(z, **kwargs)
    assert not result and result.code == code, result
    before = torch.random.get_rng_state().clone()
    with (
        patch("torch.randn", side_effect=AssertionError("allocated demo weights")),
        patch("torch.ones", side_effect=AssertionError("allocated demo weights")),
        patch("torch.zeros", side_effect=AssertionError("allocated demo weights")),
        patch.object(
            _trimul, "_trimul_fwd_v1", side_effect=AssertionError("launched CUDA")
        ),
        patch.object(_trimul, "_get_ext", side_effect=AssertionError("loaded CUDA")),
        pytest.raises(ValueError, match=rf"unsupported \[{code}\]"),
    ):
        turbo_trimul(z, **kwargs)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("name", tuple(WEIGHT_SHAPES))
def test_each_functional_parameter_rejects_non_tensor_before_allocation(
    fake_hopper, name
):
    z = fake_hopper((1, 128, 128, 32))
    _assert_rejected_without_side_effects(z, {name: object()}, "parameter_type")


@pytest.mark.parametrize(
    "invalid", ("parameter_device", "parameter_layout", "mask_broadcast")
)
def test_functional_rejects_bad_device_layout_and_mask_before_allocation(
    fake_hopper, invalid
):
    z = fake_hopper((1, 128, 128, 32))
    if invalid == "parameter_device":
        kwargs = {"p_out_weight": torch.zeros(32, 32)}
    elif invalid == "parameter_layout":
        kwargs = {"g_in_weight": fake_hopper((32, 64), torch.float32).transpose(0, 1)}
    else:
        kwargs = {"mask": fake_hopper((127, 128), torch.float32)}
    _assert_rejected_without_side_effects(z, kwargs, invalid)


@pytest.mark.parametrize("style", ("positional", "mixed", "keywords"))
def test_legacy_module_calls_keep_dispatch_and_aliases(style):
    module = object()
    z = torch.zeros(1, 128, 128, 32, dtype=torch.bfloat16)
    mask = torch.ones(128, 128)
    with (
        patch(
            "turbopairformer.trimul_support.check_trimul_support",
            return_value=SupportResult(True, "supported", "legacy dispatch"),
        ),
        patch.object(_trimul, "_run_supported_trimul", return_value=z) as run,
    ):
        if style == "positional":
            result = turbo_trimul(module, z, mask)
        elif style == "mixed":
            result = turbo_trimul(module, z=z, mask=mask)
        else:
            result = turbo_trimul(module=module, z=z, mask=mask)
    assert result is z
    run.assert_called_once_with(module, z, mask)


@pytest.mark.parametrize(
    "call",
    (
        lambda z: turbo_trimul(z, z=z),
        lambda z: turbo_trimul(z, "incoming", direction="outgoing"),
        lambda z: turbo_trimul(z, "outgoing", None, mask=None),
        lambda z: turbo_trimul(object(), z, direction="incoming"),
        lambda z: turbo_trimul(object(), z, p_in_weight=torch.ones(64, 32)),
    ),
)
def test_duplicate_or_mixed_contracts_fail_before_execution(call):
    z = torch.zeros(1, 128, 128, 32, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        call(z)


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="installed Hopper sm_90a CUDA kernels are required",
)


@requires_hopper
@pytest.mark.parametrize("outgoing", (True, False))
def test_installed_functional_and_legacy_outputs_and_all_gradients_match(outgoing):
    from test_optional_mask import _TriangleWeights

    torch.manual_seed(387)
    module = _TriangleWeights(32, outgoing).cuda()
    z = torch.randn(
        1, 128, 128, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.ones(128, 128, device="cuda")
    mask[:, -7:] = 0
    weights = dict(
        p_in_weight=torch.cat((module.linear_a_p.weight, module.linear_b_p.weight)),
        g_in_weight=torch.cat((module.linear_a_g.weight, module.linear_b_g.weight)),
        p_out_weight=module.linear_z.weight,
        g_out_weight=module.linear_g.weight,
        norm_in_weight=module.layer_norm_in.weight,
        norm_in_bias=module.layer_norm_in.bias,
        norm_out_weight=module.layer_norm_out.weight,
        norm_out_bias=module.layer_norm_out.bias,
    )
    functional = turbo_trimul(
        z, direction="outgoing" if outgoing else "incoming", mask=mask, **weights
    )
    legacy = turbo_trimul(module, z, mask)
    dout = torch.randn_like(legacy)
    targets = (z, *module.parameters())
    values = (functional, *torch.autograd.grad(functional, targets, dout))
    reference = (legacy, *torch.autograd.grad(legacy, targets, dout))
    for actual, expected in zip(values, reference, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


@requires_hopper
@pytest.mark.parametrize("direction", ("outgoing", "incoming"))
def test_installed_demo_weights_support_backward_and_seeded_replay(direction):
    z = torch.randn(
        1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    values = []
    for _ in range(2):
        torch.manual_seed(95)
        output = turbo_trimul(z, direction=direction)
        (gradient,) = torch.autograd.grad(output.float().square().mean(), z)
        values.append((output, gradient))
    for actual, replay in zip(*values, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.equal(
            actual.contiguous().view(torch.uint8), replay.contiguous().view(torch.uint8)
        )


@requires_hopper
@pytest.mark.parametrize("channels", (64, 96))
@pytest.mark.parametrize("sequence,prefix", ((101, (2, 1)), (129, (1, 2))))
@pytest.mark.parametrize("outgoing", (True, False))
def test_installed_packed_leaf_gradients_match_module_for_tails_and_strided_batches(
    channels, sequence, prefix, outgoing
):
    from test_optional_mask import _TriangleWeights

    torch.manual_seed(871 + channels + sequence + int(outgoing))
    module = _TriangleWeights(channels, outgoing).cuda()
    z = torch.randn(
        *prefix, sequence, sequence, channels, device="cuda", dtype=torch.bfloat16
    ).transpose(-3, -2)
    assert not z.requires_grad and not z.is_contiguous()
    mask = torch.rand(sequence, sequence, device="cuda") > 0.15
    weights = {
        "norm_in_weight": module.layer_norm_in.weight.detach().clone().requires_grad_(),
        "norm_in_bias": module.layer_norm_in.bias.detach().clone().requires_grad_(),
        "p_in_weight": torch.cat((module.linear_a_p.weight, module.linear_b_p.weight))
        .detach()
        .requires_grad_(),
        "g_in_weight": torch.cat((module.linear_a_g.weight, module.linear_b_g.weight))
        .detach()
        .requires_grad_(),
        "norm_out_weight": module.layer_norm_out.weight.detach()
        .clone()
        .requires_grad_(),
        "norm_out_bias": module.layer_norm_out.bias.detach().clone().requires_grad_(),
        "p_out_weight": module.linear_z.weight.detach().clone().requires_grad_(),
        "g_out_weight": module.linear_g.weight.detach().clone().requires_grad_(),
    }
    assert all(value.is_leaf and value.requires_grad for value in weights.values())
    functional = turbo_trimul(
        z, direction="outgoing" if outgoing else "incoming", mask=mask, **weights
    )
    legacy = turbo_trimul(module, z, mask)
    assert functional.shape == z.shape and functional.dtype == z.dtype
    dout = torch.randn_like(functional)
    actual_gradients = torch.autograd.grad(functional, tuple(weights.values()), dout)
    named_parameters = dict(module.named_parameters())
    module_gradients = dict(
        zip(
            named_parameters,
            torch.autograd.grad(legacy, tuple(named_parameters.values()), dout),
            strict=True,
        )
    )
    expected_gradients = {
        "norm_in_weight": module_gradients["layer_norm_in.weight"],
        "norm_in_bias": module_gradients["layer_norm_in.bias"],
        "p_in_weight": torch.cat(
            (
                module_gradients["linear_a_p.weight"],
                module_gradients["linear_b_p.weight"],
            )
        ),
        "g_in_weight": torch.cat(
            (
                module_gradients["linear_a_g.weight"],
                module_gradients["linear_b_g.weight"],
            )
        ),
        "norm_out_weight": module_gradients["layer_norm_out.weight"],
        "norm_out_bias": module_gradients["layer_norm_out.bias"],
        "p_out_weight": module_gradients["linear_z.weight"],
        "g_out_weight": module_gradients["linear_g.weight"],
    }
    actual_values = (functional, *actual_gradients)
    expected_values = (legacy, *(expected_gradients[name] for name in weights))
    for name, actual, expected in zip(
        ("output", *weights), actual_values, expected_values, strict=True
    ):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        ), name
