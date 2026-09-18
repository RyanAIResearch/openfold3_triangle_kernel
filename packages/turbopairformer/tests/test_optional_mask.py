# SPDX-License-Identifier: Apache-2.0

"""Optional masks preserve the explicit unmasked operator and its gradients."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from turbopairformer import _triangle_attention as facade
from turbopairformer import (
    is_triangle_attention_supported,
    turbo_attention,
    turbo_trimul,
)


def _assert_bits_equal(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def test_none_mask_support_checks_metadata_without_allocation_or_gpu_query():
    mode = FakeTensorMode()

    def metadata(shape):
        return FakeTensor(
            mode,
            torch.empty(shape, device="meta", dtype=torch.bfloat16),
            torch.device("cuda", 0),
        )

    q = metadata((2, 1, 3, 2, 65, 7))
    bias = metadata((1, 1, 1, 2, 65, 65))
    with (
        patch.dict(facade._CAPABILITY_CACHE, {0: (9, 0)}),
        patch.dict(facade._SUPPORT_CACHE, {}, clear=True),
        patch("torch.zeros", side_effect=AssertionError("allocated a mask")),
        patch("torch.empty", side_effect=AssertionError("allocated a tensor")),
        patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("queried GPU"),
        ),
    ):
        for _ in range(2):  # also exercise the cached None-mask metadata key
            assert is_triangle_attention_supported(q, q, q, pair_bias=bias) == (
                True,
                "supported",
            )
            assert is_triangle_attention_supported(q, q, q, None, bias) == (
                True,
                "supported",
            )


@pytest.mark.parametrize("bad_input", ("q_type", "bias_type", "cpu"))
def test_invalid_optional_mask_calls_reject_before_allocation_or_cuda(bad_input):
    q = torch.zeros((1, 2, 1, 8, 16), dtype=torch.bfloat16)
    bias = torch.zeros((1, 1, 1, 8, 8), dtype=torch.bfloat16)
    q_arg = object() if bad_input == "q_type" else q
    bias_arg = object() if bad_input == "bias_type" else bias
    with (
        patch("torch.zeros", side_effect=AssertionError("allocated a mask")),
        patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("queried GPU"),
        ),
        patch.object(
            facade, "_forward_extension", side_effect=AssertionError("loaded CUDA")
        ),
    ):
        supported, reason = is_triangle_attention_supported(
            q_arg, q, q, pair_bias=bias_arg
        )
        assert not supported
        assert "Tensor instances" in reason or "CUDA tensors" in reason
        with pytest.raises(ValueError, match="unsupported TurboPairFormer"):
            turbo_attention(q_arg, q, q, pair_bias=bias_arg)


@pytest.mark.parametrize("keywords", ({}, {"pair_bias": None}))
def test_pair_bias_is_still_required(keywords):
    q = torch.zeros((1, 2, 1, 8, 16), dtype=torch.bfloat16)
    with patch("torch.zeros", side_effect=AssertionError("allocated a mask")):
        supported, reason = is_triangle_attention_supported(q, q, q, **keywords)
        assert not supported and "pair_bias is required" in reason
        with pytest.raises(ValueError, match="pair_bias is required"):
            turbo_attention(q, q, q, **keywords)


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="installed Hopper sm_90a CUDA kernels are required",
)


@requires_hopper
@pytest.mark.parametrize(
    "prefix,outer,heads,sequence,dimension,bias_prefix,bias_dtype",
    (
        ((1,), 2, 2, 64, 16, (1,), torch.bfloat16),
        ((2, 1), 3, 2, 65, 7, (1, 1), torch.float32),
        ((1, 2), 1, 1, 31, 32, (1,), torch.bfloat16),
    ),
)
def test_attention_omitted_mask_matches_zero_outputs_and_gradients(
    prefix, outer, heads, sequence, dimension, bias_prefix, bias_dtype
):
    generator = torch.Generator(device="cuda").manual_seed(20260918)
    shape = prefix + (outer, heads, sequence, dimension)
    originals = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    ]
    originals.append(
        torch.randn(
            bias_prefix + (1, heads, sequence, sequence),
            device="cuda",
            dtype=bias_dtype,
            generator=generator,
        )
    )
    zero_mask = torch.zeros(
        prefix + (outer, 1, 1, sequence), device="cuda", dtype=torch.float32
    )
    dout = torch.randn(
        prefix + (outer, sequence, heads, dimension),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    def run(mask_mode):
        q, k, v, bias = [value.detach().clone().requires_grad_() for value in originals]
        if mask_mode == "explicit":
            output = turbo_attention(q, k, v, zero_mask, bias)
        elif mask_mode == "none":
            output = turbo_attention(q, k, v, res_mask=None, pair_bias=bias)
        else:
            output = turbo_attention(q, k, v, pair_bias=bias)
        gradients = torch.autograd.grad(output, (q, k, v, bias), dout)
        return output, *gradients

    expected = run("explicit")
    for mask_mode in ("omitted", "none"):
        for actual, reference in zip(run(mask_mode), expected, strict=True):
            assert torch.isfinite(actual).all()
            _assert_bits_equal(actual, reference)


class _TriangleWeights(torch.nn.Module):
    """Minimal weight protocol, independent of an OpenFold3 installation."""

    def __init__(self, channels, outgoing):
        super().__init__()
        self.c_z = self.c_hidden = channels
        self._outgoing = outgoing
        self.layer_norm_in = torch.nn.LayerNorm(channels, eps=1e-5)
        self.layer_norm_out = torch.nn.LayerNorm(channels, eps=1e-5)
        for name in (
            "linear_a_p",
            "linear_a_g",
            "linear_b_p",
            "linear_b_g",
            "linear_g",
            "linear_z",
        ):
            setattr(self, name, torch.nn.Linear(channels, channels, bias=False))


@requires_hopper
@pytest.mark.parametrize("outgoing", (False, True))
def test_trimul_omitted_mask_matches_ones_outputs_and_gradients(outgoing):
    torch.manual_seed(20260918)
    module = _TriangleWeights(32, outgoing).cuda()
    original = torch.randn((1, 101, 101, 32), device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(original.shape[:-1], device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(original)

    def run(explicit):
        z = original.detach().clone().requires_grad_()
        output = turbo_trimul(module, z, ones) if explicit else turbo_trimul(module, z)
        gradients = torch.autograd.grad(output, (z, *module.parameters()), dout)
        return output, *gradients

    for actual, reference in zip(run(False), run(True), strict=True):
        assert torch.isfinite(actual).all()
        _assert_bits_equal(actual, reference)
