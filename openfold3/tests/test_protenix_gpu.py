# SPDX-License-Identifier: Apache-2.0
"""CUDA regression coverage for the native Protenix training integration.

The numerical oracle uses the original host dimensions and PyTorch broadcasting,
not the provider's layout normalization. These tests do not measure performance.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Protenix integration requires CUDA"
)


@pytest.fixture(autouse=True)
def _cuda_test_state():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Protenix integration requires CUDA BF16 support")
    pytest.importorskip("triton")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # The independent FP32 oracle must not silently use TF32 products.
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1729)
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _reference(q, k, v, mask, pair_bias):
    """Host attention: Q is already scaled and both biases are additive."""
    scores = q @ k.transpose(-1, -2) + mask + pair_bias
    return (scores.softmax(dim=-1) @ v).transpose(-2, -3)


def _assert_numerical_close(actual, expected, *, name, relative=0.05, atol=0.004):
    assert actual.shape == expected.shape, name
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all(), f"{name}: nonfinite native result"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    difference = actual - expected
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-7)
    rel_l2 = (torch.linalg.vector_norm(difference) / denominator).item()
    assert rel_l2 <= relative, f"{name}: relative L2 {rel_l2:.6g} > {relative}"
    # A norm check alone can conceal a single corrupted element.
    max_error = difference.abs().max().item()
    max_allowed = atol + relative * expected.abs().max().item()
    assert max_error <= max_allowed, (
        f"{name}: maximum error {max_error:.6g} > {max_allowed:.6g}"
    )


def _make_inputs(layout, sequence, head_dim):
    heads, rows, batch, templates = 2, 3, 2, 2
    if layout == "unbatched_shared":
        q_shape = (rows, heads, sequence, head_dim)
        bias_shape = (1, heads, sequence, sequence)
        mask_shape = (rows, 1, 1, sequence)
    elif layout == "chunk_distinct":
        q_shape = (rows, heads, sequence, head_dim)
        bias_shape = (rows, heads, sequence, sequence)
        mask_shape = (rows, 1, 1, sequence)
    elif layout in ("batched_distinct", "batched_shared"):
        q_shape = (batch, rows, heads, sequence, head_dim)
        bias_batch = batch if layout == "batched_distinct" else 1
        bias_shape = (bias_batch, 1, heads, sequence, sequence)
        mask_shape = (batch, rows, 1, 1, sequence)
    elif layout == "template_broadcast":
        q_shape = (batch, templates, rows, heads, sequence, head_dim)
        bias_shape = (batch, 1, 1, heads, sequence, sequence)
        mask_shape = (batch, 1, rows, 1, 1, sequence)
    elif layout == "batch_broadcast_templates":
        q_shape = (batch, templates, rows, heads, sequence, head_dim)
        bias_shape = (1, templates, 1, heads, sequence, sequence)
        mask_shape = (1, templates, rows, 1, 1, sequence)
    else:
        raise AssertionError(f"Unknown layout: {layout}")

    def random_leaf(shape, scale=1.0):
        return (
            (torch.randn(shape, device="cuda") * scale)
            .to(torch.bfloat16)
            .requires_grad_()
        )

    q = random_leaf(q_shape, head_dim**-0.5)
    k, v = random_leaf(q_shape), random_leaf(q_shape)
    pair_bias = random_leaf(bias_shape, 0.3)
    mask = torch.zeros(mask_shape, device="cuda", dtype=torch.float32)
    mask[..., 2::5] = -1e4
    if layout in ("chunk_distinct", "template_broadcast"):
        mask = mask.to(torch.bfloat16)
    return q, k, v, mask, pair_bias


def _check_provider(inputs, loss_kind):
    from openfold3.core.kernels.protenix import attention_scaled

    q, k, v, mask, pair_bias = inputs
    native_leaves = (q, k, v, pair_bias)
    reference_leaves = tuple(
        tensor.detach().float().requires_grad_() for tensor in native_leaves
    )
    rq, rk, rv, rbias = reference_leaves
    actual = attention_scaled(q, k, v, mask, pair_bias)
    expected = _reference(rq, rk, rv, mask.float(), rbias)
    assert actual.dtype == torch.bfloat16
    _assert_numerical_close(actual, expected, name="output", relative=0.035)

    if loss_kind == "linear_strided_dout":
        # Native backward assumes packed dO. This deliberately sends a strided
        # gradient, so the public provider must make it safe before the kernel.
        storage = torch.randn((*actual.shape, 2), device="cuda").to(actual.dtype)
        dout = storage[..., 0]
        assert not dout.is_contiguous()
        actual.backward(dout)
        expected.backward(dout.float())
    else:
        # A second, independent objective exercises output-dependent gradients.
        target = torch.randn(actual.shape, device="cuda")
        (0.5 * (actual.float() - target).square().sum()).backward()
        (0.5 * (expected - target).square().sum()).backward()

    for name, native, reference in zip(
        ("dQ", "dK", "dV", "dBias"), native_leaves, reference_leaves, strict=True
    ):
        assert native.grad is not None and reference.grad is not None, name
        assert torch.count_nonzero(reference.grad), f"{name}: degenerate oracle"
        _assert_numerical_close(native.grad, reference.grad, name=name)


@pytest.mark.parametrize(
    "layout,sequence,head_dim",
    [
        ("unbatched_shared", 17, 16),
        ("chunk_distinct", 17, 16),
        ("batched_distinct", 33, 32),
        ("batched_shared", 33, 32),
        ("template_broadcast", 17, 16),
        ("batch_broadcast_templates", 33, 32),
    ],
)
@pytest.mark.parametrize("loss_kind", ("linear_strided_dout", "quadratic"))
def test_provider_layout_output_and_all_gradients(
    layout, sequence, head_dim, loss_kind
):
    _check_provider(_make_inputs(layout, sequence, head_dim), loss_kind)


@pytest.mark.slow
@pytest.mark.parametrize("sequence", (384, 640, 768))
def test_provider_production_sequence_lengths(sequence):
    """Bind production sequence specializations without a full model allocation."""
    shape = (1, 2, 4, sequence, 32)
    q, k, v = [torch.randn(shape, device="cuda") for _ in range(3)]
    q = (q * 32**-0.5).to(torch.bfloat16).requires_grad_()
    k, v = [x.to(torch.bfloat16).requires_grad_() for x in (k, v)]
    bias = (
        (0.3 * torch.randn((1, 1, 4, sequence, sequence), device="cuda"))
        .to(torch.bfloat16)
        .requires_grad_()
    )
    mask = torch.zeros((1, 2, 1, 1, sequence), device="cuda")
    mask[..., 3::11] = -1e4
    _check_provider((q, k, v, mask, bias), "linear_strided_dout")


def _nonzero_parameters(module):
    # Final projections initialize to zero in the model; leave no inert branch.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0 if name.endswith("weight") else 0.0)
            else:
                parameter.normal_(std=0.08)
    return module


@pytest.mark.parametrize("starting", (True, False), ids=("starting", "ending"))
def test_triangle_attention_forward_and_backward(starting):
    from openfold3.core.model.layers.triangular_attention import TriangleAttention

    native = _nonzero_parameters(TriangleAttention(32, 16, 2, starting=starting).cuda())
    reference = copy.deepcopy(native)
    x = torch.randn((2, 17, 17, 32), device="cuda").to(torch.bfloat16)
    x = x.requires_grad_()
    reference_x = x.detach().float().requires_grad_()
    mask = torch.ones((2, 17, 17), device="cuda")
    # An asymmetric mask checks the ending-node transpose as well.
    mask[:, ::3, 2::4] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = native(x, mask=mask, use_protenix_triangle_attention=True)
    expected = reference(reference_x, mask=mask)
    _assert_numerical_close(actual, expected, name="triangle output", relative=0.06)
    dout = torch.randn_like(actual)
    actual.backward(dout)
    expected.backward(dout.float())
    _assert_numerical_close(x.grad, reference_x.grad, name="triangle dX", relative=0.08)
    for name, parameter in native.named_parameters():
        reference_grad = dict(reference.named_parameters())[name].grad
        assert parameter.grad is not None and reference_grad is not None, name
        _assert_numerical_close(
            parameter.grad, reference_grad, name=name, relative=0.08, atol=0.008
        )


def test_pair_block_uses_protenix_attention_and_cueq_trimul():
    pytest.importorskip("cuequivariance_torch")
    import openfold3.core.kernels.protenix as provider
    import openfold3.core.model.layers.triangular_multiplicative_update as trimul
    import openfold3.core.model.primitives.attention as attention
    from openfold3.core.model.latent.base_blocks import PairBlock

    # N > 100 avoids cuEq's small-sequence fallback. C=128 matches the verified
    # training route, and dropout is disabled to avoid obscuring the contract.
    block = _nonzero_parameters(
        PairBlock(
            c_z=128,
            c_hidden_mul=128,
            c_hidden_pair_att=32,
            no_heads_pair=4,
            transition_type="relu",
            transition_n=2,
            pair_dropout=0.0,
            fuse_projection_weights=False,
            inf=1e9,
        ).cuda()
    )
    parameter_ids = {name: id(p) for name, p in block.named_parameters()}
    z = torch.randn((1, 101, 101, 128), device="cuda").to(torch.bfloat16)
    z = z.requires_grad_()
    mask = torch.ones(z.shape[:-1], device="cuda")
    with (
        patch.object(
            provider, "attention_scaled", wraps=provider.attention_scaled
        ) as native_calls,
        patch.object(
            trimul,
            "triangle_multiplicative_update",
            wraps=trimul.triangle_multiplicative_update,
        ) as trimul_calls,
        patch.object(
            attention,
            "_cueq_triangle_attn",
            side_effect=AssertionError("cuEq must not replace Protenix attention"),
        ),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        output = block(
            z,
            pair_mask=mask,
            use_protenix_triangle_attention=True,
            use_cueq_triangle_kernels=True,
        )
        (output.float() * torch.randn_like(output.float())).sum().backward()

    assert native_calls.call_count == 2
    assert trimul_calls.call_count == 2
    assert {name: id(p) for name, p in block.named_parameters()} == parameter_ids
    assert output.shape == z.shape and torch.isfinite(output).all()
    assert torch.linalg.vector_norm(output.detach().float() - z.detach().float()) > 0
    assert z.grad is not None and torch.isfinite(z.grad).all()
    parameters = dict(block.named_parameters())
    for name in (
        "tri_mul_out.linear_a_p.weight",
        "tri_mul_in.linear_b_p.weight",
        "tri_att_start.mha.linear_q.weight",
        "tri_att_end.mha.linear_k.weight",
        "tri_att_start.linear_z.weight",
        "tri_att_end.linear_z.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), f"{name}: inactive backward route"
    for name, parameter in parameters.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
