# SPDX-License-Identifier: Apache-2.0

"""TurboPairFormer Triangle Attention against an FP64 oracle at training shapes.

For every case the same BF16 inputs feed three paths:

* ``oracle``: FP64 PyTorch attention, forward and backward;
* ``stock``: BF16 matmul / FP32 softmax PyTorch attention, the error level of
  the stock OpenFold3 route under autocast;
* ``turbopairformer``: the facade with the requested execution flags.

TurboPairFormer must be finite, bitwise reproducible against itself, exactly zero
on fully masked planes, and within 1.1 times the stock path's relative error
plus 1e-6, as encoded by this test.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="TurboPairFormer kernels need a Hopper GPU (sm_90a, capability 9.0)",
)

DEV = torch.device("cuda")
NEG = -1e4
PROD_FLAGS = {"TURBOPAIRFORMER_SR_BACKWARD": "1", "TURBOPAIRFORMER_SR_SHADOW": "1"}
CLASSIC_FLAGS = {"TURBOPAIRFORMER_SR_BACKWARD": "0", "TURBOPAIRFORMER_SR_SHADOW": "0"}


def _finite_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isinf(mask), torch.full_like(mask, NEG), mask)


def _oracle(q, k, v, mask, bias, dout):
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    b64 = bias.detach().double().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q64, k64.transpose(-1, -2)) * scale
    s = s + _finite_mask(mask).double() + b64
    out = torch.matmul(torch.softmax(s, dim=-1), v64).transpose(-2, -3)
    out.backward(dout.double())
    return out.detach(), q64.grad, k64.grad, v64.grad, b64.grad


def _stock(q, k, v, mask, bias, dout):
    q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    b_ = bias.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q_, k_.transpose(-1, -2)).float() * scale
    s = s + _finite_mask(mask) + b_.float()
    out = torch.matmul(torch.softmax(s, dim=-1).to(torch.bfloat16), v_)
    out = out.transpose(-2, -3)
    out.backward(dout)
    return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad


def _turbopairformer(q, k, v, mask, bias, dout):
    from turbopairformer import turbo_attention

    q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    b_ = bias.detach().clone().requires_grad_(True)
    out = turbo_attention(q_, k_, v_, mask, b_)
    out.backward(dout)
    return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _case(B, O, H, N, D, mask_frac, full_planes, bias_dtype):
    q = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    k = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    v = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    mask = torch.zeros(B, O, 1, 1, N, device=DEV)
    if mask_frac:
        mask[torch.rand(B, O, 1, 1, N, device=DEV) < mask_frac] = float("-inf")
    if full_planes:
        mask[:, :full_planes] = float("-inf")
    bias = (torch.randn(1, 1, H, N, N, device=DEV) * 2).to(bias_dtype)
    dout = torch.randn(B, O, N, H, D, device=DEV).to(torch.bfloat16)
    if full_planes:
        dout[:, :full_planes] = 0
    return q, k, v, mask, bias, dout


CASES = [
    (
        "prod N=128 D=32",
        PROD_FLAGS,
        dict(B=1, O=8, H=4, N=128, D=32, mask_frac=0.0, full_planes=0),
    ),
    (
        "prod N=100 pad D=32 masked",
        PROD_FLAGS,
        dict(B=1, O=8, H=4, N=100, D=32, mask_frac=0.3, full_planes=0),
    ),
    (
        "prod N=384 D=24",
        PROD_FLAGS,
        dict(B=1, O=4, H=4, N=384, D=24, mask_frac=0.2, full_planes=0),
    ),
    (
        "prod N=200 D=20 padD",
        PROD_FLAGS,
        dict(B=2, O=4, H=4, N=200, D=20, mask_frac=0.2, full_planes=0),
    ),
    (
        "prod N=256 D=64 full planes",
        PROD_FLAGS,
        dict(B=1, O=6, H=4, N=256, D=64, mask_frac=0.2, full_planes=2),
    ),
    (
        "prod N=384 D=48",
        PROD_FLAGS,
        dict(B=1, O=2, H=4, N=384, D=48, mask_frac=0.1, full_planes=0),
    ),
    (
        "prod N=768 D=16 H=8",
        PROD_FLAGS,
        dict(B=1, O=2, H=8, N=768, D=16, mask_frac=0.1, full_planes=0),
    ),
    (
        "classic N=384 D=24",
        CLASSIC_FLAGS,
        dict(B=1, O=4, H=4, N=384, D=24, mask_frac=0.2, full_planes=0),
    ),
    (
        "classic N=256 D=64 full planes",
        CLASSIC_FLAGS,
        dict(B=1, O=6, H=4, N=256, D=64, mask_frac=0.2, full_planes=2),
    ),
]


@pytest.mark.parametrize("name,flags,shape", CASES, ids=[c[0] for c in CASES])
def test_triangle_attention_fp64_error_stays_within_stock_envelope(
    name, flags, shape, monkeypatch
):
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    torch.manual_seed(0)
    q, k, v, mask, bias, dout = _case(bias_dtype=torch.bfloat16, **shape)
    ref = _oracle(q, k, v, mask, bias, dout)
    stock = _stock(q, k, v, mask, bias, dout)
    actual = _turbopairformer(q, k, v, mask, bias, dout)
    replayed = _turbopairformer(q, k, v, mask, bias, dout)

    assert all(torch.equal(a, b) for a, b in zip(actual, replayed, strict=True)), (
        "not replayable"
    )
    assert all(torch.isfinite(t).all() for t in actual), "non-finite output"

    full = shape["full_planes"]
    if full:
        for tensor in (
            actual[0][:, :full],
            actual[1][:, :full],
            actual[2][:, :full],
            actual[3][:, :full],
        ):
            assert tensor.abs().max().item() == 0.0, (
                "masked planes must be exactly zero"
            )
        ref = (ref[0][:, full:], *ref[1:])
        stock = (stock[0][:, full:], *stock[1:])
        actual = (actual[0][:, full:], *actual[1:])

    # TurboPairFormer error must not exceed the stock BF16 path by more than 10 %.
    for label, s_val, st_val, r_val in zip(
        ("out", "dq", "dk", "dv", "dbias"), actual, stock, ref, strict=True
    ):
        e_actual, e_stock = _rel(s_val, r_val), _rel(st_val, r_val)
        assert e_actual <= 1.1 * e_stock + 1e-6, (
            f"{name} {label}: actual={e_actual:.2e} stock={e_stock:.2e}"
        )


def test_fp32_bias_is_accepted_and_returns_fp32_gradient():
    torch.manual_seed(0)
    q, k, v, mask, bias, dout = _case(
        B=1,
        O=4,
        H=4,
        N=128,
        D=32,
        mask_frac=0.0,
        full_planes=0,
        bias_dtype=torch.float32,
    )
    out, dq, dk, dv, dbias = _turbopairformer(q, k, v, mask, bias, dout)
    assert out.dtype == torch.bfloat16
    assert dbias.dtype == torch.float32
    assert torch.isfinite(dbias).all()
