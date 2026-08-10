#!/usr/bin/env python3
"""Module-level smoke for the StableFold production wiring.

D=16 and D=32 modules route to StableFold: the criterion is
tolerance vs stock plus bitwise replay against the same backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openfold3.core.model.primitives.attention as attention_module  # noqa: E402
from openfold3.core.kernels.cuda.stablefold_triangle_attention import (  # noqa: E402
    invocation_count,
)
from openfold3.core.model.layers.triangular_attention import (  # noqa: E402
    TriangleAttention,
)

_STOCK_CALLS = {"count": 0}
_ORIG_ATTENTION = attention_module._attention


def _counting_attention(*args, **kwargs):
    _STOCK_CALLS["count"] += 1
    return _ORIG_ATTENTION(*args, **kwargs)


attention_module._attention = _counting_attention


def nbad(a, b):
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def build(c_in, c_hidden, heads, seed):
    torch.manual_seed(seed)
    module = (
        TriangleAttention(
            c_in=c_in, c_hidden=c_hidden, no_heads=heads, starting=True
        )
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn_like(parameter, dtype=torch.float32)
                .mul(0.05)
                .to(torch.bfloat16)
            )
    return module


def run_pass(module, x_base, mask, grad_seed, use_stable):
    module.zero_grad(set_to_none=True)
    x = x_base.clone().requires_grad_()
    stock_before = _STOCK_CALLS["count"]
    stablefold_before = invocation_count()
    out = module(x, mask=mask, use_stable_triangle_attention=use_stable)
    out.backward(grad_seed)
    torch.cuda.synchronize()
    # Routing regression locks (2026-08-09 overwrite bug): a stable pass
    # must invoke the StableFold facade and must NEVER touch stock attention.
    if use_stable:
        if _STOCK_CALLS["count"] != stock_before:
            raise AssertionError(
                "stable pass reached stock _attention (overwrite regression)"
            )
        if invocation_count() == stablefold_before:
            raise AssertionError(
                "stable pass did not invoke the StableFold facade"
            )
    return {
        "out": out.detach().clone(),
        "x_grad": x.grad.detach().clone(),
        "grads": {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        },
    }


def case(label, c_in, c_hidden, heads, n, seed, bitwise_required,
         mask_dtype=None):
    module = build(c_in, c_hidden, heads, seed)
    x_base = torch.randn((1, n, n, c_in), device="cuda").to(torch.bfloat16)
    mask = (torch.rand((1, n, n), device="cuda") > 0.1).float()
    grad_seed = torch.randn((1, n, n, c_in), device="cuda").to(torch.bfloat16)

    if mask_dtype is not None:
        mask = mask.to(mask_dtype)
    stock = run_pass(module, x_base, mask.float(), grad_seed, False)
    stable = run_pass(module, x_base, mask, grad_seed, True)
    replay = run_pass(module, x_base, mask, grad_seed, True)

    replay_bad = (
        nbad(stable["out"], replay["out"])
        + nbad(stable["x_grad"], replay["x_grad"])
        + sum(
            nbad(stable["grads"][key], replay["grads"][key])
            for key in stable["grads"]
        )
    )
    if bitwise_required:
        bad = (
            nbad(stock["out"], stable["out"])
            + nbad(stock["x_grad"], stable["x_grad"])
            + sum(
                nbad(stock["grads"][key], stable["grads"][key])
                for key in stock["grads"]
            )
        )
        passed = bad == 0 and replay_bad == 0
        print(
            f"{label}: bitwise_bad={bad} replay_bad={replay_bad} "
            f"passed={passed}"
        )
        return passed
    out_gap = max_abs(stock["out"], stable["out"])
    x_gap = max_abs(stock["x_grad"], stable["x_grad"])
    grad_gap = max(
        max_abs(stock["grads"][key], stable["grads"][key])
        for key in stock["grads"]
    )
    passed = replay_bad == 0 and out_gap < 0.1 and x_gap < 0.1
    print(
        f"{label}: out_gap={out_gap:.2e} x_grad_gap={x_gap:.2e} "
        f"param_grad_gap={grad_gap:.2e} replay_bad={replay_bad} "
        f"passed={passed}"
    )
    return passed


def main():
    results = [
        case("trunk D=32 N=384 (StableFold)", 128, 32, 4, 384, 11, False),
        case("trunk D=32 N=640 (StableFold)", 128, 32, 4, 640, 12, False),
        case("template D=16 N=640 (StableFold)", 64, 16, 4, 640, 13, False),
        case("trunk D=32 N=384 bf16-mask (real AMP dtype)", 128, 32, 4,
             384, 14, False, mask_dtype=torch.bfloat16),
    ]
    print(f"ALL_PASSED={all(results)}")


if __name__ == "__main__":
    main()
