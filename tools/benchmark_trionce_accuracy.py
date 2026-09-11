#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare accuracy and run-to-run determinism per kernel provider.

Speed is only half of what the Trionce kernels are claimed to buy over the
stock route; the other half is numerical behaviour. This reports both of the
properties that matter, so packaging the kernels as a pip package can be shown
not to have moved either:

* **accuracy** -- L2 relative error against an FP64 oracle, for the forward
  output and every gradient, alongside the stock BF16 path measured the same
  way. The methodology mirrors ``trionce/tests/test_numerics_vs_fp64.py``,
  whose bound is that the kernel stays within ``1.1 * stock + 1e-6``.
* **determinism** -- whether a second identical call returns bitwise-identical
  results. The kernels use no floating-point atomics, so they replay exactly;
  a provider that lost that property would show up here.

Usage::

    python tools/benchmark_trionce_accuracy.py
    python tools/benchmark_trionce_accuracy.py --json accuracy.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Callable

import torch

DEV = torch.device("cuda")
# The oracle needs a finite mask; -inf would poison the FP64 matmul.
NEG = -1e4

# name -> (batch, outer, heads, sequence, head_dim, masked_fraction)
CASES: dict[str, tuple[int, int, int, int, int, float]] = {
    "trunk_n384_d32": (1, 384, 4, 384, 32, 0.0),
    "trunk_n384_d32_masked": (1, 384, 4, 384, 32, 0.25),
    "trunk_n640_d32": (1, 640, 4, 640, 32, 0.0),
    "n384_d16": (1, 384, 4, 384, 16, 0.0),
    "n256_d64": (1, 256, 2, 256, 64, 0.0),
    "apb_outer24_n384_d32": (2, 24, 4, 384, 32, 0.1),
}

TENSOR_NAMES = ("out", "dq", "dk", "dv", "dbias")


def _finite_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isinf(mask), torch.full_like(mask, NEG), mask)


def _rel(value: torch.Tensor, reference: torch.Tensor) -> float:
    """L2 relative error, as the package's FP64-oracle test defines it."""

    return float(
        (value.double() - reference.double()).norm()
        / reference.double().norm().clamp_min(1e-30)
    )


def _case_inputs(spec: tuple[int, int, int, int, int, float], seed: int):
    batch, outer, heads, seq, head_dim, masked = spec
    torch.manual_seed(seed)
    q = torch.randn(batch, outer, heads, seq, head_dim, device=DEV).to(torch.bfloat16)
    k = torch.randn(batch, outer, heads, seq, head_dim, device=DEV).to(torch.bfloat16)
    v = torch.randn(batch, outer, heads, seq, head_dim, device=DEV).to(torch.bfloat16)
    mask = torch.zeros(batch, outer, 1, 1, seq, device=DEV)
    if masked:
        mask[torch.rand(batch, outer, 1, 1, seq, device=DEV) < masked] = float("-inf")
    bias = (torch.randn(1, 1, heads, seq, seq, device=DEV) * 2).to(torch.bfloat16)
    d_out = torch.randn(batch, outer, seq, heads, head_dim, device=DEV).to(
        torch.bfloat16
    )
    return q, k, v, mask, bias, d_out


def _oracle(q, k, v, mask, bias, d_out):
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    b64 = bias.detach().double().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q64, k64.transpose(-1, -2)) * scale
    scores = scores + _finite_mask(mask).double() + b64
    out = torch.matmul(torch.softmax(scores, dim=-1), v64).transpose(-2, -3)
    out.backward(d_out.double())
    return out.detach(), q64.grad, k64.grad, v64.grad, b64.grad


def _stock(q, k, v, mask, bias, d_out):
    """BF16 matmul with an FP32 softmax: the stock OpenFold3 route."""

    q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    b_ = bias.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q_, k_.transpose(-1, -2)).float() * scale
    scores = scores + _finite_mask(mask) + b_.float()
    out = torch.matmul(torch.softmax(scores, dim=-1).to(torch.bfloat16), v_)
    out = out.transpose(-2, -3)
    out.backward(d_out)
    return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad


def _kernel(attention: Callable[..., Any]):
    def run(q, k, v, mask, bias, d_out):
        q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        b_ = bias.detach().clone().requires_grad_(True)
        out = attention(q_, k_, v_, mask, b_)
        out.backward(d_out)
        return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad

    return run


def _providers(names: list[str]) -> dict[str, Callable[..., Any]]:
    resolved: dict[str, Callable[..., Any]] = {}
    for name in names:
        if name == "stock":
            resolved[name] = _stock
        elif name == "trionce":
            from trionce import triangle_attention

            resolved[name] = _kernel(triangle_attention)
        elif name == "intree":
            from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
                stablefold_triangle_attention,
            )

            resolved[name] = _kernel(stablefold_triangle_attention)
        else:
            raise SystemExit(f"unknown provider: {name}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers", nargs="+", default=["stock", "intree", "trionce"]
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--json", type=str)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    print(f"device   {torch.cuda.get_device_name(0)}")
    print(f"torch    {torch.__version__} / CUDA {torch.version.cuda}")
    print("oracle   FP64 PyTorch attention; error is L2 relative, lower is better")
    print()

    providers = _providers(arguments.providers)
    report: dict[str, Any] = {}

    for case_name in arguments.cases:
        spec = CASES[case_name]
        inputs = _case_inputs(spec, seed=hash(case_name) % (2**31))
        reference = _oracle(*inputs)

        results: dict[str, Any] = {}
        for provider, runner in providers.items():
            first = runner(*inputs)
            second = runner(*inputs)
            deterministic = all(
                torch.equal(a, b) for a, b in zip(first, second, strict=True)
            )
            errors = {
                name: _rel(value, ref)
                for name, value, ref in zip(
                    TENSOR_NAMES, first, reference, strict=True
                )
            }
            results[provider] = {"errors": errors, "deterministic": deterministic}
        report[case_name] = results

        print(f"### {case_name}  {spec[:5]}"
              + (f"  masked={spec[5]:.0%}" if spec[5] else ""))
        header = f"{'tensor':<10}" + "".join(
            f"{p:>14}" for p in arguments.providers
        )
        print(header)
        print("-" * len(header))
        for tensor in TENSOR_NAMES:
            row = "".join(
                f"{results[p]['errors'][tensor]:>14.3e}" for p in arguments.providers
            )
            print(f"{tensor:<10}{row}")
        replay = "".join(
            f"{('bitwise' if results[p]['deterministic'] else 'VARIES'):>14}"
            for p in arguments.providers
        )
        print(f"{'replay':<10}{replay}")
        print()

    print("### summary")
    print("Within the package's own bound, a kernel provider must stay at or below")
    print("1.1 x the stock relative error + 1e-6 on every tensor.")
    failures = 0
    for case_name, results in report.items():
        for provider in arguments.providers:
            if provider == "stock":
                continue
            for tensor in TENSOR_NAMES:
                observed = results[provider]["errors"][tensor]
                bound = 1.1 * results["stock"]["errors"][tensor] + 1e-6
                if observed > bound:
                    failures += 1
                    print(f"  OVER BOUND {case_name} {provider} {tensor}: "
                          f"{observed:.3e} > {bound:.3e}")
            if not results[provider]["deterministic"]:
                failures += 1
                print(f"  NONDETERMINISTIC {case_name} {provider}")
    if failures == 0:
        print("  every provider is within the stock envelope and replays bitwise")

    if "trionce" in providers and "intree" in providers:
        moved = [
            (case_name, tensor)
            for case_name, results in report.items()
            for tensor in TENSOR_NAMES
            if results["trionce"]["errors"][tensor]
            != results["intree"]["errors"][tensor]
        ]
        if moved:
            print(f"  packaging moved the error on {len(moved)} tensors, e.g. {moved[:3]}")
            failures += 1
        else:
            print("  trionce and intree report identical error on every tensor")

    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {arguments.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
