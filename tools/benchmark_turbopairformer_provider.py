#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Time the TurboPairFormer triangle kernels per provider, and against stock attention.

Two questions this answers, at the pair-stack layouts the training run actually
issues:

1. **Did packaging cost anything?**  ``turbopairformer`` (the installed AOT extensions)
   against ``intree`` (the vendored copy).  They compile byte-identical ``.cu``
   sources, so any gap is build or dispatch overhead, not kernel work.  Note
   that in a source checkout the in-tree copy is served by the development JIT
   lane, which compiles with a shorter flag list than the AOT build; the
   comparison is therefore "as each is actually served", which is what a user
   experiences.
2. **Is the speedup still there?**  Either kernel provider against OpenFold3's
   stock PyTorch attention at the same shapes and dtype.

Timing uses CUDA events around a synchronised region, after warmup, and reports
the median of the measured iterations.  Forward-plus-backward is the headline
number because training pays both.

Usage::

    python tools/benchmark_turbopairformer_provider.py                    # default set
    python tools/benchmark_turbopairformer_provider.py --iters 30 --json out.json
    python tools/benchmark_turbopairformer_provider.py --shapes trunk_n640_d32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Callable
from typing import Any

import torch

# name -> (batch, outer, heads, sequence, head_dim), taken from the shapes the
# repository's own SR benchmarks use for production sites.
SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "trunk_n384_d32": (1, 384, 4, 384, 32),
    "trunk_n640_d32": (1, 640, 4, 640, 32),
    "trunk_n768_d32": (1, 768, 4, 768, 32),
    "n384_d16": (1, 384, 4, 384, 16),
    "n384_d64": (1, 384, 2, 384, 64),
    "apb_outer24_n384_d32": (2, 24, 4, 384, 32),
}

WARMUP = 5


def _inputs(shape: tuple[int, int, int, int, int], seed: int, *, grad: bool):
    batch, outer, heads, seq, head_dim = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*size):
        return torch.randn(
            *size, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    q = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    k = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    v = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    mask = torch.zeros(batch, outer, 1, 1, seq, device="cuda", dtype=torch.bfloat16)
    bias = randn(batch, 1, heads, seq, seq).requires_grad_(grad)
    return q, k, v, mask, bias


def _stock_attention(q, k, v, mask, bias):
    """OpenFold3's stock path, which applies the 1/sqrt(d) scale itself."""

    from openfold3.core.model.primitives.attention import _attention

    scale = 1.0 / math.sqrt(q.shape[-1])
    return _attention(q * scale, k, v, [mask, bias]).transpose(-2, -3)


def _load_provider(name: str) -> Callable[..., Any]:
    if name == "turbopairformer":
        from turbopairformer import triangle_attention

        return triangle_attention
    if name == "intree":
        from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_attention import (
            turbopairformer_triangle_attention,
        )

        return turbopairformer_triangle_attention
    if name == "stock":
        return _stock_attention
    raise SystemExit(f"unknown provider: {name}")


def _time(fn, iters: int) -> list[float]:
    """Return per-iteration milliseconds, measured with CUDA events."""

    samples: list[float] = []
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _measure(attention, shape, iters, *, backward: bool) -> dict[str, Any] | None:
    inputs = _inputs(shape, seed=7, grad=backward)
    q, k, v, mask, bias = inputs
    try:
        if backward:
            grad_out = torch.randn_like(attention(*inputs))

            def step():
                for tensor in (q, k, v, bias):
                    tensor.grad = None
                out = attention(*inputs)
                out.backward(grad_out)

        else:

            def step():
                with torch.no_grad():
                    attention(*inputs)

        torch.cuda.reset_peak_memory_stats()
        samples = _time(step, iters)
    except (RuntimeError, ValueError, torch.OutOfMemoryError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "peak_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["turbopairformer", "intree", "stock"],
        help="which implementations to time",
    )
    parser.add_argument(
        "--shapes", nargs="+", default=list(SHAPES), help="which layouts to time"
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", type=str, help="also write the raw report here")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    device = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability()
    print(f"device   {device} (sm_{capability[0]}{capability[1]})")
    print(f"torch    {torch.__version__} / CUDA {torch.version.cuda}")
    print(
        f"flags    SR_BACKWARD={os.environ.get('OPENFOLD3_SR_BACKWARD')} "
        f"SR_SHADOW={os.environ.get('OPENFOLD3_SR_SHADOW')} "
        f"ALLOW_JIT={os.environ.get('OPENFOLD3_TRIANGLE_ALLOW_JIT')}"
    )
    print(f"iters    {arguments.iters} (after {WARMUP} warmup)")
    print()

    attentions = {name: _load_provider(name) for name in arguments.providers}
    report: dict[str, Any] = {"device": device, "shapes": {}}

    for pass_name, backward in (("forward", False), ("forward+backward", True)):
        print(f"### {pass_name} (median ms, lower is better)")
        header = f"{'layout':<24}" + "".join(f"{n:>16}" for n in arguments.providers)
        print(header)
        print("-" * len(header))
        for shape_name in arguments.shapes:
            shape = SHAPES[shape_name]
            cells = []
            results = {}
            for provider in arguments.providers:
                result = _measure(
                    attentions[provider], shape, arguments.iters, backward=backward
                )
                results[provider] = result
                if result is None or "error" in result:
                    cells.append(f"{'n/a':>16}")
                else:
                    cells.append(f"{result['median_ms']:>16.4f}")
            report["shapes"].setdefault(shape_name, {})[pass_name] = results
            print(f"{shape_name:<24}" + "".join(cells))

        # Ratios that answer the two questions directly.
        if {"turbopairformer", "intree"} <= set(arguments.providers):
            print()
            print(
                f"{'layout':<24}{'turbopairformer/intree':>23}"
                f"{'stock/turbopairformer':>23}"
            )
            for shape_name in arguments.shapes:
                results = report["shapes"][shape_name][pass_name]
                packaging = _ratio(results, "turbopairformer", "intree")
                speedup = _ratio(results, "stock", "turbopairformer")
                print(f"{shape_name:<24}{packaging:>23}{speedup:>23}")
        print()

    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {arguments.json}")
    return 0


def _ratio(results: dict[str, Any], numerator: str, denominator: str) -> str:
    top = results.get(numerator)
    bottom = results.get(denominator)
    if not top or not bottom or "error" in top or "error" in bottom:
        return "n/a"
    return f"{top['median_ms'] / bottom['median_ms']:.2f}x"


if __name__ == "__main__":
    sys.exit(main())
