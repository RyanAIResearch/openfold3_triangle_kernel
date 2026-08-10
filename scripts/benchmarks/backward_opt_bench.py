#!/usr/bin/env python3
"""Backward-chain baseline bench + quick gate for the opt/backward-round1 work.

Times each backward stage (delta, dK/dV, dQ/dBias incl. waves+finalize) and
the full chain with CUDA events; runs a fast correctness gate (replay bitwise
vs self, error ratio vs outer-chunked fp64 reference) so every optimization
lands with a receipt against this exact script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENFOLD3_TRIANGLE_ALLOW_JIT", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openfold3.core.kernels.cuda.fused_triangle_dq_dbias_v9 import (  # noqa: E402
    fused_triangle_dq_dbias_v9,
)
from openfold3.core.kernels.cuda.stable_triangle_dkdv_v2 import (  # noqa: E402
    stable_triangle_dkdv_v2,
)
from openfold3.core.kernels.cuda.streaming_triangle_delta_v1 import (  # noqa: E402
    load_extension as load_delta,
)
from openfold3.core.kernels.cuda.streaming_triangle_forward_v1 import (  # noqa: E402
    load_extension as load_forward,
)


def make_case(sequence, heads, head_dim, seed):
    torch.manual_seed(seed)
    shape = (1, sequence, heads, sequence, head_dim)
    q = torch.randn(shape, device="cuda").to(torch.bfloat16)
    k = torch.randn(shape, device="cuda").to(torch.bfloat16)
    v = torch.randn(shape, device="cuda").to(torch.bfloat16)
    mask_bits = torch.rand((1, sequence, 1, 1, sequence), device="cuda") > 0.1
    mask = torch.where(
        mask_bits, torch.zeros((), device="cuda"),
        torch.full((), -1e9, device="cuda"))
    bias = torch.randn(
        (1, 1, heads, sequence, sequence), device="cuda").to(torch.bfloat16)
    d_out = torch.randn(shape, device="cuda").to(torch.bfloat16)
    return q, k, v, mask, bias, d_out


def forward_state(fwd, dlt, q, k, v, mask, bias, d_out):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    fwd.streaming_triangle_forward_out(q, k, v, mask, bias, out, lse)
    delta = torch.empty_like(lse)
    dlt.streaming_triangle_delta_out(d_out, out, delta)
    torch.cuda.synchronize()
    return out, lse, delta


def time_fn(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def gate(q, k, v, mask, bias, d_out, lse, delta, chunk=16):
    """Replay + error ratio for the backward chain at this shape."""
    def run():
        dk, dv = stable_triangle_dkdv_v2(q, k, v, d_out, lse, delta, mask, bias)
        dq, dbias = fused_triangle_dq_dbias_v9(
            q, k, v, d_out, lse, delta, mask, bias)
        torch.cuda.synchronize()
        return {"dq": dq, "dk": dk, "dv": dv, "dbias": dbias}

    first = run()
    replay = run()
    replay_bits = sum(
        int((first[key].view(torch.uint8) != replay[key].view(torch.uint8))
            .sum().item())
        for key in first)
    del replay
    torch.cuda.empty_cache()

    sequence, head_dim = q.shape[1], q.shape[-1]
    names = ("dq", "dk", "dv")
    ours_sq = {n: 0.0 for n in names}
    stock_sq = {n: 0.0 for n in names}
    dbias_truth = torch.zeros_like(bias, dtype=torch.float64)
    dbias_stock = torch.zeros_like(bias, dtype=torch.float64)
    for start in range(0, sequence, chunk):
        sl = (slice(None), slice(start, start + chunk))
        for dtype, ours, dbias_acc in (
            (torch.float64, None, dbias_truth),
            (torch.bfloat16, None, dbias_stock),
        ):
            leaves = [t[sl].to(dtype).requires_grad_() for t in (q, k, v)]
            bias_leaf = bias.to(dtype).requires_grad_()
            qc, kc, vc = leaves
            scores = torch.einsum(
                "...qc,...kc->...qk", qc / math.sqrt(head_dim), kc)
            scores = scores + mask[sl].to(dtype) + bias_leaf
            probability = torch.softmax(scores, dim=-1)
            output = torch.einsum("...qk,...kc->...qc", probability, vc)
            grads = torch.autograd.grad(
                output, leaves + [bias_leaf], d_out[sl].to(dtype))
            key = "truth" if dtype == torch.float64 else "stock"
            for n, g in zip(names, grads[:3]):
                t64 = g.double()
                cand = first[n][sl].double()
                if key == "truth":
                    ours_sq[n] += (cand - t64).pow(2).sum().item()
                    stock_sq[n] -= t64.pow(2).sum().item() * 0  # placeholder
                    setattr(gate, f"_t_{n}", t64)
                else:
                    truth64 = getattr(gate, f"_t_{n}")
                    stock_sq[n] += (t64 - truth64).pow(2).sum().item()
            dbias_acc += grads[3].double()
            del leaves, qc, kc, vc, bias_leaf, scores, probability, output
            torch.cuda.empty_cache()
    ratios = {
        n: math.sqrt(ours_sq[n]) / max(math.sqrt(stock_sq[n]), 1e-30)
        for n in names
    }
    t64 = dbias_truth
    ratios["dbias"] = (
        (first["dbias"].double() - t64).norm()
        / max((dbias_stock - t64).norm().item(), 1e-30)
    ).item()
    return replay_bits, {k: round(v, 4) for k, v in ratios.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, nargs="+", default=[640, 768])
    parser.add_argument("--gate-sequence", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    fwd, dlt = load_forward(), load_delta()
    report = {}

    gq, gk, gv, gmask, gbias, gdo = make_case(args.gate_sequence, 4, 32, 7)
    _, glse, gdelta = forward_state(fwd, dlt, gq, gk, gv, gmask, gbias, gdo)
    replay_bits, ratios = gate(gq, gk, gv, gmask, gbias, gdo, glse, gdelta)
    report["gate"] = {
        "sequence": args.gate_sequence,
        "replay_mismatch_bytes": replay_bits,
        "ratios": ratios,
        "pass": replay_bits == 0 and all(r <= 2.0 for r in ratios.values()),
    }
    print(json.dumps(report["gate"]), flush=True)
    del gq, gk, gv, gmask, gbias, gdo, glse, gdelta
    torch.cuda.empty_cache()

    for sequence in args.sequences:
        q, k, v, mask, bias, d_out = make_case(sequence, 4, 32, 11)
        out, lse, delta_t = forward_state(fwd, dlt, q, k, v, mask, bias, d_out)
        delta_buf = torch.empty_like(lse)
        row = {
            "delta": time_fn(
                lambda: dlt.streaming_triangle_delta_out(
                    d_out, out, delta_buf),
                args.warmup, args.samples),
            "dkdv": time_fn(
                lambda: stable_triangle_dkdv_v2(
                    q, k, v, d_out, lse, delta_t, mask, bias),
                args.warmup, args.samples),
            "dq_dbias": time_fn(
                lambda: fused_triangle_dq_dbias_v9(
                    q, k, v, d_out, lse, delta_t, mask, bias),
                args.warmup, args.samples),
        }
        def full_chain():
            dlt.streaming_triangle_delta_out(d_out, out, delta_buf)
            stable_triangle_dkdv_v2(
                q, k, v, d_out, lse, delta_buf, mask, bias)
            fused_triangle_dq_dbias_v9(
                q, k, v, d_out, lse, delta_buf, mask, bias)
        row["backward_chain"] = time_fn(full_chain, args.warmup, args.samples)
        row = {k2: round(v2, 3) for k2, v2 in row.items()}
        report[sequence] = row
        print(json.dumps({sequence: row}), flush=True)
        del q, k, v, mask, bias, d_out, out, lse, delta_t, delta_buf
        torch.cuda.empty_cache()

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
