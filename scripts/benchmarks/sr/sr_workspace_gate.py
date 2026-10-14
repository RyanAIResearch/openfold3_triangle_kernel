#!/usr/bin/env python3
"""Bitwise workspace-plan invariance gate for TurboPairFormer SR backward.

This gate intentionally crosses query-stripe, leaf-wave, batch-chunk, and
storage-dimension boundaries.  It also compares dK/dV against the independent
two-pass producer so a consistently wrong workspace plan cannot self-certify.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict

import torch

from openfold3.core.kernels.turbopairformer import (
    turbopairformer_triangle_attention as facade,
)
from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_dkdv import (
    turbopairformer_triangle_dkdv,
)
from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_dq_dbias import (
    sr_workspace_plan,
    turbopairformer_triangle_sr_backward,
)

CASES = {
    "trunk_n640_d32": (1, 640, 4, 640, 32),
    "trunk_n768_d32": (1, 768, 4, 768, 32),
    "batch2_outer24_n384_d32": (2, 24, 4, 384, 32),
    "n384_d16": (1, 384, 4, 384, 16),
    "n384_d64": (1, 384, 2, 384, 64),
}

BUDGETS_MIB = (64, 128, 256, 512)


def emit(kind: str, **payload: object) -> None:
    print(json.dumps({"kind": kind, **payload}, sort_keys=True), flush=True)


def make_case(name: str) -> tuple[torch.Tensor, ...]:
    batch, outer, heads, sequence, dim = CASES[name]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    shape = (batch, outer, heads, sequence, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(
        batch, outer, 1, 1, sequence, device="cuda", dtype=torch.float32
    )
    mask[..., -3:] = float("-inf")
    bias = (
        0.1
        * torch.randn(
            batch,
            1,
            heads,
            sequence,
            sequence,
            device="cuda",
            dtype=torch.float32,
        )
    ).bfloat16()
    torch.manual_seed(seed ^ 0x5A5A5A5A)
    grad_out = torch.randn(
        batch,
        outer,
        sequence,
        heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return q, k, v, mask, bias, grad_out


def forward_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, outer, heads, sequence, dim = q.shape
    out = torch.empty(
        (batch, outer, sequence, heads, dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    shadow = torch.empty_like(out)
    facade._forward_extension().turbopairformer_triangle_forward_shadow_out(
        q, k, v, mask, bias, out, logsumexp, shadow
    )
    delta = (
        (grad_out.float() * (out.float() + shadow.float()))
        .sum(-1)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    return logsumexp, delta


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def configurations(name: str) -> list[dict[str, int | None]]:
    batch, _, _, sequence, _ = CASES[name]
    configs: list[dict[str, int | None]] = [
        {
            "workspace_budget_bytes": mib * 1024 * 1024,
            "query_span": None,
            "batch_chunk": None,
        }
        for mib in BUDGETS_MIB
    ]
    for query_span in (64, 128, min(256, sequence), sequence):
        if query_span % 64 == 0:
            configs.append(
                {
                    "workspace_budget_bytes": 512 * 1024 * 1024,
                    "query_span": query_span,
                    "batch_chunk": None,
                }
            )
    if batch > 1:
        for batch_chunk in range(1, batch + 1):
            configs.append(
                {
                    "workspace_budget_bytes": 128 * 1024 * 1024,
                    "query_span": 64,
                    "batch_chunk": batch_chunk,
                }
            )
    unique: list[dict[str, int | None]] = []
    seen: set[tuple[int | None, ...]] = set()
    for config in configs:
        key = tuple(config.values())
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def run_case(name: str) -> None:
    q, k, v, mask, bias, grad_out = make_case(name)
    logsumexp, delta = forward_state(q, k, v, mask, bias, grad_out)
    two_pass_dk, two_pass_dv = turbopairformer_triangle_dkdv(
        q, k, v, grad_out, logsumexp, delta, mask, bias
    )
    reference: tuple[torch.Tensor, ...] | None = None
    reference_digests: dict[str, str] | None = None
    keys = ("dq", "dk", "dv", "dbias")
    for index, config in enumerate(configurations(name)):
        plan = sr_workspace_plan(
            sequence=int(q.shape[3]),
            heads=int(q.shape[2]),
            dim=int(q.shape[4]),
            outer=int(q.shape[1]),
            batch=int(q.shape[0]),
            **config,
        )
        torch.cuda.reset_peak_memory_stats()
        outputs = turbopairformer_triangle_sr_backward(
            q, k, v, grad_out, logsumexp, delta, mask, bias, **config
        )
        torch.cuda.synchronize()
        output_digests = {key: digest(value) for key, value in zip(keys, outputs)}
        if reference is None:
            reference = tuple(value.detach().clone() for value in outputs)
            reference_digests = output_digests
        mismatches = {
            key: int(torch.count_nonzero(value != expected).item())
            for key, value, expected in zip(keys, outputs, reference)
        }
        two_pass_mismatches = {
            "dk": int(torch.count_nonzero(outputs[1] != two_pass_dk).item()),
            "dv": int(torch.count_nonzero(outputs[2] != two_pass_dv).item()),
        }
        emit(
            "configuration",
            case=name,
            config_index=index,
            requested=config,
            plan=asdict(plan),
            digests=output_digests,
            reference_digests=reference_digests,
            mismatches=mismatches,
            two_pass_mismatches=two_pass_mismatches,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        if any(mismatches.values()) or any(two_pass_mismatches.values()):
            raise AssertionError(f"workspace invariance failure: {name} {config}")
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
    emit("case_pass", case=name, configurations=len(configurations(name)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    args = parser.parse_args()
    selected = args.case or list(CASES)
    for name in selected:
        run_case(name)
    emit("workspace_invariance_pass", cases=selected)


if __name__ == "__main__":
    main()
