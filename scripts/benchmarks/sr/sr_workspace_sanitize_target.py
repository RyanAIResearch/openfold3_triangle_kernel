#!/usr/bin/env python3
"""Small query-striped SR target for compute-sanitizer."""

from __future__ import annotations

import torch

from openfold3.core.kernels.cuda import stablefold_triangle_attention as facade
from openfold3.core.kernels.cuda.stablefold_triangle_dq_dbias import (
    stablefold_triangle_sr_backward,
)


def main() -> None:
    batch, outer, heads, sequence, dim = 2, 8, 2, 128, 32
    torch.manual_seed(640)
    shape = (batch, outer, heads, sequence, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(
        batch, outer, 1, 1, sequence, device="cuda", dtype=torch.float32
    )
    bias = torch.randn(
        batch,
        1,
        heads,
        sequence,
        sequence,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_out = torch.randn(
        batch,
        outer,
        sequence,
        heads,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = torch.empty(
        (batch, outer, sequence, heads, dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    shadow = torch.empty_like(out)
    facade._forward_extension().stablefold_triangle_forward_shadow_out(
        q, k, v, mask, bias, out, lse, shadow
    )
    delta = (
        (grad_out.float() * (out.float() + shadow.float()))
        .sum(-1)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    stablefold_triangle_sr_backward(
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        mask,
        bias,
        workspace_budget_bytes=64 * 1024 * 1024,
        query_span=64,
        batch_chunk=1,
    )
    torch.cuda.synchronize()
    print("SR_WORKSPACE_SANITIZE_TARGET_DONE", flush=True)


if __name__ == "__main__":
    main()
