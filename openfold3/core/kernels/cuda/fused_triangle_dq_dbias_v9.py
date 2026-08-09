# SPDX-License-Identifier: Apache-2.0

"""Streaming fused dQ + dBias v9 with a fixed dBias reduction tree.

Each leaf plane has a single writer and the finalizer uses a fixed order.
The implementation uses no floating-point atomics. See the CUDA source and
``PRODUCTION_MANIFEST.md`` for the numerical contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

_OUTER_LEAF = 4
_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    source = Path(__file__).with_suffix(".cu")
    return {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}


def load_extension(*, verbose: bool | None = None) -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="fused_triangle_dq_dbias_v9",
            sources=[str(Path(__file__).with_suffix(".cu"))],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=bool(verbose),
        )
    return _EXTENSION


def outer_leaf_for(sequence: int) -> int:
    """Adaptive plane policy: leaf grows with N, keeping workspace O(N^2)."""
    return _OUTER_LEAF * ((sequence + 767) // 768)


def workspace_elements(sequence: int, heads: int) -> int:
    leaf = outer_leaf_for(sequence)
    leaves = (sequence + leaf - 1) // leaf
    return leaves * heads * sequence * sequence


def fused_triangle_dq_dbias_v9(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_output: torch.Tensor,
    logsumexp: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    workspace: torch.Tensor | None = None,
    out_dq: torch.Tensor | None = None,
    out_dbias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = int(q.shape[1])
    heads = int(q.shape[2])
    if workspace is None:
        workspace = torch.empty(
            workspace_elements(sequence, heads),
            device=q.device,
            dtype=torch.bfloat16,
        )
    if out_dq is None:
        out_dq = torch.empty_like(q)
    if out_dbias is None:
        out_dbias = torch.empty(
            (1, 1, heads, sequence, sequence),
            device=q.device,
            dtype=torch.float32,
        )
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().fused_triangle_dq_dbias_v9_out(
            q,
            k,
            v,
            d_output,
            logsumexp.reshape(-1).contiguous(),
            delta.reshape(-1).contiguous(),
            mask,
            pair_bias,
            out_dq,
            out_dbias,
            workspace,
        )
    return out_dq, out_dbias


__all__ = [
    "fused_triangle_dq_dbias_v9",
    "load_extension",
    "source_sha256",
    "workspace_elements",
]
