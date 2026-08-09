# SPDX-License-Identifier: Apache-2.0

"""Streaming dK/dV v2 — key-block ownership, H/D/N general."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    from openfold3.core.kernels.cuda._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("stable_dkdv", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.cuda._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="stable_dkdv",
            aot_module="openfold3.core.kernels.cuda._ext._stable_dkdv_sm90",
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=("stable_triangle_dkdv_v2_out",),
            verbose=bool(verbose),
        )
    return _EXTENSION


def stable_triangle_dkdv_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_output: torch.Tensor,
    logsumexp: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    out_dk: torch.Tensor | None = None,
    out_dv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out_dk is None:
        out_dk = torch.empty_like(q)
    if out_dv is None:
        out_dv = torch.empty_like(q)
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().stable_triangle_dkdv_v2_out(
            q,
            k,
            v,
            d_output,
            logsumexp.reshape(-1).contiguous(),
            delta.reshape(-1).contiguous(),
            mask,
            pair_bias,
            out_dk,
            out_dv,
        )
    return out_dk, out_dv


__all__ = ["stable_triangle_dkdv_v2", "load_extension", "source_sha256"]
