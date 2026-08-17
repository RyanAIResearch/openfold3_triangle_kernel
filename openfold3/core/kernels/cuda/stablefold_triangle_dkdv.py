# SPDX-License-Identifier: Apache-2.0

"""StableFold dK/dV with key-block ownership and runtime H/D/N."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    from openfold3.core.kernels.cuda._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("stablefold_dkdv", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.cuda._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="stablefold_dkdv",
            aot_module="openfold3.core.kernels.cuda._ext._stablefold_dkdv_sm90",
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "stablefold_triangle_dkdv_out",
                "stablefold_triangle_dkdv_delta_true_out",
            ),
            verbose=bool(verbose),
        )
    return _EXTENSION


_KEY_BLOCK = 64


def delta_true_partial_elements(
    sequence: int, heads: int, batch: int, outer: int | None = None
) -> int:
    """FP32 element count of the [rows, key_blocks] T-partial workspace."""

    if outer is None:
        outer = sequence
    key_blocks = (sequence + _KEY_BLOCK - 1) // _KEY_BLOCK
    return batch * outer * heads * sequence * key_blocks


def stablefold_triangle_dkdv(
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
    emit_delta_true: bool = False,
    delta_true_partials: torch.Tensor | None = None,
    out_delta_true: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """dK/dV; optionally also the consistent delta_true = sum_k P_k dP_k.

    With ``emit_delta_true`` the kernel additionally folds the row scalars
    T = sum_k w dP and Z = sum_k w per key block (fixed order, single
    writer, zero atomics) and a follow-up reduce produces
    ``delta_true = T/Z`` in the lse/delta row layout. dK/dV bits are
    identical in both modes.
    """

    if out_dk is None:
        out_dk = torch.empty_like(q)
    if out_dv is None:
        out_dv = torch.empty_like(q)
    if not emit_delta_true:
        with torch.cuda.device(q.device), torch.no_grad():
            load_extension().stablefold_triangle_dkdv_out(
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
    batch, outer, heads = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    sequence = int(q.shape[3])
    required = delta_true_partial_elements(sequence, heads, batch, outer=outer)
    if delta_true_partials is None:
        delta_true_partials = torch.empty(
            required, device=q.device, dtype=torch.float32
        )
    elif delta_true_partials.numel() < required:
        raise ValueError(
            f"delta_true_partials has {delta_true_partials.numel()} elements, "
            f"but this shape requires {required}"
        )
    if out_delta_true is None:
        out_delta_true = torch.empty_like(logsumexp)
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().stablefold_triangle_dkdv_delta_true_out(
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
            delta_true_partials,
            out_delta_true.reshape(-1),
        )
    return out_dk, out_dv, out_delta_true


__all__ = [
    "stablefold_triangle_dkdv",
    "delta_true_partial_elements",
    "load_extension",
    "source_sha256",
]
