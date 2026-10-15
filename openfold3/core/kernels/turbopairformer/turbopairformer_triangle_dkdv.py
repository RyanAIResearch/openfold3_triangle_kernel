# SPDX-License-Identifier: Apache-2.0

"""Load and call the split TurboPairFormer dK/dV CUDA kernels.

The sibling ``turbopairformer_triangle_dkdv.cu`` owns each key block's gradients
and can also emit a recomputed ``delta_true`` row scalar. This module prepares
output buffers and optional row-scalar workspace for direct callers. The public
attention facade uses the separate single-reconstruction backward route in
``turbopairformer_triangle_dq_dbias.py``.

The binary is installed under ``openfold3.core.kernels.turbopairformer._ext.*``;
``_aot_runtime`` owns its module name, required exports, and loading policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# Load on first use, then reuse the successfully loaded extension module.
_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return ``{source filename: SHA256}`` without loading the extension.

    See ``_aot_runtime.source_sha256_for`` for manifest precedence, local-file
    fallback, and errors. A recorded hash need not match the current source.
    """
    from openfold3.core.kernels.turbopairformer._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("turbopairformer_dkdv", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Load once using the shared AOT/JIT policy; return the cached module.

    See ``_aot_runtime.load_extension_module`` for validation and fallback.
    ``verbose`` controls JIT logging; ``None`` means ``False``.
    """
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.turbopairformer._aot_runtime import (
            _PRODUCTION_MODULE_CONTRACT,
            load_extension_module,
        )

        module_key = "turbopairformer_dkdv"
        aot_module, expected_symbols = _PRODUCTION_MODULE_CONTRACT[module_key]
        _EXTENSION = load_extension_module(
            module_key=module_key,
            aot_module=aot_module,
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=expected_symbols,
            verbose=bool(verbose),
        )
    return _EXTENSION


_KEY_BLOCK = 64  # Keys owned by one CUDA block; matches kKeyBlock in the source.


def delta_true_partial_elements(
    sequence: int, heads: int, batch: int, outer: int | None = None
) -> int:
    """Count FP32 elements for the optional ``delta_true`` partial workspace.

    There is one partial for every query row and 64-key block, so the count is
    ``batch * outer * heads * sequence * ceil(sequence / 64)``. Physical order
    is ``[batch, outer, heads, key_blocks, sequence]``. ``outer=None`` uses the
    historical square case ``outer=sequence``; explicit outer counts may differ.
    """

    if outer is None:
        outer = sequence
    key_blocks = (sequence + _KEY_BLOCK - 1) // _KEY_BLOCK
    return batch * outer * heads * sequence * key_blocks


def turbopairformer_triangle_dkdv(
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
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute dK/dV, optionally returning a recomputed ``delta_true`` as well.

    Q/K/V and dK/dV use ``[B, O, H, N, D]`` BF16; ``d_output`` uses
    ``[B, O, N, H, D]`` BF16. ``logsumexp`` from forward and ``delta`` from
    backward are FP32 row statistics, flattened for the native call. Mask and
    pair bias use ``[B, O, 1, 1, N]`` FP32 and ``[B, 1, H, N, N]`` BF16.
    Missing output buffers are allocated; supplied buffers are passed through.

    Both modes form ``dS = P * (dP - delta)`` using the supplied ``delta``.
    With ``emit_delta_true=True``, the kernel also sums FP32 ``P * dP`` within
    each key block. A fixed-order reduction combines those partials into an
    additional FP32 result in the logsumexp row layout. It does not divide by
    the recomputed ``sum(P)`` or replace the delta used for these dK/dV results.
    The return tuple has two tensors normally, or three when emitting delta_true.
    """

    if out_dk is None:
        out_dk = torch.empty_like(q)
    if out_dv is None:
        out_dv = torch.empty_like(q)
    if not emit_delta_true:
        # Plain mode needs no row-scalar workspace or follow-up reduction.
        with torch.cuda.device(q.device), torch.no_grad():
            load_extension().turbopairformer_triangle_dkdv_out(
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
    # Optional mode owns one FP32 partial per query row and key block.
    batch, outer, heads = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    sequence = int(q.shape[3])
    required_partial_elements = delta_true_partial_elements(
        sequence, heads, batch, outer=outer
    )
    if delta_true_partials is None:
        delta_true_partials = torch.empty(
            required_partial_elements, device=q.device, dtype=torch.float32
        )
    elif delta_true_partials.numel() < required_partial_elements:
        raise ValueError(
            f"delta_true_partials has {delta_true_partials.numel()} elements, "
            f"but this shape requires {required_partial_elements}"
        )
    if out_delta_true is None:
        out_delta_true = torch.empty_like(logsumexp)
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().turbopairformer_triangle_dkdv_delta_true_out(
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
    "turbopairformer_triangle_dkdv",
    "delta_true_partial_elements",
    "load_extension",
    "source_sha256",
]
