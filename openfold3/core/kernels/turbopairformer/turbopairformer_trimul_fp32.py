# SPDX-License-Identifier: Apache-2.0
"""FP32-storage triangle multiplication for inference.

This is a separate implementation, not a call to the stock or cuEq module.
The TF32 path normalizes each input row once, then fuses its projections,
gates and layout packing in the sibling CUDA extension. Output normalization,
projection and gating remain fused. The triangle contraction is a vendor GEMM.
Strict FP32 and uncommon channel sizes use the independent vendor composition.
All activations, projection results, contraction results and outputs are FP32.
The GEMMs inherit the caller's TF32 policy and CUDA overrides. Autocast is
disabled locally; this module never changes global precision settings.

For each flattened batch item the stages are:
  1. Normalize input row chunks, project ap/ag/bp/bg/output-gate together, and
     pack gated, masked A/B into channel-major [H, N, N] storage.
  2. Contract output-row chunks with vendor batched GEMM: outgoing A @ B.T,
     incoming A.T @ B. Every chunk retains the complete reduction dimension N.
  3. Normalize the product across H, project to C, and apply the output gate.

Only A, B and the output gate span a complete batch item. Input projection
scratch is bounded by ``projection_workspace_bytes``; product/output scratch
is bounded by ``output_row_chunk_size``. The returned update is independent
storage. Input z is never overwritten and residual addition belongs to callers.
The TF32 path adds a normalized FP32 input chunk. Projection chunks hold a
whole number of 64-row tiles (the budget is rounded down). Its scratch uses
C / (2*C + 4*H) of the projection budget: about 21.33 MiB at C=H with the
default 128 MiB budget.
"""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

_EXTENSION: Any = None
_INVOCATIONS = 0
_MODULE_KEY = "turbopairformer_trimul_fp32"
_AOT_MODULE = (
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_trimul_fp32_sm90a"
)
_EXPORTS = (
    "trimul_fp32_gate_pack_out",
    "trimul_fp32_gate_output_out",
    "trimul_fp32_input_fused_out",
    "trimul_fp32_output_fused_out",
    "trimul_fp32_normalize_input_out",
)


def load_extension(*, verbose: bool = False) -> Any:
    """Use the shared fail-closed AOT / explicit-development-JIT policy."""
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.turbopairformer._aot_runtime import (
            load_extension_module,
        )

        _EXTENSION = load_extension_module(
            module_key=_MODULE_KEY,
            aot_module=_AOT_MODULE,
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=_EXPORTS,
            verbose=verbose,
        )
    return _EXTENSION


def invocation_count() -> int:
    """Number of completed Python dispatches through this FP32 implementation."""
    return _INVOCATIONS


def _check_parameter(tensor, shape, device, name, *, optional=False):
    if tensor is None and optional:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.dtype != torch.float32 or tensor.device != device:
        raise ValueError(f"{name} must be FP32 on {device}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )


def _linear_parts(layer, rows, channels, device, name):
    _check_parameter(layer.weight, (rows, channels), device, name + ".weight")
    _check_parameter(layer.bias, (rows,), device, name + ".bias", optional=True)
    # A per-layer cast request would change the FP32 policy. Do not silently
    # bypass such a request when using functional vendor primitives directly.
    if getattr(layer, "precision", None) not in (None, torch.float32):
        raise ValueError(f"{name}.precision requests a non-FP32 computation")
    return layer.weight, layer.bias


def _normalization_parts(layer, channels, device, name):
    _check_parameter(layer.weight, (channels,), device, name + ".weight", optional=True)
    _check_parameter(layer.bias, (channels,), device, name + ".bias", optional=True)
    if not isinstance(layer.eps, (int, float)) or not 0 < layer.eps < float("inf"):
        raise ValueError(f"{name}.eps must be finite and positive")
    return layer.weight, layer.bias, float(layer.eps)


def _packed_input_parameters(module, channels, hidden, device):
    """Pack the same equation as [ap, ag, bp, bg, output_gate]."""
    unfused_names = ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g")
    has_unfused = all(hasattr(module, name) for name in unfused_names)
    has_fused = all(hasattr(module, name) for name in ("linear_ab_p", "linear_ab_g"))
    if has_unfused == has_fused:
        raise ValueError(
            "TriMul module must expose exactly one fused or unfused projection layout"
        )
    if has_unfused:
        parts = [
            _linear_parts(getattr(module, name), hidden, channels, device, name)
            for name in unfused_names
        ]
    else:
        projection, projection_bias = _linear_parts(
            module.linear_ab_p, 2 * hidden, channels, device, "linear_ab_p"
        )
        gate, gate_bias = _linear_parts(
            module.linear_ab_g, 2 * hidden, channels, device, "linear_ab_g"
        )
        parts = [
            (
                projection[:hidden],
                None if projection_bias is None else projection_bias[:hidden],
            ),
            (gate[:hidden], None if gate_bias is None else gate_bias[:hidden]),
            (
                projection[hidden:],
                None if projection_bias is None else projection_bias[hidden:],
            ),
            (gate[hidden:], None if gate_bias is None else gate_bias[hidden:]),
        ]
    parts.append(_linear_parts(module.linear_g, channels, channels, device, "linear_g"))
    packed_weight = torch.cat([weight for weight, _ in parts], dim=0)
    packed_bias = None
    if any(bias is not None for _, bias in parts):
        packed_bias = torch.cat(
            [
                weight.new_zeros(weight.shape[0]) if bias is None else bias
                for weight, bias in parts
            ]
        )
    return packed_weight, packed_bias


def run_trimul_fp32(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    output_row_chunk_size: int | None = None,
    projection_workspace_bytes: int = 128 * 1024 * 1024,
    contraction_workspace_bytes: int = 512 * 1024 * 1024,
) -> torch.Tensor:
    """Return a pure FP32 TriMul update for ``z[*B,N,N,C]`` under no_grad.

    Both AF2 separate projections and AF2-Multimer fused projections are
    accepted, with arbitrary positive C/H and incoming/outgoing contraction.
    A bool or FP32 mask must broadcast exactly to ``z.shape[:-1]``. Logical
    tails and non-contiguous tensors are supported; no shape falls back to a
    different implementation. Invalid inputs raise before extension dispatch.

    The triangle contraction runs over chunks of outer rows. By default a
    chunk holds as many outer rows as fit ``contraction_workspace_bytes`` of
    FP32 product scratch ([H, rows, N]); ``output_row_chunk_size`` sets the
    number of rows directly instead.
    """
    global _INVOCATIONS
    if torch.is_grad_enabled():
        raise RuntimeError(
            "run_trimul_fp32 is inference-only; "
            "enter torch.no_grad() or inference_mode()"
        )
    if (
        not isinstance(z, torch.Tensor)
        or z.device.type != "cuda"
        or z.dtype != torch.float32
    ):
        raise ValueError("z must be a CUDA FP32 tensor")
    if z.ndim < 3 or any(dimension <= 0 for dimension in z.shape):
        raise ValueError("z must have nonempty shape [*B,N,N,C]")
    sequence, pair_columns, channels = z.shape[-3:]
    if sequence != pair_columns:
        raise ValueError("z must have square pair axes")
    hidden = getattr(module, "c_hidden", None)
    if (
        type(hidden) is not int
        or hidden <= 0
        or getattr(module, "c_z", None) != channels
    ):
        raise ValueError("module channels must match z, with positive c_hidden")
    if type(getattr(module, "_outgoing", None)) is not bool:
        raise ValueError("module._outgoing must be a bool")
    if output_row_chunk_size is None:
        # Few large contraction calls: at N=768, 128-row calls ran cuBLAS at
        # about 28% of TF32 peak and one full call at about 64%. On the
        # qualified H100 stack every chunk size from 128 rows to N gave
        # bitwise-identical outputs; only speed and scratch memory change.
        if (
            type(contraction_workspace_bytes) is not int
            or contraction_workspace_bytes <= 0
        ):
            raise ValueError("contraction_workspace_bytes must be a positive integer")
        contraction_bytes_per_outer_row = hidden * sequence * z.element_size()
        output_row_chunk_size = max(
            1,
            min(
                sequence, contraction_workspace_bytes // contraction_bytes_per_outer_row
            ),
        )
    elif type(output_row_chunk_size) is not int or output_row_chunk_size <= 0:
        raise ValueError("output_row_chunk_size must be a positive integer")
    packed_channels = 4 * hidden + channels
    projection_bytes_per_row = (channels + packed_channels) * z.element_size()
    if (
        type(projection_workspace_bytes) is not int
        or projection_workspace_bytes < projection_bytes_per_row
    ):
        raise ValueError(
            "projection_workspace_bytes must fit one FP32 normalized/projected row"
        )
    projection_rows = projection_workspace_bytes // projection_bytes_per_row
    # Whole 64-row tiles keep every chunk start aligned for the persistent
    # TF32 kernel's vector stores. Each row's result is independent of the
    # chunk it belongs to, so the boundaries do not change any output value.
    if projection_rows >= 64:
        projection_rows -= projection_rows % 64
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.device != z.device:
            raise ValueError("mask must be a tensor on z's device")
        if mask.dtype not in (torch.bool, torch.float32):
            raise ValueError("mask must have dtype bool or FP32")
        try:
            mask = torch.broadcast_to(mask, z.shape[:-1])
        except RuntimeError as error:
            raise ValueError("mask must broadcast exactly to z.shape[:-1]") from error

    norm_in = _normalization_parts(
        module.layer_norm_in, channels, z.device, "layer_norm_in"
    )
    norm_out = _normalization_parts(
        module.layer_norm_out, hidden, z.device, "layer_norm_out"
    )
    output_weight, output_bias = _linear_parts(
        module.linear_z, channels, hidden, z.device, "linear_z"
    )
    with torch.cuda.device(z.device), torch.autocast("cuda", enabled=False):
        input_weight, input_bias = _packed_input_parameters(
            module, channels, hidden, z.device
        )
        extension = load_extension()
        # This is a structural/mode dispatch, independent of input values. All
        # activations still use FP32 storage. Only MMA multiplicands are rounded
        # to TF32, with explicit round-to-nearest-even in the CUDA kernels.
        fused_tf32 = (
            torch.backends.cuda.matmul.allow_tf32
            and os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
            and channels in (32, 64, 96, 128)
            and hidden in (32, 64, 96, 128)
        )
        if fused_tf32:
            norm_in = tuple(
                value.contiguous() if isinstance(value, torch.Tensor) else value
                for value in norm_in
            )
            norm_out = tuple(
                value.contiguous() if isinstance(value, torch.Tensor) else value
                for value in norm_out
            )
            output_weight = output_weight.contiguous()
            output_bias = None if output_bias is None else output_bias.contiguous()
        update = torch.empty(z.shape, device=z.device, dtype=torch.float32)
        pair_rows = sequence * sequence
        prefixes = product(*(range(dimension) for dimension in z.shape[:-3]))
        for prefix in prefixes:
            # At most one non-contiguous batch item is packed at a time. A/B
            # and the gate also belong to one item, independent of batch size.
            input_rows = z[prefix].reshape(pair_rows, channels)
            mask_rows = (
                None
                if mask is None
                else mask[prefix].reshape(pair_rows).to(torch.float32).contiguous()
            )
            a = torch.empty(
                (hidden, sequence, sequence), device=z.device, dtype=torch.float32
            )
            b = torch.empty_like(a)
            output_gate = torch.empty(
                (pair_rows, channels), device=z.device, dtype=torch.float32
            )
            for first_row in range(0, pair_rows, projection_rows):
                last_row = min(first_row + projection_rows, pair_rows)
                if fused_tf32:
                    # One normalization is shared by all channel tiles. Keep
                    # the chunk budget and free this scratch after its
                    # projection dispatch on the same CUDA stream.
                    input_chunk = input_rows[first_row:last_row].contiguous()
                    normalized_input = torch.empty_like(input_chunk)
                    extension.trimul_fp32_normalize_input_out(
                        input_chunk,
                        norm_in[0],
                        norm_in[1],
                        norm_in[2],
                        normalized_input,
                    )
                    extension.trimul_fp32_input_fused_out(
                        normalized_input,
                        input_weight,
                        input_bias,
                        norm_in[0],
                        norm_in[1],
                        norm_in[2],
                        mask_rows,
                        a,
                        b,
                        output_gate,
                        first_row,
                        hasattr(module, "linear_ab_p"),
                        True,
                    )
                    del input_chunk, normalized_input
                    continue
                normalized = F.layer_norm(
                    input_rows[first_row:last_row], (channels,), *norm_in
                )
                projected = F.linear(normalized, input_weight, input_bias)
                extension.trimul_fp32_gate_pack_out(
                    projected, mask_rows, a, b, output_gate, first_row
                )
                del normalized, projected
            del input_rows, mask_rows

            output_rows = update[prefix].view(pair_rows, channels)
            right = b.transpose(1, 2) if module._outgoing else b
            for first_outer_row in range(0, sequence, output_row_chunk_size):
                last_outer_row = min(first_outer_row + output_row_chunk_size, sequence)
                left = (
                    a[:, first_outer_row:last_outer_row, :]
                    if module._outgoing
                    else a.transpose(1, 2)[:, first_outer_row:last_outer_row, :]
                )
                contracted = torch.bmm(left, right)
                first_row = first_outer_row * sequence
                last_row = last_outer_row * sequence
                if fused_tf32:
                    extension.trimul_fp32_output_fused_out(
                        contracted,
                        output_weight,
                        output_bias,
                        norm_out[0],
                        norm_out[1],
                        norm_out[2],
                        output_gate[first_row:last_row],
                        output_rows[first_row:last_row],
                    )
                    del contracted, left
                    continue
                product_rows = contracted.permute(1, 2, 0).contiguous().view(-1, hidden)
                del contracted
                normalized = F.layer_norm(product_rows, (hidden,), *norm_out)
                del product_rows
                projected = F.linear(normalized, output_weight, output_bias)
                del normalized
                extension.trimul_fp32_gate_output_out(
                    projected,
                    output_gate[first_row:last_row],
                    output_rows[first_row:last_row],
                )
                del projected, left
            del a, b, right, output_gate, output_rows
        _INVOCATIONS += 1
        return update


__all__ = ["run_trimul_fp32", "invocation_count", "load_extension"]
