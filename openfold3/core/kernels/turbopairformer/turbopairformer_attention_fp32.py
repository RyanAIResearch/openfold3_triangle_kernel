# SPDX-License-Identifier: Apache-2.0

"""Inference-only FP32 Triangle Attention with an explicit TF32 policy.

Q, K, V, additive mask, pair bias, softmax state, and output remain FP32.
TF32-enabled inference uses a fused CUDA online-softmax kernel with explicitly
rounded TF32 Tensor Core multiplicands. Three TF32 products retain QK leading
and cross terms to reduce softmax-sensitive rounding error; PV uses ordinary
TF32 rounding. Scores and probabilities stay on chip, and output is written
in its public layout directly. Matching Q/K/V strides with contiguous head
channels are read directly, including the model's transposed projection views.
Strict FP32 uses bounded
cuBLAS PEDANTIC QK/PV plus native CUDA softmax. Neither route dispatches to
cuEquivariance, PyTorch attention, or the BF16 implementation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any = None
_CAPABILITIES: dict[int, tuple[int, int]] = {}
_DEFAULT_WORKSPACE_BYTES = 64 * 1024 * 1024
_MAX_BLAS_DIMENSION = 2_147_483_647


def tf32_enabled() -> bool:
    """Read the current policy; NVIDIA's explicit disable always wins.

    Only NVIDIA_TF32_OVERRIDE=0 has documented disable semantics. Other values do
    not grant permission that the PyTorch matmul flag has denied. The extension's
    explicit PEDANTIC mode also prevents a host cuBLAS override from silently
    turning the disabled path into a Tensor Core GEMM.
    """
    return os.environ.get("NVIDIA_TF32_OVERRIDE") != "0" and bool(
        torch.backends.cuda.matmul.allow_tf32
    )


def is_attention_fp32_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    """Check metadata and inference mode without reading tensor values.

    Q/K/V are [*B, O, H, N, D], with at least one batch dimension. The mask
    broadcasts to [*B, O, 1, 1, N]; bias broadcasts to [*B, 1, H, N, N]. O can
    be an outer chunk, independently of the complete key/query sequence N.
    """
    if torch.is_grad_enabled():
        return False, "FP32 Triangle Attention is inference-only; use torch.no_grad()"
    named = {"q": q, "k": k, "v": v, "res_mask": res_mask, "pair_bias": pair_bias}
    if any(not isinstance(tensor, torch.Tensor) for tensor in named.values()):
        return False, "q/k/v, res_mask, and pair_bias must be tensors"
    if any(tensor.dtype != torch.float32 for tensor in named.values()):
        return False, "q/k/v, res_mask, and pair_bias must all be FP32"
    if not q.is_cuda or any(tensor.device != q.device for tensor in named.values()):
        return False, "all inputs must share one CUDA device"
    if q.ndim < 5 or any(dimension <= 0 for dimension in q.shape):
        return False, "q must have positive shape [*B, O, H, N, D]"
    if k.shape != q.shape or v.shape != q.shape:
        return False, "q/k/v must have identical shapes"
    outer, heads, sequence, head_dim = q.shape[-4:]
    if not 1 <= head_dim <= 64:
        return False, "head dimension must be between 1 and 64"
    if sequence > _MAX_BLAS_DIMENSION:
        return False, "sequence exceeds the cuBLAS integer dimension limit"
    prefix = tuple(q.shape[:-4])
    targets = {
        "res_mask": prefix + (outer, 1, 1, sequence),
        "pair_bias": prefix + (1, heads, sequence, sequence),
    }
    for name, target in targets.items():
        try:
            broadcast = torch.broadcast_shapes(named[name].shape, target)
        except RuntimeError:
            return False, f"{name} does not broadcast to {target}"
        if tuple(broadcast) != target:
            return False, f"{name} must broadcast exactly to {target}"
    device_index = q.device.index
    capability = _CAPABILITIES.get(device_index)
    if capability is None:
        capability = torch.cuda.get_device_capability(q.device)
        _CAPABILITIES[device_index] = capability
    if capability != (9, 0):
        return False, "FP32 Triangle Attention currently requires Hopper CC 9.0"
    return True, "supported inference-only FP32 Triangle Attention"


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> None:
    """Raise before loading an extension if the inference contract is invalid."""
    supported, reason = is_attention_fp32_supported(q, k, v, res_mask, pair_bias)
    if not supported:
        raise ValueError(reason)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Use the shared fail-closed AOT loader and its explicit development opt-in."""
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.turbopairformer._aot_runtime import (
            load_extension_module,
        )

        _EXTENSION = load_extension_module(
            module_key="turbopairformer_attention_fp32",
            aot_module=(
                "openfold3.core.kernels.turbopairformer._ext."
                "_turbopairformer_attention_fp32_sm90a"
            ),
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "turbopairformer_attention_fp32_forward",
                "turbopairformer_attention_fp32_module_projections",
                "turbopairformer_attention_fp32_module_layer_norm",
                "turbopairformer_attention_fp32_module_pair_bias",
                "turbopairformer_attention_fp32_module_chunk_projections",
                "turbopairformer_attention_fp32_gated_forward",
            ),
            verbose=bool(verbose),
        )
    return _EXTENSION


def turbopairformer_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Compute raw-Q attention, returning contiguous FP32 [*B, O, N, H, D].

    The kernel applies 1/sqrt(D) exactly once. Finite negative masks, including
    an entire row of -1e9, use ordinary additive softmax. A row whose final
    logits are all true -inf produces zero. NaNs are not sanitized.

    ``workspace_bytes`` bounds global score and temporary output tiles; the
    fused TF32 route does not allocate these tiles. It excludes the returned
    output, optional contiguous input copies, shared-memory tiles, and cuBLAS's
    own workspace. No autograd implementation is provided.
    """
    validate_inputs(q, k, v, res_mask, pair_bias)
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int):
        raise TypeError("workspace_bytes must be an integer")
    outer, heads, sequence, head_dim = q.shape[-4:]
    if not 4 * (sequence + head_dim) <= workspace_bytes <= 2**63 - 1:
        raise ValueError("workspace_bytes must hold at least one FP32 score/output row")
    prefix = tuple(q.shape[:-4])
    batch = math.prod(prefix)
    with torch.autocast("cuda", enabled=False), torch.cuda.device(q.device):
        # Flatten batch/sample axes without forcing the model's transposed
        # projection views into three temporary contiguous buffers. The fused
        # kernel accepts common Q/K/V strides with unit channel stride, and
        # guards 16-byte cp.async alignment independently in its C++ launcher.
        allow_tf32 = tf32_enabled()
        packed = [
            tensor.reshape(batch, outer, heads, sequence, head_dim)
            for tensor in (q, k, v)
        ]
        # A size-one axis has no address variation and need not share a stride.
        # contiguous() also legitimately preserves arbitrary singleton strides.
        common_strides = tuple(
            stride if size > 1 else 0
            for size, stride in zip(packed[0].shape, packed[0].stride())
        )
        direct_layout = (head_dim == 1 or packed[0].stride(-1) == 1) and all(
            tuple(
                stride if size > 1 else 0
                for size, stride in zip(tensor.shape, tensor.stride())
            )
            == common_strides
            for tensor in packed[1:]
        )
        if not allow_tf32 or not direct_layout:
            # Preserve values: layout copies, never dtype conversions. Strict
            # cuBLAS needs contiguous operands; arbitrary strides use this same
            # supported packing route before the fused kernel.
            packed = [tensor.contiguous() for tensor in packed]
        mask = (
            res_mask.expand(prefix + (outer, 1, 1, sequence))
            .reshape(batch, outer, sequence)
            .contiguous()
        )
        bias = (
            pair_bias.expand(prefix + (1, heads, sequence, sequence))
            .reshape(batch, heads, sequence, sequence)
            .contiguous()
        )
        output = load_extension().turbopairformer_attention_fp32_forward(
            *packed, mask, bias, allow_tf32, workspace_bytes
        )
    return output.reshape(prefix + (outer, sequence, heads, head_dim))


_MODULE_CHANNELS = 128
# The fused projections reproduce cuBLAS's TF32 GEMMs only where cuBLAS runs
# one ascending-K chain per output. For smaller matrices it selects other
# kernels whose bits differ. Measured on H100 with the Torch 2.10 / CUDA 12.9
# stack over 1050 row counts up to 160000: the [rows, 128] x [128, 128]
# projections match from 122 rows, the pair-bias projection to 4 heads from
# 4096 rows and to 8 heads from 2048 rows. The unfused module runs these
# GEMMs per batch item when it chunks, so the bound applies to N^2.
_MINIMUM_PAIR_ROWS = {4: 4096, 8: 2048}
# With outer-row chunks the unfused module runs the 128-wide GEMMs (q, k, v,
# gate, output) per chunk on chunk_rows * N rows; every chunk, including the
# last and smallest, must reach the 122-row bound above (rounded up to 128).
_MINIMUM_CHUNK_GEMM_ROWS = 128


def _aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % 16 == 0


def is_triangle_attention_module_fp32_supported(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    layer_norm_weight: torch.Tensor | None,
    layer_norm_bias: torch.Tensor | None,
    projection_weights: tuple[torch.Tensor, ...],
    projection_biases: tuple[torch.Tensor | None, ...],
    heads: int,
    head_dim: int,
    chunk_size: int | None = None,
) -> tuple[bool, str]:
    """Check the fused TriangleAttention module route without reading values.

    The route covers TF32-enabled inference of the trunk configuration: a
    square pair representation x [*, N, N, 128] (any leading strides, unit
    channel stride, 16-byte-aligned rows), an affine LayerNorm, bias-free
    q/k/v/gate/output projections to heads * head_dim = 128 with 4 or 8
    heads, and N^2 at least _MINIMUM_PAIR_ROWS[heads]. With outer-row chunks
    smaller than N, every chunk must hold at least _MINIMUM_CHUNK_GEMM_ROWS pair
    rows. Other calls keep the module's unfused route.
    """
    if torch.is_grad_enabled():
        return False, "inference-only"
    if not tf32_enabled():
        return False, "strict FP32 keeps the unfused module"
    if x.ndim < 4 or x.dtype != torch.float32 or not x.is_cuda:
        return False, "x must be FP32 CUDA [*, N, N, C]"
    if x.shape[-1] != _MODULE_CHANNELS or x.shape[-2] != x.shape[-3] or x.numel() == 0:
        return False, "x must be a non-empty square [*, N, N, 128] pair representation"
    if heads * head_dim != _MODULE_CHANNELS or heads not in _MINIMUM_PAIR_ROWS:
        return False, "requires heads * head_dim = 128 with 4 or 8 heads"
    sequence = x.shape[-2]
    if sequence * sequence < _MINIMUM_PAIR_ROWS[heads]:
        return (
            False,
            "below the cuBLAS large-matrix size; the unfused route keeps its bits",
        )
    if chunk_size is not None and chunk_size < sequence:
        if chunk_size < 1:
            return False, "chunk_size must be positive"
        smallest_chunk = sequence % chunk_size or chunk_size
        if smallest_chunk * sequence < _MINIMUM_CHUNK_GEMM_ROWS:
            return False, "a chunk is below the cuBLAS large-matrix size"
    if layer_norm_weight is None or layer_norm_bias is None:
        return False, "requires an affine LayerNorm"
    if any(bias is not None for bias in projection_biases):
        return False, "requires bias-free projections"
    if any(
        tuple(weight.shape) != (_MODULE_CHANNELS, _MODULE_CHANNELS)
        for weight in projection_weights[:5]
    ):
        return False, "q/k/v/gate/output weights must be [128, 128]"
    parameters = (layer_norm_weight, layer_norm_bias) + tuple(projection_weights)
    if any(
        p.dtype != torch.float32
        or p.device != x.device
        or not p.is_contiguous()
        or not _aligned(p)
        for p in parameters
    ):
        return False, "parameters must be contiguous, aligned FP32 on x's device"
    if (
        mask.shape != x.shape[:-1]
        or mask.dtype != torch.float32
        or mask.device != x.device
    ):
        return False, "mask must be FP32 [*, N, N] on x's device"
    if x.stride(-1) != 1 or not _aligned(x):
        return False, "x needs unit channel stride and 16-byte alignment"
    try:
        # A view (never a copy) with one batch axis, as the launch uses it.
        pair_rows = x.view(-1, *x.shape[-3:])
    except RuntimeError:
        return False, "x's batch axes must flatten without a copy"
    leading = zip(pair_rows.shape[:3], pair_rows.stride()[:3])
    if any(size > 1 and stride % 4 for size, stride in leading):
        return False, "x rows must be 16-byte aligned"
    device_index = x.device.index
    capability = _CAPABILITIES.get(device_index)
    if capability is None:
        capability = torch.cuda.get_device_capability(x.device)
        _CAPABILITIES[device_index] = capability
    if capability != (9, 0):
        return False, "requires Hopper CC 9.0"
    return True, "supported fused TF32 TriangleAttention module"


def turbopairformer_triangle_attention_module_fp32(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
    layer_norm_epsilon: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    output_weight: torch.Tensor,
    bias_weight: torch.Tensor,
    heads: int,
    head_dim: int,
    inf: float,
    transpose_bias: bool,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """TriangleAttention's forward (after its optional input transpose), fused.

    Bit-identical to the module's unfused TF32 route: PyTorch LayerNorm,
    cuBLAS TF32 linears, torch.sigmoid, the fused attention above, the
    ``output * gate`` product and linear_o. LayerNorm and the q/k/v/gate/
    pair-bias projections run in two kernels that write the operands once;
    the gate is applied inside the attention kernel. linear_o is the same
    nn.functional.linear call as the module's. The caller checks
    is_triangle_attention_module_fp32_supported first.

    With chunk_size < N the outer rows are processed in chunks, like the
    module's _chunk_stable: LayerNorm and the pair bias first cover every row,
    then each chunk runs its projections, gated attention and linear_o.
    """
    prefix = tuple(x.shape[:-3])
    sequence = x.shape[-2]
    batch = math.prod(prefix)
    if chunk_size is not None and chunk_size < sequence:
        return _module_fp32_chunked(
            x.view(batch, sequence, sequence, _MODULE_CHANNELS),
            mask,
            prefix,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_epsilon,
            q_weight,
            k_weight,
            v_weight,
            gate_weight,
            output_weight,
            bias_weight,
            heads,
            head_dim,
            inf,
            transpose_bias,
            chunk_size,
        )
    with torch.autocast("cuda", enabled=False), torch.cuda.device(x.device):
        extension = load_extension()
        pair_rows = x.view(batch, sequence, sequence, _MODULE_CHANNELS)
        projections = extension.turbopairformer_attention_fp32_module_projections
        q, k, v, gate, pair_bias = projections(
            pair_rows,
            layer_norm_weight,
            layer_norm_bias,
            layer_norm_epsilon,
            q_weight,
            k_weight,
            v_weight,
            gate_weight,
            bias_weight,
            transpose_bias,
        )
        # The module's additive mask, inf * (mask - 1), in the kernel's [B, O, N]
        # layout.
        mask_bias = (inf * (mask - 1)).reshape(batch, sequence, sequence)
        mask_bias = mask_bias.contiguous()

        def heads_view(tensor: torch.Tensor) -> torch.Tensor:
            heads_last = tensor.view(batch, sequence, sequence, heads, head_dim)
            return heads_last.transpose(-2, -3)

        gated = extension.turbopairformer_attention_fp32_gated_forward(
            heads_view(q), heads_view(k), heads_view(v), mask_bias, pair_bias, gate
        )
        output = torch.nn.functional.linear(
            gated.view(batch, sequence, sequence, heads * head_dim), output_weight
        )
    return output.view(prefix + (sequence, sequence, output.shape[-1]))


def _module_fp32_chunked(
    pair_rows: torch.Tensor,
    mask: torch.Tensor,
    prefix: tuple[int, ...],
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
    layer_norm_epsilon: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    output_weight: torch.Tensor,
    bias_weight: torch.Tensor,
    heads: int,
    head_dim: int,
    inf: float,
    transpose_bias: bool,
    chunk_size: int,
) -> torch.Tensor:
    """The chunked fused module; pair_rows is x as [B, N, N, 128].

    Memory follows the unfused _chunk_stable: x_ln and the pair bias cover
    every row, and q/k/v/gate exist for one chunk at a time. Each chunk's output
    overwrites the x_ln rows that its projections have already read (a later
    chunk's projections only read its own rows), so no separate output tensor
    is allocated.
    """
    batch, sequence = pair_rows.shape[0], pair_rows.shape[1]
    with torch.autocast("cuda", enabled=False), torch.cuda.device(pair_rows.device):
        extension = load_extension()
        normalized = extension.turbopairformer_attention_fp32_module_layer_norm(
            pair_rows, layer_norm_weight, layer_norm_bias, layer_norm_epsilon
        )
        pair_bias = extension.turbopairformer_attention_fp32_module_pair_bias(
            normalized, bias_weight, batch, sequence, transpose_bias
        )
        mask_bias = (inf * (mask - 1)).reshape(batch, sequence, sequence)
        mask_bias = mask_bias.contiguous()
        output = normalized.view(batch, sequence, sequence, _MODULE_CHANNELS)
        chunk_projections = (
            extension.turbopairformer_attention_fp32_module_chunk_projections
        )
        for item in range(batch):
            for start in range(0, sequence, chunk_size):
                rows = min(chunk_size, sequence - start)
                first_row = (item * sequence + start) * sequence
                q, k, v, gate = chunk_projections(
                    normalized,
                    first_row,
                    rows * sequence,
                    q_weight,
                    k_weight,
                    v_weight,
                    gate_weight,
                )

                def heads_view(tensor: torch.Tensor, rows: int = rows) -> torch.Tensor:
                    heads_last = tensor.view(1, rows, sequence, heads, head_dim)
                    return heads_last.transpose(-2, -3)

                gated = extension.turbopairformer_attention_fp32_gated_forward(
                    heads_view(q),
                    heads_view(k),
                    heads_view(v),
                    mask_bias[item : item + 1, start : start + rows],
                    pair_bias[item : item + 1],
                    gate.view(1, rows, sequence, heads * head_dim),
                )
                output[item, start : start + rows] = torch.nn.functional.linear(
                    gated.view(rows, sequence, heads * head_dim), output_weight
                )
    return output.view(prefix + (sequence, sequence, _MODULE_CHANNELS))


__all__ = [
    "is_attention_fp32_supported",
    "is_triangle_attention_module_fp32_supported",
    "load_extension",
    "tf32_enabled",
    "turbopairformer_attention_fp32",
    "turbopairformer_triangle_attention_module_fp32",
    "validate_inputs",
]
