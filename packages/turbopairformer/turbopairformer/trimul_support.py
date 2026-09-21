# SPDX-License-Identifier: Apache-2.0

"""Metadata-only support contract for the TurboPairFormer TriMul fast path.

This module deliberately does not import :mod:`turbopairformer._trimul`, load an
extension, or launch a kernel.  It separates the Hopper sm90a
fast-path domain from future selector/fallback policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any

import torch

SUPPORTED_CHANNELS = frozenset((32, 64, 96, 128))
SUPPORTED_MASK_DTYPES = frozenset((torch.bool, torch.bfloat16, torch.float32))

# Fixed selector boundary. This is intentionally not environment-configurable:
# short TriMul sites remain on the stock PyTorch route, while the AOT contract
# begins at the first sequence above this threshold.
TRIMUL_STOCK_FALLBACK_THRESHOLD = 100

# Hopper launch limits used by the current CUDA implementation.  Grid x has
# the larger signed-int limit; grid y/z use the legacy 16-bit limit.
MAX_GRID_X = 2_147_483_647
MAX_GRID_YZ = 65_535
MAX_FLATTENED_BATCH = MAX_GRID_YZ

_DGB_BLOCKS_PER_SM = 16
_DGB_ROW_QUANTUM = 64
_DGB_MAX_CHUNK_ROWS = 2048

_LINEAR_NAMES = (
    "linear_a_p",
    "linear_a_g",
    "linear_b_p",
    "linear_b_g",
    "linear_g",
    "linear_z",
)
_LAYER_NORM_NAMES = ("layer_norm_in", "layer_norm_out")


@dataclass(frozen=True)
class SupportResult:
    """Reasoned fast-path verdict suitable for a future fail-closed selector."""

    supported: bool
    code: str
    reason: str
    batch: int | None = None
    sequence: int | None = None
    channels: int | None = None

    def __bool__(self) -> bool:
        return self.supported


@dataclass(frozen=True)
class TensorMetadata:
    """Tensor properties consumed by the pure metadata evaluator."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    contiguous: bool
    requires_grad: bool

    @property
    def device_key(self) -> tuple[str, int | None]:
        return self.device_type, self.device_index


@dataclass(frozen=True)
class NamedTensorMetadata:
    name: str
    tensor: TensorMetadata


@dataclass(frozen=True)
class TrimulSupportMetadata:
    """Normalized invocation metadata; constructible without a CUDA device."""

    z: TensorMetadata
    mask: TensorMetadata | None
    device_name: str
    capability: tuple[int, int]
    multiprocessor_count: int
    c_z: int
    c_hidden: int
    outgoing: bool
    weights: tuple[NamedTensorMetadata, ...]
    linear_bias_is_none: tuple[tuple[str, bool], ...]
    layer_norm_eps: tuple[tuple[str, float], ...]


def _accepted(batch: int, sequence: int, channels: int) -> SupportResult:
    return SupportResult(
        True,
        "supported",
        (
            "supported Hopper sm90a BF16 TriMul fast path: "
            f"batch={batch}, N={sequence}, C_z=C_hidden={channels}"
        ),
        batch=batch,
        sequence=sequence,
        channels=channels,
    )


def _rejected(code: str, reason: str) -> SupportResult:
    return SupportResult(False, code, reason)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _expected_weight_shapes(c_z: int, c_hidden: int) -> dict[str, tuple[int, ...]]:
    return {
        "linear_a_p.weight": (c_hidden, c_z),
        "linear_a_g.weight": (c_hidden, c_z),
        "linear_b_p.weight": (c_hidden, c_z),
        "linear_b_g.weight": (c_hidden, c_z),
        "linear_g.weight": (c_z, c_z),
        "linear_z.weight": (c_z, c_hidden),
        "layer_norm_in.weight": (c_z,),
        "layer_norm_in.bias": (c_z,),
        "layer_norm_out.weight": (c_hidden,),
        "layer_norm_out.bias": (c_hidden,),
    }


def _broadcasts_exactly(source: tuple[int, ...], target: tuple[int, ...]) -> bool:
    if len(source) > len(target):
        return False
    padded = (1,) * (len(target) - len(source)) + source
    return all(actual in (1, expected) for actual, expected in zip(padded, target))


def evaluate_trimul_support(
    metadata: TrimulSupportMetadata,
) -> SupportResult:
    """Evaluate the TurboPairFormer TriMul contract without CUDA/JIT side effects."""

    if metadata.capability != (9, 0):
        return _rejected(
            "architecture",
            "TurboPairFormer TriMul requires Hopper sm90a (H100/H200), "
            "compute capability 9.0",
        )
    if metadata.multiprocessor_count <= 0:
        return _rejected(
            "device_metadata",
            "Hopper multiprocessor_count must be a positive integer",
        )

    z = metadata.z
    if z.device_type != "cuda":
        return _rejected("z_device", "z must be a CUDA tensor")
    if z.dtype is not torch.bfloat16:
        return _rejected("z_dtype", "z must have dtype torch.bfloat16")
    if len(z.shape) < 3:
        return _rejected("z_rank", "z must have shape [*B, N, N, C_z]")
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in z.shape):
        return _rejected("z_shape", "every z dimension must be a positive integer")

    prefix = z.shape[:-3]
    row_sequence, col_sequence, tensor_channels = z.shape[-3:]
    if row_sequence != col_sequence:
        return _rejected(
            "sequence_not_square",
            f"z pair axes must be square, got {row_sequence}x{col_sequence}",
        )
    sequence = row_sequence
    if sequence <= 0:
        return _rejected("sequence", "TriMul requires N > 0")

    if type(metadata.c_z) is not int or type(metadata.c_hidden) is not int:
        return _rejected("module_channels", "module c_z and c_hidden must be integers")
    if metadata.c_z != tensor_channels:
        return _rejected(
            "module_channels",
            f"module c_z={metadata.c_z} does not match z channels={tensor_channels}",
        )
    if metadata.c_z != metadata.c_hidden:
        return _rejected(
            "channels_not_equal",
            (
                "current TurboPairFormer TriMul fast path requires c_z == c_hidden; "
                "unequal channels must use a future stock fallback"
            ),
        )
    channels = metadata.c_z
    if channels not in SUPPORTED_CHANNELS:
        return _rejected(
            "channels",
            "c_z == c_hidden must be one of {32, 64, 96, 128}",
        )
    if type(metadata.outgoing) is not bool:
        return _rejected(
            "direction",
            "module _outgoing must be an explicit bool",
        )

    batch = prod(prefix) if prefix else 1
    if batch > MAX_FLATTENED_BATCH:
        return _rejected(
            "batch_grid_z",
            (
                f"flattened batch={batch} exceeds CUDA grid-z limit "
                f"{MAX_FLATTENED_BATCH}"
            ),
        )

    nn = sequence * sequence
    rows = batch * nn
    transpose_tiles = _ceil_div(nn, 32)
    if transpose_tiles > MAX_GRID_X:
        return _rejected(
            "transpose_grid_x",
            "ceil(N*N/32) exceeds the CUDA grid-x limit",
        )
    transpose_chunks = transpose_tiles * batch
    if transpose_chunks > MAX_GRID_X:
        return _rejected(
            "transpose_chunks",
            "flattened transpose reduction chunks exceed signed int32",
        )
    if _ceil_div(rows, 8) > MAX_GRID_X:
        return _rejected(
            "layer_norm_grid_x",
            "ceil(batch*N*N/8) exceeds the CUDA grid-x limit",
        )
    if _ceil_div(rows * channels, 256) > MAX_GRID_X:
        return _rejected(
            "sigmoid_grid_x",
            "ceil(batch*N*N*C/256) exceeds the CUDA grid-x limit",
        )

    slices = channels // 32
    persistent_grid_y = max(
        1,
        min(
            _ceil_div(rows, 128),
            (2 * metadata.multiprocessor_count) // slices,
        ),
    )
    if persistent_grid_y > MAX_GRID_YZ:
        return _rejected(
            "persistent_grid_y",
            "persistent projection grid-y exceeds the CUDA grid-y limit",
        )

    target_chunks = metadata.multiprocessor_count * _DGB_BLOCKS_PER_SM
    chunk_rows = _ceil_div(rows, target_chunks)
    chunk_rows = _ceil_div(chunk_rows, _DGB_ROW_QUANTUM) * _DGB_ROW_QUANTUM
    chunk_rows = max(_DGB_ROW_QUANTUM, min(_DGB_MAX_CHUNK_ROWS, chunk_rows))
    dgb_chunks = _ceil_div(rows, chunk_rows)
    if dgb_chunks > MAX_GRID_YZ:
        return _rejected(
            "layer_norm_dgb_grid_y",
            (
                f"LayerNorm parameter-gradient chunks={dgb_chunks} exceed "
                f"CUDA grid-y limit {MAX_GRID_YZ}"
            ),
        )

    expected_shapes = _expected_weight_shapes(metadata.c_z, metadata.c_hidden)
    named_weights = {item.name: item.tensor for item in metadata.weights}
    if len(named_weights) != len(metadata.weights) or set(named_weights) != set(
        expected_shapes
    ):
        return _rejected(
            "module_parameters",
            "module must expose exactly the TurboPairFormer TriMul "
            "weight/LN parameters",
        )
    for name, expected_shape in expected_shapes.items():
        tensor = named_weights[name]
        if tensor.shape != expected_shape:
            return _rejected(
                "parameter_shape",
                f"{name} must have shape {expected_shape}, got {tensor.shape}",
            )
        if tensor.dtype is not torch.float32:
            return _rejected(
                "parameter_dtype",
                f"{name} must have dtype torch.float32",
            )
        if not tensor.contiguous:
            return _rejected(
                "parameter_layout",
                f"{name} must be contiguous",
            )
        if tensor.device_key != z.device_key:
            return _rejected(
                "parameter_device",
                f"{name} must be on the same CUDA device as z",
            )

    bias_states = dict(metadata.linear_bias_is_none)
    if len(bias_states) != len(metadata.linear_bias_is_none) or set(bias_states) != set(
        _LINEAR_NAMES
    ):
        return _rejected(
            "linear_biases",
            "module must report every TurboPairFormer TriMul Linear bias",
        )
    for name in _LINEAR_NAMES:
        if not bias_states[name]:
            return _rejected(
                "linear_bias",
                f"{name}.bias must be None for the TurboPairFormer fast path",
            )

    eps_values = dict(metadata.layer_norm_eps)
    if len(eps_values) != len(metadata.layer_norm_eps) or set(eps_values) != set(
        _LAYER_NORM_NAMES
    ):
        return _rejected(
            "layer_norm_eps",
            "module must report both LayerNorm epsilon values",
        )
    for name in _LAYER_NORM_NAMES:
        if eps_values[name] != 1e-5:
            return _rejected(
                "layer_norm_eps",
                f"{name}.eps must equal 1e-5 exactly",
            )

    mask = metadata.mask
    if mask is not None:
        if mask.device_key != z.device_key:
            return _rejected(
                "mask_device",
                "mask must be on the same CUDA device as z",
            )
        if mask.dtype not in SUPPORTED_MASK_DTYPES:
            return _rejected(
                "mask_dtype",
                "mask dtype must be bool, bfloat16, or float32",
            )
        if mask.requires_grad:
            return _rejected("mask_grad", "mask.requires_grad must be False")
        if any(
            not isinstance(dimension, int) or dimension <= 0 for dimension in mask.shape
        ):
            return _rejected(
                "mask_shape",
                "every mask dimension must be a positive integer",
            )
        if not _broadcasts_exactly(mask.shape, z.shape[:-1]):
            return _rejected(
                "mask_broadcast",
                f"mask shape {mask.shape} cannot broadcast to {z.shape[:-1]}",
            )

    if sequence <= TRIMUL_STOCK_FALLBACK_THRESHOLD:
        return _rejected(
            "sequence_stock_fallback",
            (
                f"N={sequence} is at or below the fixed stock PyTorch fallback "
                f"threshold {TRIMUL_STOCK_FALLBACK_THRESHOLD}; "
                "TurboPairFormer TriMul AOT starts at N=101"
            ),
        )

    return _accepted(batch, sequence, channels)


def _tensor_metadata(tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device_type=tensor.device.type,
        device_index=tensor.device.index,
        contiguous=tensor.is_contiguous(),
        requires_grad=tensor.requires_grad,
    )


def check_trimul_tensor_support(
    z: torch.Tensor,
    direction: str = "outgoing",
    mask: torch.Tensor | None = None,
    *,
    norm_in_weight: torch.Tensor | None = None,
    norm_in_bias: torch.Tensor | None = None,
    p_in_weight: torch.Tensor | None = None,
    g_in_weight: torch.Tensor | None = None,
    norm_out_weight: torch.Tensor | None = None,
    norm_out_bias: torch.Tensor | None = None,
    p_out_weight: torch.Tensor | None = None,
    g_out_weight: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> SupportResult:
    """Validate functional inputs before allocating any optional parameters.

    Missing parameters are represented only by metadata here. The caller creates
    demonstration tensors after this check accepts the complete invocation.
    """
    if not isinstance(z, torch.Tensor):
        return _rejected("z_type", "z must be a torch.Tensor")
    if z.device.type != "cuda":
        return _rejected("z_device", "z must be a CUDA tensor")
    if z.ndim < 3:
        return _rejected("z_rank", "z must have shape [*B, N, N, C_z]")
    if mask is not None and not isinstance(mask, torch.Tensor):
        return _rejected("mask_type", "mask must be None or a torch.Tensor")
    if not isinstance(direction, str) or direction not in ("outgoing", "incoming"):
        return _rejected("direction", "direction must be 'outgoing' or 'incoming'")
    if not isinstance(eps, (int, float)) or eps != 1e-5:
        return _rejected("layer_norm_eps", "eps must equal 1e-5 exactly")

    channels = z.shape[-1]
    parameters = {
        "norm_in_weight": (norm_in_weight, (channels,)),
        "norm_in_bias": (norm_in_bias, (channels,)),
        "p_in_weight": (p_in_weight, (2 * channels, channels)),
        "g_in_weight": (g_in_weight, (2 * channels, channels)),
        "norm_out_weight": (norm_out_weight, (channels,)),
        "norm_out_bias": (norm_out_bias, (channels,)),
        "p_out_weight": (p_out_weight, (channels, channels)),
        "g_out_weight": (g_out_weight, (channels, channels)),
    }
    parameter_metadata: dict[str, TensorMetadata] = {}
    for name, (parameter, shape) in parameters.items():
        if parameter is None:
            parameter_metadata[name] = TensorMetadata(
                shape=shape,
                dtype=torch.float32,
                device_type=z.device.type,
                device_index=z.device.index,
                contiguous=True,
                requires_grad=False,
            )
        elif not isinstance(parameter, torch.Tensor):
            return _rejected("parameter_type", f"{name} must be None or a torch.Tensor")
        elif tuple(parameter.shape) != shape:
            return _rejected(
                "parameter_shape",
                f"{name} must have shape {shape}, got {tuple(parameter.shape)}",
            )
        else:
            parameter_metadata[name] = _tensor_metadata(parameter)

    # The native ABI uses separate a/b matrices. A contiguous [2C, C] tensor
    # yields contiguous row slices, without performing any tensor operation here.
    projection = replace(parameter_metadata["p_in_weight"], shape=(channels, channels))
    gate = replace(parameter_metadata["g_in_weight"], shape=(channels, channels))
    weights = {
        "linear_a_p.weight": projection,
        "linear_a_g.weight": gate,
        "linear_b_p.weight": projection,
        "linear_b_g.weight": gate,
        "linear_g.weight": parameter_metadata["g_out_weight"],
        "linear_z.weight": parameter_metadata["p_out_weight"],
        "layer_norm_in.weight": parameter_metadata["norm_in_weight"],
        "layer_norm_in.bias": parameter_metadata["norm_in_bias"],
        "layer_norm_out.weight": parameter_metadata["norm_out_weight"],
        "layer_norm_out.bias": parameter_metadata["norm_out_bias"],
    }
    try:
        capability = tuple(torch.cuda.get_device_capability(z.device))
        device_name = torch.cuda.get_device_name(z.device)
        multiprocessor_count = int(
            torch.cuda.get_device_properties(z.device).multi_processor_count
        )
    except (AssertionError, RuntimeError, ValueError) as error:
        return _rejected(
            "device_metadata",
            f"could not query CUDA device metadata: {error}",
        )
    return evaluate_trimul_support(
        TrimulSupportMetadata(
            z=_tensor_metadata(z),
            mask=None if mask is None else _tensor_metadata(mask),
            device_name=device_name,
            capability=(int(capability[0]), int(capability[1])),
            multiprocessor_count=multiprocessor_count,
            c_z=channels,
            c_hidden=channels,
            outgoing=direction == "outgoing",
            weights=tuple(
                NamedTensorMetadata(name, value) for name, value in weights.items()
            ),
            linear_bias_is_none=tuple((name, True) for name in _LINEAR_NAMES),
            layer_norm_eps=tuple((name, eps) for name in _LAYER_NORM_NAMES),
        )
    )


def check_trimul_support(
    module: Any,
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> SupportResult:
    """Inspect a live invocation without importing or building the extension."""

    if not isinstance(z, torch.Tensor):
        return _rejected("z_type", "z must be a torch.Tensor")
    if z.device.type != "cuda":
        return _rejected("z_device", "z must be a CUDA tensor")
    if mask is not None and not isinstance(mask, torch.Tensor):
        return _rejected("mask_type", "mask must be None or a torch.Tensor")

    if not hasattr(module, "c_z") or not hasattr(module, "c_hidden"):
        return _rejected(
            "module_structure",
            "module must expose c_z and c_hidden",
        )
    if type(getattr(module, "_outgoing", None)) is not bool:
        return _rejected(
            "direction",
            "module _outgoing must be an explicit bool",
        )
    if hasattr(module, "linear_ab_p") or hasattr(module, "linear_ab_g"):
        return _rejected(
            "fused_projection_layout",
            (
                "fused TriMul projection weights are not supported by the "
                "current TurboPairFormer AOT ABI"
            ),
        )

    weights: list[NamedTensorMetadata] = []
    bias_states: list[tuple[str, bool]] = []
    for name in _LINEAR_NAMES:
        layer = getattr(module, name, None)
        if layer is None or not isinstance(
            getattr(layer, "weight", None), torch.Tensor
        ):
            return _rejected(
                "module_structure",
                f"module must expose {name}.weight",
            )
        weights.append(
            NamedTensorMetadata(f"{name}.weight", _tensor_metadata(layer.weight))
        )
        bias_states.append((name, getattr(layer, "bias", object()) is None))

    eps_values: list[tuple[str, float]] = []
    for name in _LAYER_NORM_NAMES:
        layer = getattr(module, name, None)
        if layer is None:
            return _rejected("module_structure", f"module must expose {name}")
        for parameter_name in ("weight", "bias"):
            parameter = getattr(layer, parameter_name, None)
            if not isinstance(parameter, torch.Tensor):
                return _rejected(
                    "module_structure",
                    f"module must expose {name}.{parameter_name}",
                )
            weights.append(
                NamedTensorMetadata(
                    f"{name}.{parameter_name}", _tensor_metadata(parameter)
                )
            )
        eps = getattr(layer, "eps", None)
        if not isinstance(eps, (int, float)):
            return _rejected(
                "module_structure",
                f"module must expose numeric {name}.eps",
            )
        eps_values.append((name, float(eps)))

    try:
        capability = tuple(torch.cuda.get_device_capability(z.device))
        device_name = torch.cuda.get_device_name(z.device)
        multiprocessor_count = int(
            torch.cuda.get_device_properties(z.device).multi_processor_count
        )
    except (AssertionError, RuntimeError, ValueError) as error:
        return _rejected(
            "device_metadata",
            f"could not query CUDA device metadata: {error}",
        )

    metadata = TrimulSupportMetadata(
        z=_tensor_metadata(z),
        mask=None if mask is None else _tensor_metadata(mask),
        device_name=device_name,
        capability=(int(capability[0]), int(capability[1])),
        multiprocessor_count=multiprocessor_count,
        c_z=module.c_z,
        c_hidden=module.c_hidden,
        outgoing=module._outgoing,
        weights=tuple(weights),
        linear_bias_is_none=tuple(bias_states),
        layer_norm_eps=tuple(eps_values),
    )
    return evaluate_trimul_support(metadata)


__all__ = [
    "MAX_FLATTENED_BATCH",
    "MAX_GRID_X",
    "MAX_GRID_YZ",
    "TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "SUPPORTED_CHANNELS",
    "SUPPORTED_MASK_DTYPES",
    "NamedTensorMetadata",
    "TrimulSupportMetadata",
    "SupportResult",
    "TensorMetadata",
    "check_trimul_support",
    "check_trimul_tensor_support",
    "evaluate_trimul_support",
]
