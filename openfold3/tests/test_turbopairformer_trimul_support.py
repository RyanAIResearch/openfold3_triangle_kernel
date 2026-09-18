# SPDX-License-Identifier: Apache-2.0

"""Tests for the side-effect-free TurboPairFormer TriMul support contract."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from openfold3.core.kernels.cuda import (
    STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD,
    check_stable_trimul_support,
    evaluate_stable_trimul_support,
)
from openfold3.core.kernels.cuda.turbopairformer_trimul_support import (
    MAX_FLATTENED_BATCH,
    NamedTensorMetadata,
    StableTrimulSupportMetadata,
    TensorMetadata,
)

_WEIGHT_SHAPES = {
    "linear_a_p.weight": lambda c, h: (h, c),
    "linear_a_g.weight": lambda c, h: (h, c),
    "linear_b_p.weight": lambda c, h: (h, c),
    "linear_b_g.weight": lambda c, h: (h, c),
    "linear_g.weight": lambda c, h: (c, c),
    "linear_z.weight": lambda c, h: (c, h),
    "layer_norm_in.weight": lambda c, h: (c,),
    "layer_norm_in.bias": lambda c, h: (c,),
    "layer_norm_out.weight": lambda c, h: (h,),
    "layer_norm_out.bias": lambda c, h: (h,),
}
_LINEAR_NAMES = (
    "linear_a_p",
    "linear_a_g",
    "linear_b_p",
    "linear_b_g",
    "linear_g",
    "linear_z",
)
_NO_MASK = object()


def _tensor_metadata(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
    device_type: str = "cuda",
    device_index: int | None = 0,
    contiguous: bool = True,
    requires_grad: bool = True,
) -> TensorMetadata:
    return TensorMetadata(
        shape=shape,
        dtype=dtype,
        device_type=device_type,
        device_index=device_index,
        contiguous=contiguous,
        requires_grad=requires_grad,
    )


def _metadata(
    *,
    c_z: int = 32,
    c_hidden: int | None = None,
    sequence: int = 17,
    prefix: tuple[int, ...] = (2,),
    z_shape: tuple[int, ...] | None = None,
    z_dtype: torch.dtype = torch.bfloat16,
    z_device_type: str = "cuda",
    z_device_index: int | None = 0,
    z_contiguous: bool = True,
    z_requires_grad: bool = True,
    mask: TensorMetadata | None | object = _NO_MASK,
    device_name: str = "NVIDIA H100 80GB HBM3",
    capability: tuple[int, int] = (9, 0),
    multiprocessor_count: int = 132,
    outgoing: bool = True,
) -> StableTrimulSupportMetadata:
    if c_hidden is None:
        c_hidden = c_z
    if z_shape is None:
        z_shape = (*prefix, sequence, sequence, c_z)
    z = _tensor_metadata(
        z_shape,
        dtype=z_dtype,
        device_type=z_device_type,
        device_index=z_device_index,
        contiguous=z_contiguous,
        requires_grad=z_requires_grad,
    )
    weights = tuple(
        NamedTensorMetadata(
            name,
            _tensor_metadata(shape(c_z, c_hidden)),
        )
        for name, shape in _WEIGHT_SHAPES.items()
    )
    if mask is _NO_MASK:
        mask = None
    return StableTrimulSupportMetadata(
        z=z,
        mask=mask,
        device_name=device_name,
        capability=capability,
        multiprocessor_count=multiprocessor_count,
        c_z=c_z,
        c_hidden=c_hidden,
        outgoing=outgoing,
        weights=weights,
        linear_bias_is_none=tuple((name, True) for name in _LINEAR_NAMES),
        layer_norm_eps=(("layer_norm_in", 1e-5), ("layer_norm_out", 1e-5)),
    )


def _replace_weight(
    metadata: StableTrimulSupportMetadata,
    name: str,
    **changes: object,
) -> StableTrimulSupportMetadata:
    weights = tuple(
        replace(item, tensor=replace(item.tensor, **changes))
        if item.name == name
        else item
        for item in metadata.weights
    )
    return replace(metadata, weights=weights)


@pytest.mark.parametrize("channels", (32, 64, 96, 128))
@pytest.mark.parametrize("sequence", (384, 640, 768))
def test_pure_contract_accepts_dimension_scalable_domain(channels, sequence):
    result = evaluate_stable_trimul_support(_metadata(c_z=channels, sequence=sequence))

    assert result.supported
    assert result.code == "supported"
    assert result.batch == 2
    assert result.sequence == sequence
    assert result.channels == channels
    assert bool(result)


@pytest.mark.parametrize("sequence", (1, 31, 99, 100))
def test_valid_short_sequences_are_reasoned_stock_fallbacks(sequence):
    result = evaluate_stable_trimul_support(_metadata(sequence=sequence))

    assert not result
    assert result.code == "sequence_stock_fallback"
    assert f"N={sequence}" in result.reason


def test_fixed_short_sequence_boundary_starts_aot_at_101():
    assert STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD == 100
    result = evaluate_stable_trimul_support(_metadata(sequence=101))

    assert result.supported


@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_pure_contract_accepts_both_directions_and_autograd_modes(
    outgoing,
    requires_grad,
):
    result = evaluate_stable_trimul_support(
        _metadata(
            sequence=101,
            outgoing=outgoing,
            z_requires_grad=requires_grad,
            prefix=(),
        )
    )

    assert result.supported
    assert result.batch == 1


def test_contract_accepts_arbitrary_batch_prefix_and_packed_input_layout():
    result = evaluate_stable_trimul_support(
        _metadata(sequence=101, prefix=(2, 3), z_contiguous=False)
    )

    assert result.supported
    assert result.batch == 6


def test_unequal_channels_are_reasoned_unsupported_for_stock_fallback():
    result = evaluate_stable_trimul_support(_metadata(c_z=64, c_hidden=32))

    assert not result
    assert result.code == "channels_not_equal"
    assert "stock fallback" in result.reason


@pytest.mark.parametrize("channels", (16, 48, 160))
def test_unsupported_channels_are_rejected(channels):
    result = evaluate_stable_trimul_support(_metadata(c_z=channels))

    assert not result
    assert result.code == "channels"


def test_non_boolean_direction_is_rejected():
    result = evaluate_stable_trimul_support(_metadata(outgoing=1))

    assert not result
    assert result.code == "direction"


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"device_name": "NVIDIA H200", "capability": (9, 0)}, "architecture"),
        ({"device_name": "NVIDIA H100", "capability": (8, 0)}, "architecture"),
        ({"multiprocessor_count": 0}, "device_metadata"),
        ({"z_device_type": "cpu", "z_device_index": None}, "z_device"),
        ({"z_dtype": torch.float32}, "z_dtype"),
        ({"z_shape": (17, 32)}, "z_rank"),
        ({"z_shape": (2, 17, 19, 32)}, "sequence_not_square"),
        ({"z_shape": (2, 0, 0, 32)}, "z_shape"),
    ),
)
def test_device_and_input_contract_rejections(changes, code):
    result = evaluate_stable_trimul_support(_metadata(**changes))

    assert not result
    assert result.code == code


def test_module_channel_must_match_tensor_channel():
    result = evaluate_stable_trimul_support(_metadata(c_z=32, z_shape=(1, 17, 17, 64)))

    assert not result
    assert result.code == "module_channels"


@pytest.mark.parametrize(
    ("metadata", "code"),
    (
        (_metadata(prefix=(MAX_FLATTENED_BATCH + 1,), sequence=1), "batch_grid_z"),
        (_metadata(sequence=262_145, prefix=()), "transpose_grid_x"),
        (_metadata(sequence=185_364, prefix=(2,)), "transpose_chunks"),
        (_metadata(sequence=131_073, prefix=()), "layer_norm_grid_x"),
        (_metadata(c_z=128, sequence=65_536, prefix=()), "sigmoid_grid_x"),
        (
            _metadata(sequence=65, prefix=(32_768,)),
            "layer_norm_dgb_grid_y",
        ),
    ),
)
def test_explicit_batch_and_grid_contract(metadata, code):
    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == code


@pytest.mark.parametrize("missing_name", tuple(_WEIGHT_SHAPES))
def test_every_required_weight_and_layer_norm_parameter_is_named(missing_name):
    metadata = _metadata()
    metadata = replace(
        metadata,
        weights=tuple(item for item in metadata.weights if item.name != missing_name),
    )

    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == "module_parameters"


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"shape": (31, 32)}, "parameter_shape"),
        ({"dtype": torch.bfloat16}, "parameter_dtype"),
        ({"contiguous": False}, "parameter_layout"),
        ({"device_index": 1}, "parameter_device"),
    ),
)
def test_parameter_metadata_must_match_exact_contract(changes, code):
    metadata = _replace_weight(_metadata(), "linear_a_p.weight", **changes)

    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == code


def test_duplicate_parameter_metadata_is_rejected():
    metadata = _metadata()
    metadata = replace(metadata, weights=(*metadata.weights, metadata.weights[0]))

    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == "module_parameters"


def test_every_linear_must_report_bias_none():
    metadata = _metadata()
    states = dict(metadata.linear_bias_is_none)
    states["linear_b_g"] = False
    metadata = replace(metadata, linear_bias_is_none=tuple(states.items()))

    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == "linear_bias"


@pytest.mark.parametrize("name", ("layer_norm_in", "layer_norm_out"))
def test_layer_norm_epsilon_is_exact(name):
    metadata = _metadata()
    eps = dict(metadata.layer_norm_eps)
    eps[name] = 1.0001e-5
    metadata = replace(metadata, layer_norm_eps=tuple(eps.items()))

    result = evaluate_stable_trimul_support(metadata)

    assert not result
    assert result.code == "layer_norm_eps"


@pytest.mark.parametrize("dtype", (torch.bool, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    "shape",
    ((), (101, 101), (1, 101, 101), (2, 1, 101, 1), (2, 3, 101, 101)),
)
def test_mask_accepts_valid_dtypes_and_standard_broadcasting(dtype, shape):
    mask = _tensor_metadata(shape, dtype=dtype, requires_grad=False)
    result = evaluate_stable_trimul_support(
        _metadata(sequence=101, prefix=(2, 3), mask=mask)
    )

    assert result.supported


@pytest.mark.parametrize(
    ("mask", "code"),
    (
        (
            _tensor_metadata((17, 17), dtype=torch.float16, requires_grad=False),
            "mask_dtype",
        ),
        (
            _tensor_metadata(
                (17, 17),
                dtype=torch.float32,
                device_type="cpu",
                device_index=None,
                requires_grad=False,
            ),
            "mask_device",
        ),
        (
            _tensor_metadata((17, 17), dtype=torch.float32, requires_grad=True),
            "mask_grad",
        ),
        (
            _tensor_metadata((0, 17), dtype=torch.bool, requires_grad=False),
            "mask_shape",
        ),
        (
            _tensor_metadata((4, 17, 17), dtype=torch.bool, requires_grad=False),
            "mask_broadcast",
        ),
        (
            _tensor_metadata((1, 2, 3, 17, 17), dtype=torch.bool, requires_grad=False),
            "mask_broadcast",
        ),
    ),
)
def test_mask_rejections_are_reasoned(mask, code):
    result = evaluate_stable_trimul_support(_metadata(mask=mask))

    assert not result
    assert result.code == code


def test_live_checker_rejects_cpu_before_querying_cuda():
    result = check_stable_trimul_support(
        object(),
        torch.empty((1, 2, 2, 32), dtype=torch.bfloat16),
    )

    assert not result
    assert result.code == "z_device"


def test_package_export_does_not_import_or_build_cuda_extension():
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not old_pythonpath
        else os.pathsep.join((str(repo_root), old_pythonpath))
    )
    script = """
import sys
from openfold3.core.kernels.cuda import check_stable_trimul_support
assert callable(check_stable_trimul_support)
assert 'openfold3.core.kernels.cuda.turbopairformer_trimul' not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repo_root,
        env=environment,
    )


def _is_h100() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        torch.cuda.get_device_capability(device) == (9, 0)
        and "H100" in torch.cuda.get_device_name(device).upper()
    )


def _live_trimul_module(c_z: int, c_hidden: int, outgoing: bool, device):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicativeUpdate,
    )

    return TriangleMultiplicativeUpdate(
        c_z=c_z,
        c_hidden=c_hidden,
        _outgoing=outgoing,
    ).to(device)


@pytest.mark.skipif(not _is_h100(), reason="requires an NVIDIA H100")
@pytest.mark.parametrize("channels", (32, 64, 96, 128))
@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_h100_live_positive_metadata_contract(channels, outgoing, requires_grad):
    device = torch.device("cuda", torch.cuda.current_device())
    module = _live_trimul_module(channels, channels, outgoing, device)
    module.requires_grad_(requires_grad)
    z = torch.empty(
        (1, 101, 101, channels),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    mask = torch.ones((101, 101), dtype=torch.bool, device=device)

    result = check_stable_trimul_support(module, z, mask)

    assert result.supported
    assert result.channels == channels


@pytest.mark.skipif(not _is_h100(), reason="requires an NVIDIA H100")
def test_h100_live_negative_metadata_contracts():
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        FusedTriangleMultiplicationOutgoing,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    unequal_module = _live_trimul_module(64, 32, True, device)
    z = torch.empty((1, 3, 3, 64), dtype=torch.bfloat16, device=device)
    unequal = check_stable_trimul_support(unequal_module, z)

    module = _live_trimul_module(64, 64, False, device)
    grad_mask = torch.ones(
        (3, 3), dtype=torch.float32, device=device, requires_grad=True
    )
    bad_mask = check_stable_trimul_support(module, z, grad_mask)

    fused_module = FusedTriangleMultiplicationOutgoing(64, 64).to(device)
    fused = check_stable_trimul_support(fused_module, z)

    assert not unequal
    assert unequal.code == "channels_not_equal"
    assert not bad_mask
    assert bad_mask.code == "mask_grad"
    assert not fused
    assert fused.code == "fused_projection_layout"
