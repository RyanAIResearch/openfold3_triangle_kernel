# SPDX-License-Identifier: Apache-2.0

"""CPU-only Hopper eligibility checks; these are not GPU execution tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from stablefold.trimul_support import (
    NamedTensorMetadata,
    TensorMetadata,
    TrimulSupportMetadata,
    evaluate_trimul_support,
)


@pytest.fixture
def valid_trimul_metadata() -> TrimulSupportMetadata:
    """Describe one N128/C32 invocation without allocating CUDA tensors."""

    def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> TensorMetadata:
        return TensorMetadata(
            shape=shape,
            dtype=dtype,
            device_type="cuda",
            device_index=0,
            contiguous=True,
            requires_grad=True,
        )

    linear_names = (
        "linear_a_p",
        "linear_a_g",
        "linear_b_p",
        "linear_b_g",
        "linear_g",
        "linear_z",
    )
    weights = tuple(
        NamedTensorMetadata(f"{name}.weight", tensor((32, 32), torch.float32))
        for name in linear_names
    ) + tuple(
        NamedTensorMetadata(f"{name}.{parameter}", tensor((32,), torch.float32))
        for name in ("layer_norm_in", "layer_norm_out")
        for parameter in ("weight", "bias")
    )
    return TrimulSupportMetadata(
        z=tensor((1, 128, 128, 32), torch.bfloat16),
        mask=None,
        device_name="NVIDIA H200",
        capability=(9, 0),
        multiprocessor_count=132,
        c_z=32,
        c_hidden=32,
        outgoing=True,
        weights=weights,
        linear_bias_is_none=tuple((name, True) for name in linear_names),
        layer_norm_eps=(("layer_norm_in", 1e-5), ("layer_norm_out", 1e-5)),
    )


@pytest.mark.parametrize(
    "device_name",
    (
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 SXM5 80GB",
        "NVIDIA H100 NVL",
        "NVIDIA H200",
        "NVIDIA H200 SXM",
        "NVIDIA H200 NVL",
    ),
)
def test_trimul_accepts_h100_and_h200_metadata(valid_trimul_metadata, device_name):
    result = evaluate_trimul_support(
        replace(valid_trimul_metadata, device_name=device_name)
    )

    assert result.supported
    assert result.code == "supported"
    assert (result.batch, result.sequence, result.channels) == (1, 128, 32)


@pytest.mark.parametrize("capability", ((8, 0), (9, 1), (10, 0)))
def test_h200_name_does_not_bypass_architecture_guard(
    valid_trimul_metadata, capability
):
    result = evaluate_trimul_support(
        replace(valid_trimul_metadata, capability=capability)
    )

    assert not result.supported
    assert result.code == "architecture"


@pytest.mark.parametrize("multiprocessor_count", (0, -1))
def test_trimul_rejects_nonpositive_sm_count(
    valid_trimul_metadata, multiprocessor_count
):
    result = evaluate_trimul_support(
        replace(valid_trimul_metadata, multiprocessor_count=multiprocessor_count)
    )

    assert not result.supported
    assert result.code == "device_metadata"


def test_trimul_sm_count_guard_is_not_a_product_specific_count(valid_trimul_metadata):
    result = evaluate_trimul_support(
        replace(valid_trimul_metadata, multiprocessor_count=1)
    )

    assert result.supported
