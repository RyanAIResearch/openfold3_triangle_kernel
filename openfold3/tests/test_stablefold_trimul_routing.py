# SPDX-License-Identifier: Apache-2.0

"""Production routing tests for the StableFold TriMul fast path."""

from __future__ import annotations

import copy
import logging
from unittest.mock import patch

import pytest
import torch

import openfold3.core.model.layers.triangular_multiplicative_update as tmu
from openfold3.core.kernels.cuda.stablefold_trimul_support import SupportResult
from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationOutgoing,
)


def _nondegenerate(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    return module


def _accepted() -> SupportResult:
    return SupportResult(
        True,
        "supported",
        "test support",
        batch=1,
        sequence=3,
        channels=32,
    )


@pytest.mark.parametrize(
    "module_cls",
    (TriangleMultiplicationOutgoing, FusedTriangleMultiplicationOutgoing),
)
def test_unsupported_stable_request_falls_back_to_stock_output_and_gradients(
    module_cls,
):
    torch.manual_seed(7)
    stock_module = _nondegenerate(module_cls(8, 8))
    stable_module = copy.deepcopy(stock_module)
    z_stock = torch.randn((1, 3, 3, 8), requires_grad=True)
    z_stable = z_stock.detach().clone().requires_grad_(True)
    mask = torch.randint(0, 2, (1, 3, 3), dtype=torch.float32)
    grad = torch.randn_like(z_stock)

    stock = stock_module(z_stock, mask)
    stock.backward(grad)
    with (
        patch.object(tmu, "TRITON_AVAILABLE", False),
        patch.object(
            tmu,
            "_cueq_triangle_mult",
            side_effect=AssertionError("cuEq must not own a StableFold fallback"),
        ),
    ):
        fallback = stable_module(
            z_stable,
            mask,
            use_cueq_triangle_kernels=True,
            use_triton_triangle_kernels=True,
            use_stable_triangle_multiplicative=True,
        )
    fallback.backward(grad)

    torch.testing.assert_close(fallback, stock)
    torch.testing.assert_close(z_stable.grad, z_stock.grad)
    for (stock_name, stock_parameter), (stable_name, stable_parameter) in zip(
        stock_module.named_parameters(), stable_module.named_parameters()
    ):
        assert stable_name == stock_name
        torch.testing.assert_close(stable_parameter.grad, stock_parameter.grad)


def test_supported_metadata_calls_prechecked_runner_once():
    module = TriangleMultiplicationOutgoing(8, 8)
    z = torch.randn((1, 3, 3, 8))
    mask = torch.ones((1, 3, 3))
    expected = torch.full_like(z, 3.0)

    with (
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul_support."
            "check_stable_trimul_support",
            return_value=_accepted(),
        ) as support_check,
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul."
            "_run_supported_stable_trimul",
            return_value=expected,
        ) as runner,
    ):
        actual = module(
            z,
            mask,
            use_cueq_triangle_kernels=True,
            use_triton_triangle_kernels=True,
            use_stable_triangle_multiplicative=True,
        )

    assert actual is expected
    support_check.assert_called_once_with(module, z, mask)
    runner.assert_called_once_with(module, z, mask)


def test_supported_runner_failure_propagates_without_fallback():
    module = TriangleMultiplicationOutgoing(8, 8)
    z = torch.randn((1, 3, 3, 8))

    with (
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul_support."
            "check_stable_trimul_support",
            return_value=_accepted(),
        ),
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul."
            "_run_supported_stable_trimul",
            side_effect=RuntimeError("synthetic AOT launch failure"),
        ),
        pytest.raises(RuntimeError, match="synthetic AOT launch failure"),
    ):
        module(z, use_stable_triangle_multiplicative=True)


def test_public_runner_rejects_before_extension_load():
    from openfold3.core.kernels.cuda import stablefold_trimul

    module = TriangleMultiplicationOutgoing(32, 32)
    z = torch.randn((1, 3, 3, 32), dtype=torch.bfloat16)

    with (
        patch.object(
            stablefold_trimul,
            "_get_ext",
            side_effect=AssertionError("unsupported metadata loaded the extension"),
        ) as get_ext,
        pytest.raises(ValueError, match=r"unsupported \[z_device\]"),
    ):
        stablefold_trimul.run_stable_trimul(module, z, None)

    get_ext.assert_not_called()


def test_fallback_warning_is_logged_once(caplog):
    module = TriangleMultiplicationOutgoing(8, 8)
    z = torch.randn((1, 3, 3, 8))
    rejection = SupportResult(False, "channels", "synthetic rejection")
    tmu._STABLEFOLD_TRIMUL_FALLBACK_LOGGED.clear()

    try:
        with (
            patch(
                "openfold3.core.kernels.cuda.stablefold_trimul_support."
                "check_stable_trimul_support",
                return_value=rejection,
            ),
            caplog.at_level(logging.WARNING, logger=tmu.__name__),
            torch.no_grad(),
        ):
            module(z, use_stable_triangle_multiplicative=True)
            module(z, use_stable_triangle_multiplicative=True)
    finally:
        tmu._STABLEFOLD_TRIMUL_FALLBACK_LOGGED.clear()

    records = [
        record for record in caplog.records if "fallback to stock" in record.message
    ]
    assert len(records) == 1
    assert "[channels]" in records[0].message
