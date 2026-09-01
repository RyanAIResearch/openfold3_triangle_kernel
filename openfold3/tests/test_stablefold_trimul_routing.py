# SPDX-License-Identifier: Apache-2.0

"""Production routing tests for the StableFold TriMul fast path."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import openfold3.core.model.layers.triangular_multiplicative_update as tmu
from openfold3.core.kernels.cuda.stablefold_trimul_support import (
    SupportResult,
    check_stable_trimul_support,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold3.core.utils.distributed_contract import (
    _set_stablefold_distributed_strict,
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
def test_short_sequence_rejection_falls_back_to_stock_output_and_gradients(
    module_cls,
):
    torch.manual_seed(7)
    stock_module = _nondegenerate(module_cls(8, 8))
    stable_module = copy.deepcopy(stock_module)
    z_stock = torch.randn((1, 3, 3, 8), requires_grad=True)
    z_stable = z_stock.detach().clone().requires_grad_(True)
    mask = torch.randint(0, 2, (1, 3, 3), dtype=torch.float32)
    grad = torch.randn_like(z_stock)
    rejection = SupportResult(
        False,
        "sequence_stock_fallback",
        "N=3 follows the fixed stock PyTorch route",
    )

    stock = stock_module(z_stock, mask)
    stock.backward(grad)
    with (
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul_support."
            "check_stable_trimul_support",
            return_value=rejection,
        ) as support_check,
        patch(
            "openfold3.core.kernels.cuda.stablefold_trimul."
            "_run_supported_stable_trimul",
            side_effect=AssertionError("short N must not enter the AOT runner"),
        ) as runner,
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

    support_check.assert_called_once_with(stable_module, z_stable, mask)
    runner.assert_not_called()

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


def test_strict_ddp_route_allows_short_n_but_rejects_unexpected_fallback():
    module = TriangleMultiplicationOutgoing(8, 8)
    z = torch.randn((1, 3, 3, 8))
    environment = {
        "OPENFOLD3_SR_BACKWARD": "1",
        "OPENFOLD3_SR_SHADOW": "1",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
    }
    _set_stablefold_distributed_strict(True, environment)
    try:
        short_n = SupportResult(
            False,
            "sequence_stock_fallback",
            "declared fixed stock route",
        )
        with patch(
            "openfold3.core.kernels.cuda.stablefold_trimul_support."
            "check_stable_trimul_support",
            return_value=short_n,
        ):
            assert tmu._route_stable_trimul(module, z, None) == (False, None)

        unexpected = SupportResult(False, "channels", "synthetic rejection")
        with (
            patch(
                "openfold3.core.kernels.cuda.stablefold_trimul_support."
                "check_stable_trimul_support",
                return_value=unexpected,
            ),
            pytest.raises(RuntimeError, match="strict route rejected"),
        ):
            tmu._route_stable_trimul(module, z, None)
    finally:
        _set_stablefold_distributed_strict(False)


_EXPECTED_RELEASE_FIELDS = frozenset(
    (
        "output",
        "no_grad_output",
        "d_z",
        "d_layer_norm_in.weight",
        "d_layer_norm_in.bias",
        "d_linear_a_p.weight",
        "d_linear_a_g.weight",
        "d_linear_b_p.weight",
        "d_linear_b_g.weight",
        "d_linear_g.weight",
        "d_layer_norm_out.weight",
        "d_layer_norm_out.bias",
        "d_linear_z.weight",
    )
)


def _is_h100() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        torch.cuda.get_device_capability(device) == (9, 0)
        and "H100" in torch.cuda.get_device_name(device).upper()
    )


def _release_module(channels: int, outgoing: bool):
    module_type = (
        TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    )
    return module_type(channels, channels).cuda()


def _initialize_release_module(module) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "layer_norm" in name and name.endswith("weight"):
                parameter.normal_(mean=1.0, std=0.05)
            else:
                parameter.normal_(mean=0.0, std=0.02)


def _call_public_fields(
    module,
    z,
    mask,
    grad_out,
    *,
    use_stable: bool,
    training: bool,
):
    z_case = z.detach().clone().requires_grad_(training)
    named_parameters = tuple(module.named_parameters())
    module.train(training)
    kwargs = {
        "mask": mask,
        "inplace_safe": False,
        "use_cueq_triangle_kernels": use_stable,
        "use_triton_triangle_kernels": use_stable,
        "use_stable_triangle_multiplicative": use_stable,
    }
    if not training:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(z_case, **kwargs)
        return {"no_grad_output": output.detach()}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(z_case, **kwargs)
    gradients = torch.autograd.grad(
        output,
        (z_case, *(parameter for _, parameter in named_parameters)),
        grad_out,
    )
    fields = {"output": output.detach(), "d_z": gradients[0].detach()}
    fields.update(
        {
            f"d_{name}": gradient.detach()
            for (name, _), gradient in zip(named_parameters, gradients[1:], strict=True)
        }
    )
    return fields


def _call_stable_with_route(module, z, mask, grad_out, *, training: bool):
    from openfold3.core.kernels.cuda import stablefold_trimul as facade

    original_get_ext = facade._get_ext
    original_runner = facade._run_supported_stable_trimul
    extension_paths: list[Path] = []

    def counted_get_ext():
        extension = original_get_ext()
        extension_paths.append(Path(extension.__file__).resolve())
        return extension

    with (
        patch.object(facade, "_get_ext", side_effect=counted_get_ext) as get_ext,
        patch.object(
            facade,
            "_run_supported_stable_trimul",
            wraps=original_runner,
        ) as runner,
        patch.object(
            tmu,
            "_cueq_triangle_mult",
            side_effect=AssertionError("StableFold-owned route entered cuEq"),
        ),
        patch.object(tmu, "TRITON_AVAILABLE", False),
    ):
        fields = _call_public_fields(
            module,
            z,
            mask,
            grad_out,
            use_stable=True,
            training=training,
        )

    return fields, {
        "runner": runner.call_count,
        "get_ext": get_ext.call_count,
        "extension_paths": tuple(extension_paths),
    }


def _assert_canonical_aot_paths(paths) -> None:
    assert paths
    assert all(
        path.parent.name == "_ext"
        and path.name.startswith("_stablefold_trimul_sm90a")
        and path.suffix == ".so"
        for path in paths
    )


@pytest.mark.skipif(not _is_h100(), reason="requires an NVIDIA H100")
@pytest.mark.parametrize("sequence", (100, 101))
@pytest.mark.parametrize("channels", (32, 128))
@pytest.mark.parametrize("outgoing", (False, True), ids=("incoming", "outgoing"))
@pytest.mark.parametrize("training", (False, True), ids=("infer", "train"))
def test_h100_short_sequence_boundary_routes(
    sequence,
    channels,
    outgoing,
    training,
):
    torch.manual_seed(9100 + sequence + channels + outgoing + 7 * training)
    module = _release_module(channels, outgoing)
    _initialize_release_module(module)
    z = torch.randn(
        (1, sequence, sequence, channels),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mask = (torch.rand((1, sequence, sequence), device="cuda") > 0.08).to(
        torch.bfloat16
    )
    grad_out = torch.randn_like(z)
    support = check_stable_trimul_support(module, z, mask)
    if sequence == 100:
        assert not support
        assert support.code == "sequence_stock_fallback"
    else:
        assert support.supported

    stable_fields, route = _call_stable_with_route(
        module,
        z,
        mask,
        grad_out,
        training=training,
    )

    if sequence == 100:
        stock_fields = _call_public_fields(
            module,
            z,
            mask,
            grad_out,
            use_stable=False,
            training=training,
        )
        assert route["runner"] == 0
        assert route["get_ext"] == 0
        assert route["extension_paths"] == ()
        assert stable_fields.keys() == stock_fields.keys()
        assert all(
            torch.equal(stable_fields[name], stock_fields[name])
            for name in stable_fields
        )
        return

    assert route["runner"] == 1
    assert route["get_ext"] == (2 if training else 1)
    _assert_canonical_aot_paths(route["extension_paths"])
    assert all(torch.isfinite(value).all() for value in stable_fields.values())


@pytest.mark.skipif(not _is_h100(), reason="requires an NVIDIA H100")
def test_h100_n31_seed2997_stock_exact_all_fields_and_five_replays():
    torch.manual_seed(2997)
    module = _release_module(32, True)
    _initialize_release_module(module)
    z = torch.randn((1, 31, 31, 32), device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand((1, 31, 31), device="cuda") > 0.08).to(torch.bfloat16)
    mask.reshape(-1)[0] = 0
    mask.reshape(-1)[1] = 1
    grad_out = torch.randn_like(z)
    support = check_stable_trimul_support(module, z, mask)
    assert not support
    assert support.code == "sequence_stock_fallback"

    stock_fields = _call_public_fields(
        module,
        z,
        mask,
        grad_out,
        use_stable=False,
        training=True,
    )
    stock_fields.update(
        _call_public_fields(
            module,
            z,
            mask,
            grad_out,
            use_stable=False,
            training=False,
        )
    )
    stable_fields, train_route = _call_stable_with_route(
        module,
        z,
        mask,
        grad_out,
        training=True,
    )
    no_grad_fields, infer_route = _call_stable_with_route(
        module,
        z,
        mask,
        grad_out,
        training=False,
    )
    stable_fields.update(no_grad_fields)

    assert set(stable_fields) == _EXPECTED_RELEASE_FIELDS
    assert stable_fields.keys() == stock_fields.keys()
    assert all(
        torch.equal(stable_fields[name], stock_fields[name]) for name in stable_fields
    )
    for route in (train_route, infer_route):
        assert route["runner"] == 0
        assert route["get_ext"] == 0
        assert route["extension_paths"] == ()

    for _ in range(5):
        replay_fields, replay_train_route = _call_stable_with_route(
            module,
            z,
            mask,
            grad_out,
            training=True,
        )
        replay_no_grad, replay_infer_route = _call_stable_with_route(
            module,
            z,
            mask,
            grad_out,
            training=False,
        )
        replay_fields.update(replay_no_grad)
        assert all(
            torch.equal(stable_fields[name], replay_fields[name])
            for name in stable_fields
        )
        for route in (replay_train_route, replay_infer_route):
            assert route["runner"] == 0
            assert route["get_ext"] == 0
            assert route["extension_paths"] == ()
