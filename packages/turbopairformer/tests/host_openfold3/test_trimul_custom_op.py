# SPDX-License-Identifier: Apache-2.0

"""Contracts for the versioned TurboPairFormer TriMul ``torch.library`` boundary."""

from __future__ import annotations

import copy
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicativeUpdate,
)
from torch._subclasses.fake_tensor import FakeTensorMode

from turbopairformer import _trimul as stable

_FORWARD_SCHEMA = (
    "turbopairformer::_trimul_fwd_v1(Tensor z_bf, Tensor mask_f, "
    "Tensor w_ap, Tensor w_ag, Tensor w_bp, Tensor w_bg, Tensor w_g, "
    "Tensor w_z, Tensor g_in, Tensor b_in, Tensor g_out, Tensor b_out, "
    "bool outgoing, bool save_state) -> (Tensor, Tensor, Tensor, Tensor, "
    "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_BACKWARD_SCHEMA = (
    "turbopairformer::_trimul_bwd_v1(Tensor grad_out, Tensor z_bf, "
    "Tensor mask_f, Tensor z_ln, Tensor gate, Tensor a, Tensor b, Tensor x, "
    "Tensor xlnT, Tensor mean_in, Tensor rstd_in, Tensor mean_out, "
    "Tensor rstd_out, Tensor w_ap, Tensor w_ag, Tensor w_bp, Tensor w_bg, "
    "Tensor w_g, Tensor w_z, Tensor g_in, Tensor g_out, Tensor out, "
    "bool outgoing) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_EXPECTED_WGRAD_K_CHUNK = 16384
_WGRAD_PLANNER_ROWS = (
    1,
    _EXPECTED_WGRAD_K_CHUNK - 1,
    _EXPECTED_WGRAD_K_CHUNK,
    _EXPECTED_WGRAD_K_CHUNK + 1,
    153 * _EXPECTED_WGRAD_K_CHUNK,
    153 * _EXPECTED_WGRAD_K_CHUNK + 1,
)
_WGRAD_WAVE_CAPACITY = {32: 2457, 64: 614, 96: 273, 128: 153}


@pytest.mark.parametrize("rows", _WGRAD_PLANNER_ROWS)
@pytest.mark.parametrize("channels", (32, 64, 96, 128))
def test_wgrad_planner_has_bounded_globally_ordered_waves(rows, channels):
    assert stable._WGRAD_K_CHUNK == _EXPECTED_WGRAD_K_CHUNK
    plan = stable._plan_wgrad_fixed_fold(rows, channels, channels)
    expected_full, expected_tail = divmod(rows, stable._WGRAD_K_CHUNK)
    expected_logical = expected_full + int(expected_tail != 0)

    assert stable._WGRAD_PARTIAL_CAP_BYTES == 48 * 1024 * 1024
    assert plan.partial_bytes_per_chunk == 5 * channels * channels * 4
    assert plan.chunks_per_wave == _WGRAD_WAVE_CAPACITY[channels]
    assert (
        sum(wave.full_chunks + int(wave.tail_rows != 0) for wave in plan.waves)
        == expected_logical
    )

    next_chunk = 0
    observed_full = 0
    observed_tail = []
    for wave in plan.waves:
        wave_chunks = wave.full_chunks + int(wave.tail_rows != 0)
        assert wave.first_chunk == next_chunk
        assert 1 <= wave_chunks <= plan.chunks_per_wave
        assert (
            wave_chunks * plan.partial_bytes_per_chunk
            <= stable._WGRAD_PARTIAL_CAP_BYTES
        )
        next_chunk += wave_chunks
        observed_full += wave.full_chunks
        if wave.tail_rows:
            observed_tail.append(wave.tail_rows)
    assert next_chunk == expected_logical
    assert observed_full == expected_full
    assert observed_tail == ([expected_tail] if expected_tail else [])

    if rows == 153 * _EXPECTED_WGRAD_K_CHUNK + 1 and channels == 128:
        assert plan.waves == (
            stable._WgradWave(0, 153, 0),
            stable._WgradWave(153, 0, 1),
        )


def test_wgrad_planner_fails_when_one_partial_exceeds_fixed_cap():
    with pytest.raises(ValueError, match="one wgrad partial exceeds"):
        stable._plan_wgrad_fixed_fold(1, 2048, 2048)


def test_wgrad_full_chunk_operands_are_zero_copy_views():
    channels = 32
    rows = 3 * stable._WGRAD_K_CHUNK
    d_proj5 = torch.empty((rows, 5 * channels), dtype=torch.bfloat16)
    z_ln = torch.empty((rows, channels), dtype=torch.bfloat16)
    projection_t, normalized = stable._wgrad_full_chunk_views(d_proj5, z_ln, 1, 2)

    assert torch._C._is_alias_of(projection_t, d_proj5)
    assert torch._C._is_alias_of(normalized, z_ln)
    assert projection_t.shape == (2, 5 * channels, stable._WGRAD_K_CHUNK)
    assert normalized.shape == (2, stable._WGRAD_K_CHUNK, channels)
    assert projection_t.stride() == (
        stable._WGRAD_K_CHUNK * 5 * channels,
        1,
        5 * channels,
    )
    assert normalized.stride() == (
        stable._WGRAD_K_CHUNK * channels,
        channels,
        1,
    )


def test_wgrad_route_is_fixed_cap_tail_only_and_leaves_dwz_unchanged():
    full_view_source = inspect.getsource(stable._wgrad_full_chunk_views)
    route_source = inspect.getsource(stable._trimul_backward_combined_impl)
    planner_source = inspect.getsource(stable._plan_wgrad_fixed_fold)
    assert ".contiguous" not in full_view_source
    assert "new_zeros" not in full_view_source
    assert "torch.cat" not in full_view_source
    assert "os.environ" not in planner_source
    assert "d_w5 = _trimul_wgrad_fixed(ext, d_proj5, z_ln)" in route_source
    assert (
        "d_wz = torch.mm(d_raw.t(), xlnT.t(), out_dtype=torch.float32)" in route_source
    )


def _flat_inputs(device: torch.device | str, n: int = 17, c: int = 32):
    z = torch.empty((1, n, n, c), dtype=torch.bfloat16, device=device)
    mask = torch.empty((1, n, n), dtype=torch.float32, device=device)
    five_weights = tuple(torch.empty((c, c), device=device) for _ in range(5))
    w_z = torch.empty((c, c), device=device)
    norms = tuple(torch.empty((c,), device=device) for _ in range(4))
    return (z, mask, *five_weights, w_z, *norms)


def _assert_pairwise_no_alias(tensors: tuple[torch.Tensor, ...]) -> None:
    for index, tensor in enumerate(tensors):
        for other in tensors[index + 1 :]:
            assert not torch._C._is_alias_of(tensor, other)


def test_schema_and_fake_kernels_are_extension_lazy():
    assert str(torch.ops.turbopairformer._trimul_fwd_v1.default._schema) == (_FORWARD_SCHEMA)
    assert str(torch.ops.turbopairformer._trimul_bwd_v1.default._schema) == (
        _BACKWARD_SCHEMA
    )
    assert "Tensor(" not in _FORWARD_SCHEMA
    assert "Tensor(" not in _BACKWARD_SCHEMA

    inputs = _flat_inputs("meta")
    with patch.object(
        stable,
        "_get_ext",
        side_effect=AssertionError("fake dispatch loaded the extension"),
    ) as get_ext:
        outputs = stable._trimul_fwd_v1(*inputs, True, True)
        inference_outputs = stable._trimul_fwd_v1(*inputs, False, False)
        backward_outputs = stable._trimul_bwd_v1(
            torch.empty_like(inputs[0]),
            inputs[0],
            inputs[1],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
            outputs[5],
            outputs[6],
            outputs[7],
            outputs[8],
            outputs[9],
            outputs[10],
            *inputs[2:8],
            inputs[8],
            inputs[10],
            outputs[0],
            True,
        )

    rows = 17 * 17
    assert tuple(outputs[0].shape) == (1, 17, 17, 32)
    assert tuple(outputs[1].shape) == (rows, 32)
    assert tuple(outputs[6].shape) == (32, rows)
    assert all(tuple(tensor.shape) == (rows,) for tensor in outputs[7:])
    assert tuple(inference_outputs[1].shape) == (0, 32)
    assert tuple(inference_outputs[6].shape) == (0, rows)
    assert tuple(inference_outputs[7].shape) == (rows,)
    assert tuple(inference_outputs[8].shape) == (rows,)
    assert all(tensor.numel() == 0 for tensor in inference_outputs[9:])
    assert [tuple(tensor.shape) for tensor in backward_outputs] == [
        (1, 17, 17, 32),
        (160, 32),
        (32, 32),
        (32,),
        (32,),
        (32,),
        (32,),
    ]
    _assert_pairwise_no_alias(outputs)
    _assert_pairwise_no_alias(backward_outputs)
    get_ext.assert_not_called()


def test_clean_import_registers_ops_without_importing_aot_loader():
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
import torch
from unittest.mock import patch
from turbopairformer import _trimul as stable
assert stable._ext is None
assert 'turbopairformer.trimul_cuda' not in sys.modules
z = torch.empty((1, 5, 5, 32), device='meta', dtype=torch.bfloat16)
mask = torch.empty((1, 5, 5), device='meta', dtype=torch.float32)
weights = [torch.empty((32, 32), device='meta') for _ in range(6)]
norms = [torch.empty((32,), device='meta') for _ in range(4)]
with patch.object(stable, '_get_ext', side_effect=AssertionError('loaded')) as load:
    outputs = stable._trimul_fwd_v1(z, mask, *weights, *norms, True, True)
    assert len(outputs) == 11
    load.assert_not_called()
assert stable._ext is None
assert 'turbopairformer.trimul_cuda' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repo_root,
        env=environment,
    )


class _RecordingContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors):
        self.non_differentiable = tensors

    def set_materialize_grads(self, value):
        self.materialize_grads = value


def _recorded_context_and_outputs():
    inputs = _flat_inputs("cpu", n=3)
    outputs = stable._trimul_fwd_v1_fake(*inputs, True, True)
    ctx = _RecordingContext()
    stable._trimul_fwd_v1_setup_context(
        ctx,
        (*inputs, True, True),
        outputs,
    )
    return ctx, inputs, outputs


def test_autograd_context_and_gradient_slots_match_the_legacy_contract():
    ctx, inputs, outputs = _recorded_context_and_outputs()
    assert len(ctx.saved_tensors) == 21
    assert ctx.non_differentiable == outputs[1:]
    assert ctx.materialize_grads is False

    c = inputs[0].shape[-1]
    d_z = torch.full_like(inputs[0], 1)
    d_w5 = torch.arange(5 * c * c, dtype=torch.float32).view(5 * c, c)
    d_wz = torch.full_like(inputs[7], 2)
    d_gin = torch.full_like(inputs[8], 3)
    d_bin = torch.full_like(inputs[9], 4)
    d_gout = torch.full_like(inputs[10], 5)
    d_bout = torch.full_like(inputs[11], 6)
    combined = (d_z, d_w5, d_wz, d_gin, d_bin, d_gout, d_bout)

    with patch.object(stable, "_trimul_bwd_v1", return_value=combined):
        gradients = stable._trimul_fwd_v1_backward(
            ctx,
            torch.ones_like(outputs[0]),
            *([None] * 10),
        )

    assert len(gradients) == 14
    assert gradients[0] is d_z
    assert gradients[1] is None  # mask
    for index in range(5):
        torch.testing.assert_close(
            gradients[2 + index],
            d_w5[index * c : (index + 1) * c],
        )
    assert gradients[7] is d_wz
    assert gradients[8] is d_gin
    assert gradients[9] is d_bin  # b_in
    assert gradients[10] is d_gout
    assert gradients[11] is d_bout  # b_out
    assert gradients[12] is None  # outgoing
    assert gradients[13] is None  # save_state


def test_registered_backward_is_explicitly_first_order_only():
    ctx, _, outputs = _recorded_context_and_outputs()
    c = outputs[0].shape[-1]
    combined = (
        torch.empty_like(outputs[0]),
        torch.empty((5 * c, c)),
        torch.empty((c, c)),
        torch.empty((c,)),
        torch.empty((c,)),
        torch.empty((c,)),
        torch.empty((c,)),
    )
    grad_out = torch.ones_like(outputs[0], requires_grad=True)
    with patch.object(stable, "_trimul_bwd_v1", return_value=combined):
        gradients = stable._trimul_fwd_v1_backward(
            ctx,
            grad_out,
            *([None] * 10),
        )
    with pytest.raises(RuntimeError, match="differentiate twice"):
        gradients[0].sum().backward()


def _dummy_module(c: int = 32):
    def parameter(*shape):
        return torch.nn.Parameter(torch.empty(shape))

    return SimpleNamespace(
        linear_a_p=SimpleNamespace(weight=parameter(c, c)),
        linear_a_g=SimpleNamespace(weight=parameter(c, c)),
        linear_b_p=SimpleNamespace(weight=parameter(c, c)),
        linear_b_g=SimpleNamespace(weight=parameter(c, c)),
        linear_g=SimpleNamespace(weight=parameter(c, c)),
        linear_z=SimpleNamespace(weight=parameter(c, c)),
        layer_norm_in=SimpleNamespace(
            weight=parameter(c),
            bias=parameter(c),
        ),
        layer_norm_out=SimpleNamespace(
            weight=parameter(c),
            bias=parameter(c),
        ),
        _outgoing=True,
    )


def _dummy_parameters(module):
    return (
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_g.weight,
        module.linear_z.weight,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
    )


def test_no_grad_preserves_legacy_requires_grad_save_state_rule():
    module = _dummy_module()
    z = torch.empty((1, 3, 3, 32), dtype=torch.bfloat16)
    observed = []

    def fake_forward(*args):
        observed.append(args[-1])
        return (torch.empty_like(args[0]),)

    with patch.object(stable, "_trimul_fwd_v1", side_effect=fake_forward):
        with torch.no_grad():
            stable._run_supported_trimul(module, z, None)
        for parameter in _dummy_parameters(module):
            parameter.requires_grad_(False)
        with torch.no_grad():
            stable._run_supported_trimul(module, z, None)

    assert observed == [True, False]


def test_private_forward_rejects_gradients_without_saved_state():
    inputs = list(_flat_inputs("meta", n=3))
    inputs[0] = inputs[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match="requires save_state=True"):
        stable._trimul_fwd_v1(*inputs, True, False)


def test_unsupported_public_path_calls_neither_custom_op_nor_extension():
    module = TriangleMultiplicationOutgoing(32, 32)
    z = torch.empty((1, 3, 3, 32), dtype=torch.bfloat16)
    with (
        patch.object(
            stable,
            "_trimul_fwd_v1",
            side_effect=AssertionError("unsupported path called custom op"),
        ) as custom_op,
        patch.object(
            stable,
            "_get_ext",
            side_effect=AssertionError("unsupported path loaded extension"),
        ) as get_ext,
        pytest.raises(ValueError, match=r"unsupported \[z_device\]"),
    ):
        stable.run_trimul(module, z, None)
    custom_op.assert_not_called()
    get_ext.assert_not_called()


def _is_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(device) == (9, 0)


def _live_module(c: int, outgoing: bool):
    module = TriangleMultiplicativeUpdate(c, c, _outgoing=outgoing).cuda()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    return module


def _legacy_call(module, z, mask):
    return stable._TurboPairFormerTrimulFn.apply(
        z,
        mask,
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_g.weight,
        module.linear_z.weight,
        module.layer_norm_in.weight.float(),
        module.layer_norm_in.bias.float(),
        module.layer_norm_out.weight.float(),
        module.layer_norm_out.bias.float(),
        module._outgoing,
    )


def _run_and_collect(runner, module, z_value, mask, grad_out):
    module.zero_grad(set_to_none=True)
    z = z_value.detach().clone().requires_grad_(True)
    output = runner(module, z, mask)
    output.backward(grad_out)
    tensors = [output.detach().clone(), z.grad.detach().clone()]
    tensors.extend(parameter.grad.detach().clone() for parameter in module.parameters())
    torch.cuda.synchronize()
    return tuple(tensors)


def _assert_exact_receipts(reference, candidate):
    assert len(reference) == 12  # output + z grad + ten parameter grads
    assert len(candidate) == len(reference)
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        assert torch.equal(actual, expected), f"receipt {index} differs"


def _manual_wgrad_fixed_reference(d_proj5, z_ln):
    rows, five_hidden = d_proj5.shape
    hidden = five_hidden // 5
    channels = z_ln.shape[1]
    plan = stable._plan_wgrad_fixed_fold(rows, hidden, channels)
    output = torch.zeros(
        (five_hidden, channels), device=d_proj5.device, dtype=torch.float32
    )
    for wave in plan.waves:
        if wave.full_chunks:
            projection_t, normalized = stable._wgrad_full_chunk_views(
                d_proj5, z_ln, wave.first_chunk, wave.full_chunks
            )
            partials = torch.bmm(projection_t, normalized, out_dtype=torch.float32)
            for partial in partials.unbind(0):
                output.add_(partial)
            del partials, projection_t, normalized
        if wave.tail_rows:
            tail_start = (wave.first_chunk + wave.full_chunks) * stable._WGRAD_K_CHUNK
            projection_tail = d_proj5.new_zeros((1, stable._WGRAD_K_CHUNK, five_hidden))
            normalized_tail = z_ln.new_zeros((1, stable._WGRAD_K_CHUNK, channels))
            projection_tail[0, : wave.tail_rows].copy_(
                d_proj5.narrow(0, tail_start, wave.tail_rows)
            )
            normalized_tail[0, : wave.tail_rows].copy_(
                z_ln.narrow(0, tail_start, wave.tail_rows)
            )
            partial = torch.bmm(
                projection_tail.transpose(1, 2),
                normalized_tail,
                out_dtype=torch.float32,
            )[0]
            output.add_(partial)
            del partial, projection_tail, normalized_tail
    return output


class _RecordingWgradFold:
    def __init__(self, extension):
        self.extension = extension
        self.initialize = []

    def trimul_wgrad_fixed_fold(self, partials, accumulator, initialize):
        self.initialize.append(initialize)
        return self.extension.trimul_wgrad_fixed_fold(partials, accumulator, initialize)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("sequence", (792, 800))
def test_hopper_wgrad_c128_single_wave_large_tail_reference_replay_and_memory(
    sequence,
):
    channels = 128
    rows = sequence * sequence
    torch.manual_seed(7300 + sequence)
    d_proj5 = torch.randn((rows, 5 * channels), device="cuda", dtype=torch.bfloat16)
    z_ln = torch.randn((rows, channels), device="cuda", dtype=torch.bfloat16)
    plan = stable._plan_wgrad_fixed_fold(rows, channels, channels)
    expected_full, expected_tail = divmod(rows, stable._WGRAD_K_CHUNK)
    assert expected_tail > 0
    assert plan.waves == (stable._WgradWave(0, expected_full, expected_tail),)
    recorder = _RecordingWgradFold(stable._get_ext())

    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = stable._trimul_wgrad_fixed(recorder, d_proj5, z_ln)
    torch.cuda.synchronize()
    peak_extra_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
    assert peak_extra_bytes <= 128 * 1024 * 1024
    # Full chunks and the tail are separate zero-copy/tail-only BMMs feeding
    # one globally ordered fold chain initialized exactly once.
    assert recorder.initialize == [True, False]
    assert recorder.initialize.count(True) == 1

    reference = _manual_wgrad_fixed_reference(d_proj5, z_ln)
    replay = stable._trimul_wgrad_fixed(stable._get_ext(), d_proj5, z_ln)
    torch.cuda.synchronize()
    assert torch.equal(actual, reference)
    assert torch.equal(replay, actual)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
def test_hopper_wgrad_tail_accuracy_matches_pure_pytorch_fp32():
    channels = 128
    sequence = 65
    rows = sequence * sequence
    torch.manual_seed(8173)
    d_proj5 = torch.randn((rows, 5 * channels), device="cuda", dtype=torch.bfloat16)
    z_ln = torch.randn((rows, channels), device="cuda", dtype=torch.bfloat16)
    actual = stable._trimul_wgrad_fixed(stable._get_ext(), d_proj5, z_ln)
    reference = d_proj5.float().t() @ z_ln.float()
    relative_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
    assert relative_l2.item() <= 1.0e-2


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("channels", (32, 64, 96, 128))
@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize("sequence", (17, 33, 101, 113))
def test_hopper_legacy_custom_op_exact_parity_and_replay(
    channels,
    outgoing,
    sequence,
):
    torch.manual_seed(1000 + channels + sequence + int(outgoing))
    module = _live_module(channels, outgoing)
    z = torch.randn(
        (1, sequence, sequence, channels),
        dtype=torch.bfloat16,
        device="cuda",
    )
    mask = torch.randint(
        0,
        2,
        (1, sequence, sequence),
        dtype=torch.float32,
        device="cuda",
    )
    grad_out = torch.randn_like(z)

    legacy = _run_and_collect(_legacy_call, module, z, mask, grad_out)
    legacy_replay = _run_and_collect(_legacy_call, module, z, mask, grad_out)
    custom = _run_and_collect(
        stable._run_supported_trimul,
        module,
        z,
        mask,
        grad_out,
    )
    custom_replay = _run_and_collect(
        stable._run_supported_trimul,
        module,
        z,
        mask,
        grad_out,
    )

    _assert_exact_receipts(legacy, legacy_replay)
    _assert_exact_receipts(legacy, custom)
    _assert_exact_receipts(legacy, custom_replay)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize(("channels", "sequence"), ((32, 64), (128, 65)))
def test_hopper_custom_ops_pass_opcheck(channels, sequence, outgoing):
    torch.manual_seed(2026 + channels + sequence + int(outgoing))
    module = _live_module(channels, outgoing)
    z = torch.randn(
        (1, sequence, sequence, channels),
        dtype=torch.bfloat16,
        device="cuda",
    )
    mask = torch.ones((1, sequence, sequence), dtype=torch.float32, device="cuda")
    inputs = (
        z.requires_grad_(True),
        mask,
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_g.weight,
        module.linear_z.weight,
        module.layer_norm_in.weight,
        module.layer_norm_in.bias,
        module.layer_norm_out.weight,
        module.layer_norm_out.bias,
        outgoing,
        True,
    )
    forward_result = torch.library.opcheck(
        stable._trimul_fwd_v1,
        inputs,
        raise_exception=True,
    )
    assert all(result == "SUCCESS" for result in forward_result.values())

    with torch.no_grad():
        outputs = stable._trimul_fwd_v1(*inputs)
    backward_inputs = (
        torch.randn_like(outputs[0]),
        inputs[0],
        inputs[1],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
        outputs[9],
        outputs[10],
        *inputs[2:8],
        inputs[8],
        inputs[10],
        outputs[0],
        outgoing,
    )
    # The backward op is intentionally the once-differentiable implementation
    # detail.  Detach its opcheck inputs so this gate covers schema, fake, and
    # AOT forward dispatch without incorrectly requiring a second-order formula.
    backward_inputs = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value
        for value in backward_inputs
    )
    assert not any(
        value.requires_grad
        for value in backward_inputs
        if isinstance(value, torch.Tensor)
    )
    backward_result = torch.library.opcheck(
        stable._trimul_bwd_v1,
        backward_inputs,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=True,
    )
    assert all(result == "SUCCESS" for result in backward_result.values())


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("save_state", (False, True))
def test_hopper_actual_and_fake_forward_metadata_match(save_state):
    torch.manual_seed(2525 + int(save_state))
    module = _live_module(32, True)
    inputs = (
        torch.randn((1, 17, 17, 32), dtype=torch.bfloat16, device="cuda"),
        torch.ones((1, 17, 17), dtype=torch.float32, device="cuda"),
        module.linear_a_p.weight.detach(),
        module.linear_a_g.weight.detach(),
        module.linear_b_p.weight.detach(),
        module.linear_b_g.weight.detach(),
        module.linear_g.weight.detach(),
        module.linear_z.weight.detach(),
        module.layer_norm_in.weight.detach(),
        module.layer_norm_in.bias.detach(),
        module.layer_norm_out.weight.detach(),
        module.layer_norm_out.bias.detach(),
    )
    with torch.no_grad():
        actual = stable._trimul_fwd_v1(*inputs, True, save_state)

    fake_mode = FakeTensorMode()
    fake_inputs = tuple(fake_mode.from_tensor(tensor) for tensor in inputs)
    with (
        fake_mode,
        torch.no_grad(),
        patch.object(
            stable,
            "_get_ext",
            side_effect=AssertionError("registered fake loaded the extension"),
        ) as get_ext,
    ):
        fake = stable._trimul_fwd_v1(*fake_inputs, True, save_state)

    assert len(actual) == len(fake) == 11
    for actual_tensor, fake_tensor in zip(actual, fake):
        assert actual_tensor.shape == fake_tensor.shape
        assert actual_tensor.dtype == fake_tensor.dtype
        assert actual_tensor.device == fake_tensor.device
        assert actual_tensor.stride() == fake_tensor.stride()
    _assert_pairwise_no_alias(actual)
    _assert_pairwise_no_alias(fake)
    get_ext.assert_not_called()


class _SupportedModule(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, z, mask):
        return stable._run_supported_trimul(self.inner, z, mask)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
@pytest.mark.parametrize(("channels", "sequence"), ((32, 64), (128, 65)))
def test_hopper_eager_matches_fullgraph_forward_and_backward(
    outgoing,
    backend,
    channels,
    sequence,
):
    torch.compiler.reset()
    torch.manual_seed(3030 + channels + sequence + int(outgoing))
    eager = _SupportedModule(_live_module(channels, outgoing))
    compiled = copy.deepcopy(eager)
    compiled = torch.compile(compiled, backend=backend, fullgraph=True)
    z = torch.randn(
        (1, sequence, sequence, channels),
        dtype=torch.bfloat16,
        device="cuda",
    )
    mask = torch.ones((1, sequence, sequence), dtype=torch.float32, device="cuda")
    grad_out = torch.randn_like(z)

    eager_receipt = _run_and_collect(
        lambda wrapper, value, pair_mask: wrapper(value, pair_mask),
        eager,
        z,
        mask,
        grad_out,
    )
    compiled_receipt = _run_and_collect(
        lambda wrapper, value, pair_mask: wrapper(value, pair_mask),
        compiled,
        z,
        mask,
        grad_out,
    )
    _assert_exact_receipts(eager_receipt, compiled_receipt)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
@pytest.mark.parametrize("outgoing", (False, True))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_hopper_dynamic_fullgraph_backward_crosses_k_chunk_boundary(
    outgoing,
    backend,
):
    torch.compiler.reset()
    channels = 32
    torch.manual_seed(3737 + int(outgoing))
    eager = _SupportedModule(_live_module(channels, outgoing))
    compiled = torch.compile(
        copy.deepcopy(eager), backend=backend, fullgraph=True, dynamic=True
    )
    for sequence in (127, 129):
        z = torch.randn(
            (1, sequence, sequence, channels),
            dtype=torch.bfloat16,
            device="cuda",
        )
        mask = torch.ones((1, sequence, sequence), dtype=torch.float32, device="cuda")
        grad_out = torch.randn_like(z)
        eager_receipt = _run_and_collect(
            lambda wrapper, value, pair_mask: wrapper(value, pair_mask),
            eager,
            z,
            mask,
            grad_out,
        )
        compiled_receipt = _run_and_collect(
            lambda wrapper, value, pair_mask: wrapper(value, pair_mask),
            compiled,
            z,
            mask,
            grad_out,
        )
        _assert_exact_receipts(eager_receipt, compiled_receipt)


@pytest.mark.slow
@pytest.mark.skipif(not _is_hopper(), reason="requires a Hopper GPU with compute capability 9.0")
def test_hopper_dynamic_sequence_export_save_load_preserves_versioned_target(tmp_path):
    torch.manual_seed(4040)
    wrapper = _SupportedModule(_live_module(32, True))
    z = torch.randn((1, 17, 17, 32), dtype=torch.bfloat16, device="cuda")
    mask = torch.ones((1, 17, 17), dtype=torch.float32, device="cuda")
    sequence = torch.export.Dim("sequence", min=1, max=256)
    exported = torch.export.export(
        wrapper,
        (z, mask),
        dynamic_shapes={
            "z": {1: sequence, 2: sequence},
            "mask": {1: sequence, 2: sequence},
        },
    )
    custom_targets = [
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function" and "turbopairformer" in str(node.target)
    ]
    assert custom_targets == ["turbopairformer._trimul_fwd_v1.default"]

    artifact = tmp_path / "stablefold_trimul_dynamic.pt2"
    torch.export.save(exported, artifact)
    loaded = torch.export.load(artifact).module()
    z_dynamic = torch.randn(
        (1, 33, 33, 32),
        dtype=torch.bfloat16,
        device="cuda",
    )
    mask_dynamic = torch.ones(
        (1, 33, 33),
        dtype=torch.float32,
        device="cuda",
    )
    expected = wrapper(z_dynamic, mask_dynamic)
    actual = loaded(z_dynamic, mask_dynamic)
    assert torch.equal(actual, expected)
