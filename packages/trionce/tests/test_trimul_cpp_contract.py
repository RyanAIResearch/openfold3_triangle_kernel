from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Trionce TriMul requires CUDA"
)


@pytest.fixture(scope="module")
def extension():
    from trionce.trimul_cuda import load_extension

    return load_extension()


def _make_tensors(channels=32, sequence=16):
    batch = 1
    nn = sequence * sequence
    rows = batch * nn
    device = torch.device("cuda")
    bf16 = torch.bfloat16
    fp32 = torch.float32
    weights = [
        torch.randn(channels, channels, device=device, dtype=fp32) for _ in range(6)
    ]
    return {
        "B": batch,
        "N": sequence,
        "NN": nn,
        "C": channels,
        "H": channels,
        "rows": rows,
        "z": torch.randn(rows, channels, device=device, dtype=bf16),
        "mask": torch.ones(rows, device=device, dtype=fp32),
        "gamma": torch.ones(channels, device=device, dtype=fp32),
        "beta": torch.zeros(channels, device=device, dtype=fp32),
        "weights": weights,
        "x": torch.randn(batch * channels, nn, device=device, dtype=bf16),
        "gate": torch.randn(rows, channels, device=device, dtype=bf16),
        "grad": torch.randn(rows, channels, device=device, dtype=bf16),
        "mean": torch.zeros(rows, device=device, dtype=fp32),
        "rstd": torch.ones(rows, device=device, dtype=fp32),
    }


@pytest.fixture
def tensors():
    return _make_tensors()


def _trimul_in_args(tensors):
    w_ap, w_ag, w_bp, w_bg, w_g, _ = tensors["weights"]
    return (
        tensors["z"],
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        tensors["mask"],
        tensors["gamma"],
        tensors["beta"],
        tensors["B"],
        tensors["NN"],
        tensors["C"],
        tensors["H"],
        True,
    )


def _strict_ascending_fold(partials, initial=None):
    if initial is None:
        output = torch.zeros_like(partials[0])
    else:
        output = initial.clone()
    for chunk in partials.unbind(0):
        output.add_(chunk)
    return output


def _run_all_eight(extension, tensors):
    batch = tensors["B"]
    sequence = tensors["N"]
    nn = tensors["NN"]
    channels = tensors["C"]
    rows = tensors["rows"]
    w_ap, w_ag, w_bp, w_bg, _, w_z = tensors["weights"]

    a, b, gate, mean_in, rstd_in, zln = extension.trimul_in_fwd(
        *_trimul_in_args(tensors)
    )
    out, _, mean_out, rstd_out = extension.trimul_out_fwd(
        tensors["x"],
        w_z,
        gate,
        tensors["gamma"],
        tensors["beta"],
        batch,
        nn,
        channels,
        channels,
        True,
    )
    d_proj5 = torch.empty(rows, 5 * channels, device="cuda", dtype=torch.bfloat16)
    d_raw = extension.sig_mul_bwd_out(
        out,
        gate,
        tensors["grad"],
        d_proj5[:, 4 * channels :],
        channels,
        5 * channels,
    )
    d_x, d_gout, d_bout = extension.transpose_ln_bwd(
        tensors["x"],
        d_raw,
        sequence,
        channels,
        batch,
        tensors["gamma"],
        mean_out,
        rstd_out,
    )
    extension.trimul_gate_split_bwd(
        zln,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        tensors["mask"],
        torch.randn_like(a),
        torch.randn_like(b),
        d_proj5,
        batch,
        nn,
        channels,
        channels,
    )
    d_z = extension.ln_bwd_dx(
        tensors["z"],
        tensors["grad"],
        rows,
        channels,
        1,
        tensors["gamma"],
        mean_in,
        rstd_in,
    )
    d_gin, d_bin = extension.ln_bwd_dgb(
        tensors["z"],
        tensors["grad"],
        rows,
        channels,
        1,
        channels,
        mean_in,
        rstd_in,
    )
    partials = torch.randn(
        3, 5 * channels, channels, device="cuda", dtype=torch.float32
    )
    d_w5 = torch.empty(5 * channels, channels, device="cuda", dtype=torch.float32)
    folded = extension.trimul_wgrad_fixed_fold(partials, d_w5, True)
    return (
        out,
        d_raw,
        d_x,
        d_gout,
        d_bout,
        d_z,
        d_gin,
        d_bin,
        d_proj5,
        folded,
    )


@pytest.mark.parametrize("channels", (32, 64, 96, 128))
def test_all_eight_production_symbols_accept_valid_inputs(extension, channels):
    tensors = _make_tensors(channels=channels, sequence=8)
    outputs = _run_all_eight(extension, tensors)
    torch.cuda.synchronize()
    assert all(torch.isfinite(value).all() for value in outputs)


def test_all_symbols_follow_a_non_default_stream(extension):
    stream = torch.cuda.Stream()
    tensors = _make_tensors(channels=64, sequence=8)
    with torch.cuda.stream(stream):
        outputs = _run_all_eight(extension, tensors)
        completion = torch.cuda.Event()
        completion.record()
    completion.synchronize()
    assert all(torch.isfinite(value).all() for value in outputs)


def test_c32_c128_repeat_on_independent_streams(extension):
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    all_outputs = []
    for _ in range(4):
        for stream, channels in zip(streams, (32, 128)):
            with torch.cuda.stream(stream):
                all_outputs.extend(
                    _run_all_eight(
                        extension,
                        _make_tensors(channels=channels, sequence=8),
                    )
                )
    for stream in streams:
        stream.synchronize()
    assert all(torch.isfinite(value).all() for value in all_outputs)


@pytest.mark.parametrize("channels", (32, 64, 96, 128))
def test_wgrad_fold_matches_strict_ascending_reference(extension, channels):
    torch.manual_seed(9187 + channels)
    partials = torch.randn(
        17, 5 * channels, channels, device="cuda", dtype=torch.float32
    )
    expected = _strict_ascending_fold(partials)
    actual = torch.empty_like(expected)
    returned = extension.trimul_wgrad_fixed_fold(partials, actual, True)
    assert returned.data_ptr() == actual.data_ptr()
    assert torch.equal(actual, expected)


def test_wgrad_fold_carries_a_fixed_order_across_waves(extension):
    channels = 64
    torch.manual_seed(9923)
    first = torch.randn(7, 5 * channels, channels, device="cuda", dtype=torch.float32)
    second = torch.randn(11, 5 * channels, channels, device="cuda", dtype=torch.float32)
    expected = _strict_ascending_fold(second, _strict_ascending_fold(first))
    actual = torch.empty_like(expected)
    extension.trimul_wgrad_fixed_fold(first, actual, True)
    extension.trimul_wgrad_fixed_fold(second, actual, False)
    assert torch.equal(actual, expected)


def test_wgrad_fold_order_is_numerically_observable(extension):
    channels = 32
    partials = torch.empty(
        4, 5 * channels, channels, device="cuda", dtype=torch.float32
    )
    for chunk, value in enumerate((1.0e20, 1.0, -1.0e20, 1.0)):
        partials[chunk].fill_(value)
    actual = torch.empty_like(partials[0])
    extension.trimul_wgrad_fixed_fold(partials, actual, True)
    assert torch.equal(actual, torch.ones_like(actual))


def test_wgrad_fold_is_bitwise_replayable(extension):
    channels = 96
    torch.manual_seed(10103)
    partials = torch.randn(
        23, 5 * channels, channels, device="cuda", dtype=torch.float32
    )
    baseline = torch.empty_like(partials[0])
    extension.trimul_wgrad_fixed_fold(partials, baseline, True)
    for _ in range(5):
        replay = torch.empty_like(baseline)
        extension.trimul_wgrad_fixed_fold(partials, replay, True)
        assert torch.equal(replay, baseline)


def test_wgrad_fold_rejects_invalid_inputs(extension):
    channels = 32
    partials = torch.randn(
        3, 5 * channels, channels, device="cuda", dtype=torch.float32
    )
    accumulator = torch.empty_like(partials[0])

    with pytest.raises(RuntimeError, match="partials must be a CUDA tensor"):
        extension.trimul_wgrad_fixed_fold(partials.cpu(), accumulator, True)
    with pytest.raises(RuntimeError, match="partials has the wrong dtype"):
        extension.trimul_wgrad_fixed_fold(
            partials.to(torch.bfloat16), accumulator, True
        )
    with pytest.raises(RuntimeError, match="partials must be contiguous"):
        extension.trimul_wgrad_fixed_fold(partials.transpose(1, 2), accumulator, True)
    with pytest.raises(RuntimeError, match="chunks must be positive"):
        extension.trimul_wgrad_fixed_fold(partials[:0], accumulator, True)
    with pytest.raises(RuntimeError, match="M must be positive"):
        extension.trimul_wgrad_fixed_fold(
            torch.empty(3, 0, channels, device="cuda", dtype=torch.float32),
            torch.empty(0, channels, device="cuda", dtype=torch.float32),
            True,
        )
    with pytest.raises(RuntimeError, match="C must be positive"):
        extension.trimul_wgrad_fixed_fold(
            torch.empty(3, 5 * channels, 0, device="cuda", dtype=torch.float32),
            torch.empty(5 * channels, 0, device="cuda", dtype=torch.float32),
            True,
        )
    with pytest.raises(RuntimeError, match="accumulator has wrong shape"):
        extension.trimul_wgrad_fixed_fold(partials, accumulator[:-1], True)
    with pytest.raises(RuntimeError, match="accumulator must be a CUDA tensor"):
        extension.trimul_wgrad_fixed_fold(partials, accumulator.cpu(), True)
    with pytest.raises(RuntimeError, match="accumulator has the wrong dtype"):
        extension.trimul_wgrad_fixed_fold(
            partials, accumulator.to(torch.bfloat16), True
        )
    with pytest.raises(RuntimeError, match="accumulator must be contiguous"):
        noncontiguous = torch.empty(
            channels, 5 * channels, device="cuda", dtype=torch.float32
        ).t()
        extension.trimul_wgrad_fixed_fold(partials, noncontiguous, True)
    with pytest.raises(RuntimeError, match="written-to tensor"):
        extension.trimul_wgrad_fixed_fold(partials[:1], partials[0], True)


def test_cpp_source_pins_device_and_cuda_error_contracts():
    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("trionce/trimul_cuda.cu")
        .read_text()
    )
    assert source.count("CUDAGuard device_guard") == 8
    assert source.count("C10_CUDA_KERNEL_LAUNCH_CHECK") == 11
    assert source.count("cudaFuncSetAttribute(") == 3
    assert "kTrimulInMaxDynamicSharedBytes = 105984" in source
    assert "kTrimulOutMaxDynamicSharedBytes = 105472" in source
    assert "template <bool Initialize>" in source
    assert "if constexpr (!Initialize)" in source
    assert "trimul_wgrad_fixed_fold_kernel<true>" in source
    assert "trimul_wgrad_fixed_fold_kernel<false>" in source
    assert "sum = __fadd_rn(sum," in source
    assert 'checked_mul(chunks64, elements, op, "chunks*M*C")' in source
    assert "check_same_device(accumulator, partials" in source
    assert "kTrimulGateBwdMaxDynamicSharedBytes = 105472" in source
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in source


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (lambda value: value.cpu(), "z must be a CUDA tensor"),
        (lambda value: value.float(), "z has the wrong dtype"),
        (lambda value: value.t().contiguous().t(), "z must be contiguous"),
        (
            lambda value: torch.empty(
                value.numel() + 1, device=value.device, dtype=value.dtype
            )[1:].view_as(value),
            "z data pointer must be 8-byte aligned",
        ),
    ),
)
def test_trimul_in_rejects_invalid_primary_tensor(
    extension, tensors, replacement, message
):
    arguments = list(_trimul_in_args(tensors))
    arguments[0] = replacement(arguments[0])
    with pytest.raises(RuntimeError, match=message):
        extension.trimul_in_fwd(*arguments)


def test_trimul_in_rejects_invalid_shape_mask_grad_and_channels(extension, tensors):
    arguments = list(_trimul_in_args(tensors))
    arguments[1] = arguments[1][:-1]
    with pytest.raises(RuntimeError, match="w_ap has wrong shape"):
        extension.trimul_in_fwd(*arguments)

    arguments = list(_trimul_in_args(tensors))
    arguments[6] = arguments[6].requires_grad_()
    with pytest.raises(RuntimeError, match="mask must not require gradients"):
        extension.trimul_in_fwd(*arguments)

    arguments = list(_trimul_in_args(tensors))
    arguments[-2] = 64
    with pytest.raises(RuntimeError, match="C and H must be equal"):
        extension.trimul_in_fwd(*arguments)


def test_trimul_out_rejects_wrong_gate_shape(extension, tensors):
    *_, w_z = tensors["weights"]
    with pytest.raises(RuntimeError, match="gate has wrong shape"):
        extension.trimul_out_fwd(
            tensors["x"],
            w_z,
            tensors["gate"][:-1],
            tensors["gamma"],
            tensors["beta"],
            tensors["B"],
            tensors["NN"],
            tensors["C"],
            tensors["H"],
            True,
        )


def test_gate_split_rejects_wrong_destination_shape(extension, tensors):
    w_ap, w_ag, w_bp, w_bg, *_ = tensors["weights"]
    wrong = torch.empty(
        tensors["rows"],
        4 * tensors["H"],
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="d_proj5 has wrong shape"):
        extension.trimul_gate_split_bwd(
            tensors["z"],
            w_ap,
            w_ag,
            w_bp,
            w_bg,
            tensors["mask"],
            tensors["x"],
            tensors["x"],
            wrong,
            tensors["B"],
            tensors["NN"],
            tensors["C"],
            tensors["H"],
        )


def test_sigmoid_backward_rejects_stride_and_overlap(extension, tensors):
    destination = torch.empty_like(tensors["grad"])
    with pytest.raises(RuntimeError, match="dgate_dst strides"):
        extension.sig_mul_bwd_out(
            tensors["grad"],
            tensors["gate"],
            tensors["grad"].clone(),
            destination,
            tensors["H"],
            5 * tensors["H"],
        )

    with pytest.raises(RuntimeError, match="written-to tensor"):
        extension.sig_mul_bwd_out(
            tensors["grad"],
            tensors["gate"],
            tensors["grad"].clone(),
            tensors["grad"],
            tensors["H"],
            tensors["H"],
        )


def test_layer_norm_backward_rejects_nonproduction_stride(extension, tensors):
    with pytest.raises(RuntimeError, match="row_stride=C"):
        extension.ln_bwd_dx(
            tensors["z"],
            tensors["grad"],
            tensors["rows"],
            tensors["C"] + 1,
            1,
            tensors["gamma"],
            tensors["mean"],
            tensors["rstd"],
        )
    with pytest.raises(RuntimeError, match="row_stride=C"):
        extension.ln_bwd_dgb(
            tensors["z"],
            tensors["grad"],
            tensors["rows"],
            tensors["C"] + 1,
            1,
            tensors["C"],
            tensors["mean"],
            tensors["rstd"],
        )


def test_transpose_layer_norm_backward_rejects_bad_batch(extension, tensors):
    with pytest.raises(RuntimeError, match="B must be positive"):
        extension.transpose_ln_bwd(
            tensors["x"],
            tensors["grad"],
            tensors["N"],
            tensors["H"],
            0,
            tensors["gamma"],
            tensors["mean"],
            tensors["rstd"],
        )
