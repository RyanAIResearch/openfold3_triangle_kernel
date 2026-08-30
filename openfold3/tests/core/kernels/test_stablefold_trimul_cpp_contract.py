from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="StableFold TriMul requires CUDA"
)


@pytest.fixture(scope="module")
def extension():
    from openfold3.core.kernels.cuda.stablefold_trimul_cuda import load_extension

    return load_extension()


def _make_tensors(channels=32, sequence=16):
    batch = 1
    nn = sequence * sequence
    rows = batch * nn
    device = torch.device("cuda")
    bf16 = torch.bfloat16
    fp32 = torch.float32
    weights = [
        torch.randn(channels, channels, device=device, dtype=fp32)
        for _ in range(6)
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
        "x": torch.randn(
            batch * channels, nn, device=device, dtype=bf16
        ),
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


@pytest.mark.parametrize("channels", (32, 64, 96, 128))
def test_all_seven_production_symbols_accept_valid_inputs(extension, channels):
    tensors = _make_tensors(channels=channels, sequence=8)
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
    d_proj5 = torch.empty(
        rows, 5 * channels, device="cuda", dtype=torch.bfloat16
    )
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
    torch.cuda.synchronize()

    outputs = (out, d_raw, d_x, d_gout, d_bout, d_z, d_gin, d_bin, d_proj5)
    assert all(torch.isfinite(value).all() for value in outputs)


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
