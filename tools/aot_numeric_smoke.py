# SPDX-License-Identifier: Apache-2.0

"""TurboPairFormer Triangle Attention against an FP64 oracle at training shapes.

For every case the same BF16 inputs feed three paths:

* ``oracle``: FP64 PyTorch attention, forward and backward;
* ``stock``: BF16 matmul / FP32 softmax PyTorch attention, the error level of
  the stock OpenFold3 route under autocast;
* ``turbopairformer``: the facade with the requested execution flags.

TurboPairFormer must be finite, bitwise reproducible against itself, exactly zero
on fully masked planes, and within 1.1 times the stock path's relative error
plus 1e-6, as encoded by this test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import torch

DEV = torch.device("cuda")
NEG = -1e4
PROD_FLAGS = {"OPENFOLD3_SR_BACKWARD": "1", "OPENFOLD3_SR_SHADOW": "1"}
CLASSIC_FLAGS = {"OPENFOLD3_SR_BACKWARD": "0", "OPENFOLD3_SR_SHADOW": "0"}


def _finite_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isinf(mask), torch.full_like(mask, NEG), mask)


def _oracle(q, k, v, mask, bias, dout):
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    b64 = bias.detach().double().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q64, k64.transpose(-1, -2)) * scale
    s = s + _finite_mask(mask).double() + b64
    out = torch.matmul(torch.softmax(s, dim=-1), v64).transpose(-2, -3)
    out.backward(dout.double())
    return out.detach(), q64.grad, k64.grad, v64.grad, b64.grad


def _stock(q, k, v, mask, bias, dout):
    q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    b_ = bias.detach().clone().requires_grad_(True)
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q_, k_.transpose(-1, -2)).float() * scale
    s = s + _finite_mask(mask) + b_.float()
    out = torch.matmul(torch.softmax(s, dim=-1).to(torch.bfloat16), v_)
    out = out.transpose(-2, -3)
    out.backward(dout)
    return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad


def _kernel(q, k, v, mask, bias, dout):
    from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (
        turbopairformer_triangle_attention as triangle_attention,
    )

    q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    b_ = bias.detach().clone().requires_grad_(True)
    out = triangle_attention(q_, k_, v_, mask, b_)
    out.backward(dout)
    return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _case(B, O, H, N, D, mask_frac, full_planes, bias_dtype):
    q = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    k = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    v = torch.randn(B, O, H, N, D, device=DEV).to(torch.bfloat16)
    mask = torch.zeros(B, O, 1, 1, N, device=DEV)
    if mask_frac:
        mask[torch.rand(B, O, 1, 1, N, device=DEV) < mask_frac] = float("-inf")
    if full_planes:
        mask[:, :full_planes] = float("-inf")
    bias = (torch.randn(1, 1, H, N, N, device=DEV) * 2).to(bias_dtype)
    dout = torch.randn(B, O, N, H, D, device=DEV).to(torch.bfloat16)
    if full_planes:
        dout[:, :full_planes] = 0
    return q, k, v, mask, bias, dout


CASES = [
    (
        "prod N=128 D=32",
        PROD_FLAGS,
        dict(B=1, O=8, H=4, N=128, D=32, mask_frac=0.0, full_planes=0),
    ),
    (
        "prod N=100 pad D=32 masked",
        PROD_FLAGS,
        dict(B=1, O=8, H=4, N=100, D=32, mask_frac=0.3, full_planes=0),
    ),
    (
        "prod N=384 D=24",
        PROD_FLAGS,
        dict(B=1, O=4, H=4, N=384, D=24, mask_frac=0.2, full_planes=0),
    ),
    (
        "prod N=200 D=20 padD",
        PROD_FLAGS,
        dict(B=2, O=4, H=4, N=200, D=20, mask_frac=0.2, full_planes=0),
    ),
    (
        "prod N=256 D=64 full planes",
        PROD_FLAGS,
        dict(B=1, O=6, H=4, N=256, D=64, mask_frac=0.2, full_planes=2),
    ),
    (
        "prod N=384 D=48",
        PROD_FLAGS,
        dict(B=1, O=2, H=4, N=384, D=48, mask_frac=0.1, full_planes=0),
    ),
    (
        "prod N=768 D=16 H=8",
        PROD_FLAGS,
        dict(B=1, O=2, H=8, N=768, D=16, mask_frac=0.1, full_planes=0),
    ),
    (
        "classic N=384 D=24",
        CLASSIC_FLAGS,
        dict(B=1, O=4, H=4, N=384, D=24, mask_frac=0.2, full_planes=0),
    ),
    (
        "classic N=256 D=64 full planes",
        CLASSIC_FLAGS,
        dict(B=1, O=6, H=4, N=256, D=64, mask_frac=0.2, full_planes=2),
    ),
]


def _bits_equal(left, right):
    return torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    )


def attention_checks():
    """Use the standalone release's original oracle, shapes and tolerance."""
    receipts = []
    for name, flags, shape in CASES:
        os.environ.update(flags)
        torch.manual_seed(0)
        inputs = _case(bias_dtype=torch.bfloat16, **shape)
        reference = _oracle(*inputs)
        stock = _stock(*inputs)
        actual = _kernel(*inputs)
        replay = _kernel(*inputs)
        if not all(_bits_equal(a, b) for a, b in zip(actual, replay, strict=True)):
            raise RuntimeError(
                f"{name}: output or gradients are not bitwise replayable"
            )
        if not all(torch.isfinite(value).all().item() for value in actual):
            raise RuntimeError(f"{name}: non-finite output or gradients")
        full = shape["full_planes"]
        if full:
            if any(value[:, :full].abs().max().item() != 0 for value in actual[:4]):
                raise RuntimeError(f"{name}: fully masked planes must be zero")
            reference = (reference[0][:, full:], *reference[1:])
            stock = (stock[0][:, full:], *stock[1:])
            actual = (actual[0][:, full:], *actual[1:])
        errors = {}
        for label, value, baseline, oracle in zip(
            ("out", "dq", "dk", "dv", "dbias"), actual, stock, reference, strict=True
        ):
            error, baseline_error = _rel(value, oracle), _rel(baseline, oracle)
            limit = 1.1 * baseline_error + 1e-6
            if not error <= limit:
                raise RuntimeError(f"{name} {label}: {error} exceeds {limit}")
            errors[label] = {
                "relative_l2": error,
                "stock_relative_l2": baseline_error,
                "limit": limit,
            }
        receipts.append(
            {
                "name": name,
                "shape": shape,
                "flags": flags,
                "status": "PASS",
                "bitwise_replay": True,
                "finite": True,
                "fully_masked_zero": True if full else None,
                "errors": errors,
            }
        )
    return receipts


def trimul_fold_checks():
    """Retain the C++ contract's exact ascending-FP32 fold reference."""
    from openfold3.core.kernels.cuda.turbopairformer_trimul_cuda import load_extension

    extension = load_extension()
    receipts = []
    for channels in (32, 64, 96, 128):
        torch.manual_seed(9187 + channels)
        partials = torch.randn(
            17, 5 * channels, channels, device="cuda", dtype=torch.float32
        )
        expected = torch.zeros_like(partials[0])
        for chunk in partials.unbind(0):
            expected.add_(chunk)
        actual = torch.empty_like(expected)
        returned = extension.trimul_wgrad_fixed_fold(partials, actual, True)
        if returned.data_ptr() != actual.data_ptr() or not _bits_equal(
            actual, expected
        ):
            raise RuntimeError(f"C={channels}: TriMul FP32 fold differs from reference")
        receipts.append({"channels": channels, "status": "PASS", "bitwise_equal": True})
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("numeric qualification requires exactly one visible GPU")
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("numeric qualification requires Hopper CC 9.0")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this qualification is for H100")
    if torch.__version__.split("+", 1)[0] != "2.10.0" or torch.version.cuda != "12.9":
        raise RuntimeError("numeric qualification requires Torch 2.10 / CUDA 12.9")
    if os.environ.get("OPENFOLD3_TRIANGLE_ALLOW_JIT") not in (None, "0"):
        raise RuntimeError("numeric qualification forbids JIT")
    import openfold3
    from openfold3.core.kernels.cuda._aot_runtime import distributed_aot_identity
    from openfold3.core.utils.distributed_contract import (
        installed_distribution_identity,
    )

    if (
        Path(openfold3.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("numeric qualification must use an installed wheel")
    keys = (
        "turbopairformer_forward",
        "turbopairformer_delta",
        "turbopairformer_dkdv",
        "turbopairformer_dq_dbias",
        "turbopairformer_trimul",
    )
    identity = distributed_aot_identity(keys)
    with zipfile.ZipFile(args.wheel) as archive:
        manifest_bytes = archive.read(
            "openfold3/core/kernels/cuda/_build_manifest.json"
        )
    if identity["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise RuntimeError("loaded binaries do not belong to the selected wheel")
    if identity["git"] != {"commit": args.expected_commit, "dirty": False}:
        raise RuntimeError("loaded binaries do not belong to the selected clean commit")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "Attention FP64 error envelope and TriMul FP32 weight-gradient fold",
        "validator": {
            "path": "tools/aot_numeric_smoke.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "git_commit": args.expected_commit,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_count": 1,
        "aot_identity": identity,
        "installed_distribution": installed_distribution_identity(),
        "attention": attention_checks(),
        "trimul_fp32_fold": trimul_fold_checks(),
        "limitations": [
            "No fresh multi-GPU or end-to-end training qualification",
            "TriMul reference covers FP32 weight-gradient folding; "
            "full-operator accuracy is outside this check",
        ],
    }
    torch.cuda.synchronize()
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "attention_cases": len(result["attention"]),
                "trimul_fold_cases": len(result["trimul_fp32_fold"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
