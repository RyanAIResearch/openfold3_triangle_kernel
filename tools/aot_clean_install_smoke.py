#!/usr/bin/env python3
"""Minimal full-F+B smoke for an installed, compiler-free AOT wheel."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path

import torch

MODULES = (
    "openfold3.core.kernels.cuda._ext._stablefold_forward_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_delta_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_dkdv_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_dq_dbias_sm90a",
)


def raw_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def main() -> None:
    if shutil.which("nvcc") is not None:
        raise RuntimeError("clean-install smoke requires nvcc to be absent from PATH")
    if os.environ.get("OPENFOLD3_TRIANGLE_ALLOW_JIT") is not None:
        raise RuntimeError("clean-install smoke forbids the development JIT switch")

    binaries = [importlib.import_module(name) for name in MODULES]
    binary_paths = [Path(module.__file__).resolve() for module in binaries]
    if any(path.suffix != ".so" for path in binary_paths):
        raise RuntimeError(f"expected four AOT shared objects, got {binary_paths}")

    from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
        invocation_count,
        stablefold_triangle_attention,
    )

    torch.manual_seed(20260809)
    device = torch.device("cuda:0")
    # Exact 64-key tiles deliberately exercise the non-contiguous fused QKV
    # pack. The old N=65 smoke padded Q/K/V first and missed this production
    # fast path entirely.
    prefix, sequence, heads, head_dim = 1, 64, 4, 32

    def projection_view() -> torch.Tensor:
        base = torch.randn(
            prefix,
            sequence,
            sequence,
            heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        return base.transpose(-2, -3).detach().requires_grad_(True)

    q, k, v = projection_view(), projection_view(), projection_view()
    q_contiguous, k_contiguous, v_contiguous = (
        tensor.contiguous().detach().requires_grad_(True)
        for tensor in (q, k, v)
    )
    mask = torch.zeros(
        prefix,
        sequence,
        1,
        1,
        sequence,
        device=device,
        dtype=torch.bfloat16,
    )
    bias = torch.randn(
        prefix,
        1,
        heads,
        sequence,
        sequence,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bias_contiguous = bias.detach().clone().requires_grad_(True)
    before = invocation_count()
    output = stablefold_triangle_attention(q, k, v, mask, bias)
    output_contiguous = stablefold_triangle_attention(
        q_contiguous,
        k_contiguous,
        v_contiguous,
        mask,
        bias_contiguous,
    )
    d_output = torch.randn_like(output)
    gradients = torch.autograd.grad(output, (q, k, v, bias), d_output)
    gradients_contiguous = torch.autograd.grad(
        output_contiguous,
        (q_contiguous, k_contiguous, v_contiguous, bias_contiguous),
        d_output,
    )
    torch.cuda.synchronize()
    named = {
        "output": output,
        "dQ": gradients[0],
        "dK": gradients[1],
        "dV": gradients[2],
        "dBias": gradients[3],
    }
    if invocation_count() - before != 2:
        raise RuntimeError("public facade route counter did not advance twice")
    if not all(torch.isfinite(value).all().item() for value in named.values()):
        raise RuntimeError("AOT full F+B produced a non-finite result")
    reference = {
        "output": output_contiguous,
        "dQ": gradients_contiguous[0],
        "dK": gradients_contiguous[1],
        "dV": gradients_contiguous[2],
        "dBias": gradients_contiguous[3],
    }
    mismatched = [
        name for name, value in named.items()
        if not torch.equal(value, reference[name])
    ]
    if mismatched:
        raise RuntimeError(
            "non-contiguous fused pack differs from the contiguous path: "
            + ", ".join(mismatched)
        )

    forbidden_cache_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    if forbidden_cache_value and Path(forbidden_cache_value).exists():
        raise RuntimeError("production import touched the forbidden JIT cache")
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "binaries": [str(path) for path in binary_paths],
                "raw_sha256": {
                    name: raw_sha256(value) for name, value in named.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
