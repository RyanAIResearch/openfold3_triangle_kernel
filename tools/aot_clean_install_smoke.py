#!/usr/bin/env python3
"""Minimal full-F+B smoke for an installed, compiler-free AOT wheel."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

MODULES = (
    "openfold3.core.kernels.cuda._ext._stablefold_forward_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_delta_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_dkdv_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_dq_dbias_sm90a",
    "openfold3.core.kernels.cuda._ext._stablefold_trimul_sm90a",
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
        raise RuntimeError(f"expected five AOT shared objects, got {binary_paths}")

    from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
        invocation_count,
        stablefold_triangle_attention,
    )

    device = torch.device("cuda:0")
    # Exact 64-key tiles deliberately exercise the non-contiguous fused QKV
    # pack. The old N=65 smoke padded Q/K/V first and missed this production
    # fast path entirely.
    prefix, sequence, heads, head_dim = 1, 64, 4, 32

    def run_arm(arm: str) -> dict[str, str]:
        if arm == "sr_r":
            os.environ["OPENFOLD3_SR_BACKWARD"] = "1"
            os.environ["OPENFOLD3_SR_SHADOW"] = "1"
        else:
            os.environ.pop("OPENFOLD3_SR_BACKWARD", None)
            os.environ.pop("OPENFOLD3_SR_SHADOW", None)
        torch.manual_seed(20260809)

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
            raise RuntimeError(
                f"{arm}: public facade route counter did not advance twice"
            )
        if not all(torch.isfinite(value).all().item() for value in named.values()):
            raise RuntimeError(f"{arm}: AOT full F+B produced a non-finite result")
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
                f"{arm}: non-contiguous fused pack differs from the contiguous "
                "path: " + ", ".join(mismatched)
            )
        return {name: raw_sha256(value) for name, value in named.items()}

    arm_receipts = {}
    for arm in ("two_pass", "sr_r"):
        arm_receipts[arm] = run_arm(arm)

    from openfold3.core.kernels.cuda.stablefold_trimul import run_stable_trimul

    def run_trimul(outgoing: bool) -> dict[str, str]:
        torch.manual_seed(20260830 + int(outgoing))
        sequence, channels = 17, 32

        def parameter(*shape: int, ones: bool = False) -> torch.Tensor:
            value = (
                torch.ones(shape, device=device, dtype=torch.float32)
                if ones
                else torch.randn(shape, device=device, dtype=torch.float32)
            )
            return value.requires_grad_(True)

        named_parameters = {
            "w_ap": parameter(channels, channels),
            "w_ag": parameter(channels, channels),
            "w_bp": parameter(channels, channels),
            "w_bg": parameter(channels, channels),
            "w_g": parameter(channels, channels),
            "w_z": parameter(channels, channels),
            "g_in": parameter(channels, ones=True),
            "b_in": parameter(channels),
            "g_out": parameter(channels, ones=True),
            "b_out": parameter(channels),
        }
        module = SimpleNamespace(
            linear_a_p=SimpleNamespace(weight=named_parameters["w_ap"]),
            linear_a_g=SimpleNamespace(weight=named_parameters["w_ag"]),
            linear_b_p=SimpleNamespace(weight=named_parameters["w_bp"]),
            linear_b_g=SimpleNamespace(weight=named_parameters["w_bg"]),
            linear_g=SimpleNamespace(weight=named_parameters["w_g"]),
            linear_z=SimpleNamespace(weight=named_parameters["w_z"]),
            layer_norm_in=SimpleNamespace(
                weight=named_parameters["g_in"],
                bias=named_parameters["b_in"],
            ),
            layer_norm_out=SimpleNamespace(
                weight=named_parameters["g_out"],
                bias=named_parameters["b_out"],
            ),
            _outgoing=outgoing,
        )
        z = torch.randn(
            1,
            sequence,
            sequence,
            channels,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        mask = torch.ones(
            1, sequence, sequence, device=device, dtype=torch.float32
        )
        output = run_stable_trimul(module, z, mask)
        differentiable = (z, *named_parameters.values())
        gradients = torch.autograd.grad(
            output,
            differentiable,
            torch.randn_like(output),
        )
        torch.cuda.synchronize()
        named = {"output": output, "dZ": gradients[0]}
        named.update(
            {
                f"d{name}": gradient
                for name, gradient in zip(
                    named_parameters, gradients[1:], strict=True
                )
            }
        )
        if not all(torch.isfinite(value).all().item() for value in named.values()):
            direction = "outgoing" if outgoing else "incoming"
            raise RuntimeError(
                f"trimul {direction}: AOT full F+B produced a non-finite result"
            )
        return {name: raw_sha256(value) for name, value in named.items()}

    trimul_receipts = {
        "outgoing": run_trimul(True),
        "incoming": run_trimul(False),
    }

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
                "arms": arm_receipts,
                "trimul": trimul_receipts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
