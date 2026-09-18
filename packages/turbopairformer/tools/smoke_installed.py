#!/usr/bin/env python3
"""Exercise an installed TurboPairFormer wheel without OpenFold3 or a compiler.

Run this file from outside the checkout after installing the wheel.  This is
an execution/replay smoke test; the numerical test suite supplies the accuracy
gates.  All extensions must load from the installed package with JIT disabled.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

os.environ["TURBOPAIRFORMER_ALLOW_JIT"] = "0"
os.environ["TURBOPAIRFORMER_SR_BACKWARD"] = "1"
os.environ["TURBOPAIRFORMER_SR_SHADOW"] = "1"

import torch  # noqa: E402

import turbopairformer  # noqa: E402
from turbopairformer import turbo_attention, turbo_trimul  # noqa: E402


def bits_equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


class TriangleWeights(torch.nn.Module):
    """The documented weight protocol accepted by turbo_trimul; no host import."""

    def __init__(self, channels, outgoing):
        super().__init__()
        self.c_z = self.c_hidden = channels
        self._outgoing = outgoing
        self.layer_norm_in = torch.nn.LayerNorm(channels, eps=1e-5)
        self.layer_norm_out = torch.nn.LayerNorm(channels, eps=1e-5)
        for name in (
            "linear_a_p",
            "linear_a_g",
            "linear_b_p",
            "linear_b_g",
            "linear_g",
            "linear_z",
        ):
            setattr(self, name, torch.nn.Linear(channels, channels, bias=False))


def attention_run():
    torch.manual_seed(20260918)
    shape = (1, 8, 4, 128, 32)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    bias = torch.randn(
        (1, 1, 4, 128, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    supported, reason = turbopairformer.is_triangle_attention_supported(
        q, k, v, pair_bias=bias
    )
    assert supported, reason
    out = turbo_attention(q, k, v, pair_bias=bias)
    gradients = torch.autograd.grad(out.float().square().mean(), (q, k, v, bias))
    return (out.detach(), *(gradient.detach() for gradient in gradients))


def trimul_run(outgoing):
    torch.manual_seed(20260918)
    module = TriangleWeights(32, outgoing).cuda()
    z = torch.randn(
        (1, 128, 128, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    support = turbopairformer.check_trimul_support(module, z, None)
    assert support, support.reason
    out = turbo_trimul(module, z)
    gradients = torch.autograd.grad(
        out.float().square().mean(), (z, *module.parameters())
    )
    return (out.detach(), *(gradient.detach() for gradient in gradients))


def main():
    package_root = Path(turbopairformer.__file__).resolve().parent
    extension_paths = []
    for name in (
        "triangle_forward",
        "triangle_delta",
        "triangle_dkdv",
        "triangle_dq_dbias",
        "trimul_cuda",
    ):
        extension = importlib.import_module(f"turbopairformer.{name}").load_extension()
        path = Path(extension.__file__).resolve()
        assert path.is_relative_to(package_root), path
        extension_paths.append(str(path))
    results = {}
    for name, function in (
        ("attention", attention_run),
        ("trimul_outgoing", lambda: trimul_run(True)),
        ("trimul_incoming", lambda: trimul_run(False)),
    ):
        first, second = function(), function()
        assert all(torch.isfinite(value).all().item() for value in first), name
        assert all(bits_equal(a, b) for a, b in zip(first, second, strict=True)), name
        results[name] = {"finite": True, "bitwise_replay": True, "tensors": len(first)}
    torch.cuda.synchronize()
    loaded_libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line
            and any(
                name in line
                for name in ("libtorch", "libc10", "libcudart", "libstdc++")
            )
        }
    )
    print(
        json.dumps(
            {
                "turbopairformer": turbopairformer.__version__,
                "package": str(package_root),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "extensions": extension_paths,
                "results": results,
                "loaded_libraries": loaded_libraries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
