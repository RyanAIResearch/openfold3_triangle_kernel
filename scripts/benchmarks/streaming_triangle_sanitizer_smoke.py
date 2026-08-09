#!/usr/bin/env python3
"""Compute-Sanitizer smoke for the production Streaming facade.

This driver intentionally exercises the package entry point rather than any
of the individual extension loaders.  Its inputs reproduce OpenFold3's public
training boundary:

* q/k/v are non-contiguous ``Attention._prep_qkv`` transpose views;
* pair bias is a non-contiguous BF16 movedim/unsqueeze view;
* the additive mask is BF16 and uses the official ``-1e9`` masked value;
* the upstream gradient uses the public ``[..., O, N, H, D]`` output layout;
* one complete forward and ``torch.autograd.grad`` VJP is synchronized before
  the JSON receipt is emitted.

Run the same command under Compute Sanitizer, for example::

    compute-sanitizer --tool memcheck --error-exitcode 99 \
      python scripts/benchmarks/streaming_triangle_sanitizer_smoke.py \
      --sequence 65 --heads 4 --head-dim 32 --prefix 1 \
      --output-json evidence/streaming_memcheck_n65.json

The odd default sequence length deliberately covers the generic tail path.
Use separate fresh processes for memcheck, synccheck, and racecheck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from openfold3.core.kernels.cuda.streaming_triangle_attention import (  # noqa: E402
    invocation_count,
    is_streaming_supported,
    streaming_triangle_attention,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _tensor_receipt(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
    }


def _raw_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor bytes in bounded host chunks after the synchronized VJP."""

    digest = hashlib.sha256()
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    chunk_bytes = 16 * 1024 * 1024
    for start in range(0, raw.numel(), chunk_bytes):
        chunk = raw[start : start + chunk_bytes].cpu().numpy()
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _result_receipt(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        **_tensor_receipt(tensor),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "raw_sha256": _raw_sha256(tensor),
        "sum_fp64": float(value.sum(dtype=torch.float64).item()),
        "l2_fp64": float(torch.linalg.vector_norm(value.double()).item()),
    }


def _projection_view(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """Construct the exact view layout produced by ``_prep_qkv``.

    ``shape`` is ``prefix + (O, N, H, D)``.  A Linear produces a contiguous
    ``prefix + (O, N, H*D)`` tensor; view followed by transpose yields the
    production ``prefix + (O, H, N, D)`` non-contiguous tensor.
    """

    *prefix, outer, sequence, heads, head_dim = shape
    projected = torch.randn(
        tuple(prefix) + (outer, sequence, heads * head_dim),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    return projected.view(
        tuple(prefix) + (outer, sequence, heads, head_dim)
    ).transpose(-2, -3)


def _make_inputs(
    prefix: tuple[int, ...],
    sequence: int,
    heads: int,
    head_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    projection_shape = prefix + (sequence, sequence, heads, head_dim)
    q = _projection_view(projection_shape, generator).requires_grad_(True)
    k = _projection_view(projection_shape, generator).requires_grad_(True)
    v = _projection_view(projection_shape, generator).requires_grad_(True)

    pair_bias_physical = (
        torch.randn(
            prefix + (sequence, sequence, heads),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    pair_bias = (
        pair_bias_physical.movedim(-1, -3)
        .unsqueeze(-4)
        .requires_grad_(True)
    )

    mask = torch.zeros(
        prefix + (sequence, 1, 1, sequence),
        device="cuda",
        dtype=torch.bfloat16,
    )
    masked_keys = 1 if sequence == 1 else min(
        sequence - 1, max(1, sequence // 17)
    )
    mask[..., -masked_keys:] = -1.0e9

    d_output = torch.randn(
        prefix + (sequence, sequence, heads, head_dim),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    return {
        "q": q,
        "k": k,
        "v": v,
        "mask": mask,
        "pair_bias": pair_bias,
        "d_output": d_output,
    }


def _source_manifest() -> dict[str, str]:
    relatives = (
        "openfold3/core/kernels/cuda/streaming_triangle_attention.py",
        "openfold3/core/kernels/cuda/streaming_triangle_forward_v1.py",
        "openfold3/core/kernels/cuda/streaming_triangle_forward_v1.cu",
        "openfold3/core/kernels/cuda/streaming_triangle_delta_v1.py",
        "openfold3/core/kernels/cuda/streaming_triangle_delta_v1.cu",
        "openfold3/core/kernels/cuda/stable_triangle_dkdv_v2.py",
        "openfold3/core/kernels/cuda/stable_triangle_dkdv_v2.cu",
        "openfold3/core/kernels/cuda/fused_triangle_dq_dbias_v9.py",
        "openfold3/core/kernels/cuda/fused_triangle_dq_dbias_v9.cu",
        "scripts/benchmarks/streaming_triangle_sanitizer_smoke.py",
    )
    return {
        relative: hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in relatives
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    prefix = tuple(args.prefix)
    inputs = _make_inputs(
        prefix,
        args.sequence,
        args.heads,
        args.head_dim,
        args.seed,
    )
    layout_contract = {
        "qkv_projection_views_noncontiguous": all(
            not inputs[name].is_contiguous() for name in ("q", "k", "v")
        ),
        "pair_bias_permuted_view_noncontiguous": (
            not inputs["pair_bias"].is_contiguous()
        ),
        "pair_bias_bf16": inputs["pair_bias"].dtype == torch.bfloat16,
        "mask_bf16": inputs["mask"].dtype == torch.bfloat16,
        "mask_contains_official_negative_value": bool(
            torch.count_nonzero(inputs["mask"] != 0).item() > 0
            and torch.all(
                inputs["mask"][inputs["mask"] != 0]
                == torch.tensor(-1.0e9, device="cuda", dtype=torch.bfloat16)
            ).item()
        ),
        "upstream_gradient_public_layout_contiguous": bool(
            inputs["d_output"].is_contiguous()
        ),
    }
    layout_contract["passed"] = all(layout_contract.values())
    supported, support_reason = is_streaming_supported(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["mask"],
        inputs["pair_bias"],
    )
    if not supported:
        raise RuntimeError(f"streaming facade rejected smoke case: {support_reason}")
    if not layout_contract["passed"]:
        failed = [
            name
            for name, passed in layout_contract.items()
            if name != "passed" and not passed
        ]
        raise RuntimeError(f"production input-layout contract failed: {failed}")

    counter_before = invocation_count()
    output = streaming_triangle_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["mask"],
        inputs["pair_bias"],
    )
    leaves = (
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["pair_bias"],
    )
    gradients = torch.autograd.grad(
        output,
        leaves,
        inputs["d_output"],
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()
    counter_after = invocation_count()

    route = {
        "entrypoint": (
            "openfold3.core.kernels.cuda.streaming_triangle_attention."
            "streaming_triangle_attention"
        ),
        "support_check": support_reason,
        "counter_before": counter_before,
        "counter_after": counter_after,
        "observed_facade_calls": counter_after - counter_before,
        "expected_facade_calls": 1,
        "fallback_policy": "fail_closed; package facade has no fallback",
    }
    route["passed"] = route["observed_facade_calls"] == 1
    named_results = {
        "output": output,
        "dQ": gradients[0],
        "dK": gradients[1],
        "dV": gradients[2],
        "dBias": gradients[3],
    }
    results = {
        name: _result_receipt(value) for name, value in named_results.items()
    }
    finite = {
        "per_tensor": {name: value["finite"] for name, value in results.items()}
    }
    finite["passed"] = all(finite["per_tensor"].values())
    expected_output_shape = prefix + (
        args.sequence,
        args.sequence,
        args.heads,
        args.head_dim,
    )
    public_output_contract = {
        "expected_shape": list(expected_output_shape),
        "observed_shape": list(output.shape),
        "observed_stride": list(output.stride()),
        "dtype": str(output.dtype),
        "is_contiguous": bool(output.is_contiguous()),
    }
    public_output_contract["passed"] = bool(
        output.shape == expected_output_shape
        and output.dtype == torch.bfloat16
        and output.is_contiguous()
    )
    verdict = bool(
        layout_contract["passed"]
        and route["passed"]
        and finite["passed"]
        and public_output_contract["passed"]
    )
    properties = torch.cuda.get_device_properties(args.device)
    return {
        "schema_version": 1,
        "purpose": "production Streaming Triangle Attention full F+B sanitizer smoke",
        "verdict": "PASS" if verdict else "FAIL",
        "case": {
            "prefix": list(prefix),
            "sequence": args.sequence,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "seed": args.seed,
            "dtype": "torch.bfloat16",
            "scale_contract": f"raw q; facade applies 1/sqrt({args.head_dim})",
        },
        "environment": {
            "device_index": args.device,
            "device_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(args.device)),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "inputs": {name: _tensor_receipt(value) for name, value in inputs.items()},
        "input_layout_contract": layout_contract,
        "public_output_contract": public_output_contract,
        "route": route,
        "finite": finite,
        "checksums": {
            name: {
                "raw_sha256": value["raw_sha256"],
                "sum_fp64": value["sum_fp64"],
                "l2_fp64": value["l2_fp64"],
            }
            for name, value in results.items()
        },
        "results": results,
        "source_sha256": _source_manifest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", "--N", type=_positive, default=65)
    parser.add_argument("--heads", "--H", type=_positive, default=4)
    parser.add_argument("--head-dim", "--D", type=_positive, default=32)
    parser.add_argument(
        "--prefix",
        type=_positive,
        nargs="+",
        default=[1],
        help="one or more positive batch-prefix dimensions",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
