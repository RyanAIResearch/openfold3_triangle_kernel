#!/usr/bin/env python3
"""Compute-Sanitizer smoke for the production TurboPairFormer facade.

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
      python scripts/benchmarks/turbopairformer_triangle_sanitizer_smoke.py \
      --sequence 65 --heads 4 --head-dim 32 --prefix 1 \
      --output-json evidence/turbopairformer_memcheck_n65.json

The odd default sequence length deliberately covers the generic tail path.
Use separate fresh processes for memcheck, synccheck, and racecheck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (  # noqa: E402
    invocation_count,
    is_turbopairformer_supported,
    turbopairformer_triangle_attention,
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
    *,
    broadcast_bias: bool,
    bias_prefix_override: tuple[int, ...] | None,
    fully_masked: bool,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    projection_shape = prefix + (sequence, sequence, heads, head_dim)
    q = _projection_view(projection_shape, generator).requires_grad_(True)
    k = _projection_view(projection_shape, generator).requires_grad_(True)
    v = _projection_view(projection_shape, generator).requires_grad_(True)

    if bias_prefix_override is not None:
        if len(bias_prefix_override) != len(prefix) or any(
            bias not in (1, target)
            for target, bias in zip(
                prefix, bias_prefix_override, strict=True
            )
        ):
            raise ValueError(
                "--bias-prefix must match the target prefix rank and each "
                "dimension must be one or the target dimension"
            )
        bias_prefix = bias_prefix_override
    else:
        bias_prefix = (1,) * len(prefix) if broadcast_bias else prefix
    pair_bias_physical = (
        torch.randn(
            bias_prefix + (sequence, sequence, heads),
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
    if fully_masked:
        mask.fill_(float("-inf"))
    else:
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
        "bias_prefix": bias_prefix,
    }


def _batch_to_bias_map(
    target_prefix: tuple[int, ...], bias_prefix: tuple[int, ...]
) -> list[int]:
    bias_strides = [1] * len(bias_prefix)
    for axis in range(len(bias_prefix) - 2, -1, -1):
        bias_strides[axis] = bias_strides[axis + 1] * bias_prefix[axis + 1]
    mapping: list[int] = []
    for flat_index in range(math.prod(target_prefix)):
        remaining = flat_index
        coordinates = [0] * len(target_prefix)
        for axis in range(len(target_prefix) - 1, -1, -1):
            coordinates[axis] = remaining % target_prefix[axis]
            remaining //= target_prefix[axis]
        mapping.append(
            sum(
                (0 if bias_prefix[axis] == 1 else coordinates[axis])
                * bias_strides[axis]
                for axis in range(len(target_prefix))
            )
        )
    return mapping


def _broadcast_bias_reference(
    inputs: dict[str, Any],
    prefix: tuple[int, ...],
    observed_dbias: torch.Tensor,
) -> dict[str, Any]:
    """Compare CUDA-owned prefix reduction with independent public calls."""

    bias_prefix = tuple(inputs["bias_prefix"])
    batch = math.prod(prefix)
    bias_batch = math.prod(bias_prefix)
    q_tail = tuple(inputs["q"].shape[-4:])
    mask_tail = tuple(inputs["mask"].shape[-4:])
    bias_tail = tuple(inputs["pair_bias"].shape[-4:])
    output_tail = tuple(inputs["d_output"].shape[-4:])
    q_flat = inputs["q"].reshape((batch,) + q_tail)
    k_flat = inputs["k"].reshape((batch,) + q_tail)
    v_flat = inputs["v"].reshape((batch,) + q_tail)
    mask_flat = inputs["mask"].reshape((batch,) + mask_tail)
    d_output_flat = inputs["d_output"].reshape((batch,) + output_tail)
    bias_values = inputs["pair_bias"].reshape((bias_batch,) + bias_tail)
    accumulators = [
        torch.zeros(bias_tail, dtype=torch.float32, device="cpu")
        for _ in range(bias_batch)
    ]
    mapping = _batch_to_bias_map(prefix, bias_prefix)
    for batch_index, bias_index in enumerate(mapping):
        bias_leaf = (
            bias_values[bias_index : bias_index + 1]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        output = turbopairformer_triangle_attention(
            q_flat[batch_index : batch_index + 1].detach(),
            k_flat[batch_index : batch_index + 1].detach(),
            v_flat[batch_index : batch_index + 1].detach(),
            mask_flat[batch_index : batch_index + 1],
            bias_leaf,
        )
        (gradient,) = torch.autograd.grad(
            output,
            (bias_leaf,),
            d_output_flat[batch_index : batch_index + 1],
            retain_graph=False,
            create_graph=False,
        )
        # The production reducer defines its contract as ascending-batch FP32
        # additions of already-rounded public BF16 per-batch gradients.
        accumulators[bias_index].add_(gradient[0].detach().cpu().float())
    reference = (
        torch.stack(accumulators, dim=0)
        .to(torch.bfloat16)
        .reshape(observed_dbias.shape)
    )
    observed_cpu = observed_dbias.detach().cpu().contiguous()
    mismatch = int(torch.count_nonzero(observed_cpu != reference).item())
    max_abs = float(
        (observed_cpu.float() - reference.float()).abs().max().item()
    )
    return {
        "target_prefix": list(prefix),
        "bias_prefix": list(bias_prefix),
        "flattened_batch": batch,
        "bias_batches": bias_batch,
        "fixed_batch_order": list(range(batch)),
        "raw_different_elements": mismatch,
        "maximum_abs_delta": max_abs,
        "passed": mismatch == 0,
    }


def _source_manifest() -> dict[str, str]:
    relatives = (
        "openfold3/core/kernels/cuda/turbopairformer_triangle_attention.py",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_forward.py",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_forward.cu",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_delta.py",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_delta.cu",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_dkdv.py",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_dkdv.cu",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_dq_dbias.py",
        "openfold3/core/kernels/cuda/turbopairformer_triangle_dq_dbias.cu",
        "scripts/benchmarks/turbopairformer_triangle_sanitizer_smoke.py",
    )
    return {
        relative: hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest()
        for relative in relatives
    }


def _forward_backward(
    inputs: dict[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run the public facade and its complete VJP once."""

    output = turbopairformer_triangle_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["mask"],
        inputs["pair_bias"],
    )
    gradients = torch.autograd.grad(
        output,
        (
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["pair_bias"],
        ),
        inputs["d_output"],
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return output, gradients


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
        broadcast_bias=args.broadcast_bias,
        bias_prefix_override=(
            tuple(args.bias_prefix) if args.bias_prefix is not None else None
        ),
        fully_masked=args.fully_masked,
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
        "mask_contains_expected_negative_value": bool(
            torch.count_nonzero(inputs["mask"] != 0).item() > 0
            and (
                torch.isneginf(inputs["mask"]).all().item()
                if args.fully_masked
                else torch.all(
                    inputs["mask"][inputs["mask"] != 0]
                    == torch.tensor(-1.0e9, device="cuda", dtype=torch.bfloat16)
                ).item()
            )
        ),
        "upstream_gradient_public_layout_contiguous": bool(
            inputs["d_output"].is_contiguous()
        ),
    }
    layout_contract["passed"] = all(layout_contract.values())
    supported, support_reason = is_turbopairformer_supported(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["mask"],
        inputs["pair_bias"],
    )
    if not supported:
        raise RuntimeError(f"TurboPairFormer facade rejected smoke case: {support_reason}")
    if not layout_contract["passed"]:
        failed = [
            name
            for name, passed in layout_contract.items()
            if name != "passed" and not passed
        ]
        raise RuntimeError(f"production input-layout contract failed: {failed}")

    for _ in range(args.warmup):
        warmup_output, warmup_gradients = _forward_backward(inputs)
        del warmup_output, warmup_gradients
    torch.cuda.synchronize()

    counter_before = invocation_count()
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        torch.cuda.nvtx.range_push("turbopairformer.public_facade.full_forward_backward")
        try:
            output, gradients = _forward_backward(inputs)
        finally:
            torch.cuda.nvtx.range_pop()
        # Every measured kernel must finish before cudaProfilerStop closes the
        # capture range.  Warmup/JIT stays outside the range.
        torch.cuda.synchronize()
    finally:
        if args.cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStop()
    counter_after = invocation_count()

    route = {
        "entrypoint": (
            "openfold3.core.kernels.cuda.turbopairformer_triangle_attention."
            "turbopairformer_triangle_attention"
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
    fully_masked_zero = {
        "required": bool(args.fully_masked),
        "per_tensor": {
            name: int(torch.count_nonzero(value).item()) == 0
            for name, value in named_results.items()
        },
    }
    fully_masked_zero["passed"] = bool(
        not fully_masked_zero["required"]
        or all(fully_masked_zero["per_tensor"].values())
    )
    broadcast_bias_reference = {
        "required": bool(args.check_broadcast_reference),
        "passed": not args.check_broadcast_reference,
    }
    if args.check_broadcast_reference:
        broadcast_bias_reference = {
            "required": True,
            **_broadcast_bias_reference(
                inputs,
                prefix,
                gradients[3],
            ),
        }
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
        and fully_masked_zero["passed"]
        and broadcast_bias_reference["passed"]
        and public_output_contract["passed"]
    )
    properties = torch.cuda.get_device_properties(args.device)
    return {
        "schema_version": 1,
        "purpose": "production TurboPairFormer Triangle Attention full F+B sanitizer smoke",
        "verdict": "PASS" if verdict else "FAIL",
        "case": {
            "prefix": list(prefix),
            "sequence": args.sequence,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "seed": args.seed,
            "warmup": args.warmup,
            "cuda_profiler_range": args.cuda_profiler_range,
            "broadcast_bias": args.broadcast_bias,
            "bias_prefix": list(inputs["bias_prefix"]),
            "fully_masked": args.fully_masked,
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
        "inputs": {
            name: _tensor_receipt(value)
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        "input_layout_contract": layout_contract,
        "public_output_contract": public_output_contract,
        "route": route,
        "finite": finite,
        "fully_masked_zero": fully_masked_zero,
        "broadcast_bias_reference": broadcast_bias_reference,
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
    parser.add_argument(
        "--warmup",
        type=_positive,
        default=1,
        help="public F+B warmup iterations outside the optional profiler range",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="surround exactly one measured public F+B with cudaProfilerStart/Stop",
    )
    parser.add_argument(
        "--broadcast-bias",
        action="store_true",
        help="broadcast one shared pair-bias tensor over the batch prefix",
    )
    parser.add_argument(
        "--bias-prefix",
        type=_positive,
        nargs="+",
        help="explicit pair-bias prefix, e.g. --prefix 2 3 --bias-prefix 1 3",
    )
    parser.add_argument(
        "--check-broadcast-reference",
        action="store_true",
        help=(
            "compare the fixed CUDA prefix reduction against independent "
            "single-batch public calls"
        ),
    )
    parser.add_argument(
        "--fully-masked",
        action="store_true",
        help="use real -inf for every key and require finite zero gradients",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.broadcast_bias and args.bias_prefix is not None:
        raise ValueError("choose --broadcast-bias or --bias-prefix, not both")
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
