#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare two OpenFold3 checkpoints tensor by tensor.

Moving the stable triangle routes onto the ``turbopairformer`` pip package is meant to
change packaging only.  Running the same runner YAML with the same seed under
each provider and diffing the resulting checkpoints is the end-to-end form of
that claim: every parameter after the optimizer step must match exactly, not
merely to a tolerance.

Usage::

    python tools/compare_checkpoints.py <dir-or-file-a> <dir-or-file-b>

Exits nonzero when the two disagree, naming the first tensors that differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# How many differing tensors to describe before summarising the rest.
MAX_REPORTED = 10


def _resolve(path: Path) -> Path:
    """Accept a checkpoint file, or a run directory holding ``last.ckpt``."""

    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"no such checkpoint path: {path}")
    candidates = sorted(path.rglob("*.ckpt"))
    if not candidates:
        raise SystemExit(f"no .ckpt file under {path}")
    for candidate in candidates:
        if candidate.name == "last.ckpt":
            return candidate
    return candidates[0]


def _state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    return {
        name: value for name, value in state.items() if isinstance(value, torch.Tensor)
    }


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Compare logical tensor contents, including signed zero and NaN payloads."""
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("bitwise comparison requires dense, non-quantized tensors")
    return (
        tensor.detach()
        .resolve_conj()
        .resolve_neg()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
    )


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(_tensor_bytes(left), _tensor_bytes(right))
    )


def _describe(name: str, left: torch.Tensor, right: torch.Tensor) -> str:
    if left.shape != right.shape:
        return f"  {name}: shape {tuple(left.shape)} vs {tuple(right.shape)}"
    if left.dtype != right.dtype:
        return f"  {name}: dtype {left.dtype} vs {right.dtype}"
    different_bytes = int((_tensor_bytes(left) != _tensor_bytes(right)).sum())
    numeric_dtype = torch.complex128 if left.is_complex() else torch.float64
    delta = (left.to(numeric_dtype) - right.to(numeric_dtype)).abs()
    max_delta = delta.max().item() if delta.numel() else 0.0
    return (
        f"  {name}: max |delta| = {max_delta:.3e}, "
        f"differing bytes = {different_bytes}/{left.numel() * left.element_size()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    arguments = parser.parse_args()

    left_path = _resolve(arguments.left)
    right_path = _resolve(arguments.right)
    print(f"left : {left_path}")
    print(f"right: {right_path}")

    left = _state_dict(torch.load(left_path, map_location="cpu", weights_only=False))
    right = _state_dict(torch.load(right_path, map_location="cpu", weights_only=False))

    if not left and not right:
        print("FAIL: neither checkpoint contains any state tensors")
        return 1

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if only_left or only_right:
        print(
            f"FAIL: parameter sets differ ({len(only_left)} only on the left, "
            f"{len(only_right)} only on the right)"
        )
        for name in (only_left + only_right)[:MAX_REPORTED]:
            print(f"  {name}")
        return 1

    differing = [
        name for name in sorted(left) if not _bitwise_equal(left[name], right[name])
    ]

    total = len(left)
    elements = sum(tensor.numel() for tensor in left.values())
    if not differing:
        print(f"PASS: {total} tensors ({elements:,} elements) are bitwise identical")
        return 0

    print(f"FAIL: {len(differing)}/{total} tensors differ")
    for name in differing[:MAX_REPORTED]:
        print(_describe(name, left[name], right[name]))
    if len(differing) > MAX_REPORTED:
        print(f"  ... and {len(differing) - MAX_REPORTED} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
