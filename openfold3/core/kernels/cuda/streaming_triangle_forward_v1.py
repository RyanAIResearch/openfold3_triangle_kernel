# SPDX-License-Identifier: Apache-2.0

"""JIT loader for the production streaming Triangle Attention forward."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return the digest of the CUDA source compiled by this loader."""
    source = Path(__file__).with_suffix(".cu")
    return {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}


def load_extension(*, verbose: bool | None = None) -> Any:
    """Build and cache the streaming forward extension from package data."""
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="streaming_triangle_forward_v1",
            sources=[str(Path(__file__).with_suffix(".cu"))],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=bool(verbose),
        )
    return _EXTENSION


__all__ = ["load_extension", "source_sha256"]
