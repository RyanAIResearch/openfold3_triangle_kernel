# SPDX-License-Identifier: Apache-2.0

"""AOT loader for the StableFold Triangle Attention forward pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return the source digest recorded by the AOT build."""
    from openfold3.core.kernels.cuda._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("stablefold_forward", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Import the wheel-owned forward extension (JIT only by explicit opt-in)."""
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.cuda._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="stablefold_forward",
            aot_module=(
                "openfold3.core.kernels.cuda._ext._stablefold_forward_sm90a"
            ),
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "stablefold_triangle_forward_out",
                "stablefold_pack_qkv",
            ),
            verbose=bool(verbose),
        )
    return _EXTENSION


__all__ = ["load_extension", "source_sha256"]
