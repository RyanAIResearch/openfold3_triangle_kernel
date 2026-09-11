# SPDX-License-Identifier: Apache-2.0

"""AOT loader for the Trionce softmax-delta precompute."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return the source digest recorded by the AOT build."""
    from trionce._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("trionce_delta", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Import the wheel-owned delta extension (JIT only by explicit opt-in)."""
    global _EXTENSION
    if _EXTENSION is None:
        from trionce._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="trionce_delta",
            aot_module=("trionce._ext._trionce_delta_sm90a"),
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "stablefold_triangle_delta_out",
                "stablefold_triangle_delta_shadow_out",
            ),
            verbose=bool(verbose),
        )
    return _EXTENSION


__all__ = ["load_extension", "source_sha256"]
