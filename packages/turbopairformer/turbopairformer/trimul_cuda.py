# SPDX-License-Identifier: Apache-2.0

"""AOT loader for the TurboPairFormer triangle multiplicative update kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return the source digest recorded by the AOT build."""
    from turbopairformer._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("turbopairformer_trimul", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Import the wheel-owned TriMul extension (JIT only by explicit opt-in)."""
    global _EXTENSION
    if _EXTENSION is None:
        from turbopairformer._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="turbopairformer_trimul",
            aot_module=("turbopairformer._ext._turbopairformer_trimul_sm90a"),
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "trimul_in_fwd",
                "trimul_out_fwd",
                "trimul_gate_split_bwd",
                "sig_mul_bwd_out",
                "ln_bwd_dx",
                "ln_bwd_dgb",
                "transpose_ln_bwd",
                "trimul_wgrad_fixed_fold",
            ),
            verbose=bool(verbose),
        )
    return _EXTENSION


__all__ = ["load_extension", "source_sha256"]
