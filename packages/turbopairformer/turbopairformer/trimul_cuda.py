# SPDX-License-Identifier: Apache-2.0

"""Load the CUDA extension used by ``_trimul.py``.

The sibling ``.cu`` file implements fused stages and fixed-order reductions.
The binary is installed under ``turbopairformer._ext.*``;
``_aot_runtime`` owns its module name, required exports, and loading policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Load on first use, then reuse the successfully loaded extension module.
_EXTENSION: Any = None


def source_sha256() -> dict[str, str]:
    """Return ``{source filename: SHA256}`` without loading the extension.

    See ``_aot_runtime.source_sha256_for`` for manifest precedence, local-file
    fallback, and errors. A recorded hash need not match the current source.
    """
    from turbopairformer._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("turbopairformer_trimul", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Load once using the shared AOT/JIT policy; return the cached module.

    See ``_aot_runtime.load_extension_module`` for validation and fallback.
    ``verbose`` controls JIT logging; ``None`` means ``False``.
    """
    global _EXTENSION
    if _EXTENSION is None:
        from turbopairformer._aot_runtime import (
            _PRODUCTION_MODULE_CONTRACT,
            load_extension_module,
        )

        module_key = "turbopairformer_trimul"
        aot_module, expected_symbols = _PRODUCTION_MODULE_CONTRACT[module_key]
        _EXTENSION = load_extension_module(
            module_key=module_key,
            aot_module=aot_module,
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=expected_symbols,
            verbose=bool(verbose),
        )
    return _EXTENSION


__all__ = ["load_extension", "source_sha256"]
