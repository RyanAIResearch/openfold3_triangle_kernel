# SPDX-License-Identifier: Apache-2.0

"""TurboPairFormer: deterministic Hopper Triangle Attention and TriMul kernels.

Public entry points::

    from turbopairformer import turbo_attention, turbo_trimul

``TurboTriangleAttention`` and ``TurboTriangleMultiplicative`` are descriptive
aliases of these functions. ``triangle_attention`` and ``run_trimul`` remain
available for existing integrations.

Importing the package is deliberately light: only the pure-Python policy
module loads eagerly. The support-check modules and kernel facades, including
``torch.library`` op registration, load on first attribute access. CUDA
extensions load only when a supported GPU call needs them; importing these
functions does not load a CUDA extension. Tensor support checks may query device
metadata. Package import and command-line help do not import Torch.

The facades validate the tensor contract and fail closed on unsupported
inputs; model-level routing with a stock fallback belongs to the host model.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from turbopairformer.policy import (
    DistributedContractError,
    distributed_execution_flag,
    distributed_strict_enabled,
)

# Lazily resolved public names -> (module, attribute).  Keeping the kernel
# facades out of import time is what lets a CPU-only host import ``turbopairformer``
# for a support check without loading the CUDA extensions.
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "TRIMUL_STOCK_FALLBACK_THRESHOLD": (
        "turbopairformer.trimul_support",
        "TRIMUL_STOCK_FALLBACK_THRESHOLD",
    ),
    "SupportResult": ("turbopairformer.trimul_support", "SupportResult"),
    "TrimulSupportMetadata": (
        "turbopairformer.trimul_support",
        "TrimulSupportMetadata",
    ),
    "check_trimul_support": (
        "turbopairformer.trimul_support",
        "check_trimul_support",
    ),
    "evaluate_trimul_support": (
        "turbopairformer.trimul_support",
        "evaluate_trimul_support",
    ),
    "turbo_attention": ("turbopairformer._triangle_attention", "turbo_attention"),
    "turbo_trimul": ("turbopairformer._trimul", "turbo_trimul"),
    "TurboTriangleAttention": (
        "turbopairformer._triangle_attention",
        "turbo_attention",
    ),
    "TurboTriangleMultiplicative": ("turbopairformer._trimul", "turbo_trimul"),
    "triangle_attention": (
        "turbopairformer._triangle_attention",
        "turbo_attention",
    ),
    "is_triangle_attention_supported": (
        "turbopairformer._triangle_attention",
        "is_triangle_attention_supported",
    ),
    "triangle_attention_invocation_count": (
        "turbopairformer._triangle_attention",
        "invocation_count",
    ),
    "run_trimul": ("turbopairformer._trimul", "turbo_trimul"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(
            f"module 'turbopairformer' has no attribute {name!r}"
        ) from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


try:
    __version__ = version("turbopairformer")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+source"

__all__ = [
    "TurboTriangleAttention",
    "TurboTriangleMultiplicative",
    "turbo_attention",
    "turbo_trimul",
    "TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "DistributedContractError",
    "SupportResult",
    "TrimulSupportMetadata",
    "__version__",
    "check_trimul_support",
    "distributed_execution_flag",
    "distributed_strict_enabled",
    "evaluate_trimul_support",
    "is_triangle_attention_supported",
    "run_trimul",
    "triangle_attention",
    "triangle_attention_invocation_count",
]
