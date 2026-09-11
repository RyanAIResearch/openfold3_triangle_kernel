# SPDX-License-Identifier: Apache-2.0

"""Trionce: deterministic Hopper Triangle Attention and TriMul kernels.

Public entry points::

    from trionce import triangle_attention, is_triangle_attention_supported
    from trionce import run_trimul, check_trimul_support

Importing the package is deliberately light: only the pure-Python policy and
TriMul support-check modules load eagerly. The kernel facades, and with them
``torch.library`` op registration and the CUDA extensions, load on first
attribute access, so a host model can import ``trionce`` for a support
check without touching a GPU.

The facades validate the tensor contract and fail closed on unsupported
inputs; model-level routing with a stock fallback belongs to the host model.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from trionce.policy import (
    DistributedContractError,
    distributed_execution_flag,
    distributed_strict_enabled,
)
from trionce.trimul_support import (
    TRIMUL_STOCK_FALLBACK_THRESHOLD,
    SupportResult,
    TrimulSupportMetadata,
    check_trimul_support,
    evaluate_trimul_support,
)

# Lazily resolved public names -> (module, attribute).  Keeping the kernel
# facades out of import time is what lets a CPU-only host import ``trionce``
# for a support check without loading the CUDA extensions.
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "triangle_attention": (
        "trionce._triangle_attention",
        "triangle_attention",
    ),
    "is_triangle_attention_supported": (
        "trionce._triangle_attention",
        "is_triangle_attention_supported",
    ),
    "triangle_attention_invocation_count": (
        "trionce._triangle_attention",
        "invocation_count",
    ),
    "run_trimul": ("trionce.trimul", "run_trimul"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(f"module 'trionce' has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


try:
    __version__ = version("trionce")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+source"

__all__ = [
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
