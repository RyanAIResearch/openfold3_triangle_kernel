# SPDX-License-Identifier: Apache-2.0

"""StableFold: deterministic Hopper Triangle Attention and TriMul kernels.

Public entry points::

    from stablefold import triangle_attention, is_triangle_attention_supported
    from stablefold import run_trimul, check_trimul_support

Importing the package is deliberately light: only the pure-Python policy and
TriMul support-check modules load eagerly. The kernel facades, and with them
``torch.library`` op registration and the CUDA extensions, load on first
attribute access, so a host model can import ``stablefold`` for a support
check without touching a GPU.

The facades validate the tensor contract and fail closed on unsupported
inputs; model-level routing with a stock fallback belongs to the host model.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from stablefold.policy import (
    DistributedContractError,
    stablefold_distributed_execution_flag,
    stablefold_distributed_strict_enabled,
)
from stablefold.stablefold_trimul_support import (
    STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD,
    StableTrimulSupportMetadata,
    SupportResult,
    check_stable_trimul_support,
    evaluate_stable_trimul_support,
)

TRIMUL_STOCK_FALLBACK_THRESHOLD = STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD
check_trimul_support = check_stable_trimul_support
evaluate_trimul_support = evaluate_stable_trimul_support

# Lazily resolved public names -> (module, attribute).  The legacy names used
# by the original OpenFold3-fork integration are kept as aliases.
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "triangle_attention": (
        "stablefold.stablefold_triangle_attention",
        "stablefold_triangle_attention",
    ),
    "stablefold_triangle_attention": (
        "stablefold.stablefold_triangle_attention",
        "stablefold_triangle_attention",
    ),
    "is_triangle_attention_supported": (
        "stablefold.stablefold_triangle_attention",
        "is_stablefold_supported",
    ),
    "is_stablefold_supported": (
        "stablefold.stablefold_triangle_attention",
        "is_stablefold_supported",
    ),
    "triangle_attention_invocation_count": (
        "stablefold.stablefold_triangle_attention",
        "invocation_count",
    ),
    "run_trimul": ("stablefold.stablefold_trimul", "run_stable_trimul"),
    "run_stable_trimul": ("stablefold.stablefold_trimul", "run_stable_trimul"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(f"module 'stablefold' has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


try:
    __version__ = version("stablefold")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+source"

__all__ = [
    "STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "DistributedContractError",
    "StableTrimulSupportMetadata",
    "SupportResult",
    "TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "__version__",
    "check_stable_trimul_support",
    "check_trimul_support",
    "evaluate_stable_trimul_support",
    "evaluate_trimul_support",
    "is_stablefold_supported",
    "is_triangle_attention_supported",
    "run_stable_trimul",
    "run_trimul",
    "stablefold_distributed_execution_flag",
    "stablefold_distributed_strict_enabled",
    "stablefold_triangle_attention",
    "triangle_attention",
    "triangle_attention_invocation_count",
]
