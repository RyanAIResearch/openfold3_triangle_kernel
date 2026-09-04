# SPDX-License-Identifier: Apache-2.0

"""StableFold: deterministic Hopper Triangle Attention and TriMul kernels.

Public entry points::

    from stablefold import triangle_attention, is_triangle_attention_supported
    from stablefold import run_trimul, check_trimul_support

The facades validate the tensor contract and fail closed on unsupported
inputs; model-level routing (with a stock fallback) belongs to the host model.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from stablefold.policy import (
    DistributedContractError,
    stablefold_distributed_execution_flag,
    stablefold_distributed_strict_enabled,
)
from stablefold.stablefold_triangle_attention import (
    invocation_count as triangle_attention_invocation_count,
)
from stablefold.stablefold_triangle_attention import (
    is_stablefold_supported as is_triangle_attention_supported,
)
from stablefold.stablefold_triangle_attention import (
    stablefold_triangle_attention as triangle_attention,
)
from stablefold.stablefold_trimul import run_stable_trimul as run_trimul
from stablefold.stablefold_trimul_support import (
    STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD as TRIMUL_STOCK_FALLBACK_THRESHOLD,
)
from stablefold.stablefold_trimul_support import (
    StableTrimulSupportMetadata,
    SupportResult,
)
from stablefold.stablefold_trimul_support import (
    check_stable_trimul_support as check_trimul_support,
)
from stablefold.stablefold_trimul_support import (
    evaluate_stable_trimul_support as evaluate_trimul_support,
)

try:
    __version__ = version("stablefold")
except PackageNotFoundError:  # source checkout without an installed dist
    __version__ = "0.0.0+source"

__all__ = [
    "TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "DistributedContractError",
    "StableTrimulSupportMetadata",
    "SupportResult",
    "__version__",
    "check_trimul_support",
    "evaluate_trimul_support",
    "is_triangle_attention_supported",
    "run_trimul",
    "stablefold_distributed_execution_flag",
    "stablefold_distributed_strict_enabled",
    "triangle_attention",
    "triangle_attention_invocation_count",
]
