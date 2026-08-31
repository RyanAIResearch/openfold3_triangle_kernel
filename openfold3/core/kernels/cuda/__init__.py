"""Default-off CUDA production candidates for OpenFold3."""

from openfold3.core.kernels.cuda.stablefold_trimul_support import (
    STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD,
    StableTrimulSupportMetadata,
    SupportResult,
    check_stable_trimul_support,
    evaluate_stable_trimul_support,
)

__all__ = [
    "STABLE_TRIMUL_STOCK_FALLBACK_THRESHOLD",
    "StableTrimulSupportMetadata",
    "SupportResult",
    "check_stable_trimul_support",
    "evaluate_stable_trimul_support",
]
