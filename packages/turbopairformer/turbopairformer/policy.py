# SPDX-License-Identifier: Apache-2.0

"""Execution-flag policy shared by the TurboPairFormer facades and the DDP preflight.

This module deliberately imports neither PyTorch Lightning nor
``torch.distributed`` so that the kernel facades stay usable outside a
Lightning training loop.
"""

from __future__ import annotations

import os

# Canonical flag names.  The previous ``STABLEFOLD_*`` names and
# ``OPENFOLD3_*`` spellings remain accepted for existing launchers.
# The canonical name wins when more than one is set.
SR_BACKWARD = "TURBOPAIRFORMER_SR_BACKWARD"
SR_SHADOW = "TURBOPAIRFORMER_SR_SHADOW"
ALLOW_JIT = "TURBOPAIRFORMER_ALLOW_JIT"

EXECUTION_ENV_KEYS: tuple[str, ...] = (SR_BACKWARD, SR_SHADOW, ALLOW_JIT)
LEGACY_ENV_NAMES: dict[str, tuple[str, ...]] = {
    SR_BACKWARD: ("STABLEFOLD_SR_BACKWARD", "OPENFOLD3_SR_BACKWARD"),
    SR_SHADOW: ("STABLEFOLD_SR_SHADOW", "OPENFOLD3_SR_SHADOW"),
    ALLOW_JIT: ("STABLEFOLD_ALLOW_JIT", "OPENFOLD3_TRIANGLE_ALLOW_JIT"),
}
_DEFAULT_EXECUTION_FLAGS = {SR_BACKWARD: "1", SR_SHADOW: "1", ALLOW_JIT: "0"}
_CANONICAL_BY_ANY_NAME: dict[str, str] = {
    **{name: name for name in EXECUTION_ENV_KEYS},
    **{
        legacy: canonical
        for canonical, aliases in LEGACY_ENV_NAMES.items()
        for legacy in aliases
    },
}

_STRICT_DISTRIBUTED = False
_STRICT_EXECUTION_ENVIRONMENT: dict[str, str | None] | None = None


class DistributedContractError(RuntimeError):
    """Raised when TurboPairFormer kernel identities or flags differ across ranks."""


def canonical_flag_name(name: str) -> str:
    """Map a canonical or legacy flag name to its canonical spelling."""

    try:
        return _CANONICAL_BY_ANY_NAME[name]
    except KeyError as error:
        raise KeyError(f"unknown TurboPairFormer execution flag: {name}") from error


def read_execution_flag(name: str) -> str | None:
    """Resolve overrides or the production default without changing the environment.

    Single-reconstruction backward and its forward rounding residual are on by
    default. JIT stays off. Explicit canonical or legacy values take precedence.
    The same effective values are recorded and frozen by DDP preflight.
    """

    canonical = canonical_flag_name(name)
    value = os.environ.get(canonical)
    if value is not None:
        return value
    for legacy in LEGACY_ENV_NAMES[canonical]:
        value = os.environ.get(legacy)
        if value is not None:
            return value
    return _DEFAULT_EXECUTION_FLAGS[canonical]


def execution_environment() -> dict[str, str | None]:
    """Snapshot effective flags, including defaults, keyed by canonical name."""

    return {name: read_execution_flag(name) for name in EXECUTION_ENV_KEYS}


def distributed_strict_enabled() -> bool:
    """Return whether unexpected TurboPairFormer model-level fallbacks are fatal."""

    return _STRICT_DISTRIBUTED


def _set_distributed_strict(
    enabled: bool,
    execution_env: dict[str, str | None] | None = None,
) -> None:
    global _STRICT_DISTRIBUTED, _STRICT_EXECUTION_ENVIRONMENT
    _STRICT_DISTRIBUTED = bool(enabled)
    _STRICT_EXECUTION_ENVIRONMENT = (
        {canonical_flag_name(name): value for name, value in execution_env.items()}
        if enabled and execution_env is not None
        else None
    )


def validate_distributed_environment() -> None:
    """Reject changes to TurboPairFormer execution flags after rank preflight."""

    if _STRICT_DISTRIBUTED and (
        _STRICT_EXECUTION_ENVIRONMENT is None
        or execution_environment() != _STRICT_EXECUTION_ENVIRONMENT
    ):
        raise DistributedContractError(
            "TurboPairFormer execution environment changed after DDP kernel preflight"
        )


def distributed_execution_flag(name: str) -> bool:
    """Read a TurboPairFormer flag, from the frozen DDP policy when it is armed."""

    canonical = canonical_flag_name(name)
    if not _STRICT_DISTRIBUTED:
        return read_execution_flag(canonical) == "1"
    if _STRICT_EXECUTION_ENVIRONMENT is None:
        raise DistributedContractError(
            "strict TurboPairFormer execution policy was not initialized"
        )
    return _STRICT_EXECUTION_ENVIRONMENT[canonical] == "1"


__all__ = [
    "ALLOW_JIT",
    "EXECUTION_ENV_KEYS",
    "LEGACY_ENV_NAMES",
    "SR_BACKWARD",
    "SR_SHADOW",
    "DistributedContractError",
    "canonical_flag_name",
    "execution_environment",
    "read_execution_flag",
    "distributed_execution_flag",
    "distributed_strict_enabled",
    "validate_distributed_environment",
]
