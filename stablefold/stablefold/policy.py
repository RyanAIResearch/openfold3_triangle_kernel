# SPDX-License-Identifier: Apache-2.0

"""Execution-flag policy shared by the StableFold facades and the DDP preflight.

This module deliberately imports neither PyTorch Lightning nor
``torch.distributed`` so that the kernel facades stay usable outside a
Lightning training loop.
"""

from __future__ import annotations

import os

# Canonical flag names.  The OpenFold3 fork that first shipped StableFold used
# the ``OPENFOLD3_*`` spellings; both are honoured, the canonical name wins.
SR_BACKWARD = "STABLEFOLD_SR_BACKWARD"
SR_SHADOW = "STABLEFOLD_SR_SHADOW"
ALLOW_JIT = "STABLEFOLD_ALLOW_JIT"

EXECUTION_ENV_KEYS: tuple[str, ...] = (SR_BACKWARD, SR_SHADOW, ALLOW_JIT)
LEGACY_ENV_NAMES: dict[str, str] = {
    SR_BACKWARD: "OPENFOLD3_SR_BACKWARD",
    SR_SHADOW: "OPENFOLD3_SR_SHADOW",
    ALLOW_JIT: "OPENFOLD3_TRIANGLE_ALLOW_JIT",
}
_CANONICAL_BY_ANY_NAME: dict[str, str] = {
    **{name: name for name in EXECUTION_ENV_KEYS},
    **{legacy: name for name, legacy in LEGACY_ENV_NAMES.items()},
}

_STRICT_DISTRIBUTED_STABLEFOLD = False
_STRICT_EXECUTION_ENVIRONMENT: dict[str, str | None] | None = None


class DistributedContractError(RuntimeError):
    """Raised when StableFold kernel identities or flags differ across ranks."""


def canonical_flag_name(name: str) -> str:
    """Map a canonical or legacy flag name to its canonical spelling."""

    try:
        return _CANONICAL_BY_ANY_NAME[name]
    except KeyError as error:
        raise KeyError(f"unknown StableFold execution flag: {name}") from error


def read_execution_flag(name: str) -> str | None:
    """Return the raw environment value of a flag, honouring legacy names."""

    canonical = canonical_flag_name(name)
    value = os.environ.get(canonical)
    if value is None:
        value = os.environ.get(LEGACY_ENV_NAMES[canonical])
    return value


def execution_environment() -> dict[str, str | None]:
    """Snapshot every StableFold execution flag keyed by canonical name."""

    return {name: read_execution_flag(name) for name in EXECUTION_ENV_KEYS}


def stablefold_distributed_strict_enabled() -> bool:
    """Return whether unexpected StableFold model-level fallbacks are fatal."""

    return _STRICT_DISTRIBUTED_STABLEFOLD


def _set_stablefold_distributed_strict(
    enabled: bool,
    execution_env: dict[str, str | None] | None = None,
) -> None:
    global _STRICT_DISTRIBUTED_STABLEFOLD, _STRICT_EXECUTION_ENVIRONMENT
    _STRICT_DISTRIBUTED_STABLEFOLD = bool(enabled)
    _STRICT_EXECUTION_ENVIRONMENT = (
        {canonical_flag_name(name): value for name, value in execution_env.items()}
        if enabled and execution_env is not None
        else None
    )


def validate_stablefold_distributed_environment() -> None:
    """Reject changes to StableFold execution flags after rank preflight."""

    if _STRICT_DISTRIBUTED_STABLEFOLD and (
        _STRICT_EXECUTION_ENVIRONMENT is None
        or execution_environment() != _STRICT_EXECUTION_ENVIRONMENT
    ):
        raise DistributedContractError(
            "StableFold execution environment changed after DDP kernel preflight"
        )


def stablefold_distributed_execution_flag(name: str) -> bool:
    """Read a StableFold flag, from the frozen DDP policy when it is armed."""

    canonical = canonical_flag_name(name)
    if not _STRICT_DISTRIBUTED_STABLEFOLD:
        return read_execution_flag(canonical) == "1"
    if _STRICT_EXECUTION_ENVIRONMENT is None:
        raise DistributedContractError(
            "strict StableFold execution policy was not initialized"
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
    "stablefold_distributed_execution_flag",
    "stablefold_distributed_strict_enabled",
    "validate_stablefold_distributed_environment",
]
