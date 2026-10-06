# SPDX-License-Identifier: Apache-2.0

"""Resolve which implementation serves the stable triangle kernel routes.

Two providers expose the same facade:

``turbopairformer``
    the installed ``turbopairformer`` pip package, built from
    ``packages/turbopairformer`` in this repository (install a built wheel
    with ``pip install <wheel>``).
``intree``
    the kernel copy vendored under ``openfold3/core/kernels/cuda``.

Builds of both providers from the same current source use byte-identical
``.cu`` files; the provider-equivalence tests check their outputs and gradients
bitwise. Published wheels may contain historical sources, so this claim does
not extend to an arbitrary installed wheel.

Selection is explicit and logged once:

* ``OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer`` requires the package and
  fails closed if it cannot be imported.
* ``OPENFOLD3_TRIANGLE_PROVIDER=intree`` always uses the vendored copy.
* Unset, the vendored copy serves. Installing the standalone package never
  changes an existing OpenFold3 run's kernel provider.

The resolution is memoized: a run cannot silently change provider midway, which
would void the reproducibility receipts the distributed preflight writes.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import os
from collections.abc import Callable
from typing import Any

PROVIDER_ENV = "OPENFOLD3_TRIANGLE_PROVIDER"
TURBOPAIRFORMER = "turbopairformer"
INTREE = "intree"
PROVIDERS = (TURBOPAIRFORMER, INTREE)

_logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the requested kernel provider cannot serve the route."""


def _turbopairformer_available() -> bool:
    """Require a regular package, not a stray namespace directory on sys.path."""

    try:
        spec = importlib.util.find_spec(TURBOPAIRFORMER)
    except ImportError, ValueError:
        return False
    # A namespace package has no origin and no loader; a real one has both.
    return spec is not None and spec.origin is not None


@functools.lru_cache(maxsize=1)
def active_provider() -> str:
    """Return the memoized provider name serving the stable triangle routes."""

    requested = os.environ.get(PROVIDER_ENV)
    if requested is None:
        resolved = INTREE
        _logger.info(
            "%s is unset; serving stable triangle kernels from the %s provider",
            PROVIDER_ENV,
            resolved,
        )
        return resolved

    requested = requested.strip().lower()
    if requested in ("stablefold", "trionce"):
        raise ProviderError(
            f"{PROVIDER_ENV}={requested!r} uses a retired package name; "
            f"select {PROVIDER_ENV}=turbopairformer and install a compatible "
            "TurboPairFormer wheel"
        )
    if requested not in PROVIDERS:
        raise ProviderError(
            f"{PROVIDER_ENV}={requested!r} is not a known provider; "
            f"choose one of {', '.join(PROVIDERS)}"
        )
    if requested == TURBOPAIRFORMER and not _turbopairformer_available():
        raise ProviderError(
            f"{PROVIDER_ENV}=turbopairformer was requested but the "
            "'turbopairformer' package is "
            "not importable; install a compatible TurboPairFormer wheel with "
            "'pip install /path/to/turbopairformer.whl'"
        )
    _logger.info("serving stable triangle kernels from the %s provider", requested)
    return requested


def _import(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:  # fail closed: explicit selection must not
        raise ProviderError(  # degrade into a different backend
            f"provider {active_provider()!r} cannot import {module_name}: {error}"
        ) from error


# Module import is memoized; the attribute is resolved on every call.  The
# host used to re-run ``from ... import f`` inside each routing call, which is
# a sys.modules lookup plus a getattr and therefore sees a patched attribute.
# Caching the function object instead would silently defeat monkeypatching,
# which the routing tests and any instrumentation rely on.
@functools.lru_cache(maxsize=3)
def _module(kind: str) -> Any:
    turbopairformer_serving = active_provider() == TURBOPAIRFORMER
    names = {
        "attention": (
            # The package, not the private implementation module: the public
            # names are what the package guarantees.
            TURBOPAIRFORMER,
            "openfold3.core.kernels.cuda.turbopairformer_triangle_attention",
        ),
        "trimul_support": (
            "turbopairformer.trimul_support",
            "openfold3.core.kernels.cuda.turbopairformer_trimul_support",
        ),
        "trimul_runner": (
            "turbopairformer._trimul",
            "openfold3.core.kernels.cuda.turbopairformer_trimul",
        ),
    }[kind]
    return _import(names[0] if turbopairformer_serving else names[1])


def triangle_attention_facade() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(is_supported, triangle_attention)`` from the active provider.

    ``is_supported(q, k, v, mask, bias)`` returns ``(bool, reason)`` and
    ``triangle_attention(q, k, v, mask, bias)`` returns the attention output;
    both consume raw (unscaled) ``q`` and apply ``1/sqrt(d)`` internally.
    """

    module = _module("attention")
    if active_provider() == TURBOPAIRFORMER:
        return module.is_triangle_attention_supported, module.triangle_attention
    return (
        module.is_turbopairformer_supported,
        module.turbopairformer_triangle_attention,
    )


def trimul_support_check() -> Callable[..., Any]:
    """Return ``check_support(module, z, mask)`` from the active provider."""

    module = _module("trimul_support")
    if active_provider() == TURBOPAIRFORMER:
        return module.check_trimul_support
    return module.check_stable_trimul_support


def trimul_runner() -> Callable[..., Any]:
    """Return ``run_supported(module, z, mask)`` from the active provider.

    Resolved separately from the support check so the fused runner, and with it
    the ``torch.library`` registration its module performs at import, still
    loads only once a shape has actually passed support -- the ordering the
    host had before the routes moved behind this provider.
    """

    module = _module("trimul_runner")
    if active_provider() == TURBOPAIRFORMER:
        return module._run_supported_trimul
    return module._run_supported_stable_trimul


def validate_attention_policy() -> None:
    """Validate the selected provider's policy before any host shape fallback."""

    if active_provider() == TURBOPAIRFORMER:
        policy = _import("turbopairformer.policy")
        validator = getattr(policy, "validate_attention_policy", None)
        if validator is not None:
            validator()
        else:
            # Published wheels predate the mandatory-policy helper. Their
            # existing getters retain canonical flag precedence and DDP state.
            policy.validate_distributed_environment()
            if not policy.distributed_execution_flag(policy.SR_BACKWARD):
                raise RuntimeError(
                    "TurboPairFormer attention requires single-reconstruction "
                    "backward; the effective SR_BACKWARD flag must be 1. "
                    "Use historical source or low-level reference kernels "
                    "for ablations."
                )
        return
    from openfold3.core.utils.distributed_contract import (
        validate_turbopairformer_attention_policy,
    )

    validate_turbopairformer_attention_policy()


def strict_fallback_enabled() -> bool:
    """Return whether an unexpected model-level fallback must raise.

    The host DDP preflight and the ``turbopairformer`` package keep independent
    strict flags; either one arming the contract makes a fallback fatal.
    """

    from openfold3.core.utils.distributed_contract import (
        turbopairformer_distributed_strict_enabled,
    )

    if turbopairformer_distributed_strict_enabled():
        return True
    if active_provider() == TURBOPAIRFORMER:
        return bool(_import("turbopairformer.policy").distributed_strict_enabled())
    return False


def provider_identity() -> dict[str, Any]:
    """Return a receipt-friendly description of the serving provider."""

    provider = active_provider()
    identity: dict[str, Any] = {
        "provider": provider,
        "env": os.environ.get(PROVIDER_ENV),
    }
    if provider == TURBOPAIRFORMER:
        package = _import("turbopairformer")
        identity["turbopairformer_version"] = package.__version__
        identity["turbopairformer_path"] = package.__file__
    return identity


def _reset_for_tests() -> None:
    """Clear the memoized resolution; tests switch providers in-process."""

    active_provider.cache_clear()
    _module.cache_clear()


__all__ = [
    "INTREE",
    "PROVIDERS",
    "PROVIDER_ENV",
    "TURBOPAIRFORMER",
    "ProviderError",
    "active_provider",
    "provider_identity",
    "strict_fallback_enabled",
    "triangle_attention_facade",
    "trimul_runner",
    "trimul_support_check",
    "validate_attention_policy",
]
