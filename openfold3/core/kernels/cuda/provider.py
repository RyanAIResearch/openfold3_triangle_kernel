# SPDX-License-Identifier: Apache-2.0

"""Resolve which implementation serves the stable triangle kernel routes.

Two providers expose the same facade:

``trionce``
    the installed ``trionce`` pip package, built from ``trionce/`` in this
    repository (``pip install -e trionce``).
``intree``
    the kernel copy vendored under ``openfold3/core/kernels/cuda``.

Both compile byte-identical ``.cu`` sources, so the two providers are expected
to be numerically indistinguishable; ``openfold3/tests/test_trionce_provider_equivalence.py``
asserts that forward outputs and gradients match bitwise.

Selection is explicit and logged once:

* ``OPENFOLD3_TRIANGLE_PROVIDER=trionce`` requires the package and fails closed
  if it cannot be imported.
* ``OPENFOLD3_TRIANGLE_PROVIDER=intree`` always uses the vendored copy.
* Unset, the pip package wins when it is importable, otherwise the vendored
  copy serves, so an installation without ``trionce`` behaves exactly as before.

The resolution is memoized: a run cannot silently change provider midway, which
would void the reproducibility receipts the distributed preflight writes.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import os
from typing import Any, Callable

PROVIDER_ENV = "OPENFOLD3_TRIANGLE_PROVIDER"
TRIONCE = "trionce"
INTREE = "intree"
PROVIDERS = (TRIONCE, INTREE)

_logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the requested kernel provider cannot serve the route."""


def _trionce_available() -> bool:
    """Return whether a real ``trionce`` install -- not a stray directory -- exists.

    A directory named ``trionce`` on ``sys.path`` (the package's own project
    folder, when the interpreter starts in the repository root) is importable as
    a PEP 420 namespace package, and setuptools appends its editable finder
    *after* the standard path finder, so that empty namespace wins.  Treating it
    as an install would route the kernels at a package with no modules in it.
    """

    try:
        spec = importlib.util.find_spec(TRIONCE)
    except (ImportError, ValueError):
        return False
    # A namespace package has no origin and no loader; a real one has both.
    return spec is not None and spec.origin is not None


@functools.lru_cache(maxsize=1)
def active_provider() -> str:
    """Return the memoized provider name serving the stable triangle routes."""

    requested = os.environ.get(PROVIDER_ENV)
    if requested is None:
        resolved = TRIONCE if _trionce_available() else INTREE
        _logger.info(
            "%s is unset; serving stable triangle kernels from the %s provider",
            PROVIDER_ENV,
            resolved,
        )
        return resolved

    requested = requested.strip().lower()
    if requested not in PROVIDERS:
        raise ProviderError(
            f"{PROVIDER_ENV}={requested!r} is not a known provider; "
            f"choose one of {', '.join(PROVIDERS)}"
        )
    if requested == TRIONCE and not _trionce_available():
        raise ProviderError(
            f"{PROVIDER_ENV}=trionce was requested but the 'trionce' package is "
            "not importable; install it with 'pip install -e trionce'"
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


@functools.lru_cache(maxsize=1)
def triangle_attention_facade() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(is_supported, triangle_attention)`` from the active provider.

    ``is_supported(q, k, v, mask, bias)`` returns ``(bool, reason)`` and
    ``triangle_attention(q, k, v, mask, bias)`` returns the attention output;
    both consume raw (unscaled) ``q`` and apply ``1/sqrt(d)`` internally.
    """

    if active_provider() == TRIONCE:
        module = _import("trionce.triangle_attention")
        return module.is_triangle_attention_supported, module.triangle_attention

    module = _import("openfold3.core.kernels.cuda.stablefold_triangle_attention")
    return module.is_stablefold_supported, module.stablefold_triangle_attention


@functools.lru_cache(maxsize=1)
def trimul_support_check() -> Callable[..., Any]:
    """Return ``check_support(module, z, mask)`` from the active provider."""

    if active_provider() == TRIONCE:
        return _import("trionce.trimul_support").check_trimul_support
    return _import(
        "openfold3.core.kernels.cuda.stablefold_trimul_support"
    ).check_stable_trimul_support


@functools.lru_cache(maxsize=1)
def trimul_runner() -> Callable[..., Any]:
    """Return ``run_supported(module, z, mask)`` from the active provider.

    Kept separate from the support check so the fused runner, and with it the
    ``torch.library`` registration its module performs at import, still loads
    only once a shape has actually passed support -- the ordering the host had
    before the routes moved behind this provider.
    """

    if active_provider() == TRIONCE:
        return _import("trionce.trimul")._run_supported_trimul
    return _import(
        "openfold3.core.kernels.cuda.stablefold_trimul"
    )._run_supported_stable_trimul


def strict_fallback_enabled() -> bool:
    """Return whether an unexpected model-level fallback must raise.

    The host DDP preflight and the ``trionce`` package keep independent strict
    flags; either one arming the contract makes a fallback fatal.
    """

    from openfold3.core.utils.distributed_contract import (
        stablefold_distributed_strict_enabled,
    )

    if stablefold_distributed_strict_enabled():
        return True
    if active_provider() == TRIONCE:
        return bool(_import("trionce.policy").distributed_strict_enabled())
    return False


def provider_identity() -> dict[str, Any]:
    """Return a receipt-friendly description of the serving provider."""

    provider = active_provider()
    identity: dict[str, Any] = {"provider": provider, "env": os.environ.get(PROVIDER_ENV)}
    if provider == TRIONCE:
        package = _import("trionce")
        identity["trionce_version"] = package.__version__
        identity["trionce_path"] = package.__file__
    return identity


def _reset_for_tests() -> None:
    """Clear the memoized resolution; tests switch providers in-process."""

    active_provider.cache_clear()
    triangle_attention_facade.cache_clear()
    trimul_support_check.cache_clear()
    trimul_runner.cache_clear()


__all__ = [
    "INTREE",
    "PROVIDERS",
    "PROVIDER_ENV",
    "TRIONCE",
    "ProviderError",
    "active_provider",
    "provider_identity",
    "strict_fallback_enabled",
    "triangle_attention_facade",
    "trimul_runner",
    "trimul_support_check",
]
