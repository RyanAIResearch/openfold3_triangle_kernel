# SPDX-License-Identifier: Apache-2.0

"""Fail-closed loader for the AOT StableFold Triangle Attention extensions.

Production always imports the four wheel-owned binary modules.  Runtime JIT
exists only for source-tree development and requires the explicit
``OPENFOLD3_TRIANGLE_ALLOW_JIT=1`` opt-in.  There is deliberately no automatic
fallback when an AOT artifact is absent or ABI-incompatible.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sysconfig
from pathlib import Path
from threading import Lock
from typing import Any

import torch

_DEVELOPMENT_JIT_ENV = "OPENFOLD3_TRIANGLE_ALLOW_JIT"
_MANIFEST_PATH = Path(__file__).with_name("_build_manifest.json")
_LOCK = Lock()
_MODULE_CACHE: dict[str, Any] = {}


class AOTExtensionError(RuntimeError):
    """Raised when the production AOT extension contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    try:
        value = json.loads(_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AOTExtensionError(
            f"invalid AOT build manifest at {_MANIFEST_PATH}"
        ) from error
    if value.get("schema_version") != 1:
        raise AOTExtensionError("unsupported AOT build-manifest schema")
    return value


def _development_jit_enabled() -> bool:
    return os.environ.get(_DEVELOPMENT_JIT_ENV) == "1"


def _runtime_abi() -> dict[str, Any]:
    return {
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def _preflight_aot_artifact(module_key: str, aot_module: str) -> Path:
    """Validate the wheel-owned binary before its initializer can execute."""

    manifest = _manifest()
    if not manifest.get("built"):
        raise AOTExtensionError(
            "the installed package contains a source-tree manifest rather than "
            "an AOT build manifest"
        )
    if manifest.get("target", {}).get("cuda_arch") != "sm_90":
        raise AOTExtensionError("the extension wheel is not an sm_90 artifact")

    expected_abi = manifest.get("abi", {})
    actual_abi = _runtime_abi()
    mismatches = {
        name: (expected_abi.get(name), actual)
        for name, actual in actual_abi.items()
        if expected_abi.get(name) != actual
    }
    if mismatches:
        raise AOTExtensionError(f"AOT extension ABI mismatch: {mismatches}")

    if not torch.cuda.is_available():
        raise AOTExtensionError("the sm_90 extension requires an available CUDA GPU")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (9, 0):
        raise AOTExtensionError(
            f"the sm_90 extension requires compute capability 9.0, got {capability}"
        )

    extension = manifest.get("extensions", {}).get(module_key)
    if not isinstance(extension, dict):
        raise AOTExtensionError(f"manifest has no extension entry for {module_key}")
    if extension.get("module") != aot_module:
        raise AOTExtensionError(f"manifest module mismatch for {module_key}")
    try:
        spec = importlib.util.find_spec(aot_module)
    except (ImportError, AttributeError, ValueError) as error:
        raise AOTExtensionError(f"cannot locate AOT module {aot_module}") from error
    if spec is None or spec.origin is None:
        raise AOTExtensionError(f"cannot locate AOT module {aot_module}")
    module_path = Path(spec.origin).resolve()
    expected_filename = Path(str(extension.get("filename", ""))).name
    if module_path.name != expected_filename:
        raise AOTExtensionError(f"binary filename mismatch for {module_key}")
    expected_hash = extension.get("sha256")
    if not expected_hash or _sha256(module_path) != expected_hash:
        raise AOTExtensionError(f"binary hash mismatch for {module_key}")
    return module_path


def _validate_loaded_aot_module(
    module_key: str,
    module: Any,
    expected_symbols: tuple[str, ...],
    preflight_path: Path,
) -> None:
    """Validate symbols and ensure import did not resolve a different file."""

    if Path(module.__file__).resolve() != preflight_path:
        raise AOTExtensionError(f"loaded binary path changed for {module_key}")
    missing = [name for name in expected_symbols if not hasattr(module, name)]
    if missing:
        raise AOTExtensionError(
            f"AOT extension {module_key} is missing symbols: {missing}"
        )


def _load_development_jit(
    *,
    module_key: str,
    source: Path,
    verbose: bool,
) -> Any:
    if not source.is_file():
        raise AOTExtensionError(f"development CUDA source is missing: {source}")
    if not torch.cuda.is_available():
        raise AOTExtensionError("development JIT requires an available CUDA GPU")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (9, 0):
        raise AOTExtensionError(
            f"development JIT requires compute capability 9.0, got {capability}"
        )
    # Importing the compiler path is intentionally inside the explicit opt-in
    # branch.  Production import never touches cpp_extension or a build cache.
    from torch.utils.cpp_extension import load

    source_hash = _sha256(source)[:12]
    return load(
        name=f"openfold3_triangle_dev_{module_key}_{source_hash}",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=verbose,
    )


def load_extension_module(
    *,
    module_key: str,
    aot_module: str,
    source: Path,
    expected_symbols: tuple[str, ...],
    verbose: bool = False,
) -> Any:
    """Load one AOT module, with an explicit source-development escape hatch."""

    with _LOCK:
        cached = _MODULE_CACHE.get(module_key)
        if cached is not None:
            return cached
        try:
            artifact_path = _preflight_aot_artifact(module_key, aot_module)
            module = importlib.import_module(aot_module)
            _validate_loaded_aot_module(
                module_key, module, expected_symbols, artifact_path
            )
        except (ImportError, OSError, AOTExtensionError) as error:
            if not _development_jit_enabled():
                raise AOTExtensionError(
                    f"unable to load production AOT extension {aot_module}; install "
                    "the Python 3.14 / Torch 2.10 / CUDA 12.9 sm_90 wheel. "
                    f"Source-tree developers may explicitly set {_DEVELOPMENT_JIT_ENV}=1."
                ) from error
            module = _load_development_jit(
                module_key=module_key,
                source=source,
                verbose=verbose,
            )
            missing = [
                name for name in expected_symbols if not hasattr(module, name)
            ]
            if missing:
                raise AOTExtensionError(
                    f"development JIT module is missing symbols: {missing}"
                )
        _MODULE_CACHE[module_key] = module
        return module


def source_sha256_for(module_key: str, source: Path) -> dict[str, str]:
    """Return the source digest embedded in the AOT artifact receipt."""

    manifest = _manifest()
    sources = manifest.get("sources", {})
    recorded = sources.get(source.name)
    if not isinstance(recorded, str):
        matches = [
            digest
            for relative, digest in sources.items()
            if relative.endswith("/" + source.name)
        ]
        recorded = matches[0] if len(matches) == 1 else None
    if isinstance(recorded, str):
        return {source.name: recorded}
    if source.is_file():
        return {source.name: _sha256(source)}
    raise AOTExtensionError(
        f"no recorded source hash for {module_key}; install an AOT wheel or "
        f"set {_DEVELOPMENT_JIT_ENV}=1 in a source checkout"
    )


__all__ = [
    "AOTExtensionError",
    "load_extension_module",
    "source_sha256_for",
]
