# SPDX-License-Identifier: Apache-2.0

"""Fail-closed loader for the AOT TurboPairFormer CUDA extensions.

Production always imports wheel-owned binary modules.  Runtime JIT
exists only for source-tree development and requires the explicit
``OPENFOLD3_TRIANGLE_ALLOW_JIT=1`` opt-in.  There is deliberately no automatic
fallback when an AOT artifact is absent or ABI-incompatible.

The per-kernel Python loaders call ``load_extension_module``. Distributed
startup calls ``distributed_aot_identity`` to verify local artifacts before
the caller compares identities across ranks. ``source_sha256_for`` reports
source provenance without loading an extension.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import sysconfig
from pathlib import Path
from threading import Lock
from typing import Any

import torch

_DEVELOPMENT_JIT_ENV = "OPENFOLD3_TRIANGLE_ALLOW_JIT"
_MANIFEST_PATH = Path(__file__).with_name("_build_manifest.json")
_LOCK = Lock()
# Successful loads are cached by module key; the lock also serializes first use.
_MODULE_CACHE: dict[str, Any] = {}

# Shared binary names and exports for per-kernel loaders and distributed preload.
_PRODUCTION_MODULE_CONTRACT: dict[str, tuple[str, tuple[str, ...]]] = {
    "turbopairformer_forward": (
        "openfold3.core.kernels.turbopairformer._ext._turbopairformer_forward_sm90a",
        (
            # Attention output and log-sum-exp (LSE).
            "turbopairformer_triangle_forward_out",
            # Reorder projection Q/K/V into the kernel input layout.
            "turbopairformer_pack_qkv",
            # Also save the BF16 output rounding residual for backward.
            "turbopairformer_triangle_forward_shadow_out",
        ),
    ),
    "turbopairformer_delta": (
        "openfold3.core.kernels.turbopairformer._ext._turbopairformer_delta_sm90a",
        (
            # Sum dOutput * output over head channels.
            "turbopairformer_triangle_delta_out",
            # Add the saved rounding residual before multiplying.
            "turbopairformer_triangle_delta_shadow_out",
        ),
    ),
    "turbopairformer_dkdv": (
        "openfold3.core.kernels.turbopairformer._ext._turbopairformer_dkdv_sm90a",
        (
            # Write dK/dV using the supplied delta.
            "turbopairformer_triangle_dkdv_out",
            # Also reconstruct and reduce the independent delta_true output.
            "turbopairformer_triangle_dkdv_delta_true_out",
        ),
    ),
    "turbopairformer_dq_dbias": (
        "openfold3.core.kernels.turbopairformer._ext._turbopairformer_dq_dbias_sm90a",
        (
            # Split backward: write dQ and per-batch FP32 dBias.
            "turbopairformer_triangle_dq_dbias_out",
            # Combine per-batch dBias for a broadcast bias input.
            "turbopairformer_reduce_broadcast_dbias_out",
            # Reconstruct probabilities once for all four gradients.
            "turbopairformer_triangle_sr_backward_out",
        ),
    ),
    "turbopairformer_trimul": (
        "openfold3.core.kernels.turbopairformer._ext._turbopairformer_trimul_sm90a",
        (
            # Forward: input projections/gates, then output LN/projection/gate.
            "trimul_in_fwd",
            "trimul_out_fwd",
            # Backward: branch gates, output gate, and layer normalization.
            "trimul_gate_split_bwd",
            "sig_mul_bwd_out",
            "ln_bwd_dx",
            "ln_bwd_dgb",
            "transpose_ln_bwd",
            # Accumulate FP32 weight-gradient partials in a fixed order.
            "trimul_wgrad_fixed_fold",
        ),
    ),
}


class AOTExtensionError(RuntimeError):
    """Raised when the production AOT extension contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    """Read the manifest and require schema version 1.

    File I/O, JSON parsing, and unsupported-version failures raise
    ``AOTExtensionError``. Each consumer validates the fields it needs.
    """
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
    """Check the build, runtime ABI, device, and binary before importing it.

    The resolved path is returned so the post-import check can verify that
    Python loaded the same file whose name and digest were just checked.
    """

    manifest = _manifest()
    if not manifest.get("built"):
        raise AOTExtensionError(
            "the installed package contains a source-tree manifest rather than "
            "an AOT build manifest"
        )
    if manifest.get("target", {}).get("cuda_arch") != "sm_90a":
        raise AOTExtensionError("the extension wheel is not an sm_90a artifact")

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
        raise AOTExtensionError("the sm_90a extension requires an available CUDA GPU")
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability != (9, 0):
        raise AOTExtensionError(
            f"the sm_90a extension requires compute capability 9.0, got {capability}"
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
    """Build the local CUDA source after an explicitly allowed AOT failure.

    The source digest is part of the extension name, so distinct source bytes
    select distinct development builds. Production never calls this path.
    """
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

    # The kernels use warpgroup MMA, which ptxas only accepts for the
    # arch-specific sm_90a target; plain "9.0" builds fail to assemble.
    # Supply a default without overriding an existing developer setting.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
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
    """Return the cached module or validate and load its AOT binary.

    AOT is always tried first. Import, OS, or AOT-contract failures may fall
    back to a local build only with ``OPENFOLD3_TRIANGLE_ALLOW_JIT=1``. That
    flag does not force recompilation when a valid AOT binary is available.

    Successful modules are cached by ``module_key`` under a lock. A cache hit
    returns immediately without rechecking the manifest, environment, or
    source; failed loads are not cached. ``verbose`` affects only JIT builds.
    """

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
                    "the Python 3.14 / Torch 2.10 / CUDA 12.9 sm_90a wheel. "
                    "Source-tree developers may explicitly set "
                    f"{_DEVELOPMENT_JIT_ENV}=1."
                ) from error
            module = _load_development_jit(
                module_key=module_key,
                source=source,
                verbose=verbose,
            )
            missing = [name for name in expected_symbols if not hasattr(module, name)]
            if missing:
                raise AOTExtensionError(
                    f"development JIT module is missing symbols: {missing}"
                ) from error
        _MODULE_CACHE[module_key] = module
        return module


def distributed_aot_identity(module_keys: tuple[str, ...]) -> dict[str, Any]:
    """Preload and identify AOT modules before a distributed train step.

    Every rank calls this function independently. The caller exchanges the
    returned JSON-compatible identity; this function performs no collective.
    Local failures are expected to be caught by the
    distributed contract so all ranks can leave together instead of one rank
    failing lazily while its peers wait in a later collective.

    This path forbids development JIT and checks both packaged sources and
    requested binaries. It revalidates artifacts even if already cached by a
    per-kernel loader, then records the successfully imported modules in the
    same cache. Source paths and requested module keys are processed in sorted
    order so equivalent installations produce comparable identities.
    """

    if _development_jit_enabled():
        raise AOTExtensionError(
            "distributed production training forbids development JIT"
        )

    manifest = _manifest()
    git = manifest.get("git", {})
    if (
        git.get("dirty") is not False
        or re.fullmatch(r"[0-9a-f]{40}", str(git.get("commit", ""))) is None
    ):
        raise AOTExtensionError("distributed AOT manifest must attest dirty=false")
    target = manifest.get("target", {})
    if target.get("cuda_arch") != "sm_90a" or target.get("ptx_embedded") is not False:
        raise AOTExtensionError(
            "distributed production requires sm_90a SASS with no embedded PTX"
        )

    requested = tuple(sorted(set(module_keys)))
    unknown = [key for key in requested if key not in _PRODUCTION_MODULE_CONTRACT]
    if unknown:
        raise AOTExtensionError(f"unknown distributed AOT modules: {unknown}")

    # Verify source files against the build receipt before loading binaries.
    install_root = Path(__file__).resolve().parents[4]
    packaged_sources: dict[str, str] = {}
    for relative, expected_hash in sorted(manifest.get("sources", {}).items()):
        if not relative.startswith("openfold3/"):
            continue
        source_path = install_root / relative
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise AOTExtensionError(
                f"packaged source does not match AOT manifest: {relative}"
            )
        packaged_sources[relative] = expected_hash
    if not packaged_sources:
        raise AOTExtensionError(
            "distributed AOT manifest has no verifiable packaged sources"
        )

    extensions: dict[str, dict[str, Any]] = {}
    with _LOCK:
        for module_key in requested:
            aot_module, expected_symbols = _PRODUCTION_MODULE_CONTRACT[module_key]
            artifact_path = _preflight_aot_artifact(module_key, aot_module)
            module = importlib.import_module(aot_module)
            _validate_loaded_aot_module(
                module_key, module, expected_symbols, artifact_path
            )
            _MODULE_CACHE[module_key] = module
            extensions[module_key] = {
                "filename": artifact_path.name,
                "module": aot_module,
                "sha256": _sha256(artifact_path),
                "symbols": list(expected_symbols),
            }

    return {
        "schema_version": 1,
        "manifest_sha256": _sha256(_MANIFEST_PATH),
        "git": git,
        "target": target,
        "abi": manifest.get("abi"),
        "packaged_source_sha256": packaged_sources,
        "extensions": extensions,
    }


def source_sha256_for(module_key: str, source: Path) -> dict[str, str]:
    """Return a recorded source digest, or hash the local file as a fallback.

    Look up the basename first, then a unique path ending in that basename.
    A recorded string is returned without comparing it with the local file.
    This helper neither loads nor compiles an extension.

    The manifest must be readable JSON with supported schema version 1:
    manifest I/O, JSON parsing, and unsupported-version errors raise
    ``AOTExtensionError`` before the local-file fallback. A valid manifest with
    no usable hash and no local source file also raises ``AOTExtensionError``.
    """

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
    "distributed_aot_identity",
    "load_extension_module",
    "source_sha256_for",
]
