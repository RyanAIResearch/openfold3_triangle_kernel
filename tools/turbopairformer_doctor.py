#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check the selected TurboPairFormer provider, its runtime, and its AOT binaries.

Selection follows OPENFOLD3_TRIANGLE_PROVIDER (default: intree). The standalone
package can also be checked without OpenFold3 installed by explicitly selecting
turbopairformer. A CPU-only check verifies imports, not CUDA execution or binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

EXTENSION_FACADES = {
    "turbopairformer": {
        "stablefold_forward": "turbopairformer.triangle_forward",
        "stablefold_delta": "turbopairformer.triangle_delta",
        "stablefold_dkdv": "turbopairformer.triangle_dkdv",
        "stablefold_dq_dbias": "turbopairformer.triangle_dq_dbias",
        "stablefold_trimul": "turbopairformer.trimul_cuda",
    },
    "intree": {
        "stablefold_forward": "openfold3.core.kernels.cuda.stablefold_triangle_forward",
        "stablefold_delta": "openfold3.core.kernels.cuda.stablefold_triangle_delta",
        "stablefold_dkdv": "openfold3.core.kernels.cuda.stablefold_triangle_dkdv",
        "stablefold_dq_dbias": (
            "openfold3.core.kernels.cuda.stablefold_triangle_dq_dbias"
        ),
        "stablefold_trimul": "openfold3.core.kernels.cuda.stablefold_trimul_cuda",
    },
}


class DoctorError(RuntimeError):
    """A preflight check failed; the message names the remedy."""


def _check_gpu(require_gpu: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if require_gpu:
            raise DoctorError("no CUDA device is visible; check CUDA_VISIBLE_DEVICES")
        return {"available": False}
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    if capability != (9, 0):
        raise DoctorError(
            f"TurboPairFormer kernels are compiled for sm_90a; {name} reports "
            f"compute capability {capability[0]}.{capability[1]}"
        )
    return {
        "available": True,
        "name": name,
        "capability": f"{capability[0]}.{capability[1]}",
        "count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def _check_provider(expected: str | None) -> dict[str, Any]:
    requested = os.environ.get("OPENFOLD3_TRIANGLE_PROVIDER", "intree").strip().lower()
    if importlib.util.find_spec("openfold3") is not None:
        try:
            provider = importlib.import_module("openfold3.core.kernels.cuda.provider")
            identity = provider.provider_identity()
        except ModuleNotFoundError as error:
            if (
                error.name == "openfold3.core.kernels.cuda.provider"
                and requested == "intree"
            ):
                # Older OpenFold3 wheels always serve their vendored kernels.
                identity = {"provider": "intree", "host_provider_adapter": False}
            else:
                raise DoctorError(
                    f"cannot resolve the installed OpenFold3 provider: {error}; "
                    "the standalone provider requires a host revision containing "
                    "provider.py"
                ) from error
        except (ImportError, RuntimeError) as error:
            raise DoctorError(
                f"cannot resolve the installed OpenFold3 kernel provider: {error}; "
                "install the host revision containing provider.py"
            ) from error
    else:
        selected = requested
        if selected != "turbopairformer":
            raise DoctorError(
                "OpenFold3 is not installed; select the standalone package with "
                "OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer"
            )
        identity = {"provider": selected, "host": "not installed"}
    if expected is not None and identity["provider"] != expected:
        raise DoctorError(
            f"expected the {expected} provider but OPENFOLD3_TRIANGLE_PROVIDER "
            f"resolves to {identity['provider']}"
        )
    return identity


def _check_package(provider_name: str) -> dict[str, Any]:
    name = "turbopairformer" if provider_name == "turbopairformer" else "openfold3"
    try:
        package = importlib.import_module(name)
    except ImportError as error:
        raise DoctorError(
            f"the selected provider's package {name!r} is not importable"
        ) from error
    path = getattr(package, "__file__", None)
    if path is None:
        raise DoctorError(
            f"{name!r} resolved to a namespace package; install a built wheel"
        )
    return {
        "name": name,
        "version": getattr(package, "__version__", "unknown"),
        "path": path,
    }


def _check_extensions(provider_name: str) -> dict[str, Any]:
    runtime_name = (
        "turbopairformer._aot_runtime"
        if provider_name == "turbopairformer"
        else "openfold3.core.kernels.cuda._aot_runtime"
    )
    runtime = importlib.import_module(runtime_name)
    manifest_path = (
        runtime._manifest_path()
        if hasattr(runtime, "_manifest_path")
        else Path(runtime.__file__).with_name("_build_manifest.json")
    )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DoctorError(
            f"cannot read the {provider_name} build manifest: {error}"
        ) from error
    if not manifest.get("built"):
        raise DoctorError(
            f"the selected {provider_name} provider has no AOT build manifest; "
            "install its compiled wheel before launching"
        )

    # A doctor must never certify a JIT rebuild as the manifest's AOT artifact.
    loaded: dict[str, Any] = {}
    for key, facade_name in EXTENSION_FACADES[provider_name].items():
        try:
            entry = manifest["extensions"][key]
            runtime._preflight_aot_artifact(key, entry["module"])
            facade = importlib.import_module(facade_name)
            module = facade.load_extension()
        except Exception as error:
            raise DoctorError(f"AOT extension {key} failed to load: {error}") from error
        loaded[key] = getattr(module, "__file__", "<builtin>")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git": manifest.get("git", {}),
        "abi": manifest.get("abi", {}),
        "extensions": loaded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-provider", choices=("turbopairformer", "intree"))
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="check imports on CPU; this does not validate CUDA binaries",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    arguments = parser.parse_args()

    report: dict[str, Any] = {}
    try:
        report["provider"] = _check_provider(arguments.expect_provider)
        selected = report["provider"]["provider"]
        report["package"] = _check_package(selected)
        report["gpu"] = _check_gpu(require_gpu=not arguments.allow_no_gpu)
        if report["gpu"].get("available"):
            report["aot"] = _check_extensions(selected)
        report["scope"] = "aot-load" if "aot" in report else "imports-only"
    except (DoctorError, ImportError) as error:
        print(f"turbopairformer doctor: FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    gpu = report["gpu"]
    print(f"turbopairformer doctor: PASS ({report['scope']})")
    if gpu.get("available"):
        print(f"  gpu       {gpu['name']} (sm_{gpu['capability'].replace('.', '')}a)")
        print(f"  torch     {gpu['torch']} / CUDA {gpu['torch_cuda']}")
    else:
        print("  gpu       none visible; CUDA binaries not checked")
    print(f"  package   {report['package']['name']} {report['package']['version']}")
    print(f"            {report['package']['path']}")
    if "aot" in report:
        commit = report["aot"]["git"].get("commit", "unknown")[:12]
        dirty = " (dirty)" if report["aot"]["git"].get("dirty") else ""
        count = len(report["aot"]["extensions"])
        print(f"  aot       {count} extensions @ {commit}{dirty}")
    print(f"  provider  {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
