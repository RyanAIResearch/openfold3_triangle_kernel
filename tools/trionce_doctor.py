#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Preflight the Trionce kernel install before a training launch.

Checks, in order, everything a run needs and nothing it does not:

1. the GPU is a Hopper sm_90a device, which is what the kernels compile for;
2. the ``trionce`` package imports and reports where it was loaded from;
3. every AOT extension loads and matches the hash recorded in its build
   manifest, so a stale or partially rebuilt ``_ext`` cannot serve a run;
4. ``OPENFOLD3_TRIANGLE_PROVIDER`` resolves to the provider the caller meant.

Exits nonzero on the first failure with a message naming the fix.  This is the
editable-install counterpart of ``tools.manage_attested_sm90_wheel
verify-prebuilt``, which verifies the signed release wheel instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

# Facade module -> the AOT extension it loads.  Going through the facades is
# what a training step does, so a doctor PASS means the real call path works.
EXTENSION_FACADES = {
    "trionce_forward": "trionce.triangle_forward",
    "trionce_delta": "trionce.triangle_delta",
    "trionce_dkdv": "trionce.triangle_dkdv",
    "trionce_dq_dbias": "trionce.triangle_dq_dbias",
    "trionce_trimul": "trionce.trimul_cuda",
}


class DoctorError(RuntimeError):
    """A preflight check failed; the message names the remedy."""


def _check_gpu(require_gpu: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if require_gpu:
            raise DoctorError("no CUDA device is visible; check CUDA_VISIBLE_DEVICES")
        return {"available": False}
    capability = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    if capability != (9, 0):
        raise DoctorError(
            f"Trionce kernels are compiled for sm_90a; {name} reports "
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


def _check_package() -> dict[str, Any]:
    try:
        package = importlib.import_module("trionce")
    except ImportError as error:
        raise DoctorError(
            "the 'trionce' package is not importable; install it with "
            "'uv pip install --no-build-isolation -e packages/trionce'"
        ) from error
    path = getattr(package, "__file__", None)
    if path is None:
        raise DoctorError(
            "'trionce' resolved to a namespace package, not an install; run "
            "the launcher from outside the source checkout or reinstall it"
        )
    return {"version": package.__version__, "path": path}


def _check_extensions() -> dict[str, Any]:
    from trionce import _aot_runtime

    manifest_path = os.path.join(
        os.path.dirname(_aot_runtime.__file__), "_build_manifest.json"
    )
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except OSError as error:
        raise DoctorError(f"cannot read the build manifest: {error}") from error
    if not manifest.get("built"):
        raise DoctorError(
            "the installed trionce carries a source-tree placeholder manifest; "
            "rebuild it with TRIONCE_BUILD_AOT=1 and TORCH_CUDA_ARCH_LIST=9.0a"
        )

    loaded: dict[str, Any] = {}
    for key, facade_name in EXTENSION_FACADES.items():
        try:
            facade = importlib.import_module(facade_name)
            module = facade.load_extension()
        except Exception as error:  # the loader validates hash, path, symbols
            raise DoctorError(f"AOT extension {key} failed to load: {error}") from error
        loaded[key] = getattr(module, "__file__", "<builtin>")
    return {
        "manifest": manifest_path,
        "git": manifest.get("git", {}),
        "abi": manifest.get("abi", {}),
        "extensions": loaded,
    }


def _check_provider(expected: str | None) -> dict[str, Any]:
    from openfold3.core.kernels.cuda import provider

    provider._reset_for_tests()
    active = provider.active_provider()
    if expected is not None and active != expected:
        raise DoctorError(
            f"expected the {expected} provider but {provider.PROVIDER_ENV} "
            f"resolves to {active}"
        )
    return provider.provider_identity()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-provider",
        choices=("trionce", "intree"),
        help="fail unless the resolved kernel provider is this one",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="report a missing GPU instead of failing (CPU-only smoke checks)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON"
    )
    arguments = parser.parse_args()

    report: dict[str, Any] = {}
    try:
        report["gpu"] = _check_gpu(require_gpu=not arguments.allow_no_gpu)
        report["package"] = _check_package()
        if report["gpu"].get("available"):
            report["aot"] = _check_extensions()
        report["provider"] = _check_provider(arguments.expect_provider)
    except DoctorError as error:
        print(f"trionce doctor: FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    gpu = report["gpu"]
    print("trionce doctor: PASS")
    if gpu.get("available"):
        print(f"  gpu       {gpu['name']} (sm_{gpu['capability'].replace('.', '')}a)")
        print(f"  torch     {gpu['torch']} / CUDA {gpu['torch_cuda']}")
    else:
        print("  gpu       none visible (--allow-no-gpu)")
    print(f"  package   trionce {report['package']['version']}")
    print(f"            {report['package']['path']}")
    if "aot" in report:
        commit = report["aot"]["git"].get("commit", "unknown")[:12]
        dirty = " (dirty)" if report["aot"]["git"].get("dirty") else ""
        print(f"  aot       {len(report['aot']['extensions'])} extensions @ {commit}{dirty}")
    print(f"  provider  {report['provider']['provider']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
