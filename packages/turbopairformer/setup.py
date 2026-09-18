#!/usr/bin/env python3
"""Build the ABI-locked TurboPairFormer CUDA sm_90a extensions.

Build wheels with ``TURBOPAIRFORMER_BUILD_AOT=1`` in the pinned environment.
Source-only developer installs require ``TURBOPAIRFORMER_SOURCE_ONLY=1`` explicitly;
metadata generation remains available without importing Torch or CUDA.
The canonical wheel builder is ``tools/build_sm90_wheel.sh``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel

BUILD_AOT = os.environ.get("TURBOPAIRFORMER_BUILD_AOT") == "1"
SOURCE_ONLY = os.environ.get("TURBOPAIRFORMER_SOURCE_ONLY") == "1"
if BUILD_AOT:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
else:
    torch = None
    CUDA_HOME = None

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "turbopairformer"
EXPECTED_PYTHON = (3, 14)
EXPECTED_TORCH = "2.10.0"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_NVCC = "12.9"
EXPECTED_ARCH_LIST = "9.0a"
CXX_FLAGS = ["-O3", "-std=c++17", "-fvisibility=hidden"]
NVCC_FLAGS = [
    "-O3",
    "-lineinfo",
    "-std=c++17",
    "--frandom-seed=0",
    "-Xcompiler=-fvisibility=hidden",
]

EXTENSION_SPECS = {
    "stablefold_forward": {
        "module": "turbopairformer._ext._stablefold_forward_sm90a",
        "source": "triangle_forward.cu",
        "symbols": [
            "stablefold_triangle_forward_out",
            "stablefold_triangle_forward_shadow_out",
            "stablefold_pack_qkv",
        ],
    },
    "stablefold_delta": {
        "module": "turbopairformer._ext._stablefold_delta_sm90a",
        "source": "triangle_delta.cu",
        "symbols": [
            "stablefold_triangle_delta_out",
            "stablefold_triangle_delta_shadow_out",
        ],
    },
    "stablefold_dkdv": {
        "module": "turbopairformer._ext._stablefold_dkdv_sm90a",
        "source": "triangle_dkdv.cu",
        "symbols": [
            "stablefold_triangle_dkdv_out",
            "stablefold_triangle_dkdv_delta_true_out",
        ],
    },
    "stablefold_dq_dbias": {
        "module": "turbopairformer._ext._stablefold_dq_dbias_sm90a",
        "source": "triangle_dq_dbias.cu",
        "symbols": [
            "stablefold_triangle_dq_dbias_out",
            "stablefold_reduce_broadcast_dbias_out",
            "stablefold_triangle_sr_backward_out",
        ],
    },
    "stablefold_trimul": {
        "module": "turbopairformer._ext._stablefold_trimul_sm90a",
        "source": "trimul_cuda.cu",
        "symbols": [
            "trimul_in_fwd",
            "trimul_out_fwd",
            "trimul_gate_split_bwd",
            "sig_mul_bwd_out",
            "ln_bwd_dx",
            "ln_bwd_dgb",
            "transpose_ln_bwd",
            "trimul_wgrad_fixed_fold",
        ],
    },
}

PACKAGED_PYTHON = (
    "triangle_forward.py",
    "triangle_delta.py",
    "triangle_dkdv.py",
    "triangle_dq_dbias.py",
    "_triangle_attention.py",
    "_trimul.py",
    "trimul_cuda.py",
    "trimul_support.py",
    "_aot_runtime.py",
    "policy.py",
    "distributed.py",
    "__init__.py",
    "_ext/__init__.py",
    "PRODUCTION_MANIFEST.md",
)
PACKAGED_REPO_FILES = (
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "tools/build_sm90_wheel.sh",
    "tools/build_manylinux_wheel.sh",
    "tools/Dockerfile.manylinux",
    "tools/repair_wheel.py",
    "tools/validate_wheel.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(*command: str) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _assert_build_contract() -> None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("the first AOT wheel supports Linux x86_64 only")
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"sm_90a wheel requires Python {EXPECTED_PYTHON}, got "
            f"{sys.version_info[:2]}"
        )
    torch_base = torch.__version__.split("+", 1)[0]
    if torch_base != EXPECTED_TORCH:
        raise RuntimeError(
            f"sm_90a wheel requires Torch {EXPECTED_TORCH}, got {torch.__version__}"
        )
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"sm_90a wheel requires Torch CUDA {EXPECTED_TORCH_CUDA}, got "
            f"{torch.version.cuda}"
        )
    if os.environ.get("TORCH_CUDA_ARCH_LIST") != EXPECTED_ARCH_LIST:
        raise RuntimeError(
            "set TORCH_CUDA_ARCH_LIST=9.0a exactly; the WGMMA/TMA kernels "
            "require Hopper architecture-specific SASS and embed no PTX"
        )
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is unavailable; build in the CUDA 12.9 env")
    if shutil.which("patchelf") is None:
        raise RuntimeError(
            "AOT builds require patchelf to remove build-host runtime search paths. "
            "Install it in the build environment: python -m pip install patchelf"
        )
    nvcc_version = _command(str(Path(CUDA_HOME) / "bin/nvcc"), "--version")
    if f"release {EXPECTED_NVCC}" not in nvcc_version:
        raise RuntimeError(f"sm_90a wheel requires nvcc 12.9, got:\n{nvcc_version}")


def _source_paths() -> list[Path]:
    paths = [PACKAGE / value["source"] for value in EXTENSION_SPECS.values()]
    paths.extend(PACKAGE / name for name in PACKAGED_PYTHON)
    paths.extend(ROOT / name for name in PACKAGED_REPO_FILES)
    return paths


if BUILD_AOT:
    _BaseBuildExtension = BuildExtension.with_options(use_ninja=False)

    class AOTBuildExtension(_BaseBuildExtension):
        """Compile all extensions and embed the exact source/binary receipt."""

        def run(self) -> None:
            _assert_build_contract()
            super().run()
            self._write_build_manifest()

        def build_extension(self, extension) -> None:
            super().build_extension(extension)
            binary = Path(self.get_ext_fullpath(extension.name)).resolve()
            # CPython/Conda sysconfig can inject a build-host RPATH independently
            # of CC/CXX. Strip it before hashing and before inplace copies are
            # made; importing Torch supplies the matching runtime dependencies.
            _command("patchelf", "--remove-rpath", str(binary))
            if _command("patchelf", "--print-rpath", str(binary)):
                raise RuntimeError(f"failed to remove runtime search path: {binary}")

        def _write_build_manifest(self) -> None:
            build_lib = Path(self.build_lib).resolve()
            extension_records: dict[str, dict[str, object]] = {}
            module_to_key = {
                value["module"]: key for key, value in EXTENSION_SPECS.items()
            }
            for extension in self.extensions:
                key = module_to_key[extension.name]
                binary = Path(self.get_ext_fullpath(extension.name)).resolve()
                if not binary.is_file():
                    raise RuntimeError(f"built extension is missing: {binary}")
                try:
                    relative_binary = binary.relative_to(build_lib).as_posix()
                except ValueError:
                    relative_binary = "turbopairformer/_ext/" + binary.name
                extension_records[key] = {
                    "module": extension.name,
                    "filename": relative_binary,
                    "sha256": _sha256(binary),
                    "size_bytes": binary.stat().st_size,
                    "symbols": EXTENSION_SPECS[key]["symbols"],
                }

            source_records = {
                path.relative_to(ROOT).as_posix(): _sha256(path)
                for path in _source_paths()
            }
            git_commit = os.environ.get("TURBOPAIRFORMER_AOT_GIT_COMMIT")
            git_dirty_text = os.environ.get("TURBOPAIRFORMER_AOT_GIT_DIRTY")
            if git_commit is None or git_dirty_text is None:
                try:
                    git_commit = _command("git", "rev-parse", "HEAD")
                    git_dirty = bool(_command("git", "status", "--porcelain"))
                except OSError, subprocess.CalledProcessError:
                    git_commit = "unknown"
                    git_dirty = True
            else:
                git_dirty = git_dirty_text != "0"
            try:
                nvcc_version = _command(str(Path(CUDA_HOME) / "bin/nvcc"), "--version")
            except OSError, subprocess.CalledProcessError:
                nvcc_version = "unavailable"
            host_cxx = os.environ.get("CXX", "c++")
            try:
                host_cxx_version = _command(host_cxx, "--version")
            except OSError, subprocess.CalledProcessError:
                host_cxx_version = "unavailable"

            manifest = {
                "schema_version": 1,
                "built": True,
                "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
                "git": {"commit": git_commit, "dirty": git_dirty},
                "target": {
                    "cuda_arch": "sm_90a",
                    "torch_cuda_arch_list": EXPECTED_ARCH_LIST,
                    "ptx_embedded": False,
                },
                "abi": {
                    "python_version": platform.python_version(),
                    "python_soabi": sysconfig.get_config_var("SOABI"),
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                },
                "toolchain": {
                    "patchelf_version": _command("patchelf", "--version"),
                    "cuda_home": str(CUDA_HOME),
                    "nvcc_version": nvcc_version,
                    "host_cxx": host_cxx,
                    "host_cxx_version": host_cxx_version,
                    "python_compiler": platform.python_compiler(),
                    "glibc": list(platform.libc_ver()),
                    "cxx_flags": CXX_FLAGS,
                    "nvcc_flags": NVCC_FLAGS,
                },
                "sources": source_records,
                "extensions": extension_records,
            }
            payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            destinations = [build_lib / "turbopairformer/_build_manifest.json"]
            # An editable install builds the extensions in place, so the import
            # that matters resolves to the source tree.  Leaving only the
            # placeholder manifest there would have the loader reject its own
            # freshly built binaries as an unattested source checkout.
            if getattr(self, "inplace", 0):
                destinations.append(PACKAGE / "_build_manifest.local.json")
            for destination in destinations:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(payload)

    extensions = [
        CUDAExtension(
            name=value["module"],
            # Keep the compiler input relative to the checkout so CUDA
            # line-info does not capture a random mktemp source prefix.
            sources=[f"turbopairformer/{value['source']}"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        )
        for value in EXTENSION_SPECS.values()
    ]
    command_classes = {"build_ext": AOTBuildExtension}
else:
    extensions = []
    command_classes = {}


def _require_build_mode() -> None:
    if not BUILD_AOT and not SOURCE_ONLY:
        raise RuntimeError(
            "TurboPairFormer requires compiled CUDA extensions. Build with "
            "TURBOPAIRFORMER_BUILD_AOT=1 using tools/build_sm90_wheel.sh; for an "
            "explicit source-only developer install set TURBOPAIRFORMER_SOURCE_ONLY=1. "
            "A source-only install is not a usable precompiled release."
        )


class CheckedBuildPy(build_py):
    def run(self) -> None:
        _require_build_mode()
        super().run()


class CheckedEditableWheel(editable_wheel):
    def run(self) -> None:
        _require_build_mode()
        super().run()


command_classes.update(
    {"build_py": CheckedBuildPy, "editable_wheel": CheckedEditableWheel}
)

setup(
    ext_modules=extensions,
    cmdclass=command_classes,
    zip_safe=False,
)
