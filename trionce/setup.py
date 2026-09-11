#!/usr/bin/env python3
"""Build the ABI-locked Trionce CUDA sm_90a extensions.

Plain ``pip install .`` / ``uv pip install .`` without ``TRIONCE_BUILD_AOT=1``
produces a source-only package whose loader refuses to run (fail closed) unless
``TRIONCE_ALLOW_JIT=1`` is set for source-tree development.  The production
wheel is built by ``tools/build_sm90_wheel.sh`` inside the pinned
Python 3.14 / Torch 2.10 / CUDA 12.9 environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup

BUILD_AOT = os.environ.get("TRIONCE_BUILD_AOT") == "1"
if BUILD_AOT:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
else:
    torch = None
    CUDA_HOME = None

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "trionce"
EXPECTED_PYTHON = (3, 14)
EXPECTED_TORCH = "2.10.0"
EXPECTED_TORCH_CUDA = "12.9"
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
    "trionce_forward": {
        "module": "trionce._ext._trionce_forward_sm90a",
        "source": "triangle_forward.cu",
        "symbols": [
            "stablefold_triangle_forward_out",
            "stablefold_triangle_forward_shadow_out",
            "stablefold_pack_qkv",
        ],
    },
    "trionce_delta": {
        "module": "trionce._ext._trionce_delta_sm90a",
        "source": "triangle_delta.cu",
        "symbols": [
            "stablefold_triangle_delta_out",
            "stablefold_triangle_delta_shadow_out",
        ],
    },
    "trionce_dkdv": {
        "module": "trionce._ext._trionce_dkdv_sm90a",
        "source": "triangle_dkdv.cu",
        "symbols": [
            "stablefold_triangle_dkdv_out",
            "stablefold_triangle_dkdv_delta_true_out",
        ],
    },
    "trionce_dq_dbias": {
        "module": "trionce._ext._trionce_dq_dbias_sm90a",
        "source": "triangle_dq_dbias.cu",
        "symbols": [
            "stablefold_triangle_dq_dbias_out",
            "stablefold_reduce_broadcast_dbias_out",
            "stablefold_triangle_sr_backward_out",
        ],
    },
    "trionce_trimul": {
        "module": "trionce._ext._trionce_trimul_sm90a",
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
    "triangle_attention.py",
    "trimul.py",
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
    nvcc_version = _command(str(Path(CUDA_HOME) / "bin/nvcc"), "--version")
    if "release 12.9" not in nvcc_version:
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
                    relative_binary = "trionce/_ext/" + binary.name
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
            git_commit = os.environ.get("TRIONCE_AOT_GIT_COMMIT")
            git_dirty_text = os.environ.get("TRIONCE_AOT_GIT_DIRTY")
            if git_commit is None or git_dirty_text is None:
                try:
                    git_commit = _command("git", "rev-parse", "HEAD")
                    git_dirty = bool(_command("git", "status", "--porcelain"))
                except (OSError, subprocess.CalledProcessError):
                    git_commit = "unknown"
                    git_dirty = True
            else:
                git_dirty = git_dirty_text != "0"
            try:
                nvcc_version = _command(str(Path(CUDA_HOME) / "bin/nvcc"), "--version")
            except (OSError, subprocess.CalledProcessError):
                nvcc_version = "unavailable"
            host_cxx = os.environ.get("CXX", "c++")
            try:
                host_cxx_version = _command(host_cxx, "--version")
            except (OSError, subprocess.CalledProcessError):
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
            destinations = [build_lib / "trionce/_build_manifest.json"]
            # An editable install builds the extensions in place, so the import
            # that matters resolves to the source tree.  Leaving only the
            # placeholder manifest there would have the loader reject its own
            # freshly built binaries as an unattested source checkout.
            if getattr(self, "inplace", 0):
                destinations.append(PACKAGE / "_build_manifest.json")
            for destination in destinations:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(payload)

    extensions = [
        CUDAExtension(
            name=value["module"],
            # Keep the compiler input relative to the checkout so CUDA
            # line-info does not capture a random mktemp source prefix.
            sources=[f"trionce/{value['source']}"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        )
        for value in EXTENSION_SPECS.values()
    ]
    command_classes = {"build_ext": AOTBuildExtension}
else:
    extensions = []
    command_classes = {}

setup(
    ext_modules=extensions,
    cmdclass=command_classes,
    zip_safe=False,
)
