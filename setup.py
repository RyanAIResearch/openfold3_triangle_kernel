#!/usr/bin/env python3
"""Build the ABI-locked StableFold CUDA sm_90a wheel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CUDA_PACKAGE = ROOT / "openfold3/core/kernels/cuda"
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
    "stablefold_forward": {
        "module": (
            "openfold3.core.kernels.cuda._ext._stablefold_forward_sm90a"
        ),
        "source": "stablefold_triangle_forward.cu",
        "symbols": [
            "stablefold_triangle_forward_out",
            "stablefold_triangle_forward_shadow_out",
            "stablefold_pack_qkv",
        ],
    },
    "stablefold_delta": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_delta_sm90a",
        "source": "stablefold_triangle_delta.cu",
        "symbols": [
            "stablefold_triangle_delta_out",
            "stablefold_triangle_delta_shadow_out",
        ],
    },
    "stablefold_dkdv": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_dkdv_sm90a",
        "source": "stablefold_triangle_dkdv.cu",
        "symbols": [
            "stablefold_triangle_dkdv_out",
            "stablefold_triangle_dkdv_delta_true_out",
        ],
    },
    "stablefold_dq_dbias": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_dq_dbias_sm90a",
        "source": "stablefold_triangle_dq_dbias.cu",
        "symbols": [
            "stablefold_triangle_dq_dbias_out",
            "stablefold_reduce_broadcast_dbias_out",
            "stablefold_triangle_sr_backward_out",
        ],
    },
    "stablefold_trimul": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_trimul_sm90a",
        "source": "stablefold_trimul_cuda.cu",
        "symbols": [
            "trimul_in_fwd",
            "trimul_out_fwd",
            "trimul_gate_split_bwd",
            "sig_mul_bwd_out",
            "ln_bwd_dx",
            "ln_bwd_dgb",
            "transpose_ln_bwd",
            "pack_rows3",
            "gather_rows1",
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(*command: str) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
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
        raise RuntimeError("CUDA_HOME is unavailable; build in the CUDA 12.9 image")
    nvcc_version = _command(str(Path(CUDA_HOME) / "bin/nvcc"), "--version")
    if "release 12.9" not in nvcc_version:
        raise RuntimeError(
            f"sm_90a wheel requires nvcc 12.9, got:\n{nvcc_version}"
        )


def _source_paths() -> list[Path]:
    paths = [CUDA_PACKAGE / value["source"] for value in EXTENSION_SPECS.values()]
    paths.extend(
        CUDA_PACKAGE / name
        for name in (
            "stablefold_triangle_forward.py",
            "stablefold_triangle_delta.py",
            "stablefold_triangle_dkdv.py",
            "stablefold_triangle_dq_dbias.py",
            "stablefold_triangle_attention.py",
            "stablefold_trimul.py",
            "stablefold_trimul_cuda.py",
            "_aot_runtime.py",
        )
    )
    paths.extend(
        ROOT / name
        for name in (
            "setup.py",
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "openfold3/core/kernels/cuda/PRODUCTION_MANIFEST.md",
            "openfold3/core/kernels/cuda/_ext/__init__.py",
            "tools/build_sm90_wheel.sh",
            "tools/attest_sm90_wheel.py",
            "tools/aot_clean_install_smoke.py",
        )
    )
    return paths


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
                relative_binary = (
                    "openfold3/core/kernels/cuda/_ext/" + binary.name
                )
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
        git_commit = os.environ.get("OPENFOLD3_AOT_GIT_COMMIT")
        git_dirty_text = os.environ.get("OPENFOLD3_AOT_GIT_DIRTY")
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
        destination = (
            build_lib / "openfold3/core/kernels/cuda/_build_manifest.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


extensions = [
    CUDAExtension(
        name=value["module"],
        # Keep the compiler input relative to the fresh checkout so CUDA
        # line-info does not capture a random mktemp source prefix.
        sources=[f"openfold3/core/kernels/cuda/{value['source']}"],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )
    for value in EXTENSION_SPECS.values()
]

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": AOTBuildExtension},
    zip_safe=False,
)
