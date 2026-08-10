#!/usr/bin/env python3
"""Verify and hash the ABI-locked sm_90 wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_MEMBER = "openfold3/core/kernels/cuda/_build_manifest.json"
EXPECTED_EXTENSIONS = {
    "stablefold_forward": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_forward_sm90",
        "symbols": ["stablefold_triangle_forward_out"],
    },
    "stablefold_delta": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_delta_sm90",
        "symbols": ["stablefold_triangle_delta_out"],
    },
    "stablefold_dkdv": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_dkdv_sm90",
        "symbols": ["stablefold_triangle_dkdv_out"],
    },
    "stablefold_dq_dbias": {
        "module": "openfold3.core.kernels.cuda._ext._stablefold_dq_dbias_sm90",
        "symbols": [
            "stablefold_triangle_dq_dbias_out",
            "stablefold_reduce_broadcast_dbias_out",
        ],
    },
}
LOADER_MEMBERS = (
    "openfold3/core/kernels/cuda/stablefold_triangle_forward.py",
    "openfold3/core/kernels/cuda/stablefold_triangle_delta.py",
    "openfold3/core/kernels/cuda/stablefold_triangle_dkdv.py",
    "openfold3/core/kernels/cuda/stablefold_triangle_dq_dbias.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*values: str) -> str:
    return subprocess.run(
        values, check=True, capture_output=True, text=True
    ).stdout


def verify_binary_tools(
    archive: zipfile.ZipFile, extensions: dict[str, dict[str, object]]
) -> None:
    for tool in ("nm", "cuobjdump"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required binary inspection tool is missing: {tool}")
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        initializers: set[str] = set()
        for record in extensions.values():
            member = str(record["filename"])
            binary = destination / Path(member).name
            binary.write_bytes(archive.read(member))
            basename = str(record["module"]).rsplit(".", 1)[-1]
            initializer = f"PyInit_{basename}"
            symbols = command("nm", "-D", "--defined-only", str(binary))
            if symbols.count(initializer) != 1:
                raise RuntimeError(f"unexpected initializer symbols in {member}")
            if initializer in initializers:
                raise RuntimeError(f"duplicate initializer: {initializer}")
            initializers.add(initializer)
            elf_listing = command("cuobjdump", "--list-elf", str(binary))
            if "sm_90" not in elf_listing:
                raise RuntimeError(f"{member} has no sm_90 cubin")
            ptx = command("cuobjdump", "--dump-ptx", str(binary))
            if ".version" in ptx:
                raise RuntimeError(f"{member} unexpectedly embeds PTX")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--require-binary-tools", action="store_true")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    if not re.search(r"-cp314-cp314-linux_x86_64\.whl$", wheel.name):
        raise RuntimeError("wheel filename is not the dedicated CPython 3.14 ABI tag")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        wheel_metadata_members = [
            name for name in names if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata_members) != 1:
            raise RuntimeError("wheel must contain exactly one WHEEL metadata file")
        wheel_metadata = archive.read(wheel_metadata_members[0]).decode()
        if "Root-Is-Purelib: false" not in wheel_metadata:
            raise RuntimeError("CUDA extension wheel is incorrectly marked pure-Python")
        if "Tag: cp314-cp314-linux_x86_64" not in wheel_metadata:
            raise RuntimeError("wheel metadata is not CPython 3.14 linux_x86_64")
        if MANIFEST_MEMBER not in names:
            raise RuntimeError("wheel has no embedded AOT manifest")
        manifest_bytes = archive.read(MANIFEST_MEMBER)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != 1 or not manifest.get("built"):
            raise RuntimeError("wheel contains an unbuilt manifest")
        if manifest.get("target") != {
            "cuda_arch": "sm_90",
            "ptx_embedded": False,
            "torch_cuda_arch_list": "9.0",
        }:
            raise RuntimeError("wheel target contract is not sm_90 SASS-only")
        abi = manifest.get("abi", {})
        if not str(abi.get("python_soabi", "")).startswith("cpython-314"):
            raise RuntimeError("wheel was not built for CPython 3.14")
        if str(abi.get("torch_version", "")).split("+", 1)[0] != "2.10.0":
            raise RuntimeError("wheel was not built against Torch 2.10.0")
        if abi.get("torch_cuda_version") != "12.9":
            raise RuntimeError("wheel was not built against Torch CUDA 12.9")

        git = manifest.get("git", {})
        if git.get("dirty") is not False or not re.fullmatch(
            r"[0-9a-f]{40}", str(git.get("commit", ""))
        ):
            raise RuntimeError("wheel is not attested to one clean Git commit")

        extensions = manifest.get("extensions", {})
        if set(extensions) != set(EXPECTED_EXTENSIONS):
            raise RuntimeError("wheel does not contain exactly four extensions")
        binary_members: set[str] = set()
        for key, record in extensions.items():
            member = str(record["filename"])
            expected = EXPECTED_EXTENSIONS[key]
            if record.get("module") != expected["module"]:
                raise RuntimeError(f"unexpected extension module name: {key}")
            if record.get("symbols") != expected["symbols"]:
                raise RuntimeError(f"unexpected extension symbol contract: {key}")
            module_basename = str(record["module"]).rsplit(".", 1)[-1]
            expected_prefix = (
                "openfold3/core/kernels/cuda/_ext/" + module_basename + "."
            )
            if not member.startswith(expected_prefix) or not member.endswith(".so"):
                raise RuntimeError(
                    f"extension is outside its private package: {member}"
                )
            if member in binary_members:
                raise RuntimeError(f"duplicate extension binary member: {member}")
            binary_members.add(member)
            if member not in names:
                raise RuntimeError(f"missing extension binary: {member}")
            if sha256_bytes(archive.read(member)) != record["sha256"]:
                raise RuntimeError(f"extension hash mismatch: {key}")
        packaged_shared_objects = {
            name
            for name in names
            if name.startswith("openfold3/core/kernels/cuda/_ext/")
            and name.endswith(".so")
        }
        if packaged_shared_objects != binary_members:
            raise RuntimeError("wheel has unrecorded or missing private shared objects")
        for relative, expected in manifest.get("sources", {}).items():
            source = ROOT / relative
            if not source.is_file() or sha256_file(source) != expected:
                raise RuntimeError(f"source receipt mismatch: {relative}")
        for member in LOADER_MEMBERS:
            text = archive.read(member).decode()
            if "torch.utils.cpp_extension" in text:
                raise RuntimeError(f"production loader contains JIT code: {member}")
        if args.require_binary_tools:
            verify_binary_tools(archive, extensions)

    wheel_hash = sha256_file(wheel)
    receipt = {
        "schema_version": 1,
        "wheel": wheel.name,
        "wheel_sha256": wheel_hash,
        "embedded_manifest_sha256": sha256_bytes(manifest_bytes),
        "git": manifest["git"],
        "abi": manifest["abi"],
        "target": manifest["target"],
        "extension_sha256": {
            key: value["sha256"] for key, value in manifest["extensions"].items()
        },
    }
    wheel.with_suffix(wheel.suffix + ".sha256").write_text(
        f"{wheel_hash}  {wheel.name}\n"
    )
    wheel.with_suffix(wheel.suffix + ".attestation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
