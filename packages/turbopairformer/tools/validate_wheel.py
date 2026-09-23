#!/usr/bin/env python3
"""Validate a TurboPairFormer AOT wheel without importing Torch or running CUDA.

Use --require-pypi after a real manylinux build/audit. This checker deliberately
never changes platform tags, binaries, or receipts to make a wheel pass.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import struct
import zipfile
from email.parser import BytesParser
from pathlib import Path

MODULES = {
    "turbopairformer_forward": "turbopairformer._ext._turbopairformer_forward_sm90a",
    "turbopairformer_delta": "turbopairformer._ext._turbopairformer_delta_sm90a",
    "turbopairformer_dkdv": "turbopairformer._ext._turbopairformer_dkdv_sm90a",
    "turbopairformer_dq_dbias": "turbopairformer._ext._turbopairformer_dq_dbias_sm90a",
    "turbopairformer_trimul": "turbopairformer._ext._turbopairformer_trimul_sm90a",
}
PYPI_PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_34_x86_64")
NATIVE_SOURCE_SUFFIXES = {
    ".cu",
    ".cuh",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".ipp",
    ".tpp",
    ".ptx",
}
# These libraries are supplied by the pinned PyPI Torch distribution and its
# CUDA runtime dependency. Do not extend this list with system/build-host libs.
EXTERNAL_PROVIDERS = {
    "libc10.so": "torch",
    "libc10_cuda.so": "torch",
    "libtorch.so": "torch",
    "libtorch_cpu.so": "torch",
    "libtorch_cuda.so": "torch",
    "libtorch_python.so": "torch",
    "libcudart.so.12": "nvidia-cuda-runtime-cu12",
}


def _elf_dynamic_strings(binary: bytes, selected_tags: tuple[int, ...]) -> list[str]:
    """Read dynamic strings from the supported ELF64 little-endian x86_64 format.

    Bound every file/virtual-address lookup. Malformed or unsupported ELF files
    fail closed rather than being reported as having no runtime search paths.
    """

    def unpack(layout: str, offset: int) -> tuple:
        size = struct.calcsize(layout)
        if offset < 0 or offset + size > len(binary):
            raise ValueError("truncated ELF structure")
        return struct.unpack_from(layout, binary, offset)

    header = unpack("<16sHHIQQQIHHHHHH", 0)
    ident, elf_type, machine, version = header[:4]
    if (
        ident[:7] != b"\x7fELF\x02\x01\x01"
        or elf_type != 3
        or machine != 62
        or version != 1
        or header[8] != 64
    ):
        raise ValueError("expected an ELF64 little-endian x86_64 shared object")
    ph_offset, ph_size, ph_count = header[5], header[9], header[10]
    if ph_size != 56 or ph_count == 0 or ph_count == 0xFFFF:
        raise ValueError("unsupported ELF program header table")
    if ph_offset + ph_size * ph_count > len(binary):
        raise ValueError("truncated ELF program header table")
    loads = []
    dynamic = []
    for index in range(ph_count):
        kind, _, offset, address, _, size, _, _ = unpack(
            "<IIQQQQQQ", ph_offset + index * ph_size
        )
        if offset + size > len(binary):
            raise ValueError("ELF segment extends past the file")
        if kind == 1:  # PT_LOAD
            loads.append((address, offset, size))
        elif kind == 2:  # PT_DYNAMIC
            dynamic.append((offset, size))
    if len(dynamic) != 1:
        raise ValueError("expected one ELF dynamic segment")
    dynamic_offset, dynamic_size = dynamic[0]
    if dynamic_size % 16:
        raise ValueError("malformed ELF dynamic segment size")
    string_address = None
    string_size = None
    path_offsets = []
    terminated = False
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
        tag, value = unpack("<qQ", offset)
        if tag == 0:  # DT_NULL
            terminated = True
            break
        if tag == 5:  # DT_STRTAB (virtual address)
            string_address = value
        elif tag == 10:  # DT_STRSZ
            string_size = value
        elif tag in selected_tags:  # Dynamic string-table offset.
            path_offsets.append(value)
    if not terminated:
        raise ValueError("unterminated ELF dynamic segment")
    if string_address is None or string_size is None:
        raise ValueError("ELF dynamic string table is missing")
    tables = [
        file_offset + string_address - address
        for address, file_offset, size in loads
        if address <= string_address and string_address + string_size <= address + size
    ]
    if len(tables) != 1:
        raise ValueError("ELF dynamic string table is outside one load segment")
    table_offset = tables[0]
    table_end = table_offset + string_size
    paths = []
    for offset in path_offsets:
        if offset >= string_size:
            raise ValueError("ELF runtime path is outside the dynamic string table")
        start = table_offset + offset
        end = binary.find(b"\x00", start, table_end)
        if end == -1:
            raise ValueError("unterminated ELF runtime path")
        try:
            paths.append(binary[start:end].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("ELF runtime path is not valid UTF-8") from exc
    return paths


def elf_runtime_paths(binary: bytes) -> list[str]:
    """Read DT_RPATH and DT_RUNPATH entries."""
    return [
        path
        for entry in _elf_dynamic_strings(binary, (15, 29))
        for path in entry.split(":")
    ]


def elf_needed_libraries(binary: bytes) -> list[str]:
    """Read direct DT_NEEDED dependencies without loading untrusted ELF code."""
    return _elf_dynamic_strings(binary, (1,))


def _validate_record(archive: zipfile.ZipFile, metadata_path: str) -> None:
    record_path = metadata_path.rsplit("/", 1)[0] + "/RECORD"
    rows = list(csv.reader(io.StringIO(archive.read(record_path).decode())))
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)) or set(paths) != {
        name for name in archive.namelist() if not name.endswith("/")
    }:
        raise ValueError("wheel RECORD coverage mismatch")
    for name, digest, size in rows:
        if name == record_path:
            if digest or size:
                raise ValueError("RECORD must not hash itself")
            continue
        content = archive.read(name)
        expected = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
        if digest != "sha256=" + expected.decode().rstrip("=") or size != str(
            len(content)
        ):
            raise ValueError(f"wheel RECORD mismatch: {name}")


def validate_wheel(
    path: Path,
    *,
    require_pypi: bool = False,
    allow_dirty: bool = False,
    expected_commit: str | None = None,
) -> dict:
    """Raise ValueError for incomplete, stale, or unsupported release artifacts."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate wheel archive members")
        native_sources = [
            name
            for name in names
            if any(
                suffix.lower() in NATIVE_SOURCE_SUFFIXES
                for suffix in Path(name).suffixes
            )
        ]
        require(
            not native_sources,
            "CUDA/C++ source or PTX must not be packaged: "
            + ", ".join(sorted(native_sources)),
        )
        require(
            not any(name.endswith(".local.json") for name in names),
            "editable build receipt leaked into the wheel",
        )
        require(
            all(
                name.startswith(("turbopairformer/", "turbopairformer.libs/"))
                or (
                    name.split("/", 1)[0].startswith("turbopairformer-")
                    and name.split("/", 1)[0].endswith(".dist-info")
                )
                for name in names
            ),
            "wheel contains a host or renamed package",
        )
        metadata_paths = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_paths) == 1, "expected one METADATA file")
        metadata_path = metadata_paths[0]
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        require(metadata["Name"] == "turbopairformer", "incorrect distribution name")
        require(metadata["Requires-Python"] == "==3.14.*", "incorrect Python contract")
        dependencies = metadata.get_all("Requires-Dist", [])
        require("torch==2.10.0" in dependencies, "Torch dependency must be ==2.10.0")
        wheel_metadata = BytesParser().parsebytes(
            archive.read(metadata_path.rsplit("/", 1)[0] + "/WHEEL")
        )
        require(
            wheel_metadata["Root-Is-Purelib"] == "false", "AOT wheel must be binary"
        )
        tags = wheel_metadata.get_all("Tag", [])
        require(bool(tags), "missing wheel tags")
        for tag in tags:
            require(
                bool(
                    re.fullmatch(
                        r"cp314-cp314-(linux_x86_64|manylinux[^-]*_x86_64)", tag
                    )
                ),
                f"unsupported wheel tag: {tag}",
            )
        if require_pypi:
            require(
                all("-manylinux" in tag for tag in tags),
                "PyPI release requires audited manylinux tags, not linux_x86_64",
            )
        filename_parts = path.name.removesuffix(".whl").split("-")
        require(len(filename_parts) in (5, 6), "invalid wheel filename")
        require(
            filename_parts[0] == "turbopairformer"
            and filename_parts[1] == metadata["Version"],
            "wheel filename and package metadata disagree",
        )
        filename_tags = {
            f"{python}-{abi}-{plat}"
            for python in filename_parts[-3].split(".")
            for abi in filename_parts[-2].split(".")
            for plat in filename_parts[-1].split(".")
        }
        require(
            set(tags) == filename_tags,
            "wheel filename tags disagree with WHEEL metadata",
        )
        manifest = json.loads(archive.read("turbopairformer/_build_manifest.json"))
        require(
            manifest.get("schema_version") == 1 and manifest.get("built") is True,
            "wheel does not contain a completed AOT build receipt",
        )
        git = manifest["git"]
        require(
            bool(re.fullmatch(r"[0-9a-f]{40}", git["commit"])), "missing source commit"
        )
        require(allow_dirty or git["dirty"] is False, "dirty build cannot be published")
        if expected_commit:
            require(
                git["commit"] == expected_commit, "wheel was built from another commit"
            )
        abi = manifest["abi"]
        require(abi["python_version"].startswith("3.14."), "unsupported build Python")
        require(
            abi["torch_version"].split("+", 1)[0] == "2.10.0", "unsupported build Torch"
        )
        require(abi["torch_cuda_version"] == "12.8", "unsupported Torch CUDA runtime")
        require(
            manifest["target"]
            == {
                "cuda_arch": "sm_90a",
                "torch_cuda_arch_list": "9.0a",
                "ptx_embedded": False,
            },
            "unsupported CUDA target",
        )
        extensions = manifest["extensions"]
        require(set(extensions) == set(MODULES), "expected five AOT extensions")
        recorded_binaries = set()
        needed_libraries = set()
        for key, module in MODULES.items():
            record = extensions[key]
            require(record["module"] == module, f"incorrect module identity: {key}")
            filename = record["filename"]
            require(
                filename.startswith("turbopairformer/_ext/")
                and filename.endswith(".so"),
                f"incorrect extension path: {filename}",
            )
            binary = archive.read(filename)
            require(binary.startswith(b"\x7fELF"), f"extension is not ELF: {filename}")
            require(
                len(binary) == record["size_bytes"], f"binary size mismatch: {filename}"
            )
            require(
                hashlib.sha256(binary).hexdigest() == record["sha256"],
                f"binary hash mismatch: {filename}",
            )
            paths = elf_runtime_paths(binary)
            needed_libraries.update(elf_needed_libraries(binary))
            require(
                all(not entry.startswith("/") for entry in paths),
                f"absolute ELF runtime search path in {filename}: {paths}",
            )
            recorded_binaries.add(filename)
        require(
            recorded_binaries
            == {
                name
                for name in names
                if name.startswith("turbopairformer/_ext/") and name.endswith(".so")
            },
            "unrecorded extension binary",
        )
        if require_pypi:
            repair = manifest.get("repair", {})
            require(
                repair.get("tool") == "auditwheel"
                and bool(repair.get("tool_version"))
                and repair.get("platform") in PYPI_PLATFORMS
                and repair.get("schema_version") == 1,
                "PyPI wheel requires an auditwheel repair receipt",
            )
            require(
                tags == [f"cp314-cp314-{repair['platform']}"],
                "release tags disagree with the audited platform",
            )
            require(
                bool(re.fullmatch(r"[0-9a-f]{64}", repair.get("input_sha256", ""))),
                "missing original wheel hash",
            )
            originals = repair.get("input_extensions", {})
            require(
                set(originals) == set(MODULES), "missing pre-repair binary receipts"
            )
            for key, record in originals.items():
                require(
                    record.get("module") == MODULES[key]
                    and record.get("filename") == extensions[key]["filename"]
                    and bool(re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")))
                    and isinstance(record.get("size_bytes"), int)
                    and record["size_bytes"] > 0,
                    f"invalid original binary receipt: {key}",
                )
            external = repair.get("external_libraries", {})
            require(
                set(external) == needed_libraries.intersection(EXTERNAL_PROVIDERS),
                "external runtime dependency receipt mismatch",
            )
            for soname, provider in external.items():
                distribution = EXTERNAL_PROVIDERS[soname]
                require(
                    provider.get("distribution") == distribution
                    and provider.get("required_by") == "torch==2.10.0"
                    and bool(provider.get("path"))
                    and not provider["path"].startswith("/")
                    and ".." not in Path(provider["path"]).parts,
                    f"invalid external runtime provider: {soname}",
                )
                version = provider.get("version", "")
                require(
                    version == "2.10.0"
                    if distribution == "torch"
                    else version.startswith("12.8."),
                    f"unsupported external runtime version: {soname}",
                )
            require(
                recorded_binaries
                == {
                    name
                    for name in names
                    if not name.endswith("/")
                    and archive.read(name, pwd=None)[:4] == b"\x7fELF"
                },
                "unexpected bundled ELF library; review its release contract",
            )
            _validate_record(archive, metadata_path)
        packaged_sources = {
            name
            for name in names
            if name.startswith("turbopairformer/") and name.endswith((".py", ".md"))
        }
        recorded_hashes = manifest.get("packaged_sources")
        require(
            isinstance(recorded_hashes, dict),
            "missing packaged source receipt",
        )
        recorded_sources = set(recorded_hashes)
        require(
            recorded_sources == packaged_sources,
            "source receipt coverage mismatch: "
            f"missing={sorted(packaged_sources - recorded_sources)}, "
            f"extra={sorted(recorded_sources - packaged_sources)}",
        )
        sources = manifest.get("sources")
        require(isinstance(sources, dict), "missing build source provenance")
        for filename, expected_hash in recorded_hashes.items():
            require(
                sources.get(filename) == expected_hash,
                f"packaged source provenance mismatch: {filename}",
            )
            require(
                hashlib.sha256(archive.read(filename)).hexdigest() == expected_hash,
                f"packaged source hash mismatch: {filename}",
            )
    return {
        "wheel": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "version": metadata["Version"],
        "tags": tags,
        "git": git,
        "extensions": sorted(extensions),
        "scope": (
            "archive and ABI metadata; GPU correctness and ELF portability "
            "need separate tests"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--require-pypi", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="development builds only"
    )
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    try:
        result = validate_wheel(
            args.wheel,
            require_pypi=args.require_pypi,
            allow_dirty=args.allow_dirty,
            expected_commit=args.expected_commit,
        )
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"TurboPairFormer wheel validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
