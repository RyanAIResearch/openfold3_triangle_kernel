#!/usr/bin/env python3
"""Audit and prepare a TurboPairFormer wheel for PyPI, preserving build identity.

Torch's shared libraries and CUDA runtime remain in their declared Python
dependencies. Everything else must satisfy auditwheel's selected manylinux
policy. --check-only repeats that audit without installing Torch or using CUDA.
It rejects a wheel that would still require binary changes or bundled libraries.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from validate_wheel import (
    EXTERNAL_PROVIDERS,
    PYPI_PLATFORMS,
    elf_needed_libraries,
    validate_wheel,
)

MANIFEST = "turbopairformer/_build_manifest.json"


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }


def _elf_members(members: dict[str, bytes]) -> dict[str, bytes]:
    return {name: data for name, data in members.items() if data[:4] == b"\x7fELF"}


def _providers(sonames: set[str]) -> dict[str, dict[str, str]]:
    """Resolve exclusions only from the installed, pinned Python dependencies."""
    torch = importlib.metadata.distribution("torch")
    if torch.version.split("+", 1)[0] != "2.10.0":
        raise ValueError("repair requires the pinned torch==2.10.0 distribution")
    requirements = [Requirement(text) for text in torch.requires or []]
    providers = {}
    for soname in sorted(sonames):
        distribution_name = EXTERNAL_PROVIDERS[soname]
        distribution = importlib.metadata.distribution(distribution_name)
        version = distribution.version.split("+", 1)[0]
        if distribution_name != "torch":
            matches = [
                requirement
                for requirement in requirements
                if requirement.name == distribution_name
                and (requirement.marker is None or requirement.marker.evaluate())
                and version in requirement.specifier
            ]
            if not matches or not version.startswith("12.8."):
                raise ValueError(
                    f"{soname} is not provided by pinned Torch dependencies"
                )
        candidates = [
            file
            for file in distribution.files or []
            if file.name == soname and distribution.locate_file(file).is_file()
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one {soname} owned by {distribution_name}")
        providers[soname] = {
            "distribution": distribution_name,
            "version": version,
            "path": candidates[0].as_posix(),
            "required_by": "torch==2.10.0",
        }
    return providers


def _audit(
    source: Path, destination: Path, target: str, exclusions: list[str]
) -> tuple[Path, str]:
    command = [
        sys.executable,
        "-m",
        "auditwheel",
        "repair",
        "--plat",
        target,
        "--only-plat",
        "--wheel-dir",
        str(destination),
    ]
    for soname in exclusions:
        command.extend(("--exclude", soname))
    command.append(str(source))
    result = subprocess.run(command, text=True, capture_output=True)
    log = result.stdout + result.stderr
    if result.returncode:
        raise ValueError(f"auditwheel rejected {target}:\n{log}")
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("auditwheel did not produce exactly one wheel")
    return wheels[0], log


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    """Recalculate RECORD after adding the final-binary and audit receipts."""
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("expected one wheel RECORD")
    record = records[0]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name, data in sorted(members.items()):
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
        writer.writerow((name, "sha256=" + digest.rstrip("="), len(data)))
    writer.writerow((record, "", ""))
    members[record] = buffer.getvalue().encode()
    # SOURCE_DATE_EPOCH is optional for a separately repaired artifact. Respect
    # ZIP's representable timestamp bounds and avoid a changing wall-clock date.
    import time

    epoch = max(315532800, int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, time.gmtime(epoch)[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def check_wheel(path: Path) -> dict:
    result = validate_wheel(path, require_pypi=True)
    members = _members(path)
    receipt = json.loads(members[MANIFEST])["repair"]
    with tempfile.TemporaryDirectory(prefix="turbopairformer-audit-") as temporary:
        repaired, log = _audit(
            path,
            Path(temporary),
            receipt["platform"],
            sorted(receipt["external_libraries"]),
        )
        if _elf_members(members) != _elf_members(_members(repaired)):
            raise ValueError("wheel still requires ELF repair or library bundling")
    result["auditwheel"] = {
        "version": importlib.metadata.version("auditwheel"),
        "platform": receipt["platform"],
        "external_libraries": receipt["external_libraries"],
        "log": log,
    }
    return result


def repair_wheel(path: Path, output_dir: Path, target: str) -> dict:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("manylinux repair requires Linux x86_64")
    if target not in PYPI_PLATFORMS:
        raise ValueError(f"unsupported release platform: {target}")
    original_result = validate_wheel(path)
    members = _members(path)
    manifest = json.loads(members[MANIFEST])
    if "repair" in manifest:
        raise ValueError("already repaired; use --check-only to verify the final wheel")
    elf_files = _elf_members(members)
    sonames = {
        soname
        for data in elf_files.values()
        for soname in elf_needed_libraries(data)
        if soname in EXTERNAL_PROVIDERS
    }
    external = _providers(sonames)
    with tempfile.TemporaryDirectory(prefix="turbopairformer-repair-") as temporary:
        repaired, log = _audit(path, Path(temporary), target, sorted(external))
        repaired_members = _members(repaired)
        final_elf = _elf_members(repaired_members)
        if set(final_elf) != set(elf_files):
            raise ValueError("unexpected bundled library; review its release contract")
        manifest["repair"] = {
            "schema_version": 1,
            "tool": "auditwheel",
            "tool_version": importlib.metadata.version("auditwheel"),
            "platform": target,
            "input_wheel": path.name,
            "input_sha256": original_result["sha256"],
            "input_extensions": copy.deepcopy(manifest["extensions"]),
            "external_libraries": external,
            "scope": "ELF policy audit; Torch/CUDA libraries provided by dependencies",
        }
        for record in manifest["extensions"].values():
            data = final_elf[record["filename"]]
            record["sha256"] = hashlib.sha256(data).hexdigest()
            record["size_bytes"] = len(data)
        repaired_members[MANIFEST] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / repaired.name
        if output.exists():
            raise ValueError(
                f"refusing to overwrite an existing release artifact: {output}"
            )
        finalized = Path(temporary) / "finalized" / repaired.name
        finalized.parent.mkdir()
        _write_wheel(finalized, repaired_members)
        result = check_wheel(finalized)
        shutil.copyfile(finalized, output)
    result["repair_log"] = log
    receipt_path = output.with_suffix(".whl.validation.json")
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".whl.sha256").write_text(f"{result['sha256']}  {output.name}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/pypi"))
    parser.add_argument("--plat", choices=PYPI_PLATFORMS, default=PYPI_PLATFORMS[0])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            check_wheel(args.wheel.resolve())
            if args.check_only
            else repair_wheel(
                args.wheel.resolve(), args.output_dir.resolve(), args.plat
            )
        )
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"TurboPairFormer manylinux validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
