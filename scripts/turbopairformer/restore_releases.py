#!/usr/bin/env python3
"""Verify the three original PyPI wheels and their build-1 recovery copies.

This one-time release path preserves historical package versions and every
payload byte. Only WHEEL (Build: 1) and its RECORD entry change. It does not
certify a new kernel build or restore original PyPI upload dates or file hashes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

RECOVERY_TAG = "turbopairformer-restore-20260922"
WHEEL_TAG = "cp314-cp314-manylinux_2_28_x86_64"
# Original digests were recorded from the actual PyPI downloads before removal.
# Recovery digests pin the already reviewed build-1 archives, not arbitrary
# replacements that merely claim an earlier version number.
RELEASES = {
    "0.1.0": (
        "54a4ab1879a0029770c1f72bb7a718bc720431393f74b89fe3ebe2e4e5b57a8a",
        "e677cd4ea1a8ee245c288717b8e09e23236d0fba5652a2ecab2614f90a5f52c1",
    ),
    "0.1.1": (
        "6d44610c0ab8332c84b3f78707168bf4f1661b331541e1463d80bbbe7f3b1431",
        "57fc8e78c5c158dd75e21a35d42c3fd344d34fd9c896e60fc65c1f86e0e1e0c8",
    ),
    "0.1.1.post1": (
        "995db45c8b8e9620bf310676f7768a4cb1f43584b21683c7fb21d1ee9d0bc4e1",
        "a8348d967a0938553acc8a47157f5ffc38fa9ea4eee170620b1c477e0e8e78ec",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filename(version: str, *, restored: bool) -> str:
    build = "-1" if restored else ""
    return f"turbopairformer-{version}{build}-{WHEEL_TAG}.whl"


def archive_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate archive member")
        for info in archive.infolist():
            name = info.filename
            parsed = PurePosixPath(name)
            require(
                not info.is_dir()
                and not parsed.is_absolute()
                and ".." not in parsed.parts
                and "\\" not in name
                and str(parsed) == name,
                f"unsafe archive path: {name}",
            )
            require(
                not stat.S_ISLNK(info.external_attr >> 16),
                f"archive symlink: {name}",
            )
        return {name: archive.read(name) for name in names}


def validate_record(files: dict[str, bytes], record_name: str) -> None:
    require(record_name in files, "missing RECORD")
    rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"))))
    require(all(len(row) == 3 for row in rows), "invalid RECORD row")
    names = [row[0] for row in rows]
    require(len(names) == len(set(names)), "duplicate RECORD entry")
    require(set(names) == set(files), "RECORD coverage mismatch")
    for name, encoded_hash, size in rows:
        if name == record_name:
            require(encoded_hash == size == "", "RECORD must not hash itself")
            continue
        payload = files[name]
        expected_hash = base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).decode("ascii").rstrip("=")
        require(encoded_hash == f"sha256={expected_hash}", f"RECORD hash: {name}")
        require(size == str(len(payload)), f"RECORD size: {name}")


def validate_pair(
    original: Path,
    restored: Path,
    version: str,
    original_sha256: str,
    restored_sha256: str,
) -> dict:
    require(original.name == filename(version, restored=False), "original filename")
    require(restored.name == filename(version, restored=True), "recovery filename")
    require(digest(original) == original_sha256, "original SHA256 mismatch")
    require(digest(restored) == restored_sha256, "recovery SHA256 mismatch")
    before = archive_files(original)
    after = archive_files(restored)
    require(before.keys() == after.keys(), "wheel member sets differ")
    metadata_root = f"turbopairformer-{version}.dist-info/"
    wheel_name = metadata_root + "WHEEL"
    record_name = metadata_root + "RECORD"
    metadata_name = metadata_root + "METADATA"
    require(wheel_name in before and metadata_name in before, "missing metadata")
    validate_record(before, record_name)
    validate_record(after, record_name)
    changed = sorted(name for name in before if before[name] != after[name])
    require(changed == sorted([wheel_name, record_name]), "payload changed")
    require(before[wheel_name].endswith(b"\n\n"), "unexpected original WHEEL format")
    require(
        after[wheel_name] == before[wheel_name][:-1] + b"Build: 1\n\n",
        "WHEEL change is not exactly Build: 1",
    )
    parser = email.parser.BytesParser()
    wheel = parser.parsebytes(after[wheel_name])
    require(wheel.get_all("Build") == ["1"], "invalid build field")
    require(wheel.get_all("Tag") == [WHEEL_TAG], "wheel tag changed")
    metadata = parser.parsebytes(after[metadata_name])
    require(metadata.get_all("Name") == ["turbopairformer"], "distribution changed")
    require(metadata.get_all("Version") == [version], "package version changed")
    sources = sorted(name for name in after if name.endswith(".cu"))
    extensions = sorted(name for name in after if name.endswith(".so"))
    require(len(sources) == len(extensions) == 5, "historical kernel files missing")
    return {
        "version": version,
        "original_filename": original.name,
        "original_sha256": original_sha256,
        "restored_filename": restored.name,
        "restored_sha256": restored_sha256,
        "build": 1,
        "members": len(after),
        "changed_members": changed,
        "payload_unchanged": True,
        "cuda_sources": sources,
        "extensions": extensions,
    }


def validate_directory(
    directory: Path, tag: str, output_dir: Path | None = None
) -> dict:
    require(tag == RECOVERY_TAG, "not the authorized recovery release tag")
    expected_names = {
        filename(version, restored=restored)
        for version in RELEASES
        for restored in (False, True)
    }
    require(
        {path.name for path in directory.iterdir()} == expected_names,
        "recovery input must contain exactly the six pinned wheels",
    )
    require(
        all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
        "recovery input must contain regular files",
    )
    results = [
        validate_pair(
            directory / filename(version, restored=False),
            directory / filename(version, restored=True),
            version,
            *hashes,
        )
        for version, hashes in RELEASES.items()
    ]
    if output_dir is not None:
        require(
            not output_dir.exists() or not any(output_dir.iterdir()),
            "publication directory must be empty",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Nothing is copied until every original and recovery archive passes.
        for version in RELEASES:
            path = directory / filename(version, restored=True)
            shutil.copyfile(path, output_dir / path.name)
    return {
        "schema_version": 1,
        "scope": "historical-build-1-restoration",
        "tag": tag,
        "restored_versions": results,
        "new_gpu_validation": False,
        "original_upload_dates_and_file_hashes_restored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = validate_directory(args.directory, args.tag, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
