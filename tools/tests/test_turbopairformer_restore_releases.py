"""The historical recovery lane must never publish a changed kernel payload."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "restore_releases",
    Path(__file__).resolve().parents[2] / "scripts/turbopairformer/restore_releases.py",
)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def write_wheel(path, files, record_name, *, invalid_record=False):
    files = dict(files)
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    for name, content in files.items():
        if name == record_name:
            continue
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
        writer.writerow([name, "sha256=" + encoded.decode().rstrip("="), len(content)])
    writer.writerow([record_name, "", ""])
    files[record_name] = text.getvalue().encode()
    if invalid_record:
        files[record_name] = files[record_name].replace(b"sha256=", b"sha256=x", 1)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


@pytest.fixture
def pair(tmp_path):
    version = "0.1.1"
    root = f"turbopairformer-{version}.dist-info/"
    files = {
        "turbopairformer/__init__.py": b"# historical code\n",
        root + "METADATA": b"Name: turbopairformer\nVersion: 0.1.1\n\n",
        root + "WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            f"Tag: {restore.WHEEL_TAG}\n\n"
        ).encode(),
    }
    files.update({f"turbopairformer/kernel{i}.cu": b"source" for i in range(5)})
    files.update({f"turbopairformer/_ext/kernel{i}.so": b"binary" for i in range(5)})
    original = tmp_path / restore.filename(version, restored=False)
    restored = tmp_path / restore.filename(version, restored=True)
    record_name = root + "RECORD"
    write_wheel(original, files, record_name)
    files[root + "WHEEL"] = files[root + "WHEEL"][:-1] + b"Build: 1\n\n"
    write_wheel(restored, files, record_name)
    return original, restored, version, files, record_name


def validate(pair):
    original, restored, version, _, _ = pair
    return restore.validate_pair(
        original, restored, version, restore.digest(original), restore.digest(restored)
    )


def test_exact_build_tag_only_change_passes(pair):
    result = validate(pair)
    assert result["payload_unchanged"]
    assert result["build"] == 1
    assert len(result["changed_members"]) == 2


@pytest.mark.parametrize("kind", ["source", "binary", "metadata", "member", "build", "tag"])
def test_rejects_changed_payload_even_with_valid_record_and_new_sha(pair, kind):
    _, restored, _, files, record_name = pair
    if kind == "source":
        files["turbopairformer/kernel0.cu"] += b"changed"
    elif kind == "binary":
        files["turbopairformer/_ext/kernel0.so"] += b"changed"
    elif kind == "metadata":
        files[record_name.replace("RECORD", "METADATA")] += b"changed"
    elif kind == "member":
        files["extra.py"] = b"changed"
    elif kind == "build":
        files[record_name.replace("RECORD", "WHEEL")] = files[
            record_name.replace("RECORD", "WHEEL")
        ].replace(b"Build: 1", b"Build: 2")
    elif kind == "tag":
        files[record_name.replace("RECORD", "WHEEL")] = files[
            record_name.replace("RECORD", "WHEEL")
        ].replace(b"cp314-cp314", b"cp313-cp313")
    write_wheel(restored, files, record_name)
    with pytest.raises(ValueError):
        validate(pair)


def test_rejects_invalid_record(pair):
    _, restored, _, files, record_name = pair
    write_wheel(restored, files, record_name, invalid_record=True)
    with pytest.raises(ValueError, match="RECORD hash"):
        validate(pair)


@pytest.mark.parametrize("which", ["original", "recovery"])
def test_rejects_unpinned_archive(pair, which):
    original, restored, version, _, _ = pair
    hashes = [restore.digest(original), restore.digest(restored)]
    hashes[which == "recovery"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restore.validate_pair(original, restored, version, *hashes)


@pytest.mark.parametrize("unsafe", ["../outside.py", "/absolute.py", "a\\b.py", "a/./b.py"])
def test_rejects_unsafe_archive_paths(pair, unsafe):
    _, restored, _, files, record_name = pair
    files[unsafe] = b"content"
    write_wheel(restored, files, record_name)
    with pytest.raises(ValueError, match="unsafe archive path"):
        validate(pair)


def test_rejects_duplicate_archive_members(pair):
    _, restored, _, _, _ = pair
    with pytest.warns(UserWarning), zipfile.ZipFile(restored, "a") as archive:
        archive.writestr("turbopairformer/kernel0.cu", b"source")
    with pytest.raises(ValueError, match="duplicate archive"):
        validate(pair)


def test_unknown_tag_cannot_publish(tmp_path):
    with pytest.raises(ValueError, match="authorized recovery release tag"):
        restore.validate_directory(tmp_path, "turbopairformer-restore-arbitrary")


def test_rejects_missing_originals(tmp_path):
    with pytest.raises(ValueError, match="exactly the six pinned wheels"):
        restore.validate_directory(tmp_path, restore.RECOVERY_TAG)


def test_all_payloads_verified_before_publication_directory_created(pair, monkeypatch):
    original, restored, version, _, _ = pair
    monkeypatch.setattr(
        restore, "RELEASES", {version: (restore.digest(original), "0" * 64)}
    )
    destination = original.parent / "dist"
    with pytest.raises(ValueError, match="recovery SHA256 mismatch"):
        restore.validate_directory(original.parent, restore.RECOVERY_TAG, destination)
    assert not destination.exists()


def test_publication_directory_contains_only_recovery_wheel(pair, monkeypatch, tmp_path):
    original, restored, version, _, _ = pair
    monkeypatch.setattr(
        restore, "RELEASES", {version: (restore.digest(original), restore.digest(restored))}
    )
    destination = tmp_path / "dist"
    restore.validate_directory(original.parent, restore.RECOVERY_TAG, destination)
    assert [path.name for path in destination.iterdir()] == [restored.name]
    assert (destination / restored.name).read_bytes() == restored.read_bytes()
