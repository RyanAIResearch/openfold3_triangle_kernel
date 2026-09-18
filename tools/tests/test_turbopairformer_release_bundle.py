"""Publication gates reject stale binaries, failed tests and CPU-only evidence."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "release_bundle",
    Path(__file__).resolve().parents[2] / "scripts/turbopairformer/release_bundle.py",
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
COMMIT = "a" * 40


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    wheel = tmp_path / "turbopairformer-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("turbopairformer/_build_manifest.json", '{"built":true}')
    info = {"wheel": wheel.name, "sha256": release.digest(wheel), "version": "0.1.0"}
    monkeypatch.setattr(release, "validate_wheel", lambda path, commit: info)
    doctor = tmp_path / "doctor.json"
    package = "/venv/lib/python3.14/site-packages/turbopairformer"
    report = {
        "scope": "aot-load",
        "provider": {"provider": "turbopairformer"},
        "package": {
            "name": "turbopairformer",
            "version": "0.1.0",
            "path": package + "/__init__.py",
        },
        "gpu": {"available": True, "name": "NVIDIA H100", "capability": "9.0"},
        "aot": {
            "git": {"commit": COMMIT, "dirty": False},
            "manifest_sha256": release.wheel_manifest_hash(wheel),
            "extensions": {
                name: f"{package}/_ext/{name}.so" for name in release.EXTENSIONS
            },
        },
    }
    doctor.write_text(json.dumps(report))
    junit = tmp_path / "tests.xml"
    suite = ET.Element("testsuite", failures="0", errors="0")
    for name in release.REQUIRED_GPU_CASES.values():
        ET.SubElement(suite, "testcase", name=name + "[case]")
    ET.ElementTree(suite).write(junit)
    output = tmp_path / "bundle"
    release.create_bundle(
        Namespace(
            wheel=wheel,
            doctor=doctor,
            junit=junit,
            directory=output,
            expected_commit=COMMIT,
            doctor_command="python doctor.py --json",
            gpu_command="python -m pytest --junitxml=tests.xml",
        )
    )
    return output, info, report


def test_candidate_validates_without_publication_then_same_bundle_can_release(bundle):
    output, _, _ = bundle
    candidate = release.validate_bundle(
        output, COMMIT, "turbopairformer-v0.1.0-rc.1", True
    )
    stable = release.validate_bundle(output, COMMIT, "turbopairformer-v0.1.0", False)
    assert candidate["publish_to_pypi"] is False
    assert stable["publish_to_pypi"] is True
    assert candidate["wheel"]["sha256"] == stable["wheel"]["sha256"]


@pytest.mark.parametrize(
    "tag,prerelease",
    [
        ("turbopairformer-v0.1.0-rc.1", False),
        ("turbopairformer-v0.1.0", True),
        ("turbopairformer-v0.2.0", False),
    ],
)
def test_mismatched_release_identity_is_rejected(bundle, tag, prerelease):
    with pytest.raises(ValueError, match="tag/version"):
        release.validate_bundle(bundle[0], COMMIT, tag, prerelease)


def test_changed_gpu_evidence_is_rejected(bundle):
    output, info, _ = bundle
    (output / "standalone-gpu.xml").write_text("<testsuite/>")
    with pytest.raises(ValueError, match="stale evidence"):
        release.validate_receipt(output, info, COMMIT)


def test_gpu_tests_must_execute_each_required_scope(bundle):
    output, _, _ = bundle
    path = output / "standalone-gpu.xml"
    tree = ET.parse(path)
    ET.SubElement(tree.getroot().find("testcase"), "skipped")
    tree.write(path)
    with pytest.raises(ValueError, match="no executed attention support"):
        release.validate_junit(path)


def test_failed_gpu_test_is_rejected(bundle):
    path = bundle[0] / "standalone-gpu.xml"
    tree = ET.parse(path)
    ET.SubElement(tree.getroot().find("testcase"), "failure")
    tree.write(path)
    with pytest.raises(ValueError, match="failures"):
        release.validate_junit(path)


@pytest.mark.parametrize(
    "change,match",
    [
        ("manifest", "different installed wheel"),
        ("source", "source checkout"),
        ("gpu", "Hopper"),
        ("commit", "source commit"),
    ],
)
def test_doctor_must_identify_exact_installed_gpu_build(bundle, change, match):
    output, info, report = bundle
    manifest_hash = report["aot"]["manifest_sha256"]
    if change == "manifest":
        report["aot"]["manifest_sha256"] = "b" * 64
    elif change == "source":
        report["package"]["path"] = "/checkout/turbopairformer/__init__.py"
    elif change == "gpu":
        report["gpu"]["available"] = False
    else:
        report["aot"]["git"]["commit"] = "b" * 40
    path = output / "installed-aot.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        release.validate_doctor(path, COMMIT, info["version"], manifest_hash)


def test_wheel_digest_mismatch_is_rejected(bundle):
    output, info, _ = bundle
    with pytest.raises(ValueError, match="another wheel or digest"):
        release.validate_receipt(output, {**info, "sha256": "b" * 64}, COMMIT)
