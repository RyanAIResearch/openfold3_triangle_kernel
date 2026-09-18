#!/usr/bin/env python3
"""Create or verify the wheel and GPU evidence attached to a GitHub release.

Run GPU checks against the installed final wheel first. This script never runs
GPU tests itself and never treats a successful archive check as GPU validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_GPU_CASES = {
    "attention support": "test_accepts_the_declared_bf16_contract",
    "attention FP64": "test_triangle_attention_fp64_error_stays_within_stock_envelope",
    "TriMul execution": "test_all_eight_production_symbols_accept_valid_inputs",
    "TriMul numerics": "test_wgrad_fold_matches_strict_ascending_reference",
}
EXTENSIONS = {
    "stablefold_forward",
    "stablefold_delta",
    "stablefold_dkdv",
    "stablefold_dq_dbias",
    "stablefold_trimul",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_wheel(path: Path, commit: str) -> dict:
    spec = importlib.util.spec_from_file_location(
        "turbopairformer_wheel_validator",
        ROOT / "packages/turbopairformer/tools/validate_wheel.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_wheel(path, require_pypi=True, expected_commit=commit)


def validate_doctor(path: Path, commit: str, version: str, manifest_hash: str) -> dict:
    report = json.loads(path.read_text())
    require(report.get("scope") == "aot-load", "doctor did not load AOT extensions")
    require(
        report["provider"]["provider"] == "turbopairformer",
        "doctor tested another provider",
    )
    require(
        report["package"]["name"] == "turbopairformer"
        and report["package"]["version"] == version,
        "doctor tested another package version",
    )
    require(
        report["gpu"]["available"] is True and report["gpu"]["capability"] == "9.0",
        "doctor did not validate a Hopper GPU",
    )
    require(
        report["aot"]["git"]["commit"] == commit
        and report["aot"]["git"]["dirty"] is False,
        "doctor tested another source commit or a dirty build",
    )
    require(
        set(report["aot"]["extensions"]) == EXTENSIONS,
        "doctor did not load all five extensions",
    )
    require(
        report["aot"]["manifest_sha256"] == manifest_hash,
        "doctor tested different installed wheel bytes",
    )
    package_root = Path(report["package"]["path"]).parent
    require("site-packages" in package_root.parts, "doctor tested a source checkout")
    require(
        all(
            Path(value).parent == package_root / "_ext"
            for value in report["aot"]["extensions"].values()
        ),
        "doctor loaded an extension from outside the installed package",
    )
    return report["gpu"]


def wheel_manifest_hash(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(
            archive.read("turbopairformer/_build_manifest.json")
        ).hexdigest()


def validate_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    require(root.tag in {"testsuites", "testsuite"}, "unsupported JUnit document")
    require(not list(root.iter("failure")), "GPU tests contain failures")
    require(not list(root.iter("error")), "GPU tests contain errors")
    for suite in root.iter("testsuite"):
        require(
            int(suite.get("failures", "0")) == 0 and int(suite.get("errors", "0")) == 0,
            "GPU test suite reports failures or errors",
        )
    cases = list(root.iter("testcase"))
    passed = [case for case in cases if case.find("skipped") is None]
    names = {case.get("name", "").split("[", 1)[0] for case in passed}
    for scope, name in REQUIRED_GPU_CASES.items():
        require(name in names, f"no executed {scope} test in GPU receipt")
    return {"passed": len(passed), "skipped": len(cases) - len(passed)}


def validate_receipt(directory: Path, wheel: dict, commit: str) -> dict:
    receipt = json.loads((directory / "release-validation.json").read_text())
    require(receipt.get("schema_version") == 1, "unsupported release receipt")
    require(receipt["source_commit"] == commit, "release receipt commit mismatch")
    require(
        receipt["wheel"] == {"filename": wheel["wheel"], "sha256": wheel["sha256"]},
        "release receipt identifies another wheel or digest",
    )
    checks = receipt["checks"]
    require(isinstance(checks, list), "release checks must be a list")
    by_name = {check["name"]: check for check in checks}
    require(len(checks) == len(by_name), "duplicate release check names")
    for name, evidence in (
        ("installed_aot", "installed-aot.json"),
        ("standalone_gpu", "standalone-gpu.xml"),
    ):
        check = by_name[name]
        require(check["exit_code"] == 0, f"failed release check: {name}")
        require(bool(check["command"].strip()), f"missing check command: {name}")
        require(check["evidence"] == evidence, f"unexpected evidence path: {name}")
        require(
            check["sha256"] == digest(directory / evidence), f"stale evidence: {name}"
        )
    return {
        "gpu": validate_doctor(
            directory / "installed-aot.json",
            commit,
            wheel["version"],
            wheel_manifest_hash(directory / wheel["wheel"]),
        ),
        "tests": validate_junit(directory / "standalone-gpu.xml"),
    }


def validate_bundle(directory: Path, commit: str, tag: str, prerelease: bool) -> dict:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "expected a full commit SHA")
    wheels = list(directory.glob("turbopairformer-*.whl"))
    require(len(wheels) == 1, "release must contain exactly one TurboPairFormer wheel")
    wheel = validate_wheel(wheels[0], commit)
    base_tag = "turbopairformer-v" + wheel["version"]
    expected_tag = re.escape(base_tag) + (r"-rc\.[1-9][0-9]*" if prerelease else "")
    require(
        bool(re.fullmatch(expected_tag, tag)), "release tag/version/prerelease mismatch"
    )
    evidence = validate_receipt(directory, wheel, commit)
    return {"tag": tag, "publish_to_pypi": not prerelease, "wheel": wheel, **evidence}


def create_bundle(args: argparse.Namespace) -> dict:
    wheel = validate_wheel(args.wheel, args.expected_commit)
    validate_doctor(
        args.doctor,
        args.expected_commit,
        wheel["version"],
        wheel_manifest_hash(args.wheel),
    )
    validate_junit(args.junit)
    args.directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.wheel, args.directory / args.wheel.name)
    checks = []
    for name, source, filename, command in (
        ("installed_aot", args.doctor, "installed-aot.json", args.doctor_command),
        ("standalone_gpu", args.junit, "standalone-gpu.xml", args.gpu_command),
    ):
        shutil.copy2(source, args.directory / filename)
        checks.append(
            {
                "name": name,
                "exit_code": 0,
                "command": command,
                "evidence": filename,
                "sha256": digest(source),
            }
        )
    receipt = {
        "schema_version": 1,
        "source_commit": args.expected_commit,
        "wheel": {"filename": wheel["wheel"], "sha256": wheel["sha256"]},
        "checks": checks,
    }
    (args.directory / "release-validation.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create = subparsers.add_parser(
        "create", help="bundle completed GPU checks with their wheel"
    )
    create.add_argument("directory", type=Path, help="new output directory")
    create.add_argument("--wheel", required=True, type=Path)
    create.add_argument("--doctor", required=True, type=Path)
    create.add_argument("--junit", required=True, type=Path)
    create.add_argument("--doctor-command", required=True)
    create.add_argument("--gpu-command", required=True)
    create.add_argument("--expected-commit", required=True)
    validate = subparsers.add_parser("validate", help="check downloaded release assets")
    validate.add_argument("directory", type=Path)
    validate.add_argument("--expected-commit", required=True)
    validate.add_argument("--tag", required=True)
    validate.add_argument("--prerelease", choices=("true", "false"), required=True)
    args = parser.parse_args()
    try:
        result = (
            create_bundle(args)
            if args.operation == "create"
            else validate_bundle(
                args.directory,
                args.expected_commit,
                args.tag,
                args.prerelease == "true",
            )
        )
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as exc:
        parser.exit(1, f"TurboPairFormer release validation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
