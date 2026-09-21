#!/usr/bin/env python3
"""Run the installed console/module CLI and attach evidence to an AOT doctor JSON.

Run with the validation venv's Python outside the checkout. This records actual
installed entry-point execution; source-tree documentation tests are separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ARGUMENTS = (("--help",), ("--version",), ("api", "--all"))


def smoke() -> dict:
    package = importlib.import_module("turbopairformer")
    cli = importlib.import_module("turbopairformer.cli")
    package_path = Path(package.__file__).resolve()
    cli_module_path = Path(cli.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise ValueError("CLI smoke requires an installed wheel outside the checkout")
    if cli_module_path != package_path.parent / "cli.py":
        raise ValueError("CLI and package imports resolve to different installations")
    distribution = importlib.metadata.distribution("turbopairformer")
    entry_points = [
        entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "turbopairformer"
    ]
    if entry_points != ["turbopairformer.cli:main"]:
        raise ValueError(
            "installed wheel has no unique turbopairformer console entry point"
        )
    python_path = Path(sys.executable).absolute()
    console_path = python_path.parent / "turbopairformer"
    if not console_path.is_file():
        raise ValueError(f"installed console script is missing: {console_path}")
    version = distribution.version
    if package.__version__ != version:
        raise ValueError("package and installed distribution versions disagree")
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8")
    checks = []
    outputs = {}
    for launcher, command in (
        ("console", [str(console_path)]),
        ("module", [str(python_path), "-m", "turbopairformer"]),
    ):
        for arguments in ARGUMENTS:
            result = subprocess.run(
                [*command, *arguments],
                cwd="/tmp",
                env=environment,
                capture_output=True,
                timeout=30,
                check=True,
            )
            text = result.stdout.decode("utf-8")
            if result.stderr:
                raise ValueError(f"{launcher} CLI emitted stderr: {result.stderr!r}")
            if arguments == ("--version",):
                valid = text == f"turbopairformer {version}\n"
            elif arguments == ("--help",):
                valid = all(value in text for value in ("usage:", "api", "--help"))
            else:
                valid = all(
                    value in text
                    for value in (
                        "turbo_attention",
                        "turbo_trimul",
                        "Triangle Attention",
                        "Triangle Multiplicative Update",
                    )
                )
            if not valid:
                raise ValueError(
                    f"{launcher} CLI returned unexpected output for {arguments}"
                )
            outputs[(launcher, arguments)] = result.stdout
            checks.append(
                {
                    "launcher": launcher,
                    "arguments": list(arguments),
                    "exit_code": result.returncode,
                    "stderr": "",
                    "output_validated": True,
                    "stdout_bytes": len(result.stdout),
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                }
            )
    for arguments in ARGUMENTS:
        if outputs[("console", arguments)] != outputs[("module", arguments)]:
            raise ValueError(
                f"installed console/module outputs disagree for {arguments}"
            )
    return {
        "schema_version": 1,
        "scope": "installed-cli",
        "version": version,
        "package_path": str(package_path),
        "cli_module_path": str(cli_module_path),
        "entry_point": entry_points[0],
        "python_path": str(python_path),
        "console_path": str(console_path),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "doctor", type=Path, help="completed installed-aot.json to extend"
    )
    args = parser.parse_args()
    report = json.loads(args.doctor.read_text())
    evidence = smoke()
    if (
        report["package"]["version"] != evidence["version"]
        or report["package"]["path"] != evidence["package_path"]
    ):
        raise ValueError("CLI and AOT doctor tested different package installations")
    report["installed_cli"] = evidence
    temporary = args.doctor.with_name(args.doctor.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.doctor)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
