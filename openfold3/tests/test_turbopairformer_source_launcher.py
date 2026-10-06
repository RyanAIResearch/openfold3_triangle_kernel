"""The source overlay must reject stale, modified, or binary package contents."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from scripts.run_current_turbopairformer import (
    read_pin,
    source_environment,
    verify_site,
)


def test_source_environment_preserves_explicit_policy_and_overrides_architecture(
    tmp_path,
):
    inherited = {
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;9.0",
        "PYTHONPATH": "/historical/package",
        "OPENFOLD3_SR_BACKWARD": "0",
        "TURBOPAIRFORMER_SR_SHADOW": "0",
    }
    env = source_environment(tmp_path, inherited)
    assert env["PYTHONPATH"].split(":") == [str(tmp_path), "/historical/package"]
    assert env["TORCH_CUDA_ARCH_LIST"] == "9.0a"
    assert env["TURBOPAIRFORMER_ALLOW_JIT"] == "1"
    assert env["TURBOPAIRFORMER_SOURCE_ONLY"] == "1"
    assert env["TURBOPAIRFORMER_BUILD_AOT"] == "0"
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["OPENFOLD3_SR_BACKWARD"] == "0"
    assert "TURBOPAIRFORMER_SR_BACKWARD" not in env
    assert env["TURBOPAIRFORMER_SR_SHADOW"] == "0"
    assert inherited["TORCH_CUDA_ARCH_LIST"] == "8.0;8.6;9.0"


def test_command_script_cannot_shadow_the_verified_overlay(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "turbopairformer.py").write_text("identity = 'pinned source'\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "turbopairformer.py").write_text("identity = 'stale wheel'\n")
    command = scripts / "command.py"
    command.write_text("import turbopairformer; print(turbopairformer.identity)\n")
    result = subprocess.check_output(
        [sys.executable, str(command)],
        env=source_environment(site, os.environ),
        text=True,
    )
    assert result.strip() == "pinned source"


def test_source_pin_rejects_moving_branch_name(tmp_path):
    path = tmp_path / "pin.json"
    path.write_text(json.dumps({"revision": "feat/turbopairformer-pip-package"}))
    with pytest.raises(RuntimeError, match="full Git SHA"):
        read_pin(path)


@pytest.fixture
def source_and_site(tmp_path):
    checkout = tmp_path / "checkout"
    package = checkout / "packages/turbopairformer/turbopairformer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# pinned source\n")
    (package / "_build_manifest.json").write_text('{"built": false}\n')
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "source fixture",
        ],
        cwd=checkout,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()
    site = tmp_path / "site"
    shutil.copytree(package, site / "turbopairformer")
    return (
        checkout,
        site,
        {"revision": revision, "subdirectory": "packages/turbopairformer"},
    )


def test_overlay_verifies_source_bytes(source_and_site):
    checkout, site, pin = source_and_site
    hashes = verify_site(checkout, site, pin)
    assert set(hashes) == {
        "turbopairformer/__init__.py",
        "turbopairformer/_build_manifest.json",
    }


@pytest.mark.parametrize("change", ["modified", "missing", "binary", "revision"])
def test_overlay_rejects_untrusted_cache(source_and_site, change):
    checkout, site, pin = source_and_site
    init = site / "turbopairformer/__init__.py"
    if change == "modified":
        init.write_text("# stale wheel\n")
    elif change == "missing":
        init.unlink()
    elif change == "binary":
        (site / "turbopairformer/old_aot.so").write_bytes(b"binary")
    else:
        pin["revision"] = "0" * 40
    with pytest.raises(RuntimeError):
        verify_site(checkout, site, pin)
