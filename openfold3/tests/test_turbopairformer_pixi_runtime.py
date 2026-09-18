"""Guard the installed package/runtime boundary of the release environment."""

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WHEEL_PATH = (
    "release_artifacts/turbopairformer/0.1.0/"
    "turbopairformer-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "54a4ab1879a0029770c1f72bb7a718bc720431393f74b89fe3ebe2e4e5b57a8a"


def test_package_environment_uses_pypi_torch_and_current_noneditable_host():
    manifest = tomllib.loads((ROOT / "pixi.toml").read_text())
    features = manifest["environments"]["openfold3-cuda12"]["features"]
    assert "turbopairformer-hopper" in features
    assert not set(features) & {
        "stablefold-h100",
        "pytorch-conda-cuda",
        "deepspeed",
        "deepspeed-build",
        "openfold3-editable",
        "openfold3-editable-full",
    }
    runtime = manifest["feature"]["turbopairformer-hopper"]
    assert runtime["platforms"] == ["linux-64-cuda12"]
    assert runtime["dependencies"]["python"] == "3.14.*"
    assert runtime["pypi-dependencies"] == {
        "torch": "==2.10.0",
        "pytorch-lightning": "==2.6.1",
        "turbopairformer": {"path": WHEEL_PATH},
        "openfold3": {"path": ".", "editable": False},
    }
    assert runtime["pypi-options"]["no-build"] == ["turbopairformer"]
    env = runtime["activation"]["env"]
    assert env["OPENFOLD3_TRIANGLE_PROVIDER"] == "turbopairformer"
    assert env["TURBOPAIRFORMER_PYTHON"] == "$CONDA_PREFIX/bin/python"
    assert env["OPENFOLD3_BUILD_AOT"] == "0"
    assert env["LD_LIBRARY_PATH"] == "$CONDA_PREFIX/lib"
    assert env["TURBOPAIRFORMER_ALLOW_JIT"] == "0"
    assert env["OPENFOLD3_TRIANGLE_ALLOW_JIT"] == "0"
    tasks = runtime["tasks"]
    assert tasks["train-single-gpu"]["cmd"] == [
        "bash",
        "scripts/turbopairformer/run_single_gpu.sh",
    ]
    assert "train-multinode" not in tasks
    assert "train-single-node" not in tasks


def test_package_environment_exposes_installed_runtime_benchmark_tasks():
    manifest = tomllib.loads((ROOT / "pixi.toml").read_text())
    tasks = manifest["feature"]["turbopairformer-hopper"]["tasks"]
    for task in ("benchmark-single-gpu", "benchmark-single-node"):
        assert tasks[task]["cmd"] == ["bash", "tools/run_backend_benchmark.sh"]
    assert tasks["plot-backend-comparison"]["cmd"] == [
        "python",
        "tools/plot_backend_comparison.py",
    ]


@pytest.mark.parametrize("provider", ["turbopairformer", "intree"])
@pytest.mark.parametrize("preflight_fails", [False, True])
def test_benchmark_wrapper_checks_selected_install_before_launch(
    tmp_path, provider, preflight_fails
):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    prefix = tmp_path / "env"
    source.mkdir()
    run_dir.mkdir()
    (prefix / "bin").mkdir(parents=True)
    call_log = tmp_path / "calls.jsonl"
    interpreter = prefix / "bin/python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BENCHMARK_CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
        "is_preflight = any('turbopairformer_doctor' in arg or "
        "'manage_attested_sm90_wheel' in arg for arg in sys.argv[1:])\n"
        "if is_preflight and os.environ['BENCHMARK_PREFLIGHT_FAILS'] == '1':\n"
        "    sys.exit(7)\n"
    )
    interpreter.chmod(0o755)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")
    }
    environment.update(
        CONDA_PREFIX=str(prefix),
        PIXI_PROJECT_ROOT=str(source),
        OF3_RUN_CWD=str(run_dir),
        OPENFOLD3_TRIANGLE_PROVIDER=provider,
        BENCHMARK_CALL_LOG=str(call_log),
        BENCHMARK_PREFLIGHT_FAILS=str(int(preflight_fails)),
    )
    backend = "turbopairformer" if provider == "turbopairformer" else "stablefold"
    arguments = ["--runner-yaml", "runner.yaml", "--backend", backend]
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_backend_benchmark.sh"), *arguments],
        cwd=source,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == (7 if preflight_fails else 0), result.stderr
    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
    assert calls[0]["args"] == [
        str(source / "tools/validate_training_data.py"),
        str(source / "runner.yaml"),
        "--base-dir",
        str(run_dir),
    ]
    if provider == "turbopairformer":
        assert calls[1] == {
            "args": [
                str(source / "tools/turbopairformer_doctor.py"),
                "--expect-provider",
                "turbopairformer",
            ],
            "cwd": str(run_dir),
        }
    else:
        assert calls[1] == {
            "args": ["-m", "tools.manage_attested_sm90_wheel", "verify-prebuilt"],
            "cwd": str(source),
        }
    if preflight_fails:
        assert len(calls) == 2
    else:
        assert calls[2] == {
            "args": [str(source / "tools/benchmark_training_backend.py"), *arguments],
            "cwd": str(run_dir),
        }
        assert len(calls) == 3


@pytest.mark.parametrize(
    "arguments",
    [["--devices", "2"], ["--devices=2"], ["--num-nodes", "2"], ["--num-nodes=2"]],
)
def test_package_benchmark_rejects_unqualified_distributed_launch(tmp_path, arguments):
    source = tmp_path / "source"
    run_dir = tmp_path / "run"
    prefix = tmp_path / "env"
    for directory in (source, run_dir, prefix):
        directory.mkdir()
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")
    }
    environment.update(
        CONDA_PREFIX=str(prefix),
        PIXI_PROJECT_ROOT=str(source),
        OF3_RUN_CWD=str(run_dir),
        OPENFOLD3_TRIANGLE_PROVIDER="turbopairformer",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_backend_benchmark.sh"), *arguments],
        cwd=source,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "require --devices 1 --num-nodes 1" in result.stderr
    assert "multi-GPU validation is pending" in result.stderr


def test_package_lock_pins_bundled_candidate_and_pypi_torch():
    lock = yaml.safe_load((ROOT / "pixi.lock").read_text())
    packages = lock["environments"]["openfold3-cuda12"]["packages"]
    assert set(packages) == {"linux-64-cuda12"}
    selected = packages["linux-64-cuda12"]
    assert not any(
        "pytorch-" in entry.get("conda", "") or "libtorch-" in entry.get("conda", "")
        for entry in selected
    )
    urls = [entry["pypi"] for entry in selected if "pypi" in entry]
    assert "./" in urls
    assert all(not entry.get("editable", False) for entry in selected)
    package_urls = [url for url in urls if "turbopairformer-" in url]
    assert len(package_urls) == 1
    assert package_urls[0] == "./" + WHEEL_PATH
    record = next(
        item for item in lock["packages"] if item.get("pypi") == "./" + WHEEL_PATH
    )
    assert record["name"] == "turbopairformer"
    assert record["version"] == "0.1.0"
    assert record["sha256"] == WHEEL_SHA256
    torch_urls = [url for url in urls if "/torch-" in url]
    assert len(torch_urls) == 1
    assert "/torch-2.10.0-" in torch_urls[0]
    assert any("nvidia_cuda_runtime_cu12-12.8." in url for url in urls)


def test_bundled_wheel_matches_release_digest_and_numerical_receipts():
    wheel = ROOT / WHEEL_PATH
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == WHEEL_SHA256
    expected_sidecar = f"{WHEEL_SHA256}  {wheel.name}\n"
    assert (wheel.parent / "SHA256SUMS").read_text() == expected_sidecar
    receipt = json.loads((wheel.parent / "release-validation.json").read_text())
    assert receipt["source_commit"] == "97d2292ca7eebd09c99936feb6f40d13af93caac"
    assert receipt["wheel"] == {"filename": wheel.name, "sha256": WHEEL_SHA256}
    for check in receipt["checks"]:
        assert check["exit_code"] == 0
        evidence = wheel.parent / check["evidence"]
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == check["sha256"]
