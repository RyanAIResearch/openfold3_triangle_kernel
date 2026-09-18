"""Guard the installed package/runtime boundary of the release environment."""

import hashlib
import json
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WHEEL_PATH = (
    "release_artifacts/turbopairformer/0.1.0/"
    "turbopairformer-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "dd7da8690ae89be29a789825ac6c5796c7aa96b7eb129a481e9eef0ab1e626fe"


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
    assert receipt["source_commit"] == "d77e8d06cba4918fa551ce1f4ec41cbb21d055af"
    assert receipt["wheel"] == {"filename": wheel.name, "sha256": WHEEL_SHA256}
    for check in receipt["checks"]:
        assert check["exit_code"] == 0
        evidence = wheel.parent / check["evidence"]
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == check["sha256"]
