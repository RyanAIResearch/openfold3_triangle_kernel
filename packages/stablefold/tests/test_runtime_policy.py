"""Runtime policy and receipt checks that do not need a CUDA device."""

import json

import pytest

from stablefold import _aot_runtime as runtime
from stablefold import policy


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    for canonical, aliases in policy.LEGACY_ENV_NAMES.items():
        for name in (canonical, *aliases):
            monkeypatch.delenv(name, raising=False)
    policy._set_distributed_strict(False)
    yield
    policy._set_distributed_strict(False)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"STABLEFOLD_ALLOW_JIT": "1"}, True),
        ({"OPENFOLD3_TRIANGLE_ALLOW_JIT": "1"}, True),
        ({"STABLEFOLD_ALLOW_JIT": "0", "OPENFOLD3_TRIANGLE_ALLOW_JIT": "1"}, False),
        ({"STABLEFOLD_ALLOW_JIT": "1", "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0"}, True),
    ],
)
def test_loader_and_policy_agree_on_jit(environment, expected, monkeypatch):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert runtime._development_jit_enabled() is expected
    assert policy.distributed_execution_flag(policy.ALLOW_JIT) is expected


def test_jit_uses_frozen_policy_and_drift_is_rejected(monkeypatch):
    monkeypatch.setenv("STABLEFOLD_ALLOW_JIT", "0")
    policy._set_distributed_strict(True, policy.execution_environment())
    monkeypatch.setenv("STABLEFOLD_ALLOW_JIT", "1")
    assert not runtime._development_jit_enabled()
    with pytest.raises(policy.DistributedContractError, match="environment changed"):
        policy.validate_distributed_environment()


def test_editable_receipt_leaves_source_placeholder_intact(tmp_path, monkeypatch):
    package = tmp_path / "stablefold"
    package.mkdir()
    embedded = package / "_build_manifest.json"
    placeholder = '{"schema_version": 1, "built": false}'
    embedded.write_text(placeholder)
    local = package / "_build_manifest.local.json"
    local.write_text('{"schema_version": 1, "built": true}')
    (tmp_path / "setup.py").write_text("")
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.setattr(runtime, "_MANIFEST_PATH", embedded)
    assert runtime._manifest_path() == local
    assert runtime._manifest()["built"] is True
    assert embedded.read_text() == placeholder


def test_installed_wheel_ignores_unowned_local_receipt(tmp_path, monkeypatch):
    embedded = tmp_path / "_build_manifest.json"
    embedded.write_text(json.dumps({"schema_version": 1, "built": False}))
    embedded.with_name("_build_manifest.local.json").write_text(
        json.dumps({"schema_version": 1, "built": True})
    )
    monkeypatch.setattr(runtime, "_MANIFEST_PATH", embedded)
    assert runtime._manifest_path() == embedded
    assert runtime._manifest()["built"] is False
