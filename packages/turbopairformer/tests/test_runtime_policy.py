"""Runtime policy and receipt checks that do not need a CUDA device."""

import json
import os
import subprocess
import sys

import pytest

from turbopairformer import _aot_runtime as runtime
from turbopairformer import policy


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    for canonical, aliases in policy.LEGACY_ENV_NAMES.items():
        for name in (canonical, *aliases):
            monkeypatch.delenv(name, raising=False)
    policy._set_distributed_strict(False)
    yield
    policy._set_distributed_strict(False)


def test_default_policy_stores_residual_without_mutating_environment():
    environment = dict(os.environ)

    assert policy.execution_environment() == {
        policy.SR_BACKWARD: "1",
        policy.SR_SHADOW: "1",
        policy.ALLOW_JIT: "0",
    }
    assert policy.distributed_execution_flag(policy.SR_BACKWARD)
    assert policy.distributed_execution_flag(policy.SR_SHADOW)
    assert not policy.distributed_execution_flag(policy.ALLOW_JIT)
    assert not runtime._development_jit_enabled()
    assert dict(os.environ) == environment


@pytest.mark.parametrize(
    ("canonical", "explicit_name"),
    [
        (canonical, name)
        for canonical, aliases in policy.LEGACY_ENV_NAMES.items()
        for name in (canonical, *aliases)
    ],
)
def test_explicit_zero_overrides_default_for_every_flag_spelling(
    canonical, explicit_name, monkeypatch
):
    monkeypatch.setenv(explicit_name, "0")

    assert policy.execution_environment()[canonical] == "0"
    for name in (canonical, *policy.LEGACY_ENV_NAMES[canonical]):
        assert policy.read_execution_flag(name) == "0"
        assert not policy.distributed_execution_flag(name)
    if canonical == policy.ALLOW_JIT:
        assert not runtime._development_jit_enabled()


@pytest.mark.parametrize("spelling_index", (0, 1, 2))
def test_strict_default_snapshot_matches_explicit_residual_enablement(
    spelling_index, monkeypatch
):
    snapshot = policy.execution_environment()
    assert snapshot[policy.SR_BACKWARD] == snapshot[policy.SR_SHADOW] == "1"
    policy._set_distributed_strict(True, snapshot)
    names = [
        (canonical, *policy.LEGACY_ENV_NAMES[canonical])[spelling_index]
        for canonical in (policy.SR_BACKWARD, policy.SR_SHADOW)
    ]

    for name in names:
        monkeypatch.setenv(name, "1")
    assert policy.execution_environment() == snapshot
    policy.validate_distributed_environment()
    assert policy.distributed_execution_flag(policy.SR_BACKWARD)
    assert policy.distributed_execution_flag(policy.SR_SHADOW)

    for name in names:
        monkeypatch.delenv(name)
    assert policy.execution_environment() == snapshot
    policy.validate_distributed_environment()


@pytest.mark.parametrize(
    ("canonical", "explicit_name"),
    [
        (canonical, name)
        for canonical in (policy.SR_BACKWARD, policy.SR_SHADOW)
        for name in (canonical, *policy.LEGACY_ENV_NAMES[canonical])
    ],
)
def test_strict_policy_rejects_default_residual_disablement(
    canonical, explicit_name, monkeypatch
):
    policy._set_distributed_strict(True, policy.execution_environment())
    monkeypatch.setenv(explicit_name, "0")

    assert policy.read_execution_flag(canonical) == "0"
    assert policy.distributed_execution_flag(canonical)
    with pytest.raises(policy.DistributedContractError, match="environment changed"):
        policy.validate_distributed_environment()


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"TURBOPAIRFORMER_ALLOW_JIT": "1"}, True),
        ({"OPENFOLD3_TRIANGLE_ALLOW_JIT": "1"}, True),
        ({"STABLEFOLD_ALLOW_JIT": "1"}, True),
        ({"TURBOPAIRFORMER_ALLOW_JIT": "0", "STABLEFOLD_ALLOW_JIT": "1"}, False),
        ({"TURBOPAIRFORMER_ALLOW_JIT": "1", "STABLEFOLD_ALLOW_JIT": "0"}, True),
        ({"STABLEFOLD_ALLOW_JIT": "0", "OPENFOLD3_TRIANGLE_ALLOW_JIT": "1"}, False),
        (
            {"TURBOPAIRFORMER_ALLOW_JIT": "0", "OPENFOLD3_TRIANGLE_ALLOW_JIT": "1"},
            False,
        ),
        ({"TURBOPAIRFORMER_ALLOW_JIT": "1", "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0"}, True),
    ],
)
def test_loader_and_policy_agree_on_jit(environment, expected, monkeypatch):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert runtime._development_jit_enabled() is expected
    assert policy.distributed_execution_flag(policy.ALLOW_JIT) is expected


def test_jit_uses_frozen_policy_and_drift_is_rejected(monkeypatch):
    monkeypatch.setenv("TURBOPAIRFORMER_ALLOW_JIT", "0")
    policy._set_distributed_strict(True, policy.execution_environment())
    monkeypatch.setenv("TURBOPAIRFORMER_ALLOW_JIT", "1")
    assert not runtime._development_jit_enabled()
    with pytest.raises(policy.DistributedContractError, match="environment changed"):
        policy.validate_distributed_environment()


def test_editable_receipt_leaves_source_placeholder_intact(tmp_path, monkeypatch):
    package = tmp_path / "turbopairformer"
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


@pytest.mark.parametrize("canonical", policy.EXECUTION_ENV_KEYS)
def test_execution_aliases_follow_canonical_precedence(canonical, monkeypatch):
    stablefold_alias, openfold_alias = policy.LEGACY_ENV_NAMES[canonical]
    monkeypatch.setenv(openfold_alias, "1")
    assert policy.distributed_execution_flag(canonical)
    monkeypatch.setenv(stablefold_alias, "0")
    assert not policy.distributed_execution_flag(canonical)
    monkeypatch.setenv(canonical, "1")
    assert policy.distributed_execution_flag(canonical)
    assert policy.distributed_execution_flag(stablefold_alias)
    assert policy.distributed_execution_flag(openfold_alias)
    policy._set_distributed_strict(True, policy.execution_environment())
    monkeypatch.setenv(openfold_alias, "0")
    policy.validate_distributed_environment()
    monkeypatch.setenv(canonical, "0")
    assert policy.distributed_execution_flag(canonical)
    with pytest.raises(policy.DistributedContractError, match="environment changed"):
        policy.validate_distributed_environment()


def test_public_import_registers_torch_ops_without_loading_cuda_extensions():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import turbopairformer
assert "turbopairformer._trimul" not in sys.modules
from turbopairformer import run_trimul
import torch
assert callable(run_trimul)
for name in ("_trimul_fwd_v1", "_trimul_bwd_v1"):
    schema = torch._C._dispatch_find_schema_or_throw(
        "turbopairformer::" + name, ""
    ).schema()
    assert schema.name == "turbopairformer::" + name
assert not any(
    name.startswith("turbopairformer._ext._stablefold_")
    for name in sys.modules
)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
