# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the StableFold-only distributed kernel contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stablefold.distributed import (
    DistributedContractError,
    StableFoldKernelDDPCallback,
    _set_stablefold_distributed_strict,
    _validate_kernel_records,
    _write_receipt,
    required_stablefold_aot_modules,
    stablefold_aot_modules_from_config,
    stablefold_distributed_execution_flag,
    stablefold_distributed_strict_enabled,
    validate_stablefold_distributed_environment,
)


def _config(
    *,
    train_attention: bool = False,
    train_trimul: bool = False,
    eval_attention: bool = False,
    eval_trimul: bool = False,
):
    return SimpleNamespace(
        settings=SimpleNamespace(
            debug=SimpleNamespace(log_iteration_time=False),
            memory=SimpleNamespace(
                train={
                    "use_stable_triangle_attention": train_attention,
                    "use_stable_triangle_multiplicative": train_trimul,
                },
                eval={
                    "use_stable_triangle_attention": eval_attention,
                    "use_stable_triangle_multiplicative": eval_trimul,
                },
            ),
        )
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config(), ()),
        (
            _config(train_attention=True),
            (
                "stablefold_forward",
                "stablefold_delta",
                "stablefold_dkdv",
                "stablefold_dq_dbias",
            ),
        ),
        (_config(train_trimul=True), ("stablefold_trimul",)),
        (
            _config(train_trimul=True, eval_attention=True),
            (
                "stablefold_forward",
                "stablefold_delta",
                "stablefold_dkdv",
                "stablefold_dq_dbias",
                "stablefold_trimul",
            ),
        ),
    ],
)
def test_stablefold_aot_module_discovery_uses_train_eval_union(config, expected):
    assert stablefold_aot_modules_from_config(config) == expected
    module = SimpleNamespace(config=config)
    assert required_stablefold_aot_modules(module) == expected


def test_stablefold_aot_module_discovery_handles_missing_settings():
    assert stablefold_aot_modules_from_config(None) == ()
    assert stablefold_aot_modules_from_config(SimpleNamespace()) == ()


def test_strict_execution_flags_are_immutable(monkeypatch):
    expected = {
        "OPENFOLD3_SR_BACKWARD": "1",
        "OPENFOLD3_SR_SHADOW": "1",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    _set_stablefold_distributed_strict(True, expected)
    try:
        assert stablefold_distributed_strict_enabled()
        assert stablefold_distributed_execution_flag("OPENFOLD3_SR_BACKWARD")
        assert not stablefold_distributed_execution_flag("OPENFOLD3_TRIANGLE_ALLOW_JIT")

        monkeypatch.setenv("OPENFOLD3_SR_SHADOW", "0")
        with pytest.raises(DistributedContractError, match="environment changed"):
            validate_stablefold_distributed_environment()

        # Route decisions stay pinned to the preflight snapshot.  The boundary
        # audit above is what terminates a job after an environment mutation.
        assert stablefold_distributed_execution_flag("OPENFOLD3_SR_SHADOW")
        with pytest.raises(KeyError, match="unknown StableFold execution flag"):
            stablefold_distributed_execution_flag("UNRELATED_FLAG")
    finally:
        _set_stablefold_distributed_strict(False)
    assert not stablefold_distributed_strict_enabled()


def _kernel_record(rank: int) -> dict:
    return {
        "ok": True,
        "global_rank": rank,
        "local_rank": rank,
        "node_rank": 0,
        "hostname": "node-a",
        "cuda_visible_devices": "0,1",
        "expected_world_size": 2,
        "expected_num_nodes": 1,
        "expected_devices_per_node": 2,
        "declared_aot_modules": ["stablefold_forward"],
        "trainer_world_size": 2,
        "process_group_rank": rank,
        "process_group_world_size": 2,
        "process_group_backend": "nccl",
        "strategy_class": "DDPStrategy",
        "is_ddp_strategy": True,
        "device_index": rank,
        "device_uuid": f"GPU-00000000-0000-0000-0000-{rank:012d}",
        "pci_address": {"domain": 0, "bus": rank, "device": 0},
        "device_name": "NVIDIA H100 80GB HBM3",
        "device_total_memory_bytes": 80 * 1024**3,
        "device_multiprocessor_count": 120,
        "compute_capability": [9, 0],
        "execution_environment": {
            "OPENFOLD3_SR_BACKWARD": "1",
            "OPENFOLD3_SR_SHADOW": "1",
            "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
        },
        "runtime_identity": {
            "python": "3.14.4",
            "torch": "2.10.0",
            "torch_cuda": "12.9",
            "lightning": "2.6.1",
            "nccl": [2, 28, 9],
            "installed_distribution": {
                "name": "openfold3",
                "version": "3.0.0",
                "installed_package_tree_sha256": "b" * 64,
            },
        },
        "required_aot_modules": ["stablefold_forward"],
        "aot_identity": {
            "manifest_sha256": "a" * 64,
            "modules": {"stablefold_forward": {"sha256": "c" * 64}},
        },
    }


def test_kernel_rank_identity_accepts_one_rank_per_gpu():
    _validate_kernel_records([_kernel_record(0), _kernel_record(1)])


def test_kernel_rank_identity_rejects_aot_drift():
    second = _kernel_record(1)
    second["aot_identity"] = {
        "manifest_sha256": "f" * 64,
        "modules": {"stablefold_forward": {"sha256": "e" * 64}},
    }
    with pytest.raises(DistributedContractError, match="aot_identity"):
        _validate_kernel_records([_kernel_record(0), second])


def test_kernel_rank_identity_rejects_package_drift():
    second = _kernel_record(1)
    second["runtime_identity"]["installed_distribution"] = {
        **second["runtime_identity"]["installed_distribution"],
        "installed_package_tree_sha256": "f" * 64,
    }
    with pytest.raises(DistributedContractError, match="runtime_identity"):
        _validate_kernel_records([_kernel_record(0), second])


def test_kernel_rank_identity_rejects_duplicate_physical_gpu():
    second = _kernel_record(1)
    second["device_uuid"] = _kernel_record(0)["device_uuid"]
    with pytest.raises(DistributedContractError, match="same physical CUDA device"):
        _validate_kernel_records([_kernel_record(0), second])


def test_kernel_rank_identity_rejects_declared_module_drift():
    second = _kernel_record(1)
    second["declared_aot_modules"] = ["stablefold_trimul"]
    with pytest.raises(DistributedContractError, match="declared_aot_modules"):
        _validate_kernel_records([_kernel_record(0), second])


def test_kernel_rank_identity_rejects_discovery_mismatch():
    first = _kernel_record(0)
    second = _kernel_record(1)
    first["declared_aot_modules"] = ["stablefold_trimul"]
    second["declared_aot_modules"] = ["stablefold_trimul"]
    with pytest.raises(DistributedContractError, match="module discovery"):
        _validate_kernel_records([first, second])


def test_kernel_rank_identity_rejects_node_rank_hostname_alias():
    first = _kernel_record(0)
    second = _kernel_record(1)
    first.update(
        {
            "expected_num_nodes": 2,
            "expected_devices_per_node": 1,
            "node_rank": 0,
            "local_rank": 0,
        }
    )
    second.update(
        {
            "expected_num_nodes": 2,
            "expected_devices_per_node": 1,
            "node_rank": 1,
            "local_rank": 0,
        }
    )
    with pytest.raises(DistributedContractError, match="hostnames"):
        _validate_kernel_records([first, second])


def test_kernel_rank_identity_reports_local_aot_failure():
    failure = {
        "ok": False,
        "global_rank": 1,
        "error": "AOTExtensionError: injected rank failure",
    }
    with pytest.raises(DistributedContractError, match="injected rank failure"):
        _validate_kernel_records([_kernel_record(0), failure])


def test_kernel_receipt_has_content_hash_sidecar(tmp_path: Path):
    receipt = tmp_path / "stablefold_kernel_ddp.json"
    digest = _write_receipt(receipt, {"schema_version": 1, "status": "PASS"})

    assert digest == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert json.loads(receipt.read_text()) == {
        "schema_version": 1,
        "status": "PASS",
    }
    assert receipt.with_name(receipt.name + ".sha256").read_text() == (
        f"{digest}  {receipt.name}\n"
    )


def test_fit_start_rejects_missing_kernel_preflight(tmp_path: Path):
    callback = StableFoldKernelDDPCallback(
        expected_world_size=2,
        expected_num_nodes=1,
        expected_devices_per_node=2,
        expected_aot_modules=("stablefold_forward",),
        receipt_path=tmp_path / "stablefold_kernel_ddp.json",
    )
    with pytest.raises(DistributedContractError, match="process group"):
        callback.on_fit_start(SimpleNamespace(), SimpleNamespace())
