# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TurboPairFormer-only distributed kernel contract."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from openfold3.core.utils import distributed_contract
from openfold3.core.utils.distributed_contract import (
    DistributedContractError,
    TurboPairFormerKernelDDPCallback,
    _set_turbopairformer_distributed_strict,
    _validate_kernel_records,
    _write_receipt,
    required_turbopairformer_aot_modules,
    turbopairformer_aot_modules_from_config,
    turbopairformer_distributed_execution_flag,
    turbopairformer_distributed_strict_enabled,
    validate_turbopairformer_distributed_environment,
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
            )
        )
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config(), ()),
        (
            _config(train_attention=True),
            (
                "turbopairformer_forward",
                "turbopairformer_delta",
                "turbopairformer_dkdv",
                "turbopairformer_dq_dbias",
            ),
        ),
        (_config(train_trimul=True), ("turbopairformer_trimul",)),
        (
            _config(train_trimul=True, eval_attention=True),
            (
                "turbopairformer_forward",
                "turbopairformer_delta",
                "turbopairformer_dkdv",
                "turbopairformer_dq_dbias",
                "turbopairformer_trimul",
            ),
        ),
    ],
)
def test_turbopairformer_aot_module_discovery_uses_train_eval_union(config, expected):
    assert turbopairformer_aot_modules_from_config(config) == expected
    module = SimpleNamespace(config=config)
    assert required_turbopairformer_aot_modules(module) == expected


def test_turbopairformer_aot_module_discovery_handles_missing_settings():
    assert turbopairformer_aot_modules_from_config(None) == ()
    assert turbopairformer_aot_modules_from_config(SimpleNamespace()) == ()


def test_strict_execution_flags_are_immutable(monkeypatch):
    expected = {
        "OPENFOLD3_SR_BACKWARD": "1",
        "OPENFOLD3_SR_SHADOW": "1",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    _set_turbopairformer_distributed_strict(True, expected)
    try:
        assert turbopairformer_distributed_strict_enabled()
        assert turbopairformer_distributed_execution_flag("OPENFOLD3_SR_BACKWARD")
        assert not turbopairformer_distributed_execution_flag("OPENFOLD3_TRIANGLE_ALLOW_JIT")

        monkeypatch.setenv("OPENFOLD3_SR_SHADOW", "0")
        with pytest.raises(DistributedContractError, match="environment changed"):
            validate_turbopairformer_distributed_environment()

        # Route decisions stay pinned to the preflight snapshot.  The boundary
        # audit above is what terminates a job after an environment mutation.
        assert turbopairformer_distributed_execution_flag("OPENFOLD3_SR_SHADOW")
        with pytest.raises(KeyError, match="unknown TurboPairFormer execution flag"):
            turbopairformer_distributed_execution_flag("UNRELATED_FLAG")
    finally:
        _set_turbopairformer_distributed_strict(False)
    assert not turbopairformer_distributed_strict_enabled()


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
        "declared_aot_modules": ["turbopairformer_forward"],
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
        "required_aot_modules": ["turbopairformer_forward"],
        "aot_identity": {
            "manifest_sha256": "a" * 64,
            "modules": {"turbopairformer_forward": {"sha256": "c" * 64}},
        },
    }


@pytest.fixture
def collect_mocked_kernel_record(monkeypatch):
    """Exercise the actual per-rank gate without CUDA or a process group."""
    from openfold3.core.kernels.cuda import _aot_runtime

    monkeypatch.setattr(distributed_contract.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed_contract.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_contract.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed_contract.pl, "__version__", "2.6.1")
    monkeypatch.setattr(distributed_contract, "_runtime_identity", lambda: {})
    for name, value in _kernel_record(0)["execution_environment"].items():
        monkeypatch.setenv(name, value)

    aot_calls = []

    def aot_identity(modules):
        aot_calls.append(modules)
        return {"modules": list(modules)}

    monkeypatch.setattr(_aot_runtime, "distributed_aot_identity", aot_identity)
    cuda = distributed_contract.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)

    def collect(
        *,
        device_name="NVIDIA H200",
        capability=(9, 0),
        memory_gib=141,
        multiprocessors=132,
        device_uuid="GPU-00000000-0000-0000-0000-000000000001",
    ):
        properties = SimpleNamespace(
            name=device_name,
            total_memory=memory_gib * 1024**3,
            multi_processor_count=multiprocessors,
            uuid=device_uuid,
        )
        monkeypatch.setattr(cuda, "get_device_properties", lambda index: properties)
        monkeypatch.setattr(cuda, "get_device_capability", lambda index: capability)
        monkeypatch.setattr(cuda, "get_device_name", lambda index: device_name)
        module = SimpleNamespace(
            config=_config(train_attention=True, train_trimul=True)
        )
        trainer = SimpleNamespace(
            global_rank=0,
            local_rank=0,
            node_rank=0,
            world_size=2,
            strategy=distributed_contract.DDPStrategy(),
        )
        return distributed_contract._local_kernel_record(
            trainer,
            module,
            expected_world_size=2,
            expected_num_nodes=1,
            expected_devices_per_node=2,
            expected_aot_modules=required_turbopairformer_aot_modules(module),
        )

    return collect, aot_calls


@pytest.mark.parametrize(
    ("device_name", "memory_gib"),
    (("NVIDIA H100 80GB HBM3", 80), ("NVIDIA H200", 141), ("NVIDIA H200 NVL", 141)),
)
def test_local_kernel_record_accepts_full_h100_and_h200(
    collect_mocked_kernel_record, device_name, memory_gib
):
    collect, aot_calls = collect_mocked_kernel_record
    record = collect(device_name=device_name, memory_gib=memory_gib)

    assert record["ok"], record.get("error")
    assert record["device_name"] == device_name
    assert record["compute_capability"] == [9, 0]
    assert len(aot_calls) == 1
    assert set(aot_calls[0]) == {
        "turbopairformer_forward",
        "turbopairformer_delta",
        "turbopairformer_dkdv",
        "turbopairformer_dq_dbias",
        "turbopairformer_trimul",
    }


@pytest.mark.parametrize(
    "changes",
    (
        {"capability": (8, 0)},
        {"capability": (10, 0)},
        {"device_name": "NVIDIA H800"},
        {"device_name": "NVIDIA B200"},
        {"device_name": "NVIDIA H200 MIG 7g.141gb"},
        {"device_uuid": "MIG-00000000-0000-0000-0000-000000000001"},
        {"device_uuid": ""},
        {"memory_gib": 74},
        {"multiprocessors": 99},
    ),
)
def test_local_kernel_record_retains_device_rejections(
    collect_mocked_kernel_record, changes
):
    collect, aot_calls = collect_mocked_kernel_record
    record = collect(**changes)

    assert not record["ok"]
    assert "DistributedContractError" in record["error"]
    assert aot_calls == []


def test_kernel_rank_identity_accepts_h200_and_rejects_mixed_models():
    first, second = _kernel_record(0), _kernel_record(1)
    for record in (first, second):
        record.update(
            device_name="NVIDIA H200",
            device_total_memory_bytes=141 * 1024**3,
            device_multiprocessor_count=132,
        )
    _validate_kernel_records([first, second])

    with pytest.raises(DistributedContractError, match="device_name"):
        _validate_kernel_records([first, _kernel_record(1)])


def test_kernel_rank_identity_accepts_one_rank_per_gpu():
    _validate_kernel_records([_kernel_record(0), _kernel_record(1)])


def test_kernel_rank_identity_rejects_aot_drift():
    second = _kernel_record(1)
    second["aot_identity"] = {
        "manifest_sha256": "f" * 64,
        "modules": {"turbopairformer_forward": {"sha256": "e" * 64}},
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
    second["declared_aot_modules"] = ["turbopairformer_trimul"]
    with pytest.raises(DistributedContractError, match="declared_aot_modules"):
        _validate_kernel_records([_kernel_record(0), second])


def test_kernel_rank_identity_rejects_discovery_mismatch():
    first = _kernel_record(0)
    second = _kernel_record(1)
    first["declared_aot_modules"] = ["turbopairformer_trimul"]
    second["declared_aot_modules"] = ["turbopairformer_trimul"]
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
    receipt = tmp_path / "turbopairformer_kernel_ddp.json"
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
    callback = TurboPairFormerKernelDDPCallback(
        expected_world_size=2,
        expected_num_nodes=1,
        expected_devices_per_node=2,
        expected_aot_modules=("turbopairformer_forward",),
        receipt_path=tmp_path / "turbopairformer_kernel_ddp.json",
    )
    with pytest.raises(DistributedContractError, match="process group"):
        callback.on_fit_start(SimpleNamespace(), SimpleNamespace())


def test_training_runner_keeps_aqlab_ddp_and_adds_only_kernel_callback(
    tmp_path: Path,
):
    try:
        runner_module = importlib.import_module(
            "openfold3.entry_points.experiment_runner"
        )
    except ModuleNotFoundError as error:
        pytest.skip(f"full OpenFold3 training dependencies are unavailable: {error}")

    runner = object.__new__(runner_module.TrainingExperimentRunner)
    runner.seed = 42
    runner.model_config = _config(train_attention=True, train_trimul=True)
    runner.pl_trainer_args = SimpleNamespace(
        devices=2,
        num_nodes=1,
        mpi_plugin=False,
        precision="bf16-mixed",
        distributed_timeout=timedelta(minutes=30),
    )
    runner.deepspeed_config_path = None
    runner.logging_config = SimpleNamespace(log_lr=False, wandb_config=None)
    runner.checkpoint_config = None
    runner.memory_snapshot = SimpleNamespace(enabled=False)
    runner.experiment_config = SimpleNamespace(
        experiment_settings=SimpleNamespace(
            output_dir=tmp_path,
            log_dir=tmp_path / "logs",
        )
    )

    from pytorch_lightning.strategies import DDPStrategy

    assert isinstance(runner.strategy, DDPStrategy)
    kernel_callbacks = [
        callback
        for callback in runner.callbacks
        if isinstance(callback, TurboPairFormerKernelDDPCallback)
    ]
    assert len(kernel_callbacks) == 1
    assert kernel_callbacks[0].expected_world_size == 2
    assert kernel_callbacks[0].expected_aot_modules == (
        "turbopairformer_forward",
        "turbopairformer_delta",
        "turbopairformer_dkdv",
        "turbopairformer_dq_dbias",
        "turbopairformer_trimul",
    )
