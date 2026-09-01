# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.distributed_contract import (
    DistributedContractCallback,
    DistributedContractError,
    _receipt_readback_status,
    _set_stablefold_distributed_strict,
    _validate_rank_records,
    distributed_checkpoint_identity,
    install_rank_file_handler,
    required_stablefold_aot_modules,
    stablefold_distributed_execution_flag,
    stablefold_distributed_strict_enabled,
    validate_multinode_launcher,
    validate_stablefold_distributed_checkpoint_config,
    validate_stablefold_distributed_environment,
)
from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage
from openfold3.core.utils.grad_manager import _bucket_tensors_by_numel


def test_multinode_launcher_requires_real_world_size():
    with pytest.raises(DistributedContractError, match="requires torchrun"):
        validate_multinode_launcher(
            num_nodes=2,
            devices_per_node=4,
            mpi_plugin=False,
            environ={},
        )

    with pytest.raises(DistributedContractError, match="does not match"):
        validate_multinode_launcher(
            num_nodes=2,
            devices_per_node=4,
            mpi_plugin=False,
            environ={"WORLD_SIZE": "4"},
        )

    validate_multinode_launcher(
        num_nodes=2,
        devices_per_node=4,
        mpi_plugin=False,
        environ={"WORLD_SIZE": "8", "SLURM_NTASKS": "8"},
    )

    # torchrun's child process group is authoritative when an ambient Slurm
    # allocation describes only one launcher task per node.
    validate_multinode_launcher(
        num_nodes=2,
        devices_per_node=4,
        mpi_plugin=False,
        environ={"WORLD_SIZE": "8", "SLURM_NTASKS": "2"},
    )


def test_single_node_lightning_spawn_needs_no_external_launcher():
    validate_multinode_launcher(
        num_nodes=1,
        devices_per_node=8,
        mpi_plugin=False,
        environ={},
    )

    with pytest.raises(DistributedContractError, match="does not match"):
        validate_multinode_launcher(
            num_nodes=1,
            devices_per_node=8,
            mpi_plugin=False,
            environ={"WORLD_SIZE": "4"},
        )


def test_distributed_checkpoint_identity_seals_explicit_file(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"sealed checkpoint")
    identity = distributed_checkpoint_identity(str(checkpoint))
    assert identity == {
        "kind": "file",
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
    }
    assert distributed_checkpoint_identity(None)["kind"] == "fresh"
    for alias in ("last", "hpc", "registry"):
        with pytest.raises(ValueError, match="aliases are not qualified"):
            distributed_checkpoint_identity(alias)


def test_stablefold_checkpoint_config_rejects_mutable_resume_modes(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"sealed checkpoint")
    validate_stablefold_distributed_checkpoint_config(
        checkpoint=str(checkpoint),
        manual_checkpoint_loading=False,
        preemption_safe_resume=False,
    )
    validate_stablefold_distributed_checkpoint_config(
        checkpoint=None,
        manual_checkpoint_loading=False,
        preemption_safe_resume=False,
    )
    with pytest.raises(ValueError, match="manual checkpoint"):
        validate_stablefold_distributed_checkpoint_config(
            checkpoint=str(checkpoint),
            manual_checkpoint_loading=True,
            preemption_safe_resume=False,
        )
    with pytest.raises(ValueError, match="preemption_safe_resume"):
        validate_stablefold_distributed_checkpoint_config(
            checkpoint=str(checkpoint),
            manual_checkpoint_loading=False,
            preemption_safe_resume=True,
        )
    with pytest.raises(ValueError, match="aliases are not qualified"):
        validate_stablefold_distributed_checkpoint_config(
            checkpoint="last",
            manual_checkpoint_loading=False,
            preemption_safe_resume=False,
        )
    checkpoint_dir = tmp_path / "deepspeed-checkpoint"
    checkpoint_dir.mkdir()
    with pytest.raises(ValueError, match="DeepSpeed checkpoint directories"):
        validate_stablefold_distributed_checkpoint_config(
            checkpoint=str(checkpoint_dir),
            manual_checkpoint_loading=False,
            preemption_safe_resume=False,
        )


def test_rank_file_handler_is_idempotent_when_root_is_configured(tmp_path):
    log_path = tmp_path / "console_logs.rank3.log"
    root_logger = logging.getLogger()
    before = set(root_logger.handlers)
    try:
        install_rank_file_handler(log_path, logging.INFO)
        install_rank_file_handler(log_path, logging.INFO)
        added = [handler for handler in root_logger.handlers if handler not in before]
        assert len(added) == 1
        logging.getLogger(__name__).warning("rank-local receipt marker")
        added[0].flush()
        assert "rank-local receipt marker" in log_path.read_text()
    finally:
        for handler in list(root_logger.handlers):
            if handler not in before:
                root_logger.removeHandler(handler)
                handler.close()


def test_mpi_plugin_requires_mpi_or_pmi_environment():
    with pytest.raises(DistributedContractError, match="mpi_plugin=true"):
        validate_multinode_launcher(
            num_nodes=2,
            devices_per_node=2,
            mpi_plugin=True,
            environ={"WORLD_SIZE": "4"},
        )

    validate_multinode_launcher(
        num_nodes=2,
        devices_per_node=2,
        mpi_plugin=True,
        environ={"OMPI_COMM_WORLD_SIZE": "4"},
    )


def _dummy_module(
    *,
    attention: bool,
    trimul: bool,
    eval_attention: bool = False,
    eval_trimul: bool = False,
):
    train_memory = {
        "use_stable_triangle_attention": attention,
        "use_stable_triangle_multiplicative": trimul,
    }
    eval_memory = {
        "use_stable_triangle_attention": eval_attention,
        "use_stable_triangle_multiplicative": eval_trimul,
    }
    config = SimpleNamespace(
        settings=SimpleNamespace(
            memory=SimpleNamespace(train=train_memory, eval=eval_memory),
        )
    )
    return SimpleNamespace(config=config)


@pytest.mark.parametrize(
    ("attention", "trimul", "expected"),
    [
        (False, False, ()),
        (True, False, (
            "stablefold_forward",
            "stablefold_delta",
            "stablefold_dkdv",
            "stablefold_dq_dbias",
        )),
        (False, True, ("stablefold_trimul",)),
    ],
)
def test_required_stablefold_aot_modules(attention, trimul, expected):
    assert required_stablefold_aot_modules(
        _dummy_module(attention=attention, trimul=trimul)
    ) == expected


def test_required_stablefold_aot_modules_includes_eval_union():
    assert required_stablefold_aot_modules(
        _dummy_module(
            attention=False,
            trimul=True,
            eval_attention=True,
        )
    ) == (
        "stablefold_forward",
        "stablefold_delta",
        "stablefold_dkdv",
        "stablefold_dq_dbias",
        "stablefold_trimul",
    )


def _rank_record(global_rank: int, *, env_value: str = "1"):
    return {
        "ok": True,
        "global_rank": global_rank,
        "local_rank": global_rank,
        "node_rank": 0,
        "hostname": "node-a",
        "cuda_visible_devices": None,
        "declared_world_size": 2,
        "declared_num_nodes": 1,
        "declared_devices_per_node": 2,
        "trainer_world_size": 2,
        "process_group_rank": global_rank,
        "process_group_world_size": 2,
        "process_group_backend": "nccl",
        "strategy_class": "DDPStrategy",
        "is_ddp_strategy": True,
        "device_index": global_rank,
        "device_name": "NVIDIA H100 80GB HBM3",
        "device_total_memory_bytes": 80 * 1024**3,
        "device_multiprocessor_count": 120,
        "compute_capability": [9, 0],
        "execution_environment": {
            "OPENFOLD3_SR_BACKWARD": env_value,
            "OPENFOLD3_SR_SHADOW": "1",
            "OPENFOLD3_TRIANGLE_ALLOW_JIT": None,
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        "runtime_identity": {"source_sha256": {"x": "a" * 64}},
        "training_contract_sha256": "b" * 64,
        "checkpoint_identity": {"kind": "fresh", "sha256": None},
        "required_aot_modules": ["stablefold_forward"],
        "aot_identity": {"manifest_sha256": "a" * 64},
        "device_uuid": f"uuid-{global_rank}",
    }


def test_rank_record_contract_accepts_one_process_per_gpu():
    _validate_rank_records(
        [_rank_record(0), _rank_record(1)],
    )


def test_rank_record_contract_rejects_environment_drift():
    with pytest.raises(DistributedContractError, match="execution_environment"):
        _validate_rank_records(
            [_rank_record(0), _rank_record(1, env_value="0")],
        )


def test_rank_record_contract_rejects_checkpoint_content_drift():
    second = _rank_record(1)
    second["checkpoint_identity"] = {
        "kind": "file",
        "sha256": "f" * 64,
        "size_bytes": 123,
    }
    with pytest.raises(DistributedContractError, match="checkpoint_identity"):
        _validate_rank_records(
            [_rank_record(0), second],
        )


def test_rank_record_contract_rejects_duplicate_physical_gpu():
    duplicate = _rank_record(1)
    duplicate["device_uuid"] = _rank_record(0)["device_uuid"]
    with pytest.raises(DistributedContractError, match="same visible CUDA device"):
        _validate_rank_records(
            [_rank_record(0), duplicate],
        )


def test_rank_record_contract_rejects_one_host_as_two_nodes():
    second = _rank_record(1)
    second["node_rank"] = 1
    second["local_rank"] = 0
    second["declared_num_nodes"] = 2
    second["declared_devices_per_node"] = 1
    first = _rank_record(0)
    first["declared_num_nodes"] = 2
    first["declared_devices_per_node"] = 1
    with pytest.raises(DistributedContractError, match="hostname"):
        _validate_rank_records([first, second])


def test_gradient_buckets_preserve_parameter_order_and_bound_small_tensors():
    tensors = [torch.zeros(3), torch.zeros(4), torch.zeros(5)]
    buckets = _bucket_tensors_by_numel(tensors, max_float32_bytes=7 * 4)
    assert buckets == [tensors[:2], tensors[2:]]


def test_ema_initializes_full_state_only_after_ddp_sync_point():
    model = torch.nn.Linear(2, 2)
    model.register_buffer("version_tensor", torch.tensor([1, 0, 0]))
    ema = ExponentialMovingAverage(model, decay=0.99)
    assert set(ema.params) == {"version_tensor"}

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.init_params(model)
    assert set(ema.params) == set(model.state_dict())
    assert torch.equal(ema.params["weight"], model.weight)


def test_distributed_resume_requires_and_restores_synced_generator_state():
    source = np.random.default_rng(123)
    source.integers(0, 100, size=7)
    checkpoint = {"synced_generator_state": source.bit_generator.state}

    target = SimpleNamespace(
        model=SimpleNamespace(synced_generator=np.random.default_rng(999)),
        _require_distributed_resume_state=True,
    )
    ModelRunner._restore_synced_generator_state(target, checkpoint)
    assert np.array_equal(
        target.model.synced_generator.integers(0, 100, size=8),
        source.integers(0, 100, size=8),
    )

    with pytest.raises(RuntimeError, match="synced_generator_state"):
        ModelRunner._restore_synced_generator_state(target, {})
    target._require_distributed_resume_state = False
    ModelRunner._restore_synced_generator_state(target, {})


def test_strict_distributed_mode_rejects_environment_drift(monkeypatch):
    expected = {
        "OPENFOLD3_SR_BACKWARD": "1",
        "OPENFOLD3_SR_SHADOW": "1",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    _set_stablefold_distributed_strict(True, expected)
    try:
        assert stablefold_distributed_strict_enabled()
        assert stablefold_distributed_execution_flag("OPENFOLD3_SR_SHADOW")
        monkeypatch.setenv("OPENFOLD3_SR_SHADOW", "0")
        with pytest.raises(DistributedContractError, match="environment changed"):
            validate_stablefold_distributed_environment()
        # Route decisions remain pinned to the preflight value until the
        # batch-boundary audit fails the job.
        assert stablefold_distributed_execution_flag("OPENFOLD3_SR_SHADOW")
    finally:
        _set_stablefold_distributed_strict(False)
    assert not stablefold_distributed_strict_enabled()


def test_receipt_readback_requires_payload_and_sidecar_match(tmp_path):
    receipt = tmp_path / "distributed_contract.json"
    receipt.write_text('{"status":"PASS"}\n')
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.with_name(receipt.name + ".sha256").write_text(
        f"{digest}  {receipt.name}\n"
    )
    assert _receipt_readback_status(receipt, digest)["ok"]

    receipt.write_text('{"status":"CHANGED"}\n')
    assert not _receipt_readback_status(receipt, digest)["ok"]


def test_fit_start_rejects_a_late_or_missing_preflight(tmp_path):
    callback = DistributedContractCallback(
        expected_world_size=2,
        expected_num_nodes=1,
        expected_devices_per_node=2,
        expected_aot_modules=("stablefold_forward",),
        training_contract_sha256="a" * 64,
        checkpoint_path=None,
        receipt_path=tmp_path / "receipt.json",
    )
    with pytest.raises(DistributedContractError, match="before DDP setup"):
        callback.on_fit_start(SimpleNamespace(), SimpleNamespace())
