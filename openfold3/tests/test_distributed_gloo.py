# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""CPU multi-process regressions for the distributed production contract."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

import openfold3.core.utils.distributed_contract as contract
import openfold3.core.utils.grad_manager as grad_manager_module
from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage
from openfold3.core.utils.grad_manager import PerSampleGradManager
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

_CONTRACT_SHA256 = "c" * 64


class _ReduceStrategy:
    @staticmethod
    def reduce(tensor: torch.Tensor, reduce_op: dist.ReduceOp) -> torch.Tensor:
        dist.all_reduce(tensor, op=reduce_op)
        return tensor


class _ToyModel(torch.nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.all_samples = torch.nn.Parameter(torch.tensor([float(rank + 1)]))
        self.sparse = torch.nn.Parameter(torch.tensor([float(10 + rank)]))
        self.never = torch.nn.Parameter(torch.tensor([float(20 + rank)]))
        self.oversized = torch.nn.Parameter(torch.arange(9, dtype=torch.float32))
        self.register_buffer("version_tensor", torch.tensor([1, 0, 0]))


def _init_gloo(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )


def _ema_and_grad_worker(rank: int, world_size: int, rendezvous: str) -> None:
    _init_gloo(rank, world_size, rendezvous)
    try:
        # EMA is deliberately constructed before DDP, matching OpenFold3.
        model = _ToyModel(rank)
        ema = ExponentialMovingAverage(model, decay=0.99)
        assert set(ema.params) == {"version_tensor"}

        ddp = DistributedDataParallel(model)
        ema.init_params(ddp.module)
        weight = ddp.module.all_samples.detach().clone()
        gathered_weights = [torch.empty_like(weight) for _ in range(world_size)]
        dist.all_gather(gathered_weights, weight)
        assert all(
            torch.equal(value, gathered_weights[0]) for value in gathered_weights
        )
        assert torch.equal(ema.params["all_samples"], gathered_weights[0])

        manager = PerSampleGradManager(
            gradient_clip_val=None,
            accumulate_grad_batches=1,
            log_grad_norm=False,
        )
        trainer = SimpleNamespace(
            strategy=_ReduceStrategy(),
            global_rank=rank,
            global_step=0,
            world_size=world_size,
        )
        manager.setup(ddp.module, trainer, logger=None)
        manager.accum_count = 1
        # Force a four-float bucket so ``oversized`` exercises sliced reduction.
        grad_manager_module._GRAD_REDUCE_BUCKET_BYTES = 4 * 4

        # all_samples is active on both ranks; sparse only on rank 0; never is
        # inactive globally. This exercises per-parameter active counts.
        manager.grad_accumulator["all_samples"].fill_(1.0 if rank == 0 else 3.0)
        manager.parameter_participation_counts["all_samples"] = 1
        manager.grad_accumulator["oversized"].fill_(
            2.0 if rank == 0 else 4.0
        )
        manager.parameter_participation_counts["oversized"] = 1
        if rank == 0:
            manager.grad_accumulator["sparse"].fill_(10.0)
            manager.parameter_participation_counts["sparse"] = 1
        manager.sync_and_average_grads()

        assert torch.equal(ddp.module.all_samples.grad, torch.tensor([2.0]))
        assert torch.equal(ddp.module.sparse.grad, torch.tensor([10.0]))
        assert torch.equal(ddp.module.never.grad, torch.tensor([0.0]))
        assert torch.equal(ddp.module.oversized.grad, torch.full((9,), 3.0))
    finally:
        dist.destroy_process_group()


class _LossHarness:
    _accumulate_distributed_step_losses = (
        OpenFold3AllAtom._accumulate_distributed_step_losses
    )
    _finalize_distributed_step_losses = (
        OpenFold3AllAtom._finalize_distributed_step_losses
    )
    _reset_distributed_loss_accumulator = (
        OpenFold3AllAtom._reset_distributed_loss_accumulator
    )

    def __init__(self, rank: int, world_size: int) -> None:
        self.trainer = SimpleNamespace(
            global_rank=rank,
            world_size=world_size,
            strategy=_ReduceStrategy(),
        )
        self._reset_distributed_loss_accumulator()


def _loss_worker(rank: int, world_size: int, rendezvous: str) -> None:
    _init_gloo(rank, world_size, rendezvous)
    try:
        harness = _LossHarness(rank, world_size)
        microbatches = (
            [
                {"loss": 1.0, "diffusion_loss": 2.0, "distogram_loss": 3.0},
                {"loss": 3.0, "diffusion_loss": 4.0},
            ]
            if rank == 0
            else [
                {"loss": 9.0, "distogram_loss": 11.0}
            ]
        )
        for values in microbatches:
            harness._accumulate_distributed_step_losses(
                {name: torch.tensor(value) for name, value in values.items()}
            )
        result = harness._finalize_distributed_step_losses()
        assert result is not None
        expected = {
            "loss": 13.0 / 3.0,
            "diffusion_loss": 6.0 / 2.0,
            "distogram_loss": 14.0 / 2.0,
        }
        for name, value in expected.items():
            assert torch.allclose(result[name], torch.tensor(value))
        assert harness._distributed_loss_sums == {}
        assert harness._distributed_loss_counts == {}

        packed = torch.stack([result[name] for name in sorted(result)])
        gathered = [torch.empty_like(packed) for _ in range(world_size)]
        dist.all_gather(gathered, packed)
        assert all(torch.equal(item, gathered[0]) for item in gathered)
    finally:
        dist.destroy_process_group()


def _rank_record(rank: int, *, fail: bool = False) -> dict:
    if fail:
        return {
            "ok": False,
            "global_rank": rank,
            "error": "AOTExtensionError: injected rank failure",
        }
    return {
        "ok": True,
        "global_rank": rank,
        "local_rank": rank,
        "node_rank": 0,
        "hostname": "node-a",
        "cuda_visible_devices": "0,1",
        "declared_world_size": 2,
        "declared_num_nodes": 1,
        "declared_devices_per_node": 2,
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
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        "runtime_identity": {
            "python": "3.14.4",
            "torch": "2.10.0",
            "torch_cuda": "12.9",
            "lightning": "2.6.1",
            "nccl": [2, 30, 4],
            "source_sha256": {"distributed_contract.py": "d" * 64},
        },
        "training_contract_sha256": _CONTRACT_SHA256,
        "checkpoint_identity": {"kind": "fresh", "sha256": None},
        "required_aot_modules": ["stablefold_forward"],
        "aot_identity": {"manifest_sha256": "a" * 64},
    }


def _callback_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    receipt_path: str,
    failure_mode: str | None,
) -> None:
    _init_gloo(rank, world_size, rendezvous)
    original_local_record = contract._local_rank_record
    original_get_backend = contract.dist.get_backend
    environment_names = (
        "OPENFOLD3_SR_BACKWARD",
        "OPENFOLD3_SR_SHADOW",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    )
    previous_environment = {
        name: os.environ.get(name) for name in environment_names
    }
    try:
        os.environ.update(
            {
                "OPENFOLD3_SR_BACKWARD": "1",
                "OPENFOLD3_SR_SHADOW": "1",
                "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )
        contract._local_rank_record = lambda *args, **kwargs: _rank_record(
            rank,
            fail=failure_mode == "rank1_aot" and rank == 1,
        )
        # The production callback requires NCCL. Keep CPU/Gloo collectives while
        # replacing only the identity query exercised by the gate.
        contract.dist.get_backend = lambda: "nccl"
        callback = contract.DistributedContractCallback(
            expected_world_size=world_size,
            expected_num_nodes=1,
            expected_devices_per_node=world_size,
            expected_aot_modules=(
                ("stablefold_trimul",)
                if failure_mode == "module_scope"
                else ("stablefold_forward",)
            ),
            training_contract_sha256=_CONTRACT_SHA256,
            checkpoint_path=None,
            receipt_path=Path(receipt_path),
        )
        trainer = SimpleNamespace(
            world_size=world_size,
            global_rank=rank,
            local_rank=rank,
            node_rank=0,
            strategy=DDPStrategy(),
        )

        caught_error = None
        try:
            callback._run_preflight(trainer, SimpleNamespace())
        except contract.DistributedContractError as error:
            caught_error = str(error)
        caught = torch.tensor(int(caught_error is not None))
        dist.all_reduce(caught, op=dist.ReduceOp.SUM)

        if failure_mode is not None:
            assert caught.item() == world_size
            assert not contract.stablefold_distributed_strict_enabled()
        else:
            assert caught.item() == 0, caught_error
            assert contract.stablefold_distributed_strict_enabled()
            callback.on_fit_end(trainer, SimpleNamespace())
            assert not contract.stablefold_distributed_strict_enabled()
            assert not callback._validated
            callback._run_preflight(trainer, SimpleNamespace())
            assert contract.stablefold_distributed_strict_enabled()
            callback.teardown(trainer, SimpleNamespace(), "fit")
            assert not contract.stablefold_distributed_strict_enabled()
            assert not callback._validated
    finally:
        contract._local_rank_record = original_local_record
        contract.dist.get_backend = original_get_backend
        contract._set_stablefold_distributed_strict(False)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_two_process_gloo_ddp_ema_and_active_count_reduction(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _ema_and_grad_worker,
        args=(2, str(tmp_path / "ema-grad-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_two_process_gloo_loss_accumulator_is_microbatch_weighted(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _loss_worker,
        args=(2, str(tmp_path / "loss-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_callback_seals_shared_receipt_and_resets_strict_mode(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    mp.spawn(
        _callback_worker,
        args=(2, str(tmp_path / "success-rendezvous"), str(receipt), None),
        nprocs=2,
        join=True,
    )

    payload = json.loads(receipt.read_text())
    assert payload["status"] == "PASS"
    assert payload["backend"] == "nccl"
    assert payload["expected_world_size"] == 2
    assert len(payload["ranks"]) == 2
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert receipt.with_name("receipt.json.sha256").read_text() == (
        f"{digest}  receipt.json\n"
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
@pytest.mark.parametrize(
    "failure_mode", ["receipt", "rank1_aot", "module_scope"]
)
def test_preflight_failure_reaches_all_ranks_without_hang(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    if failure_mode == "receipt":
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("block receipt directory creation")
        receipt = blocker / "receipt.json"
    else:
        receipt = tmp_path / "rank-failure-receipt.json"
    mp.spawn(
        _callback_worker,
        args=(
            2,
            str(tmp_path / f"{failure_mode}-rendezvous"),
            str(receipt),
            failure_mode,
        ),
        nprocs=2,
        join=True,
    )
