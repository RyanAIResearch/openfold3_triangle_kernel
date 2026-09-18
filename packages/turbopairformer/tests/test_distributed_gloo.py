# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""CPU multi-process tests for the TurboPairFormer kernel DDP preflight."""

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

import turbopairformer.distributed as contract


def _init_gloo(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )


def _kernel_record(rank: int, failure_mode: str | None) -> dict:
    if failure_mode == "rank1_aot_load" and rank == 1:
        return {
            "ok": False,
            "global_rank": rank,
            "error": "AOTExtensionError: injected rank-local load failure",
        }
    aot_digest = "f" * 64 if failure_mode == "aot_mismatch" and rank == 1 else "a" * 64
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
        "declared_aot_modules": [
            "stablefold_trimul"
            if failure_mode == "declared_mismatch" and rank == 1
            else "stablefold_forward"
        ],
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
            "manifest_sha256": aot_digest,
            "modules": {"stablefold_forward": {"sha256": aot_digest}},
        },
    }


def _callback_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    receipt_path: str,
    failure_mode: str | None,
) -> None:
    _init_gloo(rank, world_size, rendezvous)
    original_local_record = contract._local_kernel_record
    original_get_backend = contract.dist.get_backend
    original_receipt_readback = contract._receipt_readback_status
    environment_names = (
        "OPENFOLD3_SR_BACKWARD",
        "OPENFOLD3_SR_SHADOW",
        "OPENFOLD3_TRIANGLE_ALLOW_JIT",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}
    try:
        os.environ.update(
            {
                "OPENFOLD3_SR_BACKWARD": "1",
                "OPENFOLD3_SR_SHADOW": "1",
                "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
            }
        )
        contract._local_kernel_record = lambda *args, **kwargs: _kernel_record(
            rank, failure_mode
        )
        # Keep CPU/Gloo collectives but present the qualified NCCL backend to
        # the kernel identity gate.
        contract.dist.get_backend = lambda *args, **kwargs: "nccl"
        if (
            failure_mode
            in (
                "rank1_receipt_readback",
                "rank1_final_receipt_readback",
            )
            and rank == 1
        ):
            readback_calls = 0

            def injected_readback(*args, **kwargs):
                nonlocal readback_calls
                readback_calls += 1
                fail_on_call = 1 if failure_mode == "rank1_receipt_readback" else 2
                if readback_calls == fail_on_call:
                    return {
                        "ok": False,
                        "error": "injected rank-local receipt readback failure",
                    }
                return original_receipt_readback(*args, **kwargs)

            contract._receipt_readback_status = injected_readback
        callback = contract.TurboPairFormerKernelDDPCallback(
            expected_world_size=world_size,
            expected_num_nodes=1,
            expected_devices_per_node=world_size,
            expected_aot_modules=("stablefold_forward",),
            receipt_path=Path(receipt_path),
        )
        trainer = SimpleNamespace(
            world_size=world_size,
            global_rank=rank,
            local_rank=rank,
            node_rank=0,
            strategy=DDPStrategy(),
        )

        error = None
        try:
            callback.setup(trainer, SimpleNamespace(), "fit")
            contract.dist.get_backend = original_get_backend
            callback.on_fit_start(trainer, SimpleNamespace())
        except contract.DistributedContractError as caught:
            error = str(caught)
        finally:
            # Restore the real Gloo identity before the assertion collective.
            contract.dist.get_backend = original_get_backend

        caught_count = torch.tensor(int(error is not None))
        dist.all_reduce(caught_count, op=dist.ReduceOp.SUM)
        if failure_mode is None:
            assert caught_count.item() == 0, error
            assert contract.distributed_strict_enabled()
            callback.on_fit_end(trainer, SimpleNamespace())
            assert not contract.distributed_strict_enabled()
        else:
            assert caught_count.item() == world_size
            assert not contract.distributed_strict_enabled()
    finally:
        contract._local_kernel_record = original_local_record
        contract.dist.get_backend = original_get_backend
        contract._receipt_readback_status = original_receipt_readback
        contract._set_distributed_strict(False)
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
def test_two_process_kernel_preflight_seals_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "turbopairformer_kernel_ddp.json"
    mp.spawn(
        _callback_worker,
        args=(2, str(tmp_path / "success-rendezvous"), str(receipt), None),
        nprocs=2,
        join=True,
    )

    payload = json.loads(receipt.read_text())
    assert payload["status"] == "PASS"
    assert payload["expected_world_size"] == 2
    assert len(payload["ranks"]) == 2
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert receipt.with_name(receipt.name + ".sha256").read_text() == (
        f"{digest}  {receipt.name}\n"
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
@pytest.mark.parametrize(
    "failure_mode",
    [
        "rank1_aot_load",
        "aot_mismatch",
        "declared_mismatch",
        "rank1_receipt_readback",
        "rank1_final_receipt_readback",
    ],
)
def test_two_process_kernel_failure_reaches_every_rank_without_hang(
    tmp_path: Path, failure_mode: str
) -> None:
    receipt = tmp_path / "turbopairformer_kernel_ddp.json"
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
    if failure_mode == "rank1_receipt_readback":
        assert json.loads(receipt.read_text())["status"] == (
            "PENDING_ALL_RANK_READBACK"
        )
    elif failure_mode == "rank1_final_receipt_readback":
        assert json.loads(receipt.read_text())["status"] == "FAIL_FINAL_READBACK"


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_two_process_receipt_failure_reaches_every_rank_without_hang(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block receipt directory creation")
    mp.spawn(
        _callback_worker,
        args=(
            2,
            str(tmp_path / "receipt-failure-rendezvous"),
            str(blocker / "turbopairformer_kernel_ddp.json"),
            "receipt_failure",
        ),
        nprocs=2,
        join=True,
    )
