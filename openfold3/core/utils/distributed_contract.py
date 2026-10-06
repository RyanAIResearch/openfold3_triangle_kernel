# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""Kernel-only distributed preflight for the TurboPairFormer AOT backends.

The OpenFold3 training loop, DDP strategy, gradients, EMA, losses, logging, and
checkpoint behavior remain owned by AQLab.  This module only makes sure that
every DDP rank is about to execute the same attested TurboPairFormer kernels on a
distinct compatible GPU, then freezes the TurboPairFormer route policy.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy

_ATTENTION_AOT_MODULES = (
    "turbopairformer_forward",
    "turbopairformer_delta",
    "turbopairformer_dkdv",
    "turbopairformer_dq_dbias",
)
_TRIMUL_AOT_MODULES = ("turbopairformer_trimul",)
_EXECUTION_ENV_KEYS = (
    "OPENFOLD3_SR_BACKWARD",
    "OPENFOLD3_SR_SHADOW",
    "OPENFOLD3_TRIANGLE_ALLOW_JIT",
)
_DEFAULT_EXECUTION_FLAGS = {
    "OPENFOLD3_SR_BACKWARD": "1",
    "OPENFOLD3_SR_SHADOW": "1",
}

_STRICT_DISTRIBUTED_TURBOPAIRFORMER = False
_STRICT_EXECUTION_ENVIRONMENT: dict[str, str | None] | None = None


class DistributedContractError(RuntimeError):
    """Raised when TurboPairFormer kernel identities differ across DDP ranks."""


def _execution_flag_value(name: str) -> str | None:
    return os.environ.get(name, _DEFAULT_EXECUTION_FLAGS.get(name))


def _execution_environment() -> dict[str, str | None]:
    """Snapshot effective flags so unset SR defaults match explicit ones."""

    return {name: _execution_flag_value(name) for name in _EXECUTION_ENV_KEYS}


def turbopairformer_distributed_strict_enabled() -> bool:
    """Return whether unexpected TurboPairFormer model-level fallbacks are fatal."""

    return _STRICT_DISTRIBUTED_TURBOPAIRFORMER


def _set_turbopairformer_distributed_strict(
    enabled: bool,
    execution_environment: dict[str, str | None] | None = None,
) -> None:
    global _STRICT_DISTRIBUTED_TURBOPAIRFORMER, _STRICT_EXECUTION_ENVIRONMENT
    _STRICT_DISTRIBUTED_TURBOPAIRFORMER = bool(enabled)
    _STRICT_EXECUTION_ENVIRONMENT = (
        dict(execution_environment)
        if enabled and execution_environment is not None
        else None
    )


def validate_turbopairformer_distributed_environment() -> None:
    """Reject changes to TurboPairFormer execution flags after rank preflight."""

    if _STRICT_DISTRIBUTED_TURBOPAIRFORMER and (
        _STRICT_EXECUTION_ENVIRONMENT is None
        or _execution_environment() != _STRICT_EXECUTION_ENVIRONMENT
    ):
        raise DistributedContractError(
            "TurboPairFormer execution environment changed after DDP kernel preflight"
        )


def turbopairformer_distributed_execution_flag(name: str) -> bool:
    """Read a TurboPairFormer flag from the immutable DDP policy when armed."""

    if name not in _EXECUTION_ENV_KEYS:
        raise KeyError(f"unknown TurboPairFormer execution flag: {name}")
    if not _STRICT_DISTRIBUTED_TURBOPAIRFORMER:
        return _execution_flag_value(name) == "1"
    if _STRICT_EXECUTION_ENVIRONMENT is None:
        raise DistributedContractError(
            "strict TurboPairFormer execution policy was not initialized"
        )
    return _STRICT_EXECUTION_ENVIRONMENT[name] == "1"


def validate_turbopairformer_attention_policy() -> None:
    """Require single reconstruction whenever TurboPairFormer attention is selected."""

    if not turbopairformer_distributed_execution_flag("OPENFOLD3_SR_BACKWARD"):
        raise RuntimeError(
            "TurboPairFormer attention requires single-reconstruction backward; "
            "OPENFOLD3_SR_BACKWARD must be unset or 1. Setting it to 0 cannot "
            "select the old algorithm. Use historical source or low-level "
            "reference kernels for ablations."
        )


def _setting(settings: Any, name: str) -> bool:
    if hasattr(settings, "get"):
        return bool(settings.get(name, False))
    return bool(getattr(settings, name, False))


def turbopairformer_aot_modules_from_config(config: Any) -> tuple[str, ...]:
    """Return the TurboPairFormer AOT union selected by train and eval configs."""

    try:
        memory = config.settings.memory
    except AttributeError:
        return ()
    modes = tuple(
        mode
        for mode_name in ("train", "eval")
        if (mode := getattr(memory, mode_name, None)) is not None
    )
    needs_attention = any(
        _setting(mode, "use_stable_triangle_attention") for mode in modes
    )
    needs_trimul = any(
        _setting(mode, "use_stable_triangle_multiplicative") for mode in modes
    )
    return (_ATTENTION_AOT_MODULES if needs_attention else ()) + (
        _TRIMUL_AOT_MODULES if needs_trimul else ()
    )


def required_turbopairformer_aot_modules(
    pl_module: pl.LightningModule,
) -> tuple[str, ...]:
    """Return the TurboPairFormer AOT union selected by a Lightning module."""

    return turbopairformer_aot_modules_from_config(getattr(pl_module, "config", None))


def installed_distribution_identity() -> dict[str, Any]:
    """Verify the active non-editable wheel RECORD and return its identity."""

    try:
        distribution = importlib.metadata.distribution("openfold3")
    except importlib.metadata.PackageNotFoundError as error:
        raise DistributedContractError(
            "TurboPairFormer DDP requires an installed openfold3 wheel"
        ) from error

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as error:
            raise DistributedContractError(
                "installed openfold3 direct_url.json is invalid"
            ) from error
        if direct_url.get("dir_info", {}).get("editable") is True:
            raise DistributedContractError(
                "TurboPairFormer DDP rejects editable OpenFold3 installs"
            )

    files = distribution.files or ()
    runtime_members = [
        member
        for member in files
        if str(member) == "openfold3/core/utils/distributed_contract.py"
    ]
    if len(runtime_members) != 1:
        raise DistributedContractError(
            "installed wheel does not contain one distributed contract module"
        )
    installed_runtime = Path(distribution.locate_file(runtime_members[0])).resolve()
    if installed_runtime != Path(__file__).resolve():
        raise DistributedContractError(
            "current OpenFold3 import is not the verified installed wheel"
        )

    verified_members: list[tuple[str, bytes]] = []
    for member in sorted(files, key=str):
        relative = str(member)
        if not relative.startswith("openfold3/") or member.hash is None:
            continue
        path = Path(distribution.locate_file(member))
        if not path.is_file():
            raise DistributedContractError(
                f"installed wheel member is missing: {relative}"
            )
        payload = path.read_bytes()
        digest = hashlib.new(member.hash.mode, payload).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if encoded != member.hash.value:
            raise DistributedContractError(
                f"installed wheel RECORD hash mismatch: {relative}"
            )
        if member.size is not None and len(payload) != member.size:
            raise DistributedContractError(
                f"installed wheel RECORD size mismatch: {relative}"
            )
        verified_members.append((relative, digest))
    if not verified_members:
        raise DistributedContractError(
            "installed openfold3 RECORD has no hashed package members"
        )

    tree_hasher = hashlib.sha256()
    for relative, digest in verified_members:
        encoded_path = relative.encode()
        tree_hasher.update(len(encoded_path).to_bytes(8, "big"))
        tree_hasher.update(encoded_path)
        tree_hasher.update(digest)
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "verified_package_files": len(verified_members),
        "installed_package_tree_sha256": tree_hasher.hexdigest(),
        "runtime_from_verified_distribution": True,
    }


def _runtime_identity() -> dict[str, Any]:
    try:
        nccl_version = list(torch.cuda.nccl.version())
    except Exception:
        nccl_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightning": pl.__version__,
        "nccl": nccl_version,
        "installed_distribution": installed_distribution_identity(),
    }


def _local_kernel_record(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    *,
    expected_world_size: int,
    expected_num_nodes: int,
    expected_devices_per_node: int,
    expected_aot_modules: tuple[str, ...],
) -> dict[str, Any]:
    """Collect one rank's kernel identity without escaping before gather."""

    record: dict[str, Any] = {"ok": False}
    try:
        record.update(
            {
                "global_rank": int(trainer.global_rank),
                "local_rank": int(trainer.local_rank),
                "node_rank": int(trainer.node_rank),
                "hostname": socket.gethostname(),
                "expected_world_size": int(expected_world_size),
                "expected_num_nodes": int(expected_num_nodes),
                "expected_devices_per_node": int(expected_devices_per_node),
                "declared_aot_modules": list(expected_aot_modules),
                "trainer_world_size": int(trainer.world_size),
                "process_group_rank": int(dist.get_rank()),
                "process_group_world_size": int(dist.get_world_size()),
                "process_group_backend": str(dist.get_backend()).lower(),
                "strategy_class": type(trainer.strategy).__qualname__,
                "is_ddp_strategy": isinstance(trainer.strategy, DDPStrategy),
            }
        )
        if record["process_group_backend"] != "nccl":
            raise DistributedContractError(
                "TurboPairFormer kernel DDP preflight requires the NCCL backend"
            )
        if not record["is_ddp_strategy"]:
            raise DistributedContractError(
                "TurboPairFormer kernel DDP preflight requires AQLab DDPStrategy"
            )
        if pl.__version__ != "2.6.1":
            raise DistributedContractError(
                "TurboPairFormer kernel DDP preflight is qualified with "
                f"pytorch-lightning 2.6.1, got {pl.__version__}"
            )

        required_modules = required_turbopairformer_aot_modules(pl_module)
        if not required_modules:
            raise DistributedContractError(
                "TurboPairFormer kernel callback was installed with no selected module"
            )
        execution_environment = _execution_environment()
        if execution_environment["OPENFOLD3_TRIANGLE_ALLOW_JIT"] != "0":
            raise DistributedContractError(
                "TurboPairFormer kernel DDP requires OPENFOLD3_TRIANGLE_ALLOW_JIT=0"
            )
        if "turbopairformer_forward" in required_modules and (
            execution_environment["OPENFOLD3_SR_BACKWARD"] != "1"
            or execution_environment["OPENFOLD3_SR_SHADOW"] != "1"
        ):
            raise DistributedContractError(
                "TurboPairFormer Attention DDP requires OPENFOLD3_SR_BACKWARD=1 "
                "and OPENFOLD3_SR_SHADOW=1"
            )
        if not torch.cuda.is_available():
            raise DistributedContractError("TurboPairFormer kernel DDP requires CUDA")

        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        device_uuid = str(getattr(properties, "uuid", ""))
        if not device_uuid:
            raise DistributedContractError("CUDA device UUID is unavailable")
        if (
            capability != (9, 0)
            or not any(model in device_name.upper() for model in ("H100", "H200"))
            or "MIG" in device_name.upper()
            or device_uuid.upper().startswith("MIG")
            or int(properties.total_memory) < 75 * 1024**3
            or int(properties.multi_processor_count) < 100
        ):
            raise DistributedContractError(
                "TurboPairFormer kernel DDP requires full H100 or H200 GPUs "
                "with compute capability 9.0, at least 75 GiB and 100 SMs, "
                "and MIG disabled"
            )

        from openfold3.core.kernels.cuda._aot_runtime import (
            distributed_aot_identity,
        )

        record.update(
            {
                "device_index": int(device_index),
                "device_uuid": device_uuid,
                "device_name": device_name,
                "device_total_memory_bytes": int(properties.total_memory),
                "device_multiprocessor_count": int(properties.multi_processor_count),
                "compute_capability": list(capability),
                "execution_environment": execution_environment,
                "runtime_identity": _runtime_identity(),
                "required_aot_modules": list(required_modules),
                "aot_identity": distributed_aot_identity(required_modules),
                "ok": True,
            }
        )
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _validate_kernel_records(records: list[dict[str, Any]]) -> None:
    """Validate gathered rank records without inspecting training state."""

    if not records:
        raise DistributedContractError("no TurboPairFormer kernel rank records gathered")
    failures = {
        record.get("global_rank", index): record.get("error", "unknown error")
        for index, record in enumerate(records)
        if not record.get("ok")
    }
    if failures:
        raise DistributedContractError(
            f"TurboPairFormer kernel rank preflight failed: {failures}"
        )

    first = records[0]
    world_size = int(first["expected_world_size"])
    num_nodes = int(first["expected_num_nodes"])
    devices_per_node = int(first["expected_devices_per_node"])
    if len(records) != world_size:
        raise DistributedContractError(
            f"gathered {len(records)} kernel records, expected {world_size}"
        )
    for slot, record in enumerate(records):
        if (
            int(record["global_rank"]) != slot
            or int(record["process_group_rank"]) != slot
            or int(record["process_group_world_size"]) != world_size
            or int(record["trainer_world_size"]) != world_size
        ):
            raise DistributedContractError(
                "trainer rank metadata disagrees with the process group"
            )

    node_ranks = sorted({int(record["node_rank"]) for record in records})
    if node_ranks != list(range(num_nodes)):
        raise DistributedContractError(f"invalid node rank set: {node_ranks}")
    hostnames_by_node = {
        node_rank: {
            str(record["hostname"])
            for record in records
            if int(record["node_rank"]) == node_rank
        }
        for node_rank in node_ranks
    }
    if any(len(hostnames) != 1 for hostnames in hostnames_by_node.values()) or len(
        {next(iter(hostnames)) for hostnames in hostnames_by_node.values()}
    ) != len(node_ranks):
        raise DistributedContractError(
            f"node ranks do not map one-to-one to hostnames: {hostnames_by_node}"
        )
    for node_rank in node_ranks:
        local_ranks = sorted(
            int(record["local_rank"])
            for record in records
            if int(record["node_rank"]) == node_rank
        )
        if local_ranks != list(range(devices_per_node)):
            raise DistributedContractError(
                f"node {node_rank} has invalid local ranks: {local_ranks}"
            )

    invariant_fields = (
        "expected_world_size",
        "expected_num_nodes",
        "expected_devices_per_node",
        "declared_aot_modules",
        "trainer_world_size",
        "process_group_backend",
        "strategy_class",
        "is_ddp_strategy",
        "device_name",
        "device_total_memory_bytes",
        "device_multiprocessor_count",
        "compute_capability",
        "execution_environment",
        "runtime_identity",
        "required_aot_modules",
        "aot_identity",
    )
    for field in invariant_fields:
        if any(record[field] != first[field] for record in records[1:]):
            raise DistributedContractError(
                f"TurboPairFormer kernel ranks disagree on {field}"
            )
    if first["process_group_backend"] != "nccl" or not first["is_ddp_strategy"]:
        raise DistributedContractError(
            "TurboPairFormer kernel preflight requires AQLab DDPStrategy over NCCL"
        )
    if (
        not first["required_aot_modules"]
        or first["declared_aot_modules"] != first["required_aot_modules"]
    ):
        raise DistributedContractError(
            "TurboPairFormer module discovery differs from the runner declaration: "
            f"declared={first['declared_aot_modules']}, "
            f"required={first['required_aot_modules']}"
        )
    if len({record["device_uuid"] for record in records}) != world_size:
        raise DistributedContractError(
            "multiple DDP ranks are bound to the same physical CUDA device"
        )


def _write_receipt(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(f"{digest}  {path.name}\n")
    sidecar_tmp.replace(sidecar)
    return digest


def _receipt_readback_status(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        fields = path.with_name(path.name + ".sha256").read_text().split()
        if (
            actual != expected_sha256
            or len(fields) != 2
            or fields[0] != expected_sha256
            or fields[1] != path.name
        ):
            raise DistributedContractError(
                "TurboPairFormer kernel receipt readback hash mismatch"
            )
        return {"ok": True, "sha256": actual}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


class TurboPairFormerKernelDDPCallback(pl.Callback):
    """Preload and compare TurboPairFormer kernels before the first DDP forward."""

    def __init__(
        self,
        *,
        expected_world_size: int,
        expected_num_nodes: int,
        expected_devices_per_node: int,
        expected_aot_modules: tuple[str, ...],
        receipt_path: Path,
    ) -> None:
        super().__init__()
        self.expected_world_size = int(expected_world_size)
        self.expected_num_nodes = int(expected_num_nodes)
        self.expected_devices_per_node = int(expected_devices_per_node)
        self.expected_aot_modules = tuple(expected_aot_modules)
        self.receipt_path = Path(receipt_path)
        self._validated = False

    def _run_preflight(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedContractError(
                "DDP process group is not initialized for TurboPairFormer preflight"
            )

        process_group_world_size = int(dist.get_world_size())
        local_record = _local_kernel_record(
            trainer,
            pl_module,
            expected_world_size=self.expected_world_size,
            expected_num_nodes=self.expected_num_nodes,
            expected_devices_per_node=self.expected_devices_per_node,
            expected_aot_modules=self.expected_aot_modules,
        )
        records: list[dict[str, Any] | None] = [
            None for _ in range(process_group_world_size)
        ]
        dist.all_gather_object(records, local_record)
        gathered = [record for record in records if record is not None]
        _validate_kernel_records(gathered)

        receipt_payload: dict[str, Any] | None = None
        receipt_status: list[dict[str, Any] | None] = [None]
        if int(dist.get_rank()) == 0:
            try:
                compact_records = []
                for record in gathered:
                    compact = dict(record)
                    compact.pop("aot_identity", None)
                    compact_records.append(compact)
                receipt_payload = {
                    "schema_version": 1,
                    "status": "PENDING_ALL_RANK_READBACK",
                    "scope": "turbopairformer_kernel_ddp_only",
                    "backend": gathered[0]["process_group_backend"],
                    "expected_world_size": self.expected_world_size,
                    "expected_num_nodes": self.expected_num_nodes,
                    "expected_devices_per_node": self.expected_devices_per_node,
                    "aot_identity": gathered[0]["aot_identity"],
                    "ranks": compact_records,
                }
                digest = _write_receipt(self.receipt_path, receipt_payload)
                receipt_status[0] = {"ok": True, "sha256": digest}
            except Exception as error:
                receipt_status[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(receipt_status, src=0)
        if not receipt_status[0] or not receipt_status[0].get("ok"):
            raise DistributedContractError(
                f"failed to seal TurboPairFormer kernel DDP receipt: {receipt_status[0]}"
            )

        local_readback = _receipt_readback_status(
            self.receipt_path,
            str(receipt_status[0]["sha256"]),
        )
        readbacks: list[dict[str, Any] | None] = [
            None for _ in range(process_group_world_size)
        ]
        dist.all_gather_object(readbacks, local_readback)
        failures = {
            rank: status
            for rank, status in enumerate(readbacks)
            if not status or not status.get("ok")
        }
        if failures:
            raise DistributedContractError(
                f"TurboPairFormer kernel receipt is not readable on every rank: {failures}"
            )

        final_status: list[dict[str, Any] | None] = [None]
        if int(dist.get_rank()) == 0:
            try:
                if receipt_payload is None:
                    raise DistributedContractError(
                        "rank zero lost the TurboPairFormer kernel receipt payload"
                    )
                receipt_payload["status"] = "PASS"
                final_status[0] = {
                    "ok": True,
                    "sha256": _write_receipt(self.receipt_path, receipt_payload),
                }
            except Exception as error:
                final_status[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(final_status, src=0)
        if not final_status[0] or not final_status[0].get("ok"):
            raise DistributedContractError(
                f"failed to finalize TurboPairFormer kernel DDP receipt: {final_status[0]}"
            )

        final_readback = _receipt_readback_status(
            self.receipt_path,
            str(final_status[0]["sha256"]),
        )
        final_readbacks: list[dict[str, Any] | None] = [
            None for _ in range(process_group_world_size)
        ]
        dist.all_gather_object(final_readbacks, final_readback)
        final_failures = {
            rank: status
            for rank, status in enumerate(final_readbacks)
            if not status or not status.get("ok")
        }
        if final_failures:
            downgrade_status: list[dict[str, Any] | None] = [None]
            if int(dist.get_rank()) == 0:
                try:
                    if receipt_payload is None:
                        raise DistributedContractError(
                            "rank zero lost the TurboPairFormer kernel receipt payload"
                        )
                    receipt_payload["status"] = "FAIL_FINAL_READBACK"
                    receipt_payload["final_readback_failures"] = final_failures
                    downgrade_status[0] = {
                        "ok": True,
                        "sha256": _write_receipt(
                            self.receipt_path,
                            receipt_payload,
                        ),
                    }
                except Exception as error:
                    downgrade_status[0] = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
            dist.broadcast_object_list(downgrade_status, src=0)
            raise DistributedContractError(
                "final TurboPairFormer kernel receipt is not readable on every rank: "
                f"{final_failures}; downgrade={downgrade_status[0]}"
            )

        _set_turbopairformer_distributed_strict(
            True,
            gathered[0]["execution_environment"],
        )
        self._validated = True

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        if stage == "fit":
            self._run_preflight(trainer, pl_module)

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedContractError(
                "DDP process group is not initialized for TurboPairFormer fit"
            )
        self._audit_environment()

    def _audit_environment(self) -> None:
        local_error = None
        try:
            if not self._validated or not turbopairformer_distributed_strict_enabled():
                raise DistributedContractError(
                    "TurboPairFormer kernel strict policy is not armed"
                )
            validate_turbopairformer_distributed_environment()
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if str(dist.get_backend()).lower() == "nccl"
            else torch.device("cpu")
        )
        status = torch.tensor(
            int(local_error is None), device=device, dtype=torch.int32
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if int(status.item()) == 0:
            errors: list[str | None] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(errors, local_error)
            raise DistributedContractError(
                f"TurboPairFormer execution flag drift across DDP ranks: {errors}"
            )

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._audit_environment()

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._audit_environment()

    def _reset(self) -> None:
        _set_turbopairformer_distributed_strict(False)
        self._validated = False

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._reset()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._reset()

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        del trainer, pl_module, stage
        self._reset()


__all__ = [
    "DistributedContractError",
    "TurboPairFormerKernelDDPCallback",
    "installed_distribution_identity",
    "required_turbopairformer_aot_modules",
    "turbopairformer_aot_modules_from_config",
    "turbopairformer_distributed_execution_flag",
    "turbopairformer_distributed_strict_enabled",
    "validate_turbopairformer_distributed_environment",
]
