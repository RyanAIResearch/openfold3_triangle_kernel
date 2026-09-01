# Copyright 2026 AlQuraishi Laboratory
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed distributed launch contract for StableFold training."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import socket
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy

_ATTENTION_AOT_MODULES = (
    "stablefold_forward",
    "stablefold_delta",
    "stablefold_dkdv",
    "stablefold_dq_dbias",
)
_TRIMUL_AOT_MODULES = ("stablefold_trimul",)
_EXECUTION_ENV_KEYS = (
    "OPENFOLD3_SR_BACKWARD",
    "OPENFOLD3_SR_SHADOW",
    "OPENFOLD3_TRIANGLE_ALLOW_JIT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)
_LAUNCH_WORLD_SIZE_KEYS = (
    "WORLD_SIZE",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
)
_MPI_WORLD_SIZE_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE")

_STRICT_DISTRIBUTED_STABLEFOLD = False
_STRICT_EXECUTION_ENVIRONMENT: dict[str, str | None] | None = None


class DistributedContractError(RuntimeError):
    """Raised before training when distributed ranks do not share one contract."""


def stablefold_distributed_strict_enabled() -> bool:
    """Return whether unexpected StableFold fallbacks must fail the job."""

    return _STRICT_DISTRIBUTED_STABLEFOLD


def validate_stablefold_distributed_environment() -> None:
    """Check the process environment once at a distributed batch boundary."""

    if _STRICT_DISTRIBUTED_STABLEFOLD and (
        _STRICT_EXECUTION_ENVIRONMENT is None
        or _execution_environment() != _STRICT_EXECUTION_ENVIRONMENT
    ):
        raise DistributedContractError(
            "StableFold execution environment changed after distributed preflight"
        )


def stablefold_distributed_execution_flag(name: str) -> bool:
    """Read a flag from the immutable distributed StableFold policy."""

    if name not in _EXECUTION_ENV_KEYS:
        raise KeyError(f"unknown StableFold execution flag: {name}")
    if _STRICT_DISTRIBUTED_STABLEFOLD:
        if _STRICT_EXECUTION_ENVIRONMENT is None:
            raise DistributedContractError(
                "strict StableFold execution policy was not initialized"
            )
        return _STRICT_EXECUTION_ENVIRONMENT[name] == "1"
    return os.environ.get(name) == "1"


def _set_stablefold_distributed_strict(
    enabled: bool,
    execution_environment: dict[str, str | None] | None = None,
) -> None:
    global _STRICT_DISTRIBUTED_STABLEFOLD, _STRICT_EXECUTION_ENVIRONMENT
    _STRICT_DISTRIBUTED_STABLEFOLD = enabled
    _STRICT_EXECUTION_ENVIRONMENT = (
        dict(execution_environment) if enabled and execution_environment else None
    )


def _positive_int_env(environ: Mapping[str, str], name: str) -> int | None:
    value = environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise DistributedContractError(
            f"distributed launcher variable {name} must be an integer"
        ) from error
    if parsed < 1:
        raise DistributedContractError(
            f"distributed launcher variable {name} must be positive"
        )
    return parsed


def distributed_checkpoint_identity(checkpoint: str | None) -> dict[str, Any]:
    """Seal one explicit checkpoint artifact for StableFold DDP v1."""

    if checkpoint is None:
        return {"kind": "fresh", "sha256": None, "size_bytes": 0}
    if checkpoint in {"last", "hpc", "registry"}:
        raise ValueError(
            "Distributed production v1 requires an explicit immutable checkpoint "
            "file; Lightning checkpoint aliases are not qualified."
        )
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise ValueError(
            "Distributed production v1 supports explicit single-file checkpoints "
            f"only, got {checkpoint_path}"
        )
    before = checkpoint_path.stat()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = checkpoint_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"checkpoint changed while hashing: {checkpoint_path}")
    return {
        "kind": "file",
        "path": str(checkpoint_path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(after.st_size),
    }


def validate_stablefold_distributed_checkpoint_config(
    *,
    checkpoint: str | None,
    manual_checkpoint_loading: bool,
    preemption_safe_resume: bool,
) -> None:
    """Reject checkpoint modes whose concrete bytes are unknown before fit."""

    if manual_checkpoint_loading:
        raise ValueError(
            "StableFold distributed production v1 does not support manual "
            "checkpoint loading; use an explicit Lightning checkpoint file."
        )
    if preemption_safe_resume:
        raise ValueError(
            "StableFold distributed production v1 does not support "
            "preemption_safe_resume because it resolves to the mutable "
            "Lightning 'last' alias. Use an explicit immutable checkpoint file."
        )
    if checkpoint in {"last", "hpc", "registry"}:
        raise ValueError(
            "StableFold distributed production v1 requires an explicit "
            "immutable checkpoint file; Lightning aliases are not qualified."
        )
    if checkpoint is not None and Path(checkpoint).is_dir():
        raise ValueError(
            "StableFold distributed production v1 does not support DeepSpeed "
            "checkpoint directories."
        )


def install_rank_file_handler(log_path: Path, level: int) -> None:
    """Add an idempotent rank-local log file without replacing other handlers."""

    resolved = log_path.resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and (
            Path(handler.baseFilename).resolve() == resolved
        ):
            handler.setLevel(level)
            return
    handler = logging.FileHandler(resolved, mode="w")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(handler)
    if root_logger.level > level:
        root_logger.setLevel(level)


def validate_multinode_launcher(
    *,
    num_nodes: int,
    devices_per_node: int,
    mpi_plugin: bool,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Reject a multi-node config that was not started by a real launcher."""

    if num_nodes < 1 or devices_per_node < 1:
        raise DistributedContractError("num_nodes and devices must both be positive")
    env = os.environ if environ is None else environ
    expected_world_size = num_nodes * devices_per_node
    observed = {
        key: value
        for key in _LAUNCH_WORLD_SIZE_KEYS
        if (value := _positive_int_env(env, key)) is not None
    }
    if num_nodes == 1 and not mpi_plugin and not observed:
        # Lightning may spawn a single node's local workers itself.
        return
    if not observed:
        raise DistributedContractError(
            "multi-node training requires torchrun, srun/SLURM, or mpirun; "
            "setting num_nodes alone cannot start remote workers"
        )
    # Scheduler variables can describe an allocation or the number of
    # torchrun launchers rather than the child process group. Prefer the
    # active launcher family instead of treating every ambient variable as
    # authoritative.
    if "WORLD_SIZE" in observed:
        authoritative = {"WORLD_SIZE": observed["WORLD_SIZE"]}
    elif mpi_plugin:
        authoritative = {
            key: observed[key]
            for key in _MPI_WORLD_SIZE_KEYS
            if key in observed
        }
    elif "OMPI_COMM_WORLD_SIZE" in observed:
        authoritative = {
            "OMPI_COMM_WORLD_SIZE": observed["OMPI_COMM_WORLD_SIZE"]
        }
    elif "PMI_SIZE" in observed:
        authoritative = {"PMI_SIZE": observed["PMI_SIZE"]}
    else:
        authoritative = {"SLURM_NTASKS": observed["SLURM_NTASKS"]}
    if any(value != expected_world_size for value in authoritative.values()):
        raise DistributedContractError(
            "launcher world size does not match devices*num_nodes: "
            f"expected {expected_world_size}, observed {authoritative}"
        )
    if mpi_plugin and not any(key in observed for key in _MPI_WORLD_SIZE_KEYS):
        raise DistributedContractError(
            "mpi_plugin=true requires an mpirun/PMI launcher environment"
        )


def _setting(settings: Any, name: str) -> bool:
    if hasattr(settings, "get"):
        return bool(settings.get(name, False))
    return bool(getattr(settings, name, False))


def stablefold_aot_modules_from_config(config: Any) -> tuple[str, ...]:
    """Return the AOT module union selected by fit-time train and eval modes."""
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
    return (
        (*_ATTENTION_AOT_MODULES,) if needs_attention else ()
    ) + ((*_TRIMUL_AOT_MODULES,) if needs_trimul else ())


def required_stablefold_aot_modules(pl_module: pl.LightningModule) -> tuple[str, ...]:
    """Return the AOT module union selected by a Lightning module."""

    return stablefold_aot_modules_from_config(getattr(pl_module, "config", None))


def _execution_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _EXECUTION_ENV_KEYS}


def _installed_distribution_identity() -> dict[str, Any]:
    """Verify the installed wheel RECORD and return its immutable identity."""

    try:
        distribution = importlib.metadata.distribution("openfold3")
    except importlib.metadata.PackageNotFoundError as error:
        raise DistributedContractError(
            "StableFold distributed production requires an installed openfold3 wheel"
        ) from error
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        if direct_url.get("dir_info", {}).get("editable") is True:
            raise DistributedContractError(
                "StableFold distributed production rejects editable installs"
            )
    files = distribution.files or ()
    record_members = [path for path in files if str(path).endswith(".dist-info/RECORD")]
    if len(record_members) != 1:
        raise DistributedContractError("installed openfold3 wheel has no unique RECORD")
    record_path = Path(distribution.locate_file(record_members[0]))
    if not record_path.is_file():
        raise DistributedContractError("installed openfold3 RECORD is missing")
    runtime_members = [
        path
        for path in files
        if str(path) == "openfold3/core/utils/distributed_contract.py"
    ]
    if len(runtime_members) != 1:
        raise DistributedContractError(
            "installed wheel does not contain the distributed contract module"
        )
    installed_runtime_path = Path(
        distribution.locate_file(runtime_members[0])
    ).resolve()
    current_runtime_path = Path(__file__).resolve()
    if installed_runtime_path != current_runtime_path:
        raise DistributedContractError(
            "current openfold3 import is not the verified installed wheel: "
            f"imported={current_runtime_path}, installed={installed_runtime_path}"
        )

    tree_hasher = hashlib.sha256()
    verified_files = 0
    for member in sorted(files, key=str):
        relative = str(member)
        if not relative.startswith("openfold3/") or member.hash is None:
            continue
        path = Path(distribution.locate_file(member))
        if not path.is_file():
            raise DistributedContractError(
                f"installed wheel member is missing: {relative}"
            )
        digest = hashlib.new(member.hash.mode, path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if encoded != member.hash.value:
            raise DistributedContractError(
                f"installed wheel RECORD hash mismatch: {relative}"
            )
        if member.size is not None and path.stat().st_size != member.size:
            raise DistributedContractError(
                f"installed wheel RECORD size mismatch: {relative}"
            )
        relative_bytes = relative.encode()
        tree_hasher.update(len(relative_bytes).to_bytes(8, "big"))
        tree_hasher.update(relative_bytes)
        tree_hasher.update(digest)
        verified_files += 1
    if verified_files < 1:
        raise DistributedContractError(
            "installed openfold3 RECORD has no hashed package members"
        )
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "verified_package_files": verified_files,
        "installed_package_tree_sha256": tree_hasher.hexdigest(),
        "runtime_from_verified_distribution": True,
    }


def _cuda_driver_version() -> int:
    driver = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    result = int(driver.cuDriverGetVersion(ctypes.byref(version)))
    if result != 0:
        raise DistributedContractError(
            f"cuDriverGetVersion failed with CUDA driver status {result}"
        )
    return int(version.value)


def _runtime_identity() -> dict[str, Any]:
    """Return the software identity that must match on every worker."""

    try:
        nccl_version = list(torch.cuda.nccl.version())
    except Exception:  # handled as an invariant value by the rank gather
        nccl_version = None
    package_root = Path(__file__).resolve().parents[2]
    source_paths = (
        "core/data/framework/data_module.py",
        "core/data/framework/stochastic_sampler_dataset.py",
        "core/kernels/cuda/_aot_runtime.py",
        "core/kernels/cuda/stablefold_triangle_attention.py",
        "core/model/layers/triangular_multiplicative_update.py",
        "core/model/primitives/attention.py",
        "core/runners/model_runner.py",
        "core/utils/distributed_contract.py",
        "core/utils/exponential_moving_average.py",
        "core/utils/grad_manager.py",
        "entry_points/experiment_runner.py",
        "projects/of3_all_atom/runner.py",
    )
    source_sha256 = {}
    for relative_path in source_paths:
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise DistributedContractError(
                f"distributed runtime source is missing: {source_path}"
            )
        source_sha256[relative_path] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    release_suffixes = {
        ".cu",
        ".cuh",
        ".cpp",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".yaml",
        ".yml",
    }
    release_hasher = hashlib.sha256()
    release_file_count = 0
    for source_path in sorted(package_root.rglob("*")):
        if not source_path.is_file() or source_path.suffix not in release_suffixes:
            continue
        relative = source_path.relative_to(package_root)
        if "__pycache__" in relative.parts or relative.parts[0] == "tests":
            continue
        relative_path = relative.as_posix()
        encoded_path = relative_path.encode()
        payload = source_path.read_bytes()
        release_hasher.update(len(encoded_path).to_bytes(8, "big"))
        release_hasher.update(encoded_path)
        release_hasher.update(len(payload).to_bytes(8, "big"))
        release_hasher.update(payload)
        release_file_count += 1
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_driver": _cuda_driver_version(),
        "cudnn": torch.backends.cudnn.version(),
        "lightning": pl.__version__,
        "nccl": nccl_version,
        "source_sha256": source_sha256,
        "release_source_file_count": release_file_count,
        "release_source_tree_sha256": release_hasher.hexdigest(),
        "installed_distribution": _installed_distribution_identity(),
    }


def _local_rank_record(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    *,
    declared_world_size: int,
    declared_num_nodes: int,
    declared_devices_per_node: int,
    training_contract_sha256: str,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"ok": False}
    try:
        record.update(
            {
                "global_rank": int(trainer.global_rank),
                "local_rank": int(trainer.local_rank),
                "node_rank": int(trainer.node_rank),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "declared_world_size": int(declared_world_size),
                "declared_num_nodes": int(declared_num_nodes),
                "declared_devices_per_node": int(declared_devices_per_node),
                "trainer_world_size": int(trainer.world_size),
                "training_contract_sha256": training_contract_sha256,
            }
        )
        record["checkpoint_identity"] = distributed_checkpoint_identity(
            checkpoint_path
        )
        record["process_group_rank"] = int(dist.get_rank())
        record["process_group_world_size"] = int(dist.get_world_size())
        record["process_group_backend"] = str(dist.get_backend()).lower()
        if record["process_group_backend"] != "nccl":
            raise DistributedContractError(
                "StableFold distributed production v1 requires the NCCL backend"
            )
        record["strategy_class"] = type(trainer.strategy).__qualname__
        from pytorch_lightning.strategies import DDPStrategy

        record["is_ddp_strategy"] = isinstance(trainer.strategy, DDPStrategy)
        required_modules = required_stablefold_aot_modules(pl_module)
        record["required_aot_modules"] = list(required_modules)
        record["execution_environment"] = _execution_environment()
        record["runtime_identity"] = _runtime_identity()
        if pl.__version__ != "2.6.1":
            raise DistributedContractError(
                "StableFold distributed v1 is qualified only with "
                f"pytorch-lightning 2.6.1, got {pl.__version__}"
            )
        if not torch.cuda.is_available():
            raise DistributedContractError(
                "distributed OpenFold3 production training requires CUDA"
            )
        if record["execution_environment"]["TORCH_NCCL_ASYNC_ERROR_HANDLING"] != "1":
            raise DistributedContractError(
                "distributed production requires "
                "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 before process-group startup"
            )
        device_index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        properties = torch.cuda.get_device_properties(device_index)
        device_uuid = str(getattr(properties, "uuid", ""))
        if not device_uuid:
            raise DistributedContractError(
                "CUDA device UUID is unavailable; physical rank binding cannot "
                "be proven"
            )
        record.update(
            {
                "device_index": int(device_index),
                "device_name": device_name,
                "device_uuid": device_uuid,
                "device_total_memory_bytes": int(properties.total_memory),
                "device_multiprocessor_count": int(properties.multi_processor_count),
                "pci_address": {
                    "domain": int(properties.pci_domain_id),
                    "bus": int(properties.pci_bus_id),
                    "device": int(properties.pci_device_id),
                },
                "compute_capability": list(capability),
            }
        )
        if required_modules:
            if (
                capability != (9, 0)
                or "H100" not in device_name
                or "MIG" in device_name.upper()
                or device_uuid.upper().startswith("MIG")
                or int(properties.total_memory) < 75 * 1024**3
                or int(properties.multi_processor_count) < 100
            ):
                raise DistributedContractError(
                    "StableFold distributed v1 requires full homogeneous H100 "
                    "80GB-class sm_90 GPUs with MIG disabled"
                )
            env = record["execution_environment"]
            if env["OPENFOLD3_TRIANGLE_ALLOW_JIT"] != "0":
                raise DistributedContractError(
                    "StableFold distributed production requires "
                    "OPENFOLD3_TRIANGLE_ALLOW_JIT=0"
                )
            if "stablefold_forward" in required_modules and (
                env["OPENFOLD3_SR_BACKWARD"] != "1"
                or env["OPENFOLD3_SR_SHADOW"] != "1"
            ):
                raise DistributedContractError(
                    "StableFold Attention distributed production requires "
                    "OPENFOLD3_SR_BACKWARD=1 and OPENFOLD3_SR_SHADOW=1"
                )
            from openfold3.core.kernels.cuda._aot_runtime import (
                distributed_aot_identity,
            )

            record["aot_identity"] = distributed_aot_identity(required_modules)
        else:
            record["aot_identity"] = None
        record["ok"] = True
    except Exception as error:  # every rank must still enter the status gather
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _validate_rank_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise DistributedContractError("no distributed rank records were gathered")
    failures = {
        record.get("global_rank"): record.get("error", "unknown local failure")
        for record in records
        if not record.get("ok")
    }
    if failures:
        raise DistributedContractError(f"distributed rank preflight failed: {failures}")

    first = records[0]
    expected_world_size = int(first["declared_world_size"])
    expected_num_nodes = int(first["declared_num_nodes"])
    expected_devices_per_node = int(first["declared_devices_per_node"])
    if len(records) != expected_world_size:
        raise DistributedContractError(
            f"gathered {len(records)} rank records, expected {expected_world_size}"
        )

    ranks = sorted(int(record["global_rank"]) for record in records)
    if ranks != list(range(expected_world_size)):
        raise DistributedContractError(f"invalid global rank set: {ranks}")
    if any(
        int(record["process_group_rank"]) != int(record["global_rank"])
        or int(record["process_group_world_size"]) != expected_world_size
        or int(record["trainer_world_size"]) != expected_world_size
        for record in records
    ):
        raise DistributedContractError(
            "trainer rank metadata disagrees with the process group"
        )
    node_ranks = sorted({int(record["node_rank"]) for record in records})
    if node_ranks != list(range(expected_num_nodes)):
        raise DistributedContractError(f"invalid node rank set: {node_ranks}")
    for node_rank in node_ranks:
        local_ranks = sorted(
            int(record["local_rank"])
            for record in records
            if int(record["node_rank"]) == node_rank
        )
        if local_ranks != list(range(expected_devices_per_node)):
            raise DistributedContractError(
                f"node {node_rank} has invalid local ranks: {local_ranks}"
            )

    invariant_fields = (
        "declared_world_size",
        "declared_num_nodes",
        "declared_devices_per_node",
        "trainer_world_size",
        "process_group_backend",
        "device_name",
        "device_total_memory_bytes",
        "device_multiprocessor_count",
        "compute_capability",
        "execution_environment",
        "runtime_identity",
        "training_contract_sha256",
        "checkpoint_identity",
        "required_aot_modules",
        "aot_identity",
        "strategy_class",
        "is_ddp_strategy",
    )
    for field in invariant_fields:
        if any(record[field] != first[field] for record in records[1:]):
            raise DistributedContractError(
                f"distributed ranks disagree on {field}"
            )
    if first["process_group_backend"] != "nccl" or not first["is_ddp_strategy"]:
        raise DistributedContractError(
            "StableFold distributed production v1 requires DDPStrategy over NCCL"
        )

    node_rank_to_hosts = {
        node_rank: {
            record["hostname"]
            for record in records
            if int(record["node_rank"]) == node_rank
        }
        for node_rank in node_ranks
    }
    if any(len(hosts) != 1 for hosts in node_rank_to_hosts.values()):
        raise DistributedContractError(
            f"each node rank must map to one hostname: {node_rank_to_hosts}"
        )
    host_to_node_ranks: dict[str, set[int]] = {}
    for record in records:
        host_to_node_ranks.setdefault(record["hostname"], set()).add(
            int(record["node_rank"])
        )
    if any(len(ranks_for_host) != 1 for ranks_for_host in host_to_node_ranks.values()):
        raise DistributedContractError(
            f"each hostname must map to one node rank: {host_to_node_ranks}"
        )

    bindings = {record["device_uuid"] for record in records}
    if len(bindings) != expected_world_size:
        raise DistributedContractError(
            "multiple ranks are bound to the same visible CUDA device"
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
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = path.with_name(path.name + ".sha256").read_text().split()
        sidecar_sha256 = sidecar[0] if sidecar else ""
        if actual_sha256 != expected_sha256 or sidecar_sha256 != expected_sha256:
            raise DistributedContractError(
                "distributed receipt readback hash mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}, "
                f"sidecar={sidecar_sha256}"
            )
        return {"ok": True, "sha256": actual_sha256}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


class DistributedContractCallback(pl.Callback):
    """Validate every rank and preload StableFold AOT before the first step."""

    def __init__(
        self,
        *,
        expected_world_size: int,
        expected_num_nodes: int,
        expected_devices_per_node: int,
        expected_aot_modules: tuple[str, ...],
        training_contract_sha256: str,
        checkpoint_path: str | None,
        receipt_path: Path,
    ) -> None:
        super().__init__()
        self.expected_world_size = expected_world_size
        self.expected_num_nodes = expected_num_nodes
        self.expected_devices_per_node = expected_devices_per_node
        self.expected_aot_modules = tuple(expected_aot_modules)
        self.training_contract_sha256 = training_contract_sha256
        self.checkpoint_path = checkpoint_path
        self.receipt_path = Path(receipt_path)
        self._validated = False

    def _run_preflight(
        self, trainer: Any, pl_module: pl.LightningModule | None
    ) -> None:
        """Run immediately after process-group initialization."""

        if self._validated:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedContractError(
                "distributed process group is not initialized at fit start"
            )
        process_group_world_size = int(dist.get_world_size())
        local_record = _local_rank_record(
            trainer,
            pl_module,
            declared_world_size=self.expected_world_size,
            declared_num_nodes=self.expected_num_nodes,
            declared_devices_per_node=self.expected_devices_per_node,
            training_contract_sha256=self.training_contract_sha256,
            checkpoint_path=self.checkpoint_path,
        )
        records: list[dict[str, Any] | None] = [
            None for _ in range(process_group_world_size)
        ]
        dist.all_gather_object(records, local_record)
        gathered = [record for record in records if record is not None]
        _validate_rank_records(gathered)

        required_modules = tuple(gathered[0]["required_aot_modules"])
        if not required_modules or required_modules != self.expected_aot_modules:
            raise DistributedContractError(
                "StableFold module discovery disagrees with the predeclared "
                f"contract: expected={self.expected_aot_modules}, "
                f"observed={required_modules}"
            )
        receipt_status: list[dict[str, Any] | None] = [None]
        if int(trainer.global_rank) == 0:
            try:
                common_aot = gathered[0]["aot_identity"]
                rank_records = []
                for record in gathered:
                    compact = dict(record)
                    compact.pop("aot_identity", None)
                    rank_records.append(compact)
                digest = _write_receipt(
                    self.receipt_path,
                    {
                        "schema_version": 1,
                        "status": "PASS",
                        "backend": dist.get_backend(),
                        "expected_world_size": self.expected_world_size,
                        "expected_num_nodes": self.expected_num_nodes,
                        "expected_devices_per_node": self.expected_devices_per_node,
                        "strict_stablefold_routes": bool(required_modules),
                        "training_contract_sha256": self.training_contract_sha256,
                        "checkpoint_identity": gathered[0]["checkpoint_identity"],
                        "aot_identity": common_aot,
                        "ranks": rank_records,
                    },
                )
                receipt_status[0] = {"ok": True, "sha256": digest}
            except Exception as error:
                receipt_status[0] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        # A rank-zero filesystem error must reach every worker before any rank
        # enters a barrier or the first training collective.
        dist.broadcast_object_list(receipt_status, src=0)
        if not receipt_status[0] or not receipt_status[0].get("ok"):
            raise DistributedContractError(
                "failed to seal distributed preflight receipt: "
                f"{receipt_status[0]}"
            )
        readback = _receipt_readback_status(
            self.receipt_path,
            str(receipt_status[0]["sha256"]),
        )
        readbacks: list[dict[str, Any] | None] = [None for _ in gathered]
        dist.all_gather_object(readbacks, readback)
        readback_failures = {
            rank: status
            for rank, status in enumerate(readbacks)
            if not status or not status.get("ok")
        }
        if readback_failures:
            raise DistributedContractError(
                "distributed receipt is not identically readable on every rank: "
                f"{readback_failures}"
            )
        _set_stablefold_distributed_strict(
            True,
            gathered[0]["execution_environment"],
        )
        dist.barrier()
        self._validated = True

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Confirm strategy-level preflight ran before Lightning setup hooks."""

        del trainer, pl_module, stage
        if not self._validated:
            raise DistributedContractError(
                "StableFold strategy preflight did not complete before setup hooks"
            )

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del trainer, pl_module
        if not self._validated:
            raise DistributedContractError(
                "StableFold distributed preflight did not complete before DDP setup"
            )

    def _synchronized_environment_audit(self) -> None:
        local_error = None
        if not self._validated:
            local_error = "DistributedContractError: callback is not validated"
        elif not stablefold_distributed_strict_enabled():
            local_error = "DistributedContractError: strict mode was disabled"
        else:
            try:
                validate_stablefold_distributed_environment()
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
        device = torch.device("cuda", torch.cuda.current_device())
        status = torch.tensor(
            int(local_error is None), device=device, dtype=torch.int32
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if int(status.item()) == 0:
            errors: list[str | None] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(errors, local_error)
            raise DistributedContractError(
                f"distributed StableFold environment drift: {errors}"
            )

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._synchronized_environment_audit()

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self._synchronized_environment_audit()

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del trainer, pl_module
        _set_stablefold_distributed_strict(False)
        self._validated = False

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        _set_stablefold_distributed_strict(False)
        self._validated = False

    def teardown(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        del trainer, pl_module, stage
        _set_stablefold_distributed_strict(False)
        self._validated = False


class StableFoldDDPStrategy(DDPStrategy):
    """DDP strategy that seals StableFold before any Lightning setup hook."""

    def __init__(
        self,
        *,
        contract_callback: DistributedContractCallback,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stablefold_contract_callback = contract_callback

    def setup_environment(self) -> None:
        super().setup_environment()
        context = SimpleNamespace(
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            node_rank=self.node_rank,
            world_size=self.world_size,
            strategy=self,
        )
        self._stablefold_contract_callback._run_preflight(
            context,
            self.lightning_module,
        )

__all__ = [
    "DistributedContractCallback",
    "DistributedContractError",
    "StableFoldDDPStrategy",
    "distributed_checkpoint_identity",
    "install_rank_file_handler",
    "required_stablefold_aot_modules",
    "stablefold_aot_modules_from_config",
    "stablefold_distributed_execution_flag",
    "stablefold_distributed_strict_enabled",
    "validate_stablefold_distributed_checkpoint_config",
    "validate_stablefold_distributed_environment",
    "validate_multinode_launcher",
]
