#!/usr/bin/env python3
"""Benchmark one OpenFold3 training backend with matched data and seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

try:
    from tools.benchmark_full_crop import install_full_crop_collator
    from tools.select_training_backend import BACKEND_FLAGS, configure_training_backend
except ModuleNotFoundError:
    from benchmark_full_crop import install_full_crop_collator  # type: ignore[no-redef]
    from select_training_backend import (  # type: ignore[no-redef]
        BACKEND_FLAGS,
        configure_training_backend,
    )


def configure_benchmark(
    config: dict[str, Any],
    *,
    backend: str,
    output_dir: Path,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
    num_workers: int,
    devices: int,
    num_nodes: int = 1,
) -> dict[str, Any]:
    """Make a validation-free, from-scratch benchmark config for one topology.

    ``devices`` is the GPU count per node and ``num_nodes`` the node count;
    ``epoch_len`` scales with the world size so every topology runs the same
    number of optimizer steps at a per-GPU batch size of one.
    """

    total_steps = warmup_steps + measure_steps
    configure_training_backend(config, backend=backend)

    config.setdefault("experiment_settings", {}).update(
        output_dir=str(output_dir),
        restart_checkpoint_path=None,
        seed=seed,
    )
    config.setdefault("data_module_args", {}).update(
        batch_size=1,
        data_seed=seed,
        num_workers=num_workers,
        num_workers_validation=0,
        epoch_len=total_steps * devices * num_nodes,
    )
    config.setdefault("logging_config", {}).update(
        wandb_config=None,
        log_grads=False,
        log_lr=False,
    )
    config.setdefault("checkpoint_config", {}).update(
        save_top_k=0,
        save_last=False,
    )
    config.setdefault("pl_trainer_args", {}).update(
        devices=devices,
        num_nodes=num_nodes,
        precision="bf16-mixed",
        accumulate_grad_batches=1,
        deepspeed_config_path=None,
        mpi_plugin=False,
        max_epochs=-1,
        max_steps=total_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_model_summary=False,
        # AQLab's current runner always installs its
        # SecondsPerIterationProgressBar callback. Lightning rejects that
        # callback when the trainer-level progress bar switch is disabled.
        enable_progress_bar=True,
    )
    return config


def _loss_tensor(outputs: Any, trainer: pl.Trainer) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs.detach()
    if isinstance(outputs, Mapping):
        value = outputs.get("loss")
        if isinstance(value, torch.Tensor):
            return value.detach()
    for key in ("train/loss", "train/loss_step", "loss"):
        value = trainer.callback_metrics.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach()
    return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def aggregate_rank_metrics(
    gpu_ms: torch.Tensor,
    loss: torch.Tensor,
    wall_seconds: torch.Tensor,
) -> int:
    """Aggregate finished measurements once, outside the measured interval."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    world_size = torch.distributed.get_world_size()
    torch.distributed.all_reduce(gpu_ms, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
    loss /= world_size
    torch.distributed.all_reduce(wall_seconds, op=torch.distributed.ReduceOp.MAX)
    return world_size


def topology_name(world_size: int, num_nodes: int) -> str:
    """Name the DDP topology recorded in ``summary.json``."""

    if num_nodes > 1:
        return "multi-node-ddp"
    return "single-gpu" if world_size == 1 else "single-node-ddp"


_RESULT_FILES = ("runner.yaml", "steps.csv", "summary.json")


def has_benchmark_results(output_dir: Path) -> bool:
    """Return whether a previous benchmark already wrote results here.

    Only the result files count. Under ``srun`` every rank creates
    ``output_dir/logs`` during setup while rank 0 may still be importing, so
    testing for an empty directory would race against the other ranks.
    """

    return any((output_dir / name).is_file() for name in _RESULT_FILES)


def owns_benchmark_output(environ: Mapping[str, str]) -> bool:
    """Return whether this process checks and writes the benchmark output.

    Exactly one process per benchmark may own the output directory:

    * Lightning's local launcher spawns workers that carry ``LOCAL_RANK`` but
      no ``RANK``; the parent process, global rank 0, owns the output.
    * ``torchrun`` sets ``RANK`` on every worker, including rank 0.
    * ``srun`` sets ``SLURM_PROCID`` on every task when it starts more than one.
    """

    if "RANK" in environ:
        return environ["RANK"] == "0"
    if "SLURM_PROCID" in environ and int(environ.get("SLURM_NTASKS", "1")) > 1:
        return environ["SLURM_PROCID"] == "0"
    return "LOCAL_RANK" not in environ


def is_source_tree_import(module_path: Path, source_root: Path) -> bool:
    """Return whether OpenFold3 was imported from source instead of a wheel."""

    return module_path.resolve().is_relative_to((source_root / "openfold3").resolve())


def configure_torch_backend_for_training(*, use_tf32: bool = False) -> None:
    """Match the backend setup used by AQLab's current training CLI."""

    from openfold3.entry_points.import_utils import (
        _configure_torch_backend,
        _enable_tf32,
    )

    _configure_torch_backend()
    if use_tf32:
        _enable_tf32()


def training_crop_budget(config: Mapping[str, Any]) -> int:
    """Return the one enabled token budget shared by all training datasets."""

    train_configs = config.get("dataset_configs", {}).get("train", {})
    if not isinstance(train_configs, Mapping) or not train_configs:
        raise ValueError("runner YAML has no training dataset configuration")
    budgets = set()
    for dataset_name, dataset_spec in train_configs.items():
        try:
            token_crop = dataset_spec["config"]["crop"]["token_crop"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"training dataset {dataset_name!r} has no token-crop configuration"
            ) from error
        if not token_crop.get("enabled", False):
            raise ValueError(
                f"training dataset {dataset_name!r} does not enable token cropping"
            )
        budgets.add(int(token_crop["token_budget"]))
    if len(budgets) != 1:
        raise ValueError(
            "--require-full-crop needs one shared token budget across all "
            f"training datasets, found {sorted(budgets)}"
        )
    budget = budgets.pop()
    if budget < 1:
        raise ValueError(f"training token budget must be positive, found {budget}")
    return budget


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _checkout_identity(source_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError, subprocess.CalledProcessError:
        return None, None
    return commit, dirty


class BackendBenchmarkCallback(pl.Callback):
    """Capture loss and steady-state GPU/wall timing without per-step syncs."""

    def __init__(
        self,
        *,
        backend: str,
        output_dir: Path,
        warmup_steps: int,
        measure_steps: int,
        global_batch_size: int,
        devices_per_node: int = 1,
        num_nodes: int = 1,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.output_dir = output_dir
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.global_batch_size = global_batch_size
        self.devices_per_node = devices_per_node
        self.num_nodes = num_nodes
        self.starts: list[torch.cuda.Event] = []
        self.ends: list[torch.cuda.Event] = []
        self.recorded: list[bool] = []
        self.losses: list[tuple[int, torch.Tensor]] = []
        self.measurement_start: float | None = None
        self.measurement_end: float | None = None
        self.current_step: int | None = None

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del trainer, pl_module
        self.starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.measure_steps)
        ]
        self.ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.measure_steps)
        ]
        self.recorded = [False] * self.measure_steps

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        step = int(trainer.global_step)
        self.current_step = step
        measured_index = step - self.warmup_steps
        if measured_index == 0:
            torch.cuda.synchronize()
            self.measurement_start = time.perf_counter()
        if 0 <= measured_index < self.measure_steps:
            self.starts[measured_index].record()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx
        if self.current_step is None:
            return
        step = self.current_step
        loss = _loss_tensor(outputs, trainer)
        if loss is not None:
            self.losses.append((step, loss))

        measured_index = step - self.warmup_steps
        if 0 <= measured_index < self.measure_steps:
            self.ends[measured_index].record()
            self.recorded[measured_index] = True
            if measured_index == self.measure_steps - 1:
                torch.cuda.synchronize()
                self.measurement_end = time.perf_counter()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if not all(self.recorded):
            completed = sum(self.recorded)
            raise RuntimeError(
                f"benchmark recorded {completed}/{self.measure_steps} measured steps"
            )
        if self.measurement_start is None or self.measurement_end is None:
            raise RuntimeError("benchmark timing boundaries were not recorded")

        local_gpu_ms = [
            start.elapsed_time(end) for start, end in zip(self.starts, self.ends)
        ]
        local_wall_seconds = self.measurement_end - self.measurement_start
        total_steps = self.warmup_steps + self.measure_steps
        local_loss_by_step = {step: loss for step, loss in self.losses}
        missing_steps = [
            step for step in range(total_steps) if step not in local_loss_by_step
        ]
        if missing_steps:
            raise RuntimeError(
                "benchmark did not capture loss for training steps: "
                + ", ".join(str(step) for step in missing_steps)
            )

        device = torch.device("cuda", torch.cuda.current_device())
        gpu_ms_tensor = torch.tensor(local_gpu_ms, dtype=torch.float64, device=device)
        loss_tensor = torch.stack(
            [
                local_loss_by_step[step].float().reshape(())
                for step in range(total_steps)
            ]
        ).to(device=device, dtype=torch.float64)
        wall_tensor = torch.tensor(
            local_wall_seconds,
            dtype=torch.float64,
            device=device,
        )

        # These collectives run after the measured interval. GPU latency uses the
        # critical-path rank; loss is the global mean for each optimizer step.
        world_size = aggregate_rank_metrics(
            gpu_ms_tensor,
            loss_tensor,
            wall_tensor,
        )

        expected_world_size = self.devices_per_node * self.num_nodes
        if world_size != expected_world_size:
            raise RuntimeError(
                f"benchmark world size {world_size} does not match "
                f"--devices {self.devices_per_node} x --num-nodes {self.num_nodes}"
            )

        if not trainer.is_global_zero:
            return

        gpu_ms = gpu_ms_tensor.cpu().tolist()
        loss_by_step = {
            step: value for step, value in enumerate(loss_tensor.cpu().tolist())
        }
        steady_wall_seconds = float(wall_tensor.item())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "steps.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("backend", "step", "phase", "loss", "gpu_step_ms"),
            )
            writer.writeheader()
            for step in range(total_steps):
                measured_index = step - self.warmup_steps
                writer.writerow(
                    {
                        "backend": self.backend,
                        "step": step,
                        "phase": "measure" if measured_index >= 0 else "warmup",
                        "loss": loss_by_step.get(step, ""),
                        "gpu_step_ms": (
                            gpu_ms[measured_index] if measured_index >= 0 else ""
                        ),
                    }
                )

        summary = {
            "schema_version": 2,
            "backend": self.backend,
            "topology": topology_name(world_size, self.num_nodes),
            "devices": self.devices_per_node,
            "num_nodes": self.num_nodes,
            "world_size": world_size,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "global_batch_size": self.global_batch_size,
            "loss_reduction": "mean_across_ranks",
            "gpu_timing_reduction": "max_across_ranks",
            "steady_wall_seconds": steady_wall_seconds,
            "steady_step_wall_seconds": steady_wall_seconds / self.measure_steps,
            "steady_samples_per_second": (
                self.global_batch_size * self.measure_steps / steady_wall_seconds
            ),
            "gpu_step_mean_ms": statistics.mean(gpu_ms),
            "gpu_step_median_ms": statistics.median(gpu_ms),
            "gpu_step_p95_ms": _percentile(gpu_ms, 0.95),
            "gpu_step_min_ms": min(gpu_ms),
            "gpu_step_max_ms": max(gpu_ms),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-yaml", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=tuple(BACKEND_FLAGS))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--require-full-crop",
        action="store_true",
        help=(
            "fail in the CPU collator if any training sample contains padding "
            "inside the configured token crop"
        ),
    )
    parser.add_argument(
        "--use-tf32",
        action="store_true",
        help=(
            "Allow TF32 for FP32 matrix multiplications while retaining the "
            "configured BF16 mixed-precision training mode"
        ),
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="GPUs per node; Lightning launches local DDP when greater than one",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help=(
            "Nodes in the job; above one, an external launcher (srun or torchrun) "
            "must already have started --devices processes on every node"
        ),
    )
    return parser


def _select_kernel_provider(backend: str) -> None:
    """Keep the standalone backend label tied to the installed package."""
    if backend != "turbopairformer":
        return
    variable = "OPENFOLD3_TRIANGLE_PROVIDER"
    requested = os.environ.get(variable, "turbopairformer").strip().lower()
    if requested != "turbopairformer":
        raise ValueError(
            "--backend turbopairformer requires "
            "OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer; "
            f"got {requested!r}"
        )
    os.environ[variable] = "turbopairformer"


def main() -> None:
    args = _parser().parse_args()
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be at least 1")
    if args.measure_steps < 1:
        raise ValueError("--measure-steps must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.devices < 1:
        raise ValueError("--devices must be at least 1")
    if args.num_nodes < 1:
        raise ValueError("--num-nodes must be at least 1")
    _select_kernel_provider(args.backend)

    source_root = Path(__file__).resolve().parents[1]
    import openfold3

    installed_module = Path(openfold3.__file__).resolve()
    if is_source_tree_import(installed_module, source_root):
        raise RuntimeError("benchmark must import OpenFold3 from the Pixi installation")

    from openfold3.core.config import config_utils
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
    from openfold3.entry_points.validator import TrainingExperimentConfig

    runner_yaml = args.runner_yaml.resolve()
    runner_yaml_sha256 = _sha256(runner_yaml)
    output_dir = args.output_root.resolve() / args.backend
    output_owner = owns_benchmark_output(os.environ)
    if output_owner and has_benchmark_results(output_dir):
        raise FileExistsError(f"benchmark output already exists: {output_dir}")

    runner_dict = config_utils.load_yaml(runner_yaml)
    configure_benchmark(
        runner_dict,
        backend=args.backend,
        output_dir=output_dir,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        devices=args.devices,
        num_nodes=args.num_nodes,
    )
    expected_crop_tokens = None
    if args.require_full_crop:
        expected_crop_tokens = training_crop_budget(runner_dict)
        install_full_crop_collator(expected_crop_tokens)
    if output_owner:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "runner.yaml").write_text(
            yaml.safe_dump(runner_dict, sort_keys=False, default_flow_style=False)
        )

    configure_torch_backend_for_training(use_tf32=args.use_tf32)
    callback = BackendBenchmarkCallback(
        backend=args.backend,
        output_dir=output_dir,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        global_batch_size=args.devices * args.num_nodes,
        devices_per_node=args.devices,
        num_nodes=args.num_nodes,
    )

    class BenchmarkRunner(TrainingExperimentRunner):
        @cached_property
        def callbacks(self):
            return [*super().callbacks, callback]

    experiment = TrainingExperimentConfig.model_validate(runner_dict)
    runner = BenchmarkRunner(experiment)
    runner.setup()
    fit_start = time.perf_counter()
    runner.run()
    fit_wall_seconds = time.perf_counter() - fit_start

    if runner.trainer.is_global_zero:
        checkout_commit, checkout_dirty = _checkout_identity(source_root)
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(
            fit_wall_seconds=fit_wall_seconds,
            seed=args.seed,
            num_workers_per_rank=args.num_workers,
            runner_yaml=str(runner_yaml),
            runner_yaml_sha256=runner_yaml_sha256,
            effective_runner_sha256=_sha256(output_dir / "runner.yaml"),
            backend_flags=BACKEND_FLAGS[args.backend],
            require_full_crop=args.require_full_crop,
            full_crop_validation=("passed" if args.require_full_crop else "disabled"),
            crop_token_budget=expected_crop_tokens,
            checkout_commit=checkout_commit,
            checkout_dirty=checkout_dirty,
            openfold3_module=str(installed_module),
            openfold3_version=_package_version("openfold3"),
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
            use_tf32=args.use_tf32,
            triton_version=_package_version("triton"),
            cuequivariance_version=importlib.metadata.version("cuequivariance"),
            cuequivariance_torch_version=importlib.metadata.version(
                "cuequivariance-torch"
            ),
            cuequivariance_ops_torch_version=importlib.metadata.version(
                "cuequivariance-ops-torch-cu12"
            ),
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
