#!/usr/bin/env python3
"""Benchmark one OpenFold3 training backend with matched data and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml

try:
    from tools.select_training_backend import BACKEND_FLAGS, configure_training_backend
except ModuleNotFoundError:
    from select_training_backend import BACKEND_FLAGS, configure_training_backend


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
) -> dict[str, Any]:
    """Make a validation-free, single-node, from-scratch benchmark config."""

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
        epoch_len=total_steps * devices,
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
        num_nodes=1,
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


def is_source_tree_import(module_path: Path, source_root: Path) -> bool:
    """Return whether OpenFold3 was imported from source instead of a wheel."""

    return module_path.resolve().is_relative_to((source_root / "openfold3").resolve())


def configure_torch_backend_for_training() -> None:
    """Match the backend setup used by AQLab's current training CLI."""

    from openfold3.entry_points.import_utils import _configure_torch_backend

    _configure_torch_backend()


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
    ) -> None:
        super().__init__()
        self.backend = backend
        self.output_dir = output_dir
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.global_batch_size = global_batch_size
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
            "topology": "single-gpu" if world_size == 1 else "single-node-ddp",
            "devices": world_size,
            "num_nodes": 1,
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
        "--devices",
        type=int,
        default=1,
        help="GPUs on this node; Lightning launches local DDP when greater than one",
    )
    return parser


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

    source_root = Path(__file__).resolve().parents[1]
    import openfold3

    installed_module = Path(openfold3.__file__).resolve()
    if is_source_tree_import(installed_module, source_root):
        raise RuntimeError("benchmark must import OpenFold3 from the Pixi installation")

    from openfold3.core.config import config_utils
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
    from openfold3.entry_points.validator import TrainingExperimentConfig

    runner_yaml = args.runner_yaml.resolve()
    output_dir = (args.output_root.resolve() / args.backend)
    is_ddp_worker = "LOCAL_RANK" in os.environ
    if output_dir.exists() and any(output_dir.iterdir()) and not is_ddp_worker:
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
    )
    if not is_ddp_worker:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "runner.yaml").write_text(
            yaml.safe_dump(runner_dict, sort_keys=False, default_flow_style=False)
        )

    configure_torch_backend_for_training()
    callback = BackendBenchmarkCallback(
        backend=args.backend,
        output_dir=output_dir,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        global_batch_size=args.devices,
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
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(
            fit_wall_seconds=fit_wall_seconds,
            seed=args.seed,
            runner_yaml=str(runner_yaml),
            openfold3_module=str(installed_module),
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
