import csv
import json
from pathlib import Path

import torch
import yaml

from tools.benchmark_training_backend import (
    aggregate_rank_metrics,
    configure_torch_backend_for_training,
    configure_benchmark,
    is_source_tree_import,
)
from tools.plot_backend_comparison import build_report


def test_benchmark_config_is_matched_validation_free_training(tmp_path):
    config = {
        "experiment_settings": {"restart_checkpoint_path": "last"},
        "data_module_args": {"batch_size": 8, "num_workers": 10},
        "logging_config": {"wandb_config": {"project": "old"}},
        "checkpoint_config": {"save_top_k": 3, "save_last": True},
        "pl_trainer_args": {
            "devices": 8,
            "num_nodes": 2,
            "max_steps": 999,
            "limit_val_batches": 1,
        },
    }

    configured = configure_benchmark(
        config,
        backend="stablefold",
        output_dir=tmp_path / "stablefold",
        warmup_steps=3,
        measure_steps=7,
        seed=42,
        num_workers=0,
        devices=8,
    )

    assert configured["experiment_settings"] == {
        "restart_checkpoint_path": None,
        "output_dir": str(tmp_path / "stablefold"),
        "seed": 42,
    }
    assert configured["data_module_args"] == {
        "batch_size": 1,
        "num_workers": 0,
        "data_seed": 42,
        "num_workers_validation": 0,
        "epoch_len": 80,
    }
    assert configured["logging_config"]["wandb_config"] is None
    assert configured["checkpoint_config"] == {
        "save_top_k": 0,
        "save_last": False,
    }
    assert configured["pl_trainer_args"]["devices"] == 8
    assert configured["pl_trainer_args"]["num_nodes"] == 1
    assert configured["pl_trainer_args"]["max_steps"] == 10
    assert configured["pl_trainer_args"]["limit_val_batches"] == 0
    assert configured["pl_trainer_args"]["num_sanity_val_steps"] == 0
    assert configured["pl_trainer_args"]["enable_progress_bar"] is True
    train_memory = configured["model_update"]["custom"]["settings"]["memory"][
        "train"
    ]
    assert train_memory["use_stable_triangle_attention"] is True
    assert train_memory["use_stable_triangle_multiplicative"] is True
    assert train_memory["use_cueq_triangle_kernels"] is False


def _write_result(root: Path, backend: str, seconds: float, losses: list[float]):
    output = root / backend
    output.mkdir(parents=True)
    summary = {
        "backend": backend,
        "topology": "single-node-ddp",
        "devices": 4,
        "world_size": 4,
        "global_batch_size": 4,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "fit_wall_seconds": seconds * len(losses),
        "steady_wall_seconds": seconds * len(losses),
        "steady_step_wall_seconds": seconds,
        "steady_samples_per_second": 4 / seconds,
        "gpu_step_mean_ms": seconds * 900,
        "gpu_step_median_ms": seconds * 890,
        "gpu_step_p95_ms": seconds * 950,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    with (output / "steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("backend", "step", "phase", "loss", "gpu_step_ms"),
        )
        writer.writeheader()
        for step, loss in enumerate(losses):
            writer.writerow(
                {
                    "backend": backend,
                    "step": step,
                    "phase": "measure",
                    "loss": loss,
                    "gpu_step_ms": seconds * 1000,
                }
            )


def test_comparison_writes_tables_and_loss_timing_plots(tmp_path):
    results = tmp_path / "results"
    _write_result(results, "aqlab", 2.0, [10.0, 9.0, 8.0])
    _write_result(results, "stablefold", 1.0, [10.0, 8.9, 7.9])
    output = results / "comparison"

    report = build_report(results, output, baseline="aqlab")

    by_backend = {row["backend"]: row for row in report["backends"]}
    assert by_backend["aqlab"]["speedup_vs_baseline"] == 1.0
    assert by_backend["stablefold"]["speedup_vs_baseline"] == 2.0
    assert report["world_size"] == 4
    assert report["gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert (output / "backend_comparison.csv").is_file()
    assert (output / "backend_comparison.json").is_file()
    assert (output / "training_time.png").stat().st_size > 0
    assert (output / "loss_curves.png").stat().st_size > 0
    with (output / "backend_comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["devices"] for row in rows} == {"4"}
    assert {row["topology"] for row in rows} == {"single-node-ddp"}


def test_distributed_metrics_use_critical_path_and_global_mean(monkeypatch):
    gpu_ms = torch.tensor([3.0, 5.0])
    loss = torch.tensor([8.0, 6.0])
    wall_seconds = torch.tensor(4.0)
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op):
        calls.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2)
        elif tensor.ndim == 0:
            tensor.fill_(4.5)
        else:
            tensor.copy_(torch.tensor([4.0, 5.5]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    world_size = aggregate_rank_metrics(gpu_ms, loss, wall_seconds)

    assert world_size == 2
    assert gpu_ms.tolist() == [4.0, 5.5]
    assert loss.tolist() == [8.0, 6.0]
    assert wall_seconds.item() == 4.5
    assert calls == [
        torch.distributed.ReduceOp.MAX,
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MAX,
    ]


def test_installed_pixi_wheel_is_not_mistaken_for_source_import(tmp_path):
    source_root = tmp_path / "checkout"
    assert is_source_tree_import(
        source_root / "openfold3" / "__init__.py",
        source_root,
    )
    assert not is_source_tree_import(
        source_root
        / ".pixi"
        / "envs"
        / "openfold3-cuda12"
        / "site-packages"
        / "openfold3"
        / "__init__.py",
        source_root,
    )


def test_benchmark_uses_current_aqlab_training_backend_setup(monkeypatch):
    import openfold3.entry_points.import_utils as import_utils

    calls = []
    monkeypatch.setattr(
        import_utils,
        "_configure_torch_backend",
        lambda: calls.append("configured"),
    )

    configure_torch_backend_for_training()

    assert calls == ["configured"]


def test_benchmark_script_is_valid_yaml_serializer_input(tmp_path):
    config = configure_benchmark(
        {},
        backend="aqlab",
        output_dir=tmp_path / "aqlab",
        warmup_steps=1,
        measure_steps=1,
        seed=7,
        num_workers=0,
        devices=1,
    )
    assert yaml.safe_load(yaml.safe_dump(config)) == config
