import csv
import json
from pathlib import Path

import yaml

from tools.benchmark_training_backend import configure_benchmark
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
        "epoch_len": 10,
    }
    assert configured["logging_config"]["wandb_config"] is None
    assert configured["checkpoint_config"] == {
        "save_top_k": 0,
        "save_last": False,
    }
    assert configured["pl_trainer_args"]["devices"] == 1
    assert configured["pl_trainer_args"]["num_nodes"] == 1
    assert configured["pl_trainer_args"]["max_steps"] == 10
    assert configured["pl_trainer_args"]["limit_val_batches"] == 0
    assert configured["pl_trainer_args"]["num_sanity_val_steps"] == 0
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
        "fit_wall_seconds": seconds * len(losses),
        "steady_wall_seconds": seconds * len(losses),
        "steady_step_wall_seconds": seconds,
        "steady_samples_per_second": 1 / seconds,
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
    assert (output / "backend_comparison.csv").is_file()
    assert (output / "backend_comparison.json").is_file()
    assert (output / "training_time.png").stat().st_size > 0
    assert (output / "loss_curves.png").stat().st_size > 0


def test_benchmark_script_is_valid_yaml_serializer_input(tmp_path):
    config = configure_benchmark(
        {},
        backend="aqlab",
        output_dir=tmp_path / "aqlab",
        warmup_steps=1,
        measure_steps=1,
        seed=7,
        num_workers=0,
    )
    assert yaml.safe_load(yaml.safe_dump(config)) == config
