import csv
import json
import pickle
from functools import partial
from pathlib import Path

import pytest
import torch
import yaml

from tools.benchmark_full_crop import require_full_crop_collate
from tools.benchmark_training_backend import (
    aggregate_rank_metrics,
    configure_benchmark,
    configure_torch_backend_for_training,
    has_benchmark_results,
    is_source_tree_import,
    owns_benchmark_output,
    topology_name,
    training_crop_budget,
)
from tools.plot_backend_comparison import build_report


@pytest.mark.parametrize("backend", ("turbopairformer", "stablefold"))
def test_benchmark_config_is_matched_validation_free_training(tmp_path, backend):
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
        backend=backend,
        output_dir=tmp_path / backend,
        warmup_steps=3,
        measure_steps=7,
        seed=42,
        num_workers=0,
        devices=8,
    )

    assert configured["experiment_settings"] == {
        "restart_checkpoint_path": None,
        "output_dir": str(tmp_path / backend),
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
    train_memory = configured["model_update"]["custom"]["settings"]["memory"]["train"]
    assert train_memory["use_stable_triangle_attention"] is True
    assert train_memory["use_stable_triangle_multiplicative"] is True
    assert train_memory["use_cueq_triangle_kernels"] is False


def test_multi_node_benchmark_config_scales_epoch_len_by_world_size(tmp_path):
    configured = configure_benchmark(
        {},
        backend="stablefold",
        output_dir=tmp_path / "stablefold",
        warmup_steps=10,
        measure_steps=400,
        seed=42,
        num_workers=4,
        devices=8,
        num_nodes=2,
    )

    assert configured["pl_trainer_args"]["devices"] == 8
    assert configured["pl_trainer_args"]["num_nodes"] == 2
    assert configured["pl_trainer_args"]["max_steps"] == 410
    assert configured["data_module_args"]["epoch_len"] == 410 * 16
    assert configured["data_module_args"]["num_workers"] == 4


def test_triton_benchmark_uses_ten_workers_and_only_triton_route(tmp_path):
    configured = configure_benchmark(
        {},
        backend="triton",
        output_dir=tmp_path / "triton",
        warmup_steps=5,
        measure_steps=20,
        seed=42,
        num_workers=10,
        devices=2,
    )

    assert configured["data_module_args"]["num_workers"] == 10
    train_memory = configured["model_update"]["custom"]["settings"]["memory"]["train"]
    assert train_memory["use_triton_triangle_kernels"] is True
    assert train_memory["use_cueq_triangle_kernels"] is False
    assert train_memory["use_stable_triangle_attention"] is False
    assert train_memory["use_stable_triangle_multiplicative"] is False


def test_full_crop_gate_accepts_real_tokens_and_rejects_padding():
    def collator(samples):
        return samples[0]

    full = {"pdb_id": ["full"], "token_mask": torch.ones(1, 768)}
    assert (
        require_full_crop_collate([full], collator=collator, expected_tokens=768)
        is full
    )

    padded = {
        "pdb_id": ["padded"],
        "token_mask": torch.cat((torch.ones(1, 767), torch.zeros(1, 1)), dim=-1),
    }
    with pytest.raises(RuntimeError, match="would use token padding"):
        require_full_crop_collate([padded], collator=collator, expected_tokens=768)


def test_full_crop_gate_function_is_forkserver_pickleable():
    from openfold3.core.data.framework.data_module import openfold_batch_collator

    collator = partial(
        require_full_crop_collate,
        collator=openfold_batch_collator,
        expected_tokens=768,
    )
    restored = pickle.loads(pickle.dumps(collator))
    assert restored.func is require_full_crop_collate
    assert restored.keywords["collator"] is openfold_batch_collator


def test_full_crop_gate_requires_one_enabled_training_budget():
    config = {
        "dataset_configs": {
            "train": {
                "pdb": {
                    "config": {
                        "crop": {"token_crop": {"enabled": True, "token_budget": 640}}
                    }
                }
            }
        }
    }
    assert training_crop_budget(config) == 640


def test_topology_name_distinguishes_single_gpu_node_and_multi_node():
    assert topology_name(1, 1) == "single-gpu"
    assert topology_name(8, 1) == "single-node-ddp"
    assert topology_name(16, 2) == "multi-node-ddp"


def test_benchmark_output_owner_is_global_rank_zero_under_each_launcher():
    # Single process, or the parent of Lightning's local launcher.
    assert owns_benchmark_output({})
    # Lightning-spawned local worker: LOCAL_RANK without RANK.
    assert not owns_benchmark_output({"LOCAL_RANK": "1"})
    assert not owns_benchmark_output({"LOCAL_RANK": "0"})
    # torchrun sets RANK on every worker, including rank zero.
    assert owns_benchmark_output({"RANK": "0", "LOCAL_RANK": "0"})
    assert not owns_benchmark_output({"RANK": "9", "LOCAL_RANK": "1"})
    # srun with one task per GPU sets SLURM_PROCID on every task.
    slurm = {"SLURM_NTASKS": "16", "SLURM_PROCID": "0", "SLURM_LOCALID": "0"}
    assert owns_benchmark_output(slurm)
    assert not owns_benchmark_output({**slurm, "SLURM_PROCID": "8"})
    # One srun task hosting Lightning's local launcher: the parent owns.
    assert owns_benchmark_output({"SLURM_NTASKS": "1", "SLURM_PROCID": "0"})
    assert not owns_benchmark_output(
        {"SLURM_NTASKS": "1", "SLURM_PROCID": "0", "LOCAL_RANK": "3"}
    )


def test_result_check_ignores_directories_created_by_other_ranks(tmp_path):
    output = tmp_path / "stablefold"
    assert not has_benchmark_results(output)
    # Under srun, other ranks create the log directory and the DDP receipt
    # before rank 0 finishes importing; that must not count as a prior result.
    (output / "logs").mkdir(parents=True)
    (output / "logs" / "stablefold_kernel_ddp.json").write_text("{}")
    assert not has_benchmark_results(output)
    (output / "summary.json").write_text("{}")
    assert has_benchmark_results(output)


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


@pytest.mark.parametrize("backend", ("turbopairformer", "stablefold"))
def test_comparison_writes_tables_and_loss_timing_plots(tmp_path, backend):
    results = tmp_path / "results"
    _write_result(results, "aqlab", 2.0, [10.0, 9.0, 8.0])
    _write_result(results, "triton", 1.5, [10.0, 8.95, 7.95])
    _write_result(results, "cueq", 1.25, [10.0, 8.92, 7.92])
    _write_result(results, backend, 1.0, [10.0, 8.9, 7.9])
    output = results / "comparison"

    report = build_report(results, output, baseline="aqlab")

    by_backend = {row["backend"]: row for row in report["backends"]}
    assert by_backend["aqlab"]["speedup_vs_baseline"] == 1.0
    assert by_backend[backend]["speedup_vs_baseline"] == 2.0
    assert [row["backend"] for row in report["backends"]] == [
        "aqlab",
        "triton",
        "cueq",
        backend,
    ]
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
    assert {row["backend"] for row in rows} == {"aqlab", "triton", "cueq", backend}


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
    monkeypatch.setattr(
        import_utils,
        "_enable_tf32",
        lambda: calls.append("tf32"),
    )

    configure_torch_backend_for_training()
    assert calls == ["configured"]

    calls.clear()
    configure_torch_backend_for_training(use_tf32=True)
    assert calls == ["configured", "tf32"]


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
