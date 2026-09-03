from pathlib import Path

import pytest
import yaml

import tools.prepare_pdb_subset as preparer
from scripts.datasets.pdb_subset_helpers import build_runner_yaml_config
from tools.validate_training_data import (
    TrainingDataError,
    find_path_problems,
    validate_runner_yaml,
)


def _runner_config(root: Path) -> dict:
    return {
        "dataset_paths": {
            "weighted-pdb": {
                "alignment_array_directory": str(root / "alignments"),
                "dataset_cache_file": str(root / "train.json"),
                "target_structures_directory": str(root / "structures"),
                "alignments_directory": "none",
                "ccd_file": None,
            },
            "val-weighted-pdb": {
                "alignment_array_directory": str(root / "alignments"),
                "dataset_cache_file": str(root / "validation.json"),
                "target_structures_directory": str(root / "structures"),
            },
        }
    }


def test_preflight_reports_all_missing_dataset_paths(tmp_path):
    config = _runner_config(tmp_path / "missing-data")

    problems = find_path_problems(config, base_dir=tmp_path)

    assert {(problem.dataset, problem.field) for problem in problems} == {
        ("weighted-pdb", "alignment_array_directory"),
        ("weighted-pdb", "dataset_cache_file"),
        ("weighted-pdb", "target_structures_directory"),
        ("val-weighted-pdb", "alignment_array_directory"),
        ("val-weighted-pdb", "dataset_cache_file"),
        ("val-weighted-pdb", "target_structures_directory"),
    }


def test_preflight_accepts_complete_paths_and_resolves_relative_values(tmp_path):
    data_root = tmp_path / "run" / "data"
    (data_root / "alignments").mkdir(parents=True)
    (data_root / "structures").mkdir()
    cache = '{"structure_data": {"sample": {}}}'
    (data_root / "train.json").write_text(cache)
    (data_root / "validation.json").write_text(cache)
    config = _runner_config(Path("data"))
    runner_yaml = tmp_path / "runner.yaml"
    runner_yaml.write_text(yaml.safe_dump(config))

    validate_runner_yaml(runner_yaml, base_dir=tmp_path / "run")


def test_preflight_error_stops_before_training_with_actionable_message(tmp_path):
    runner_yaml = tmp_path / "runner.yaml"
    runner_yaml.write_text(yaml.safe_dump(_runner_config(tmp_path / "missing")))

    with pytest.raises(TrainingDataError) as caught:
        validate_runner_yaml(runner_yaml, base_dir=tmp_path)

    message = str(caught.value)
    assert "No training process was started" in message
    assert "download and purification steps" in message
    assert "alignment_array_directory" in message


def test_preflight_rejects_empty_purified_cache(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "alignments").mkdir(parents=True)
    (data_root / "structures").mkdir()
    empty_cache = '{"structure_data": {}}'
    (data_root / "train.json").write_text(empty_cache)
    (data_root / "validation.json").write_text(empty_cache)
    runner_yaml = tmp_path / "runner.yaml"
    runner_yaml.write_text(yaml.safe_dump(_runner_config(data_root)))

    with pytest.raises(TrainingDataError, match="contains no structure_data entries"):
        validate_runner_yaml(runner_yaml, base_dir=tmp_path)


def test_generated_runner_paths_are_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = build_runner_yaml_config(
        {"train": Path("train.json"), "val": Path("validation.json")},
        Path("pdb_training_set"),
    )

    for values in config["dataset_paths"].values():
        assert Path(values["dataset_cache_file"]).is_absolute()
        assert Path(values["alignment_array_directory"]).is_absolute()


def test_prepare_subset_runs_two_download_passes_then_purifies(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(
        preparer, "_run", lambda script, *args: calls.append((script, args))
    )
    monkeypatch.setattr(preparer, "validate_runner_yaml", lambda *args, **kwargs: None)

    runner_yaml = preparer.prepare_subset(
        target_dir=tmp_path / "datasets",
        output_dir=tmp_path / "data",
        train_size=4000,
        seed=42,
        workers=24,
        download_attempts=2,
        force=False,
    )

    assert [script for script, _ in calls] == [
        "generate_subset_cache.py",
        "download_subset.py",
        "download_subset.py",
        "purify_subset.py",
    ]
    generate_args = calls[0][1]
    assert generate_args[generate_args.index("--runner-yaml") + 1] == ""
    assert runner_yaml == tmp_path / "datasets" / "train_pdb_subset.yaml"
