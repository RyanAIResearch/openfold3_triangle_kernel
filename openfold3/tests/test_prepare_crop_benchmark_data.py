import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.prepare_crop_benchmark_data import (
    CropBenchmarkDataError,
    prepare_crop_benchmark_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_structure(
    structures: Path,
    pdb_id: str,
    *,
    n_tokens: int,
    n_resolved: int,
    n_chains: int,
) -> None:
    path = structures / pdb_id / f"{pdb_id}.npz"
    path.parent.mkdir(parents=True)
    token_id = np.arange(n_tokens, dtype=np.int64)
    token_center_atom = np.ones(n_tokens, dtype=bool)
    occupancy = np.concatenate(
        (
            np.ones(n_resolved, dtype=np.float32),
            np.zeros(n_tokens - n_resolved, dtype=np.float32),
        )
    )
    chain_id = np.asarray([f"chain-{index % n_chains}" for index in range(n_tokens)])
    np.savez(
        path,
        token_id=token_id,
        token_center_atom=token_center_atom,
        occupancy=occupancy,
        chain_id=chain_id,
    )


def _write_input(
    root: Path,
    structures: Path,
    entries: tuple[str, ...],
    *,
    chain_crop_enabled: bool = True,
    n_chains: int = 2,
) -> Path:
    cache = root / "train.json"
    cache.write_text(
        json.dumps(
            {
                "_type": "WeightedPDBDatasetCache",
                "name": "fixture",
                "structure_data": {pdb_id: {"chains": {}} for pdb_id in entries},
                "reference_molecule_data": {"LIG": {"keep": True}},
            },
            indent=2,
        )
        + "\n"
    )
    runner = {
        "data_module_args": {"batch_size": 1, "num_workers": 3},
        "dataset_configs": {
            "train": {
                "weighted-pdb": {
                    "dataset_class": "WeightedPDBDataset",
                    "weight": 1.0,
                    "config": {
                        "crop": {
                            "token_crop": {"enabled": True, "token_budget": 2},
                            "chain_crop": {
                                "enabled": chain_crop_enabled,
                                "n_chains": n_chains,
                            },
                        }
                    },
                }
            }
        },
        "dataset_paths": {
            "weighted-pdb": {
                "dataset_cache_file": str(cache),
                "target_structures_directory": str(structures),
                "target_structure_file_format": "npz",
            }
        },
    }
    runner_yaml = root / "runner.yaml"
    runner_yaml.write_text(yaml.safe_dump(runner, sort_keys=False))
    return runner_yaml


def test_prepares_full_crop_runners_cache_and_provenance(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "good", n_tokens=8, n_resolved=8, n_chains=2)
    _write_structure(structures, "too-short", n_tokens=7, n_resolved=7, n_chains=2)
    _write_structure(structures, "unresolved", n_tokens=9, n_resolved=7, n_chains=2)
    _write_structure(
        structures, "too-many-chains", n_tokens=9, n_resolved=9, n_chains=3
    )
    runner_yaml = _write_input(
        tmp_path,
        structures,
        ("good", "too-short", "unresolved", "too-many-chains"),
    )
    output = tmp_path / "crop-data"

    provenance_path = prepare_crop_benchmark_data(
        runner_yaml=runner_yaml,
        output_dir=output,
        crop_sizes=(4, 6, 8),
    )

    output = output.resolve()
    filtered_cache_path = output / "training_cache_no_padding_8.json"
    filtered = json.loads(filtered_cache_path.read_text())
    assert list(filtered)[:1] == ["_type"]
    assert list(filtered["structure_data"]) == ["good"]
    assert filtered["reference_molecule_data"] == {"LIG": {"keep": True}}

    for crop_size in (4, 6, 8):
        path = output / f"runner_crop_{crop_size}.yaml"
        runner = yaml.safe_load(path.read_text())
        assert runner["data_module_args"]["num_workers"] == 10
        token_crop = runner["dataset_configs"]["train"]["weighted-pdb"]["config"][
            "crop"
        ]["token_crop"]
        assert token_crop == {"enabled": True, "token_budget": crop_size}
        paths = runner["dataset_paths"]["weighted-pdb"]
        assert paths["dataset_cache_file"] == str(filtered_cache_path)
        assert paths["target_structures_directory"] == str(structures.resolve())

    provenance = json.loads(provenance_path.read_text())
    assert provenance["configuration"]["crop_sizes"] == [4, 6, 8]
    assert provenance["configuration"]["num_workers"] == 10
    assert provenance["selection"]["selected"] == [
        {
            "pdb_id": "good",
            "token_count": 8,
            "resolved_token_center_count": 8,
            "total_unique_chains": 2,
        }
    ]
    assert provenance["selection"]["selected_count"] == 1
    assert provenance["selection"]["rejected_count"] == 3
    assert provenance["outputs"]["filtered_cache"]["sha256"] == _sha256(
        filtered_cache_path
    )
    for crop_size in (4, 6, 8):
        runner_entry = provenance["outputs"]["runner_yamls"][str(crop_size)]
        assert runner_entry["sha256"] == _sha256(Path(runner_entry["path"]))


def test_chain_limit_is_not_applied_when_chain_crop_is_disabled(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "many-chains", n_tokens=8, n_resolved=8, n_chains=8)
    runner_yaml = _write_input(
        tmp_path,
        structures,
        ("many-chains",),
        chain_crop_enabled=False,
        n_chains=1,
    )

    provenance = prepare_crop_benchmark_data(
        runner_yaml=runner_yaml,
        output_dir=tmp_path / "output",
        crop_sizes=(8,),
        num_workers=0,
    )

    report = json.loads(provenance.read_text())
    assert report["selection"]["selected_count"] == 1
    assert report["configuration"]["num_workers"] == 0
    assert report["inclusion_rule"]["chain_count"].startswith("not applied")


def test_rejects_multiple_training_datasets(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "good", n_tokens=8, n_resolved=8, n_chains=1)
    runner_yaml = _write_input(tmp_path, structures, ("good",))
    runner = yaml.safe_load(runner_yaml.read_text())
    runner["dataset_configs"]["train"]["another"] = runner["dataset_configs"]["train"][
        "weighted-pdb"
    ]
    runner_yaml.write_text(yaml.safe_dump(runner))

    with pytest.raises(
        CropBenchmarkDataError, match="must contain exactly one dataset; found 2"
    ):
        prepare_crop_benchmark_data(
            runner_yaml=runner_yaml,
            output_dir=tmp_path / "output",
            crop_sizes=(8,),
        )


def test_refuses_to_overwrite_existing_output_directory(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "good", n_tokens=8, n_resolved=8, n_chains=1)
    runner_yaml = _write_input(tmp_path, structures, ("good",))
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace")

    with pytest.raises(CropBenchmarkDataError, match="refusing to overwrite"):
        prepare_crop_benchmark_data(
            runner_yaml=runner_yaml,
            output_dir=output,
            crop_sizes=(8,),
        )

    assert sentinel.read_text() == "do not replace"


def test_no_qualifying_structure_leaves_no_partial_output(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "short", n_tokens=7, n_resolved=7, n_chains=1)
    runner_yaml = _write_input(tmp_path, structures, ("short",))
    output = tmp_path / "output"

    with pytest.raises(CropBenchmarkDataError, match="no structures satisfy"):
        prepare_crop_benchmark_data(
            runner_yaml=runner_yaml,
            output_dir=output,
            crop_sizes=(8,),
        )

    assert not output.exists()


def test_requires_an_explicit_npz_structure_format(tmp_path: Path):
    structures = tmp_path / "structures"
    _write_structure(structures, "good", n_tokens=8, n_resolved=8, n_chains=1)
    runner_yaml = _write_input(tmp_path, structures, ("good",))
    runner = yaml.safe_load(runner_yaml.read_text())
    runner["dataset_paths"]["weighted-pdb"]["target_structure_file_format"] = "pkl"
    runner_yaml.write_text(yaml.safe_dump(runner))

    with pytest.raises(CropBenchmarkDataError, match="explicitly be 'npz'"):
        prepare_crop_benchmark_data(
            runner_yaml=runner_yaml,
            output_dir=tmp_path / "output",
            crop_sizes=(8,),
        )
