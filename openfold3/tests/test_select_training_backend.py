from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from openfold3.entry_points.validator import TrainingExperimentConfig
from tools.select_training_backend import BACKEND_FLAGS, configure_training_backend

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def base_config():
    return {
        "experiment_settings": {"restart_checkpoint_path": "last"},
        "data_module_args": {"batch_size": 1, "epoch_len": 64},
        "pl_trainer_args": {
            "devices": 1,
            "num_nodes": 1,
            "deepspeed_config_path": None,
        },
        "model_update": {"custom": {"settings": {"memory": {"train": {}}}}},
    }


@pytest.mark.parametrize("backend", tuple(BACKEND_FLAGS))
def test_backend_selectors_are_exact_and_mutually_exclusive(base_config, backend):
    configured = configure_training_backend(deepcopy(base_config), backend=backend)
    train = configured["model_update"]["custom"]["settings"]["memory"]["train"]
    for key, expected in BACKEND_FLAGS[backend].items():
        assert train[key] is expected
    assert train["chunk_size"] is None


def test_aqlab_selects_the_native_stock_backend():
    assert not any(BACKEND_FLAGS["aqlab"].values())


def test_stablefold_distributed_fresh_config(base_config):
    configured = configure_training_backend(
        deepcopy(base_config),
        backend="stablefold",
        output_dir="/shared/runs/stablefold",
        devices_per_node=8,
        num_nodes=2,
        epoch_len=128,
        fresh=True,
    )
    assert configured["pl_trainer_args"]["devices"] == 8
    assert configured["pl_trainer_args"]["num_nodes"] == 2
    assert configured["data_module_args"]["epoch_len"] == 128
    assert configured["experiment_settings"]["restart_checkpoint_path"] is None
    assert configured["experiment_settings"]["output_dir"] == "/shared/runs/stablefold"


def test_rejects_epoch_len_that_cannot_be_sharded(base_config):
    with pytest.raises(ValueError, match="divisible"):
        configure_training_backend(
            deepcopy(base_config),
            backend="stablefold",
            devices_per_node=8,
            num_nodes=2,
            epoch_len=63,
        )


def test_distributed_example_is_schema_valid(tmp_path):
    config = yaml.safe_load(
        (
            REPOSITORY / "examples/training_yamls/stablefold_distributed_pdb.yml"
        ).read_text()
    )
    config["experiment_settings"]["output_dir"] = str(tmp_path / "run")

    shared_paths = {
        "alignment_array_directory": tmp_path / "alignment_arrays",
        "target_structures_directory": tmp_path / "structure_files",
        "reference_molecule_directory": tmp_path / "reference_mols",
        "template_cache_directory": tmp_path / "template_cache",
        "template_structure_array_directory": tmp_path / "template_arrays",
    }
    for path in shared_paths.values():
        path.mkdir()
    for dataset_name, dataset_paths in config["dataset_paths"].items():
        cache = tmp_path / f"{dataset_name}.json"
        cache.write_text("{}\n")
        dataset_paths["dataset_cache_file"] = str(cache)
        dataset_paths.update({key: str(path) for key, path in shared_paths.items()})

    TrainingExperimentConfig.model_validate(config)
