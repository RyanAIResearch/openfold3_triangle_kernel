from copy import deepcopy

import pytest

from tools.select_training_backend import (
    BACKEND_FLAGS,
    configure_single_gpu_smoke,
    configure_training_backend,
)


@pytest.fixture
def base_config():
    return {
        "experiment_settings": {
            "output_dir": "/shared/upstream-output",
            "restart_checkpoint_path": "last",
        },
        "data_module_args": {"batch_size": 3, "epoch_len": 65},
        "logging_config": {"wandb_config": {"project": "unchanged"}},
        "pl_trainer_args": {
            "devices": 7,
            "num_nodes": 3,
            "deepspeed_config_path": "/shared/upstream-ds.json",
        },
        "model_update": {
            "custom": {
                "settings": {
                    "memory": {
                        "train": {"chunk_size": 17, "unrelated_train_key": "keep"},
                        "eval": {"chunk_size": 23},
                    },
                },
            },
        },
    }


@pytest.mark.parametrize("backend", tuple(BACKEND_FLAGS))
def test_backend_selector_changes_only_route_flags(base_config, backend):
    expected = deepcopy(base_config)
    expected["model_update"]["custom"]["settings"]["memory"]["train"].update(
        BACKEND_FLAGS[backend]
    )

    configured = configure_training_backend(deepcopy(base_config), backend=backend)

    assert configured == expected


def test_backend_selector_creates_only_the_train_memory_path():
    configured = configure_training_backend({}, backend="stablefold")
    assert configured == {
        "model_update": {
            "custom": {
                "settings": {
                    "memory": {"train": dict(BACKEND_FLAGS["stablefold"])},
                },
            },
        },
    }


def test_turbopairformer_and_legacy_name_generate_identical_training_config(
    base_config,
):
    canonical = configure_training_backend(
        deepcopy(base_config),
        backend="turbopairformer",
    )
    legacy = configure_training_backend(deepcopy(base_config), backend="stablefold")
    assert canonical == legacy
    train_memory = canonical["model_update"]["custom"]["settings"]["memory"]["train"]
    assert train_memory["use_stable_triangle_attention"] is True
    assert train_memory["use_stable_triangle_multiplicative"] is True


def test_backend_selector_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        configure_training_backend({}, backend="not-a-backend")


def test_aqlab_selects_native_stock_routes():
    assert not any(BACKEND_FLAGS["aqlab"].values())


def test_each_accelerated_backend_has_exactly_one_operator_family():
    assert BACKEND_FLAGS["cueq"]["use_cueq_triangle_kernels"] is True
    assert BACKEND_FLAGS["stablefold"]["use_stable_triangle_attention"] is True
    assert BACKEND_FLAGS["stablefold"]["use_stable_triangle_multiplicative"] is True
    assert BACKEND_FLAGS["deepspeed-evo"]["use_deepspeed_evo_attention"] is True
    assert BACKEND_FLAGS["triton"]["use_triton_triangle_kernels"] is True
    for name, flags in BACKEND_FLAGS.items():
        if name == "aqlab":
            continue
        # Every accelerated backend enables exactly one operator family; the
        # TurboPairFormer family spans its attention and TriMul flags.
        families = {
            "cueq": flags["use_cueq_triangle_kernels"],
            "triton": flags["use_triton_triangle_kernels"],
            "deepspeed-evo": flags["use_deepspeed_evo_attention"],
            "stablefold": flags["use_stable_triangle_attention"]
            or flags["use_stable_triangle_multiplicative"],
        }
        assert sum(families.values()) == 1, name


def test_single_gpu_smoke_is_one_validation_free_training_step(base_config):
    configured = configure_single_gpu_smoke(base_config, backend="stablefold")

    assert configured["experiment_settings"] == {
        "output_dir": "./openfold3_smoke_stablefold",
        "restart_checkpoint_path": None,
    }
    assert configured["data_module_args"] == {
        "batch_size": 1,
        "epoch_len": 1,
        "num_workers": 0,
        "num_workers_validation": 0,
    }
    assert configured["logging_config"]["wandb_config"] is None
    assert configured["checkpoint_config"] == {
        "save_top_k": 0,
        "save_last": False,
    }
    assert configured["pl_trainer_args"] == {
        "devices": 1,
        "num_nodes": 1,
        "deepspeed_config_path": None,
        "mpi_plugin": False,
        "precision": "bf16-mixed",
        "max_epochs": 1,
        "max_steps": 1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
    }
