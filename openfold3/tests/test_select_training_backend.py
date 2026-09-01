from copy import deepcopy

import pytest

from tools.select_training_backend import BACKEND_FLAGS, configure_training_backend


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
