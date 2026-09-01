#!/usr/bin/env python3
"""Select one mutually exclusive OpenFold3 training-kernel backend in a YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

BACKEND_FLAGS: dict[str, dict[str, bool]] = {
    "aqlab": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "cueq": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": True,
        "use_stable_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "stablefold": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": True,
        "use_stable_triangle_multiplicative": True,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "deepspeed-evo": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": True,
        "use_lma": False,
    },
}


def configure_training_backend(
    config: dict[str, Any],
    *,
    backend: str,
    output_dir: str | None = None,
    devices_per_node: int | None = None,
    num_nodes: int | None = None,
    epoch_len: int | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Update only backend and distributed-run settings in ``config``."""

    if backend not in BACKEND_FLAGS:
        raise ValueError(f"unknown backend: {backend}")
    trainer = config.setdefault("pl_trainer_args", {})
    data = config.setdefault("data_module_args", {})
    experiment = config.setdefault("experiment_settings", {})
    model_update = config.setdefault("model_update", {})
    custom = model_update.setdefault("custom", {})
    settings = custom.setdefault("settings", {})
    memory = settings.setdefault("memory", {})
    train_memory = memory.setdefault("train", {})

    if output_dir is not None:
        experiment["output_dir"] = output_dir
    if devices_per_node is not None:
        if devices_per_node < 1:
            raise ValueError("devices_per_node must be positive")
        trainer["devices"] = devices_per_node
    if num_nodes is not None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        trainer["num_nodes"] = num_nodes
    if epoch_len is not None:
        if epoch_len < 1:
            raise ValueError("epoch_len must be positive")
        data["epoch_len"] = epoch_len
    if fresh:
        experiment["restart_checkpoint_path"] = None

    world_size = int(trainer.get("devices", 1)) * int(trainer.get("num_nodes", 1))
    resolved_epoch_len = int(data.get("epoch_len", 0))
    if resolved_epoch_len < 1 or resolved_epoch_len % world_size:
        raise ValueError(
            "data_module_args.epoch_len must be positive and divisible by "
            f"devices*num_nodes ({world_size})"
        )
    if int(data.get("batch_size", 1)) != 1 and backend == "stablefold":
        raise ValueError("StableFold distributed v1 requires batch_size=1")

    train_memory.update(BACKEND_FLAGS[backend])
    train_memory["chunk_size"] = None
    trainer["deepspeed_config_path"] = None
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_yaml", type=Path)
    parser.add_argument("--backend", required=True, choices=tuple(BACKEND_FLAGS))
    parser.add_argument(
        "--output-dir",
        help="set a backend-specific run/output directory",
    )
    parser.add_argument("--devices-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--epoch-len", type=int)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="set restart_checkpoint_path to null for a fresh run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new YAML instead of replacing runner_yaml",
    )
    args = parser.parse_args()

    with args.runner_yaml.open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("runner YAML must contain a top-level mapping")
    configure_training_backend(
        config,
        backend=args.backend,
        output_dir=args.output_dir,
        devices_per_node=args.devices_per_node,
        num_nodes=args.num_nodes,
        epoch_len=args.epoch_len,
        fresh=args.fresh,
    )
    destination = args.output or args.runner_yaml
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as target:
        yaml.safe_dump(config, target, sort_keys=False, default_flow_style=False)
    print(f"Wrote {destination} with training backend {args.backend}")


if __name__ == "__main__":
    main()
