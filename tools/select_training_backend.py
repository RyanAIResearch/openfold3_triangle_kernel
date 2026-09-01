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
) -> dict[str, Any]:
    """Update only mutually exclusive training-kernel flags in ``config``."""

    if backend not in BACKEND_FLAGS:
        raise ValueError(f"unknown backend: {backend}")
    model_update = config.setdefault("model_update", {})
    custom = model_update.setdefault("custom", {})
    settings = custom.setdefault("settings", {})
    memory = settings.setdefault("memory", {})
    train_memory = memory.setdefault("train", {})
    train_memory.update(BACKEND_FLAGS[backend])
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_yaml", type=Path)
    parser.add_argument("--backend", required=True, choices=tuple(BACKEND_FLAGS))
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
    )
    destination = args.output or args.runner_yaml
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as target:
        yaml.safe_dump(config, target, sort_keys=False, default_flow_style=False)
    print(f"Wrote {destination} with training backend {args.backend}")


if __name__ == "__main__":
    main()
