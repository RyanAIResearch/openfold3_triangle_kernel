#!/usr/bin/env python3
"""Select one mutually exclusive OpenFold3 training-kernel backend in a YAML.

TurboPairFormer uses the existing production kernel configuration flags.
"""

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
        "use_protenix_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "cueq": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": True,
        "use_stable_triangle_attention": False,
        "use_protenix_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "protenix": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": True,
        "use_stable_triangle_attention": False,
        "use_protenix_triangle_attention": True,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "triton": {
        "use_triton_triangle_kernels": True,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": False,
        "use_protenix_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "turbopairformer": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": True,
        "use_protenix_triangle_attention": False,
        "use_stable_triangle_multiplicative": True,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
    },
    "deepspeed-evo": {
        "use_triton_triangle_kernels": False,
        "use_cueq_triangle_kernels": False,
        "use_stable_triangle_attention": False,
        "use_protenix_triangle_attention": False,
        "use_stable_triangle_multiplicative": False,
        "use_deepspeed_evo_attention": True,
        "use_lma": False,
    },
}
# Preserve the historical CLI backend spelling used by production tooling.
# The YAML flags above remain the host's existing stable-triangle settings.
BACKEND_FLAGS["stablefold"] = BACKEND_FLAGS["turbopairformer"]

# Backends served by the stable triangle kernels, mapped to the provider the
# launcher must export so the run cannot silently pick the other build.
BACKEND_KERNEL_PROVIDER: dict[str, str] = {
    "turbopairformer": "turbopairformer",
    "stablefold": "intree",
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


def configure_single_gpu_smoke(
    config: dict[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    """Configure one validation-free BF16 training step on one GPU."""

    experiment_settings = config.setdefault("experiment_settings", {})
    experiment_settings.update(
        output_dir=f"./openfold3_smoke_{backend}",
        restart_checkpoint_path=None,
    )
    data_module_args = config.setdefault("data_module_args", {})
    data_module_args.update(
        batch_size=1,
        num_workers=0,
        num_workers_validation=0,
        epoch_len=1,
    )
    logging_config = config.setdefault("logging_config", {})
    logging_config["wandb_config"] = None
    checkpoint_config = config.setdefault("checkpoint_config", {})
    checkpoint_config.update(save_top_k=0, save_last=False)
    pl_trainer_args = config.setdefault("pl_trainer_args", {})
    pl_trainer_args.update(
        devices=1,
        num_nodes=1,
        precision="bf16-mixed",
        deepspeed_config_path=None,
        mpi_plugin=False,
        max_epochs=1,
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_yaml", type=Path)
    parser.add_argument("--backend", required=True, choices=tuple(BACKEND_FLAGS))
    parser.add_argument(
        "--provider",
        choices=("intree", "turbopairformer"),
        help="kernel provider for --backend turbopairformer "
        "(default: turbopairformer; legacy stablefold backend defaults to intree); "
        "prints the required export because provider selection is outside YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new YAML instead of replacing runner_yaml",
    )
    parser.add_argument(
        "--single-gpu-smoke",
        action="store_true",
        help="also configure one validation-free BF16 training step on one GPU",
    )
    args = parser.parse_args()
    if args.provider is not None and args.backend not in BACKEND_KERNEL_PROVIDER:
        parser.error("--provider requires --backend turbopairformer or stablefold")

    with args.runner_yaml.open() as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("runner YAML must contain a top-level mapping")
    configure_training_backend(
        config,
        backend=args.backend,
    )
    if args.single_gpu_smoke:
        configure_single_gpu_smoke(config, backend=args.backend)
    destination = args.output or args.runner_yaml
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as target:
        yaml.safe_dump(config, target, sort_keys=False, default_flow_style=False)
    print(f"Wrote {destination} with training backend {args.backend}")
    provider = args.provider or BACKEND_KERNEL_PROVIDER.get(args.backend)
    if provider is not None:
        print(
            f"Export OPENFOLD3_TRIANGLE_PROVIDER={provider} before launching; "
            f"the {args.backend} backend is served by the {provider} kernel build"
        )


if __name__ == "__main__":
    main()
