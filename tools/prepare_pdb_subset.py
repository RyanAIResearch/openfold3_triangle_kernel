#!/usr/bin/env python3
"""Generate, download, purify, and validate a runnable PDB training subset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from tools.validate_training_data import validate_runner_yaml
except ModuleNotFoundError:
    from validate_training_data import validate_runner_yaml

ROOT = Path(__file__).resolve().parents[1]
VAL_SIZE = 4


def _run(script: str, *arguments: str) -> None:
    subprocess.run(
        (sys.executable, "-u", str(ROOT / "scripts" / "datasets" / script), *arguments),
        check=True,
        cwd=ROOT,
    )


def prepare_subset(
    *,
    target_dir: Path,
    output_dir: Path,
    train_size: int,
    seed: int,
    workers: int,
    download_attempts: int,
    force: bool,
) -> Path:
    """Run the complete subset workflow and return its validated runner YAML."""

    target_dir = target_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    common = ("--target-dir", str(target_dir), "--train-size", str(train_size))
    generate_arguments = (
        *common,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        # The supported all-in-one path publishes only the final, purified
        # runner YAML. Avoid leaving an unpurified intermediate runner with
        # the same user-facing filename.
        "--runner-yaml",
        "",
    )
    if force:
        generate_arguments = (*generate_arguments, "--force")
    _run("generate_subset_cache.py", *generate_arguments)

    for attempt in range(1, download_attempts + 1):
        print(f"Download pass {attempt}/{download_attempts}", flush=True)
        _run(
            "download_subset.py",
            *common,
            "--val-size",
            str(VAL_SIZE),
            "--workers",
            str(workers),
            "--output-dir",
            str(output_dir),
        )

    runner_yaml = target_dir / "train_pdb_subset.yaml"
    _run(
        "purify_subset.py",
        *common,
        "--val-size",
        str(VAL_SIZE),
        "--output-dir",
        str(output_dir),
        "--runner-yaml",
        str(runner_yaml),
    )
    validate_runner_yaml(runner_yaml, base_dir=ROOT)
    return runner_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--download-attempts", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1:
        parser.error("--train-size must be positive")
    if args.workers < 1 or args.download_attempts < 1:
        parser.error("--workers and --download-attempts must be positive")

    target_dir = args.target_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else target_dir / "pdb_training_set"
    )
    runner_yaml = prepare_subset(
        target_dir=target_dir,
        output_dir=output_dir,
        train_size=args.train_size,
        seed=args.seed,
        workers=args.workers,
        download_attempts=args.download_attempts,
        force=args.force,
    )
    print(f"PDB subset is ready for training: {runner_yaml}")


if __name__ == "__main__":
    main()
