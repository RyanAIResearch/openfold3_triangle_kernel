#!/usr/bin/env python3
"""Host and data preflight for the TurboPairFormer H100 Pixi environment.

This module intentionally does not launch training or modify an OpenFold3 runner
configuration. Training remains owned by AQLab's ``run_openfold`` entry point and
PyTorch's standard ``torchrun`` launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.manage_attested_sm90_wheel import source_commit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = Path("/mnt/shared/openfold3")
DEFAULT_DATA_URI = "s3://openfold3-data/pdb_training_set/"


class WorkflowError(RuntimeError):
    """Raised when a host or data precondition is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_work_root(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise WorkflowError("training data must live outside the source checkout")
    return path


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    """Download and validate the published PDB portion of the training data."""

    work_root = _resolved_work_root(args.work_root)
    data_root = work_root / "pdb_training_set"
    if not args.skip_download:
        subprocess.run(
            (
                "aws",
                "s3",
                "sync",
                args.data_uri,
                str(data_root) + "/",
                "--no-sign-request",
            ),
            check=True,
        )
    required = (
        data_root / "dataset_caches" / "training_cache_with_templates.json",
        data_root / "dataset_caches" / "validation_cache_with_templates.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise WorkflowError(f"published PDB training data is incomplete: {missing}")
    return {
        "status": "PASS",
        "data_root": str(data_root),
        "cache_sha256": {str(path): _sha256(path) for path in required},
        "runner_yaml_modified": False,
    }


def doctor_h100(_: argparse.Namespace) -> dict[str, Any]:
    """Verify the exact local H100 runtime prerequisites."""

    import pytorch_lightning as pl
    import torch

    if sys.version_info[:2] != (3, 14):
        raise WorkflowError(f"expected Python 3.14, got {sys.version}")
    if torch.__version__.split("+", 1)[0] != "2.10.0":
        raise WorkflowError(f"expected Torch 2.10.0, got {torch.__version__}")
    if torch.version.cuda != "12.9":
        raise WorkflowError(f"expected Torch CUDA 12.9, got {torch.version.cuda}")
    if pl.__version__ != "2.6.1":
        raise WorkflowError(f"expected Lightning 2.6.1, got {pl.__version__}")
    if not torch.cuda.is_available():
        raise WorkflowError("an available CUDA GPU is required")

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        if capability != (9, 0) or "H100" not in properties.name:
            raise WorkflowError(
                f"device {index} is not a qualified full H100: "
                f"{properties.name}, cc={capability}"
            )
        if properties.total_memory < 75 * 1024**3:
            raise WorkflowError(
                f"device {index} exposes too little memory for full H100: "
                f"{properties.total_memory}"
            )
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": list(capability),
                "total_memory": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    if not devices:
        raise WorkflowError("the H100 Pixi environment exposes no CUDA devices")

    return {
        "status": "PASS",
        "commit": source_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightning": pl.__version__,
        "devices": devices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor-h100")
    data = subparsers.add_parser("prepare-training-data")
    data.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    data.add_argument("--data-uri", default=DEFAULT_DATA_URI)
    data.add_argument("--skip-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor-h100":
        result = doctor_h100(args)
    elif args.command == "prepare-training-data":
        result = prepare_data(args)
    else:  # pragma: no cover - argparse owns the command choices.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
