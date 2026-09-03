#!/usr/bin/env python3
"""Fail fast when an OpenFold3 runner YAML references missing training data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import yaml
from ijson.common import JSONError


@dataclass(frozen=True)
class PathProblem:
    dataset: str
    field: str
    expected: str
    path: Path


class TrainingDataError(RuntimeError):
    """Raised when a runner YAML cannot reach its configured dataset paths."""


def _configured_path(value: Any, base_dir: Path) -> Path | None:
    if value is None or not isinstance(value, (str, Path)):
        return None
    if str(value).strip().lower() in ("", "none"):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def find_path_problems(
    config: dict[str, Any],
    *,
    base_dir: Path,
) -> list[PathProblem]:
    """Return missing directory/file references under ``dataset_paths``."""

    dataset_paths = config.get("dataset_paths")
    if not isinstance(dataset_paths, dict) or not dataset_paths:
        raise TrainingDataError("runner YAML has no non-empty dataset_paths mapping")

    problems = []
    for dataset, values in dataset_paths.items():
        if not isinstance(values, dict):
            raise TrainingDataError(f"dataset_paths.{dataset} must be a mapping")
        for field, value in values.items():
            path = _configured_path(value, base_dir)
            if path is None:
                continue
            if field.endswith(("_directory", "_directories")):
                if not path.is_dir():
                    problems.append(
                        PathProblem(str(dataset), str(field), "directory", path)
                    )
            elif field.endswith("_file") and not path.is_file():
                problems.append(PathProblem(str(dataset), str(field), "file", path))
    return problems


def _cache_has_structure(cache_file: Path) -> bool:
    try:
        with cache_file.open("rb") as handle:
            return next(ijson.kvitems(handle, "structure_data"), None) is not None
    except (OSError, ValueError, JSONError) as error:
        raise TrainingDataError(
            f"dataset cache is unreadable: {cache_file}: {error}"
        ) from error


def validate_runner_yaml(runner_yaml: Path, *, base_dir: Path) -> None:
    """Validate every configured dataset path or raise one actionable error."""

    runner_yaml = runner_yaml.expanduser().resolve()
    if not runner_yaml.is_file():
        raise TrainingDataError(f"runner YAML is not a file: {runner_yaml}")
    config = yaml.safe_load(runner_yaml.read_text())
    if not isinstance(config, dict):
        raise TrainingDataError("runner YAML must contain a top-level mapping")

    problems = find_path_problems(config, base_dir=base_dir.expanduser().resolve())
    details = [
        f"  - dataset_paths.{problem.dataset}.{problem.field}: "
        f"expected {problem.expected}: {problem.path}"
        for problem in problems
    ]
    if not problems:
        base_dir = base_dir.expanduser().resolve()
        for dataset, values in config["dataset_paths"].items():
            cache_file = _configured_path(values.get("dataset_cache_file"), base_dir)
            if cache_file is not None and not _cache_has_structure(cache_file):
                details.append(
                    f"  - dataset_paths.{dataset}.dataset_cache_file: "
                    f"contains no structure_data entries: {cache_file}"
                )

    if details:
        raise TrainingDataError(
            f"training data preflight failed for {runner_yaml}:\n"
            + "\n".join(details)
            + "\nNo training process was started. The dataset download is "
            "incomplete, or this runner YAML points at a different download "
            "directory. Complete the download and purification steps, then "
            "regenerate the backend/smoke YAML from "
            "train_pdb_subset.yaml."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_yaml", type=Path)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="directory against which relative dataset paths are resolved",
    )
    args = parser.parse_args()
    try:
        validate_runner_yaml(args.runner_yaml, base_dir=args.base_dir)
    except TrainingDataError as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Training data preflight passed: {args.runner_yaml.resolve()}")


if __name__ == "__main__":
    main()
