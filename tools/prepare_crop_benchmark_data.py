#!/usr/bin/env python3
"""Build a no-padding PDB fixture for matched crop-size benchmarks.

The output is deliberately conservative.  A structure is retained only when
the source NPZ annotations contain at least the largest requested number of
tokens, at least that many resolved token-center atoms, and (when chain cropping
is enabled) no more chains than the configured chain-crop limit.  These static
annotations are a conservative prerequisite for full crops; the benchmark's
runtime full-crop check remains the final verification.

The source cache and runner are never modified.  The output directory must not
already exist so a previous benchmark fixture cannot be overwritten by
accident.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CROP_SIZES = (384, 640, 768)
DEFAULT_NUM_WORKERS = 10
DEFAULT_CHAIN_LIMIT = 20


class CropBenchmarkDataError(RuntimeError):
    """Raised when a safe crop benchmark fixture cannot be produced."""


@dataclass(frozen=True)
class StructureMetrics:
    """NPZ-derived metrics used by the conservative inclusion rule."""

    pdb_id: str
    token_count: int
    resolved_token_center_count: int
    total_unique_chains: int


@dataclass(frozen=True)
class TrainingDataset:
    """The one explicitly configured training dataset supported by this tool."""

    name: str
    dataset_class: str
    cache_file: Path
    structures_directory: Path
    chain_crop_enabled: bool
    chain_limit: int


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CropBenchmarkDataError(f"{location} must be a mapping")
    return value


def _resolve_path(value: Any, *, runner_yaml: Path, location: str) -> Path:
    if not isinstance(value, (str, Path)) or str(value).strip().lower() in {
        "",
        "none",
    }:
        raise CropBenchmarkDataError(f"{location} must name a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = runner_yaml.parent / path
    return path.resolve()


def _chain_crop_default(dataset_class: str) -> bool:
    # These are the two training dataset classes whose checked-in config default
    # enables chain cropping.  Other current training dataset classes inherit
    # ChainCropSettings(enabled=False).
    return dataset_class in {"WeightedPDBDataset", "DisorderedPDBDataset"}


def _extract_training_dataset(
    config: dict[str, Any], *, runner_yaml: Path
) -> TrainingDataset:
    dataset_configs = _mapping(config.get("dataset_configs"), "dataset_configs")
    train = _mapping(dataset_configs.get("train"), "dataset_configs.train")
    if len(train) != 1:
        raise CropBenchmarkDataError(
            "dataset_configs.train must contain exactly one dataset; "
            f"found {len(train)}"
        )

    dataset_name, raw_spec = next(iter(train.items()))
    location = f"dataset_configs.train.{dataset_name}"
    spec = _mapping(raw_spec, location)
    dataset_class = spec.get("dataset_class")
    if not isinstance(dataset_class, str) or not dataset_class:
        raise CropBenchmarkDataError(f"{location}.dataset_class must be explicit")

    dataset_config = _mapping(spec.get("config"), f"{location}.config")
    crop = _mapping(dataset_config.get("crop", {}), f"{location}.config.crop")
    token_crop = _mapping(
        crop.get("token_crop", {}), f"{location}.config.crop.token_crop"
    )
    token_crop_enabled = token_crop.get("enabled", True)
    if token_crop_enabled is not True:
        raise CropBenchmarkDataError(
            f"{location}.config.crop.token_crop.enabled must be true"
        )

    chain_crop = _mapping(
        crop.get("chain_crop", {}), f"{location}.config.crop.chain_crop"
    )
    chain_crop_enabled = chain_crop.get("enabled", _chain_crop_default(dataset_class))
    if not isinstance(chain_crop_enabled, bool):
        raise CropBenchmarkDataError(
            f"{location}.config.crop.chain_crop.enabled must be boolean"
        )
    chain_limit = chain_crop.get("n_chains", DEFAULT_CHAIN_LIMIT)
    if isinstance(chain_limit, bool) or not isinstance(chain_limit, int):
        raise CropBenchmarkDataError(
            f"{location}.config.crop.chain_crop.n_chains must be an integer"
        )
    if chain_limit < 1:
        raise CropBenchmarkDataError(
            f"{location}.config.crop.chain_crop.n_chains must be positive"
        )

    dataset_paths = _mapping(config.get("dataset_paths"), "dataset_paths")
    if dataset_name not in dataset_paths:
        raise CropBenchmarkDataError(
            f"dataset_paths has no entry for training dataset {dataset_name!r}"
        )
    paths = _mapping(dataset_paths[dataset_name], f"dataset_paths.{dataset_name}")
    structure_format = paths.get("target_structure_file_format")
    if not isinstance(structure_format, str) or structure_format.lower() != "npz":
        raise CropBenchmarkDataError(
            f"dataset_paths.{dataset_name}.target_structure_file_format must "
            "explicitly be 'npz'"
        )

    cache_file = _resolve_path(
        paths.get("dataset_cache_file"),
        runner_yaml=runner_yaml,
        location=f"dataset_paths.{dataset_name}.dataset_cache_file",
    )
    structures_directory = _resolve_path(
        paths.get("target_structures_directory"),
        runner_yaml=runner_yaml,
        location=f"dataset_paths.{dataset_name}.target_structures_directory",
    )
    if not cache_file.is_file():
        raise CropBenchmarkDataError(f"training cache is not a file: {cache_file}")
    if not structures_directory.is_dir():
        raise CropBenchmarkDataError(
            f"target structures directory does not exist: {structures_directory}"
        )

    return TrainingDataset(
        name=str(dataset_name),
        dataset_class=dataset_class,
        cache_file=cache_file,
        structures_directory=structures_directory,
        chain_crop_enabled=chain_crop_enabled,
        chain_limit=chain_limit,
    )


def _load_cache(cache_file: Path) -> dict[str, Any]:
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CropBenchmarkDataError(
            f"training cache is not readable JSON: {cache_file}: {error}"
        ) from error
    if not isinstance(cache, dict):
        raise CropBenchmarkDataError("training cache must contain a JSON object")
    if not isinstance(cache.get("_type"), str):
        raise CropBenchmarkDataError(
            "training cache must contain the string field '_type'"
        )
    structure_data = cache.get("structure_data")
    if not isinstance(structure_data, dict) or not structure_data:
        raise CropBenchmarkDataError(
            "training cache must contain non-empty 'structure_data'"
        )
    return cache


def _structure_path(dataset: TrainingDataset, pdb_id: str) -> Path:
    path = (dataset.structures_directory / pdb_id / f"{pdb_id}.npz").resolve()
    if dataset.structures_directory != path.parent.parent:
        raise CropBenchmarkDataError(
            f"structure ID escapes target_structures_directory: {pdb_id!r}"
        )
    return path


def read_structure_metrics(path: Path, *, pdb_id: str) -> StructureMetrics:
    """Recompute the inclusion metrics from one preprocessed structure NPZ."""

    if not path.is_file():
        raise CropBenchmarkDataError(f"structure NPZ is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as structure:
            required = {"token_id", "token_center_atom", "occupancy", "chain_id"}
            missing = sorted(required.difference(structure.files))
            if missing:
                raise CropBenchmarkDataError(
                    f"structure NPZ is missing arrays {missing}: {path}"
                )
            token_id = np.asarray(structure["token_id"])
            token_center_atom = np.asarray(structure["token_center_atom"])
            occupancy = np.asarray(structure["occupancy"])
            chain_id = np.asarray(structure["chain_id"])
    except CropBenchmarkDataError:
        raise
    except Exception as error:
        raise CropBenchmarkDataError(
            f"structure NPZ is unreadable: {path}: {type(error).__name__}: {error}"
        ) from error

    arrays = {
        "token_id": token_id,
        "token_center_atom": token_center_atom,
        "occupancy": occupancy,
        "chain_id": chain_id,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise CropBenchmarkDataError(
                f"{path}: {name} must be one-dimensional, got shape {array.shape}"
            )
        if array.shape != token_id.shape:
            raise CropBenchmarkDataError(
                f"{path}: {name} shape {array.shape} does not match token_id "
                f"shape {token_id.shape}"
            )
    if token_id.size == 0:
        raise CropBenchmarkDataError(f"structure NPZ has no atoms: {path}")

    try:
        resolved_centers = np.asarray(token_center_atom, dtype=bool) & (occupancy > 0)
    except (TypeError, ValueError) as error:
        raise CropBenchmarkDataError(
            f"{path}: token_center_atom and occupancy must be numeric/boolean arrays"
        ) from error

    return StructureMetrics(
        pdb_id=pdb_id,
        token_count=int(np.unique(token_id).size),
        resolved_token_center_count=int(np.count_nonzero(resolved_centers)),
        total_unique_chains=int(np.unique(chain_id).size),
    )


def _normalise_crop_sizes(crop_sizes: Iterable[int]) -> tuple[int, ...]:
    sizes = tuple(crop_sizes)
    if not sizes:
        raise CropBenchmarkDataError("at least one crop size is required")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise CropBenchmarkDataError("crop sizes must be integers")
    if any(size < 1 for size in sizes):
        raise CropBenchmarkDataError("crop sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise CropBenchmarkDataError("crop sizes must be unique")
    return tuple(sorted(sizes))


def _selected_by_rule(
    metrics: StructureMetrics,
    *,
    max_crop: int,
    chain_crop_enabled: bool,
    chain_limit: int,
) -> bool:
    return (
        metrics.token_count >= max_crop
        and metrics.resolved_token_center_count >= max_crop
        and (not chain_crop_enabled or metrics.total_unique_chains <= chain_limit)
    )


def _filtered_cache(
    source_cache: dict[str, Any],
    *,
    selected_ids: set[str],
    max_crop: int,
) -> dict[str, Any]:
    # Keep _type first: the OpenFold3 JSON cache reader intentionally discovers
    # the cache class from the second line of an indented JSON document.
    filtered = {"_type": source_cache["_type"]}
    filtered.update(
        (key, value) for key, value in source_cache.items() if key != "_type"
    )
    filtered["name"] = (
        f"{source_cache.get('name', 'training-cache')}-no-padding-{max_crop}"
    )
    filtered["structure_data"] = {
        pdb_id: entry
        for pdb_id, entry in source_cache["structure_data"].items()
        if pdb_id in selected_ids
    }
    return filtered


def _runner_for_crop(
    source_config: dict[str, Any],
    *,
    dataset_name: str,
    crop_size: int,
    num_workers: int,
    filtered_cache: Path,
    structures_directory: Path,
) -> dict[str, Any]:
    runner = copy.deepcopy(source_config)
    data_module_args = runner.setdefault("data_module_args", {})
    if not isinstance(data_module_args, dict):
        raise CropBenchmarkDataError("data_module_args must be a mapping")
    data_module_args["num_workers"] = num_workers

    spec = runner["dataset_configs"]["train"][dataset_name]
    dataset_config = spec.setdefault("config", {})
    crop = dataset_config.setdefault("crop", {})
    token_crop = crop.setdefault("token_crop", {})
    token_crop["enabled"] = True
    token_crop["token_budget"] = crop_size

    paths = runner["dataset_paths"][dataset_name]
    paths["dataset_cache_file"] = str(filtered_cache)
    paths["target_structures_directory"] = str(structures_directory)
    return runner


def prepare_crop_benchmark_data(
    *,
    runner_yaml: Path,
    output_dir: Path,
    crop_sizes: Iterable[int] = DEFAULT_CROP_SIZES,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Path:
    """Create the fixture and return the final provenance JSON path."""

    runner_yaml = runner_yaml.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    crop_sizes = _normalise_crop_sizes(crop_sizes)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise CropBenchmarkDataError("num_workers must be an integer")
    if num_workers < 0:
        raise CropBenchmarkDataError("num_workers must be non-negative")
    if not runner_yaml.is_file():
        raise CropBenchmarkDataError(f"runner YAML is not a file: {runner_yaml}")
    if output_dir.exists():
        raise CropBenchmarkDataError(
            f"output directory already exists; refusing to overwrite it: {output_dir}"
        )

    try:
        source_config = yaml.safe_load(runner_yaml.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CropBenchmarkDataError(
            f"runner YAML is unreadable: {runner_yaml}: {error}"
        ) from error
    source_config = _mapping(source_config, "runner YAML")
    dataset = _extract_training_dataset(source_config, runner_yaml=runner_yaml)
    source_cache = _load_cache(dataset.cache_file)
    max_crop = max(crop_sizes)

    selected_metrics: list[StructureMetrics] = []
    rejected: list[dict[str, Any]] = []
    for raw_pdb_id in source_cache["structure_data"]:
        pdb_id = str(raw_pdb_id)
        try:
            metrics = read_structure_metrics(
                _structure_path(dataset, pdb_id), pdb_id=pdb_id
            )
        except CropBenchmarkDataError as error:
            rejected.append({"pdb_id": pdb_id, "reason": str(error)})
            continue
        if _selected_by_rule(
            metrics,
            max_crop=max_crop,
            chain_crop_enabled=dataset.chain_crop_enabled,
            chain_limit=dataset.chain_limit,
        ):
            selected_metrics.append(metrics)
        else:
            failed_rules = []
            if metrics.token_count < max_crop:
                failed_rules.append("token_count")
            if metrics.resolved_token_center_count < max_crop:
                failed_rules.append("resolved_token_center_count")
            if (
                dataset.chain_crop_enabled
                and metrics.total_unique_chains > dataset.chain_limit
            ):
                failed_rules.append("total_unique_chains")
            rejected.append(
                {
                    **asdict(metrics),
                    "reason": "failed inclusion rule: " + ", ".join(failed_rules),
                }
            )

    if not selected_metrics:
        example_reasons = "; ".join(
            f"{entry['pdb_id']}: {entry['reason']}" for entry in rejected[:3]
        )
        raise CropBenchmarkDataError(
            f"no structures satisfy the no-padding rule for max crop {max_crop}"
            + (f"; examples: {example_reasons}" if example_reasons else "")
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        final_cache = output_dir / f"training_cache_no_padding_{max_crop}.json"
        staging_cache = staging_dir / final_cache.name
        selected_ids = {metrics.pdb_id for metrics in selected_metrics}
        cache = _filtered_cache(
            source_cache, selected_ids=selected_ids, max_crop=max_crop
        )
        staging_cache.write_text(json.dumps(cache, indent=2) + "\n")

        runner_outputs: dict[str, dict[str, str]] = {}
        for crop_size in crop_sizes:
            final_runner = output_dir / f"runner_crop_{crop_size}.yaml"
            staging_runner = staging_dir / final_runner.name
            runner = _runner_for_crop(
                source_config,
                dataset_name=dataset.name,
                crop_size=crop_size,
                num_workers=num_workers,
                filtered_cache=final_cache,
                structures_directory=dataset.structures_directory,
            )
            staging_runner.write_text(
                yaml.safe_dump(runner, sort_keys=False, width=1000)
            )
            runner_outputs[str(crop_size)] = {
                "path": str(final_runner),
                "sha256": sha256_file(staging_runner),
            }

        provenance = {
            "schema_version": 1,
            "inputs": {
                "runner_yaml": {
                    "path": str(runner_yaml),
                    "sha256": sha256_file(runner_yaml),
                },
                "training_cache": {
                    "path": str(dataset.cache_file),
                    "sha256": sha256_file(dataset.cache_file),
                },
                "target_structures_directory": str(dataset.structures_directory),
            },
            "configuration": {
                "training_dataset": dataset.name,
                "dataset_class": dataset.dataset_class,
                "structure_format": "npz",
                "crop_sizes": list(crop_sizes),
                "max_crop": max_crop,
                "num_workers": num_workers,
                "chain_crop_enabled": dataset.chain_crop_enabled,
                "chain_limit": dataset.chain_limit,
            },
            "inclusion_rule": {
                "metric_source": "preprocessed structure NPZ arrays",
                "token_count": f">= {max_crop} unique token_id values",
                "resolved_token_center_count": (
                    f">= {max_crop} atoms where token_center_atom and occupancy > 0"
                ),
                "chain_count": (
                    f"<= {dataset.chain_limit} unique chain_id values"
                    if dataset.chain_crop_enabled
                    else "not applied because chain cropping is disabled"
                ),
                "guarantee": (
                    "conservative static sufficient condition under the stored NPZ "
                    "annotations; runtime full-crop validation remains independent"
                ),
            },
            "selection": {
                "input_count": len(source_cache["structure_data"]),
                "selected_count": len(selected_metrics),
                "rejected_count": len(rejected),
                "selected": [asdict(metrics) for metrics in selected_metrics],
                "rejected": rejected,
            },
            "outputs": {
                "filtered_cache": {
                    "path": str(final_cache),
                    "sha256": sha256_file(staging_cache),
                },
                "runner_yamls": runner_outputs,
            },
        }
        provenance_path = staging_dir / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

        if output_dir.exists():
            raise CropBenchmarkDataError(
                "output directory appeared while preparing data; refusing to "
                f"overwrite it: {output_dir}"
            )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return output_dir / "provenance.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner_yaml", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--crop-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_CROP_SIZES,
        metavar="N",
        help="crop sizes to generate (default: 384 640 768)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="training DataLoader workers in each output runner (default: 10)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        provenance = prepare_crop_benchmark_data(
            runner_yaml=args.runner_yaml,
            output_dir=args.output_dir,
            crop_sizes=args.crop_sizes,
            num_workers=args.num_workers,
        )
    except CropBenchmarkDataError as error:
        parser.exit(2, f"ERROR: {error}\n")
    report = json.loads(provenance.read_text())
    print(
        "Crop benchmark data is ready: "
        f"{report['selection']['selected_count']}/"
        f"{report['selection']['input_count']} structures selected"
    )
    print(f"Provenance: {provenance}")


if __name__ == "__main__":
    main()
