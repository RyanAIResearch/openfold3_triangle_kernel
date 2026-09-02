#!/usr/bin/env python3
"""Remove PDB subset entries whose downloaded artifacts are incomplete.

Run this after ``download_subset.py``. A failed or permanently missing S3 object
(including a 404/NoSuchKey response) leaves no local file; this script removes
every sample that depends on such a file and writes new cache files plus a new
runner YAML. The original sampled caches are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _reference_mol_path(ccd_id: str, local_root: Path) -> Path:
    return (
        local_root
        / "preprocessed_pdb_data"
        / "standard"
        / "reference_mols"
        / f"{ccd_id}.sdf"
    )


def _structure_file_path(pdb_id: str, local_root: Path) -> Path:
    return (
        local_root
        / "preprocessed_pdb_data"
        / "standard"
        / "structure_files"
        / pdb_id
        / f"{pdb_id}.npz"
    )


def _scan_structure_residue_names(path: Path) -> set[str]:
    data = np.load(path, allow_pickle=True)
    return {str(name) for name in np.unique(data["res_name"])}


def missing_artifacts(
    pdb_id: str,
    entry: dict[str, Any],
    *,
    template_cache_subdir: str,
    local_root: Path,
) -> list[str]:
    """Return missing local artifacts required by one cache entry."""

    missing: set[str] = set()
    target = _structure_file_path(pdb_id, local_root)
    if not target.is_file():
        missing.add(str(target))

    for chain_data in entry.get("chains", {}).values():
        if rep_id := chain_data.get("alignment_representative_id"):
            alignment = local_root / "alignment_arrays" / f"{rep_id}.npz"
            template_cache = (
                local_root
                / "templates"
                / template_cache_subdir
                / f"{rep_id}.npz"
            )
            if not alignment.is_file():
                missing.add(str(alignment))
            if not template_cache.is_file():
                missing.add(str(template_cache))

        for template_id in chain_data.get("template_ids") or []:
            template_pdb = template_id.split("_")[0]
            template_root = (
                local_root
                / "templates"
                / "template_structure_arrays"
                / template_pdb
            )
            template = template_root / f"{template_id}.npz"
            moltype = template_root / "chain_id_to_moltype.npz"
            if not template.is_file():
                missing.add(str(template))
            if not moltype.is_file():
                missing.add(str(moltype))

        if reference_mol_id := chain_data.get("reference_mol_id"):
            reference_mol = _reference_mol_path(reference_mol_id, local_root)
            if not reference_mol.is_file():
                missing.add(str(reference_mol))

    if target.is_file():
        try:
            residue_names = _scan_structure_residue_names(target)
        except Exception as error:  # corrupt/truncated npz files are invalid samples
            missing.add(f"{target} (unreadable: {type(error).__name__})")
        else:
            for residue_name in residue_names:
                reference_mol = _reference_mol_path(residue_name, local_root)
                if not reference_mol.is_file():
                    missing.add(str(reference_mol))

    return sorted(missing)


def purify_cache(
    source: Path,
    destination: Path,
    *,
    template_cache_subdir: str,
    local_root: Path,
) -> dict[str, Any]:
    """Write a cache containing only entries with complete local artifacts."""

    cache = json.loads(source.read_text())
    kept: dict[str, Any] = {}
    removed: dict[str, list[str]] = {}

    for pdb_id, entry in cache["structure_data"].items():
        missing = missing_artifacts(
            pdb_id,
            entry,
            template_cache_subdir=template_cache_subdir,
            local_root=local_root,
        )
        if missing:
            removed[pdb_id] = missing
        else:
            kept[pdb_id] = entry

    purified = dict(cache)
    purified["name"] = f"{cache.get('name', source.stem)}-purified-{len(kept)}"
    purified["structure_data"] = kept
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(purified, indent=2) + "\n")

    return {
        "source": str(source),
        "output": str(destination),
        "input_count": len(cache["structure_data"]),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "removed": removed,
    }


def _purified_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_purified{source.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=None,
        help=(
            "Purified runner YAML (default: "
            "<target-dir>/train_pdb_subset.purified.yaml)"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    try:
        from .pdb_subset_helpers import (  # noqa: TID252
            default_target_dir,
            subset_cache_path,
            write_runner_yaml,
        )
    except ImportError:
        from pdb_subset_helpers import (  # type: ignore[no-redef]
            default_target_dir,
            subset_cache_path,
            write_runner_yaml,
        )

    target_dir = (args.target_dir or default_target_dir()).expanduser().resolve()
    local_root = (
        args.output_dir or (target_dir / "pdb_training_set")
    ).expanduser().resolve()
    if not local_root.is_dir():
        parser.error(
            f"download directory not found: {local_root.expanduser().resolve()}. "
            "Run download-pdb-subset successfully before purification."
        )
    train_source = subset_cache_path(
        target_dir / "training_cache_with_templates.json",
        args.train_size,
        target_dir,
    )
    val_source = subset_cache_path(
        target_dir / "validation_cache_with_templates.json",
        args.val_size,
        target_dir,
    )
    missing_sources = [
        str(path) for path in (train_source, val_source) if not path.is_file()
    ]
    if missing_sources:
        parser.error(f"subset cache(s) not found: {missing_sources}")

    train_output = _purified_path(train_source)
    val_output = _purified_path(val_source)
    train_report = purify_cache(
        train_source,
        train_output,
        template_cache_subdir="train_template_cache",
        local_root=local_root,
    )
    val_report = purify_cache(
        val_source,
        val_output,
        template_cache_subdir="val_template_cache",
        local_root=local_root,
    )
    if train_report["kept_count"] == 0:
        parser.error("purification removed every training entry")

    runner_yaml = args.runner_yaml or (
        target_dir / "train_pdb_subset.purified.yaml"
    )
    write_runner_yaml(
        runner_yaml,
        {"train": train_output, "val": val_output},
        local_root,
    )

    report_path = args.report or (target_dir / "pdb_subset_purification_report.json")
    report_path.write_text(
        json.dumps({"train": train_report, "validation": val_report}, indent=2)
        + "\n"
    )
    print(
        "Purified training subset: "
        f"kept {train_report['kept_count']}/{train_report['input_count']}, "
        f"removed {train_report['removed_count']}"
    )
    print(
        "Purified validation subset: "
        f"kept {val_report['kept_count']}/{val_report['input_count']}, "
        f"removed {val_report['removed_count']}"
    )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
