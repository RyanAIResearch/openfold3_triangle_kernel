import json
from pathlib import Path

import numpy as np

from scripts.datasets.purify_subset import purify_cache


def _entry(
    *,
    alignment_representative_id: str | None = None,
    template_ids: list[str] | None = None,
) -> dict:
    return {
        "chains": {
            "A": {
                "alignment_representative_id": alignment_representative_id,
                "template_ids": template_ids or [],
                "reference_mol_id": None,
            }
        }
    }


def _target(local_root: Path, pdb_id: str) -> Path:
    path = (
        local_root
        / "preprocessed_pdb_data"
        / "standard"
        / "structure_files"
        / pdb_id
        / f"{pdb_id}.npz"
    )
    path.parent.mkdir(parents=True)
    np.savez(path, res_name=np.array([], dtype=str))
    return path


def test_purify_cache_removes_entries_with_missing_downloads(tmp_path: Path):
    local_root = tmp_path / "pdb_training_set"
    _target(local_root, "good")

    source = tmp_path / "training_cache_subset_2.json"
    source.write_text(
        json.dumps(
            {
                "name": "test-cache",
                "reference_molecule_data": {},
                "structure_data": {
                    "good": _entry(),
                    "missing": _entry(),
                },
            }
        )
    )
    destination = tmp_path / "training_cache_subset_2_purified.json"

    report = purify_cache(
        source,
        destination,
        template_cache_subdir="train_template_cache",
        local_root=local_root,
    )

    purified = json.loads(destination.read_text())
    assert set(purified["structure_data"]) == {"good"}
    assert report["kept_count"] == 1
    assert report["removed_count"] == 1
    assert "missing" in report["removed"]


def test_purify_cache_checks_alignment_and_template_dependencies(tmp_path: Path):
    local_root = tmp_path / "pdb_training_set"
    _target(local_root, "entry")
    source = tmp_path / "training_cache_subset_1.json"
    source.write_text(
        json.dumps(
            {
                "name": "test-cache",
                "reference_molecule_data": {},
                "structure_data": {
                    "entry": _entry(
                        alignment_representative_id="entry_A",
                        template_ids=["tmpl_A"],
                    )
                },
            }
        )
    )

    report = purify_cache(
        source,
        tmp_path / "purified.json",
        template_cache_subdir="train_template_cache",
        local_root=local_root,
    )

    missing = "\n".join(report["removed"]["entry"])
    assert "alignment_arrays/entry_A.npz" in missing
    assert "train_template_cache/entry_A.npz" in missing
    assert "template_structure_arrays/tmpl/tmpl_A.npz" in missing
    assert "template_structure_arrays/tmpl/chain_id_to_moltype.npz" in missing
