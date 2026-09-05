"""Integrity checks for the sanitized single-H100 supplied-recipe receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RECEIPT = (
    REPOSITORY
    / "release_artifacts"
    / "backend-matrix-h100-1gpu-supplied-recipe-20260904"
)
RECEIPT_FILES = (
    "artifact_hashes.json",
    "losses.csv",
    "matrix_summary.json",
    "metrics.csv",
    "oom.json",
    "runtime_identity_audit.json",
)
BACKENDS = ("aqlab", "triton", "cueq", "stablefold")
CROPS = (384, 640, 768)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_single_h100_receipt_checksums_are_self_consistent():
    expected = {}
    for line in (RECEIPT / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest

    assert expected == {
        filename: _sha256(RECEIPT / filename) for filename in RECEIPT_FILES
    }


def test_single_h100_receipt_has_all_cells_without_imputed_oom_timing():
    summary = json.loads((RECEIPT / "matrix_summary.json").read_text())

    assert summary["cell_counts"] == {
        "cuda_oom": 1,
        "passed": 11,
        "planned": 12,
    }
    protocol = summary["protocol"]
    assert protocol["num_workers_per_rank"] == 10
    assert protocol["gradient_accumulation_steps"] == 16
    assert protocol["confidence_freeze"] == {
        "method": "requires_grad_false_before_trainer_fit",
        "parameter_tensor_count": 243,
        "parameter_numel": 12_881_024,
        "parameter_names_sha256": (
            "54c867c523ee3d8c3bb3b923ee6f0c709e2fbc2bd66f0311671ec0ce4cdc8fc4"
        ),
    }
    assert protocol["warmup_microbatches"] == 16
    assert protocol["measured_microbatches"] == 50
    assert protocol["completed_optimizer_updates_in_measured_interval"] == 3
    assert protocol["cuda_event_values_used_for_conclusions"] is False
    assert protocol["feature_fingerprints_recorded"] is False

    results = summary["results"]
    assert set(results) == {str(crop) for crop in CROPS}
    for crop in CROPS:
        crop_results = results[str(crop)]
        assert set(crop_results) == set(BACKENDS)
        triton_wall = crop_results["triton"]["wall_seconds_per_measured_microbatch"]
        for backend, row in crop_results.items():
            if row["status"] != "passed":
                assert (crop, backend) == (768, "aqlab")
                assert "wall_seconds_per_measured_microbatch" not in row
                assert "samples_per_second" not in row
                assert "speedup_vs_aqlab_triton" not in row
                continue
            wall = row["wall_seconds_per_measured_microbatch"]
            throughput = row["samples_per_second"]
            assert math.isfinite(wall) and wall > 0
            assert math.isfinite(throughput) and throughput > 0
            assert math.isclose(throughput, 1.0 / wall, rel_tol=1e-12)
            assert math.isclose(
                row["speedup_vs_aqlab_triton"],
                triton_wall / wall,
                rel_tol=1e-12,
            )


def test_single_h100_loss_receipt_contains_only_complete_finite_runs():
    with (RECEIPT / "losses.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    successful_cells = {
        (crop, backend)
        for crop in CROPS
        for backend in BACKENDS
        if (crop, backend) != (768, "aqlab")
    }
    assert len(rows) == len(successful_cells) * 66
    phases = Counter((int(row["crop"]), row["backend"], row["phase"]) for row in rows)
    indices: dict[tuple[int, str], list[int]] = {}
    for row in rows:
        cell = (int(row["crop"]), row["backend"])
        indices.setdefault(cell, []).append(int(row["microbatch"]))
        assert math.isfinite(float(row["loss"]))
    assert set(indices) == successful_cells
    for crop, backend in successful_cells:
        assert indices[crop, backend] == list(range(66))
        assert phases[crop, backend, "warmup"] == 16
        assert phases[crop, backend, "measure"] == 50


def test_single_h100_published_figures_match_the_receipt():
    summary = json.loads((RECEIPT / "matrix_summary.json").read_text())
    figures = summary["published_figures"]

    assert set(figures) == {"wall", "loss"}
    for figure in figures.values():
        path = REPOSITORY / figure["path"]
        assert path.is_file()
        assert figure["sha256"] == _sha256(path)


def test_single_h100_receipt_omits_device_timing_and_host_identifiers():
    with (RECEIPT / "metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert not any(
            "gpu" in field.lower() or "cuda" in field.lower()
            for field in reader.fieldnames
        )
        assert len(list(reader)) == 12

    text = "\n".join((RECEIPT / filename).read_text() for filename in RECEIPT_FILES)
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    assert not re.search(r"/(?:home|Users|root|mnt|tmp)/", text)
    assert "ubuntu" not in text.lower()
    assert "PRIVATE KEY" not in text
    assert not re.search(r"gh[pousr]_[A-Za-z0-9_]{20,}", text)
