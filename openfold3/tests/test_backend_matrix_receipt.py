"""Integrity checks for the sanitized September 4 two-H100 benchmark receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RECEIPT = REPOSITORY / "release_artifacts" / "backend-matrix-h100-2gpu-20260904"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_receipt_checksums_are_self_consistent():
    expected = {}
    for line in (RECEIPT / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest

    assert expected == {
        "matrix_summary.json": _sha256(RECEIPT / "matrix_summary.json"),
        "oom.json": _sha256(RECEIPT / "oom.json"),
        "steps.csv": _sha256(RECEIPT / "steps.csv"),
    }


def test_receipt_is_complete_and_identifies_harness_and_runtime_separately():
    summary = json.loads((RECEIPT / "matrix_summary.json").read_text())

    assert summary["cell_counts"] == {
        "cuda_oom": 1,
        "planned": 12,
        "success": 11,
    }
    assert summary["source"]["checkout_commit"] == (
        "5d4c79748fa5c76ce6a69478c6f4d38395d81b2c"
    )
    assert summary["source"]["runtime_wheel_source_commit"] == (
        "87ebe22da13c657d36997dedb4ec398fbd62c6b7"
    )
    assert summary["hardware_and_runtime"]["openfold3_version"] == (
        "0.5.1.dev5860+g87ebe22da"
    )
    figure = summary["published_loss_figure"]
    assert figure["sha256"] == _sha256(REPOSITORY / figure["path"])
    assert summary["protocol"]["num_workers_per_rank"] == 10
    assert summary["protocol"]["full_crop_validation_for_successful_cells"] == (
        "passed"
    )

    cells = summary["cells"]
    assert Counter((cell["crop_tokens"], cell["backend"]) for cell in cells) == {
        (crop, backend): 1
        for crop in (384, 640, 768)
        for backend in ("aqlab", "triton", "cueq", "stablefold")
    }
    failed = [cell for cell in cells if cell["status"] != "success"]
    assert failed == [
        {
            "backend": "aqlab",
            "backend_label": "AQLab PyTorch",
            "crop_tokens": 768,
            "evidence": "oom.json",
            "status": "cuda_oom",
        }
    ]


def test_step_receipt_has_five_warmup_and_twenty_measured_rows_per_success():
    with (RECEIPT / "steps.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 11 * 25
    phases = Counter(
        (int(row["crop_tokens"]), row["backend"], row["phase"]) for row in rows
    )
    for crop, backend in {(int(row["crop_tokens"]), row["backend"]) for row in rows}:
        assert phases[crop, backend, "warmup"] == 5
        assert phases[crop, backend, "measure"] == 20


def test_receipt_contains_no_host_identifiers_or_absolute_paths():
    text = "\n".join(
        (RECEIPT / name).read_text()
        for name in ("matrix_summary.json", "oom.json", "steps.csv")
    )
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    assert not re.search(r"/(?:home|Users|root|mnt|tmp)/", text)
    assert "ubuntu" not in text.lower()
    assert "PRIVATE KEY" not in text
