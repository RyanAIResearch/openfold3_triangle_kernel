#!/usr/bin/env python3
"""Aggregate backend benchmark results and plot timing and loss comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SUMMARY_COLUMNS = (
    "backend",
    "fit_wall_seconds",
    "steady_wall_seconds",
    "steady_step_wall_seconds",
    "steady_samples_per_second",
    "gpu_step_mean_ms",
    "gpu_step_median_ms",
    "gpu_step_p95_ms",
    "speedup_vs_baseline",
)


def build_report(results_root: Path, output_dir: Path, baseline: str) -> dict[str, Any]:
    summaries = []
    losses: dict[str, list[tuple[int, float]]] = {}
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        backend = str(summary["backend"])
        steps_path = summary_path.parent / "steps.csv"
        if not steps_path.is_file():
            raise FileNotFoundError(f"missing step metrics: {steps_path}")
        series = []
        with steps_path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("loss") not in (None, ""):
                    series.append((int(row["step"]), float(row["loss"])))
        if not series:
            raise ValueError(f"no loss values found for backend {backend}")
        summaries.append(summary)
        losses[backend] = series

    if len(summaries) < 2:
        raise ValueError("at least two backend result directories are required")
    by_backend = {str(summary["backend"]): summary for summary in summaries}
    if baseline not in by_backend:
        raise ValueError(f"baseline backend not found: {baseline}")
    baseline_seconds = float(by_backend[baseline]["steady_step_wall_seconds"])
    for summary in summaries:
        summary["speedup_vs_baseline"] = (
            baseline_seconds / float(summary["steady_step_wall_seconds"])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "backend_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in SUMMARY_COLUMNS})

    combined = {
        "schema_version": 1,
        "baseline": baseline,
        "backends": summaries,
    }
    (output_dir / "backend_comparison.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(summary["backend"]) for summary in summaries]
    step_seconds = [float(summary["steady_step_wall_seconds"]) for summary in summaries]
    speedups = [float(summary["speedup_vs_baseline"]) for summary in summaries]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(names, step_seconds, color="#4472C4")
    axis.set_ylabel("Steady-state wall seconds / training step")
    axis.set_title("OpenFold3 backend training time")
    axis.grid(axis="y", alpha=0.25)
    for bar, seconds, speedup in zip(bars, step_seconds, speedups):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{seconds:.3f}s\n{speedup:.2f}×",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "training_time.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for backend, series in losses.items():
        axis.plot(
            [step for step, _ in series],
            [loss for _, loss in series],
            marker="o",
            markersize=2.5,
            linewidth=1.3,
            label=backend,
        )
    axis.set_xlabel("Optimizer training step")
    axis.set_ylabel("Training loss")
    axis.set_title("Matched-seed backend loss curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(figure)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline", default="aqlab")
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or results_root / "comparison").resolve()
    report = build_report(results_root, output_dir, args.baseline)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
