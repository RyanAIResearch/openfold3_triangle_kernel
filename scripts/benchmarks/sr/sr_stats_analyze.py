"""Analyze the sr_stats_blocks campaign: CI, per-step walls, loss referee."""

import csv
import glob
import os
import random
import statistics

STATS = "/home/ubuntu/xzz/of3_perf_runs/sr_stats_blocks"


def read_wall(name):
    with open(os.path.join(STATS, name, "wall.time")) as handle:
        for line in handle:
            if line.startswith("wall_seconds="):
                return float(line.strip().split("=")[1])
    return None


def read_metrics(name):
    paths = glob.glob(
        os.path.join(STATS, name, "metrics/metrics/version_*/metrics.csv")
    )
    if not paths:
        return {}
    columns = {}
    with open(sorted(paths)[-1]) as handle:
        for row in csv.DictReader(handle):
            step = row.get("step")
            if step in (None, ""):
                continue
            for key, value in row.items():
                if key in ("step", "epoch") or value in (None, ""):
                    continue
                try:
                    columns.setdefault(key, {})[int(step)] = float(value)
                except ValueError:
                    continue
    return columns


blocks = [1, 2, 3, 4]
cueq_walls = [read_wall(f"b{b}_cueq") for b in blocks]
sr_walls = [read_wall(f"b{b}_sr") for b in blocks]
margins = [s - c for s, c in zip(sr_walls, cueq_walls)]
print(f"scratch walls cueq={cueq_walls}")
print(f"scratch walls sr  ={sr_walls}")
print(f"per-block margins  {['%.2f' % m for m in margins]}")
print(
    f"mean cueq {statistics.mean(cueq_walls):.2f}  "
    f"mean sr {statistics.mean(sr_walls):.2f}  "
    f"mean margin {statistics.mean(margins):.2f}s "
    f"({100 * statistics.mean(margins) / statistics.mean(cueq_walls):.2f}%)"
)
random.seed(20260822)
resamples = [
    statistics.mean(random.choices(margins, k=len(margins)))
    for _ in range(20000)
]
resamples.sort()
print(
    f"bootstrap 95% CI on mean margin: "
    f"[{resamples[500]:.2f}, {resamples[19499]:.2f}] s"
)
print(f"pretrained: cueq {read_wall('pre_cueq'):.2f} "
      f"sr {read_wall('pre_sr'):.2f} "
      f"margin {read_wall('pre_sr') - read_wall('pre_cueq'):.2f}s")

# per-step wall times (steady state = steps 2..)
def steady(name):
    columns = read_metrics(name)
    times = columns.get("train/step_wall_seconds", {})
    values = [v for step, v in sorted(times.items())][2:]
    return statistics.mean(values) if values else None


for group, names in (
    ("scratch", [f"b{b}_{arm}" for b in blocks for arm in ("cueq", "sr")]),
    ("pretrained", ["pre_cueq", "pre_sr"]),
):
    for name in names:
        value = steady(name)
        if value is not None:
            print(f"steady step s/step {name}: {value:.3f}")

# loss referee
def losses(name):
    columns = read_metrics(name)
    for key in ("train/loss", "train/loss_step", "loss"):
        if key in columns:
            return columns[key]
    candidates = [key for key in columns if key.endswith("loss")]
    return columns[candidates[0]] if candidates else {}


series = {name: losses(name) for name in
          [f"b{b}_{arm}" for b in blocks for arm in ("cueq", "sr")]
          + ["pre_cueq", "pre_sr"]}
common = None
for name in [f"b{b}_cueq" for b in blocks]:
    keys = set(series[name])
    common = keys if common is None else (common & keys)
common = sorted(common or [])
if common:
    # cueq self-envelope across its four identical-config runs
    spread = [
        max(series[f"b{b}_cueq"][step] for b in blocks)
        - min(series[f"b{b}_cueq"][step] for b in blocks)
        for step in common
    ]
    print(f"cueq 4-run self-envelope: mean {statistics.mean(spread):.3e} "
          f"max {max(spread):.3e} over {len(common)} steps")
    sr_spread = [
        max(series[f"b{b}_sr"][step] for b in blocks)
        - min(series[f"b{b}_sr"][step] for b in blocks)
        for step in common
        if all(step in series[f"b{b}_sr"] for b in blocks)
    ]
    print(f"sr   4-run self-envelope: mean {statistics.mean(sr_spread):.3e} "
          f"max {max(sr_spread):.3e}")
    gaps = [
        abs(series["b1_sr"].get(step, float("nan"))
            - series["b1_cueq"][step])
        for step in common
        if step in series["b1_sr"]
    ]
    print(f"sr-vs-cueq same-seed |dloss|: mean {statistics.mean(gaps):.3e} "
          f"max {max(gaps):.3e}")
pre_common = sorted(set(series["pre_cueq"]) & set(series["pre_sr"]))
if pre_common:
    gaps = [abs(series["pre_sr"][s] - series["pre_cueq"][s])
            for s in pre_common]
    base = statistics.mean(
        [abs(series["pre_cueq"][s]) for s in pre_common]
    )
    print(f"PRETRAINED sr-vs-cueq |dloss|: mean {statistics.mean(gaps):.3e} "
          f"max {max(gaps):.3e} (mean |loss| {base:.3f}, "
          f"{len(pre_common)} steps)")
print("ANALYZE_DONE")
