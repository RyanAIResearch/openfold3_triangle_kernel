# Result receipts

- `sr_results_bundle_20260823T0101.tar.gz` (1.2 MB)
  sha256: 048a96a3264d96b223bb1a8e7bb53ee891bbd79e6b0484fe09e3656a3ac391c4
  Contents: every SR-campaign metrics CSV (per-step loss / step wall /
  optimizer metrics), wall.time files, driver logs, run configs, and the
  metrics hook — for sr_stats_blocks (4 paired blocks + pretrained),
  sr_pre400 (400-step warmup referee), sr_fork1000 (step-1000 checkpoint
  producer + full-LR forks), sr_srfull (partial at packing time), and
  capture-run logs. EXCLUDES multi-GB binaries (checkpoints, capture
  tensors) — those live on the H100 box at the paths recorded in
  ../EXPERIMENT_MANIFEST.yaml.
  Regenerate anytime: scripts/benchmarks/sr/make_results_bundle.sh
