# Result receipts

- `H100_SUMMARY_4be8da301.md`: final correctness, sanitizer, kernel matrix,
  same-commit ABBA, production cuEq ABBA, memory/energy, and NCU summary.
- `H100_WORKSPACE_TRAINER_4be8da301.md`: mirrored trainer budget sweep and
  independent multi-stripe bitwise holdout.
- `H100_DIRECT_DELTA_4be8da301.md`: direct classic/BF16-shadow/FP32-shadow/
  FP64-perfect comparison over 26 real pretrained captures.
- `H100_AOT_eb013dd0c.md`: clean-source `sm_90a` wheel attestation and
  compiler-free/JIT-free clean-install smoke gate.
- `H100_COVERAGE_CROP768_acbc7a9ad.md`: real crop640 D16/D24/D32/D48
  runtime census, exact-layout six-backend correctness/performance and
  route-weighted ABBA, plus the final AOT attention-only crop768 ABBA with
  synchronized time, NVML energy, peak memory, loss referee, and campaign
  seal.
- `H100_PUBLISHED_COMPARATORS_acbc7a9ad.md`: source-pinned
  MegaFold/EvoFlash-3D and TriFast comparison on exact OpenFold3 layouts;
  N640/N768 n=8 paired blocks, same-size correctness/replay gates, memory
  tradeoff, the N768/D48 negative crossover, and campaign seal.
- `H100_PUBLISHED_E2E_acbc7a9ad.md`: same-OpenFold3 crop768 four-arm
  end-to-end comparison against cuEq, MegaFold/EvoFlash-3D, and the
  explicitly scoped TriFast-triangle/cuEq-APB hybrid; n=8 independent-seed
  order-balanced blocks, synchronized full training-step time, NVML energy,
  memory/loss/route gates, frozen autotune/JIT receipts, and campaign seal.

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
