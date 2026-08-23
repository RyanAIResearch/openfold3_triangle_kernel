# SR campaign scripts (imported from the 2026-08-21/22 session scratchpad)

Environment contract shared by all scripts (override via env):
- `SR_ROOT` — the worktree to test (default: /home/ubuntu/xzz/of3_sr_dq)
- `TORCH_EXTENSIONS_DIR` — prewarmed JIT cache (scripts default to a
  session-scratch path: SET THIS EXPLICITLY when reusing)
- `OPENFOLD3_TRIANGLE_ALLOW_JIT=1` — dev-JIT opt-in (set by the scripts)
- Arm flags: `OPENFOLD3_SR_BACKWARD=1` (SR-C), plus
  `OPENFOLD3_SR_SHADOW=1` (SR-R)
- Python: the pixi env
  `openfold3_workspace/src/.pixi/envs/openfold3-cuda12/bin/python`

Scripts:
- `sr_gate.py baseline|compare` — 12-shape facade battery, bitwise vs the
  a94b wheel golden. Golden tensors are NOT committed; regenerate with
  `SR_ROOT=<a94b wheel env> sr_gate.py baseline`.
  Reference golden sha256 (2026-08-22): 96b1281420e0a5651fd2e75424ce1de481e28cf1ff5c97b2f6144ad8df8ffdad
- `sr_srgate.py` — SR vs two-pass: dK/dV bitwise + replay + fp64-referenced
  dQ/dBias tolerance.
- `sr_workspace_gate.py` — workspace/query-span/leaf-wave/batch-chunk bitwise
  invariance, including the crop-640 trunk and an independent two-pass dK/dV
  oracle. Automatic SR plans compare bounded dBias slabs plus FP32 dK/dV
  continuation against a full-query slab and choose the smaller total scratch;
  forced spans remain exact diagnostic controls.
- `sr_shadowgate.py` — three-delta-contract kill-shot on sharp softmax
  (+ dV invariant under shadow).
- `sr_ladder.py [--capture DIR]` — full delta ladder
  (classic/shadow/fp32-shadow/perfect) on synthetic or captured tensors.
- `sr_apb_forensic.py` — absolute-norm/cosine forensic for saturated sites.
- `sr_capture_driver.py` — patches the facade inside run_openfold and saves
  (q,k,v,mask,bias,grad_out) per call; env: SR_CAPTURE_DIR/LIMIT/SKIP.
- `sr_stats_analyze.py` — paired-block stats + loss referee for the
  sr_stats_blocks campaign layout.
- `sr_time.py`, `sr_step0_bench.py`, `sr_build.py`,
  `sr_sanitize_target.py`, `sr_workspace_sanitize_target.py` — timing,
  per-kernel decomposition, prewarm, and standard/query-striped sanitizer
  targets.

Known debts (recorded, not yet cleaned): session-absolute default paths,
campaign shell drivers live under /home/ubuntu/xzz/of3_perf_runs/*/, and
the metrics hook (csvlogger + step-time + optimizer-metrics callbacks)
lives at of3_perf_runs/sr_stats_hook/sitecustomize.py.
