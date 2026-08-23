# StableFold: Single-Reconstruction, Workspace-Invariant Backward for Triangular Attention

Working title. Alternatives:
- "One Reconstruction Is Enough: Deterministic Backward Execution for Triangular Attention"
- "StableFold: Deterministic Triangle Kernels with a Single-Reconstruction Backward"
Target: MLSys 2027 (expected deadline ~late Oct 2026). Fallback: ICML 2027.

## One-sentence thesis

FlashAttention-style backward resolves conflicting gradient ownership through
repeated probability reconstruction; for triangular attention — which adds a
third outer reduction axis and a shared pair bias — we show that a
single-reconstruction reduction DAG with bounded BF16 partials and pipelined
transport outperforms an aggressively optimized double-reconstruction
baseline, while keeping every gradient bitwise replayable and the reduction
tree invariant to workspace scheduling.

## Contributions (C1–C5)

- **C1 — Problem.** Triangular attention's backward has *three* gradient
  ownership axes (dQ along keys, dK/dV along queries, dBias along the outer
  plane axis) plus a shared-operand duality: the pair bias is broadcast in
  forward, so its gradient is a cross-plane reduction in backward. Naïve
  solutions are unordered atomics (nondeterministic) or outer-plane-sized
  materialization (memory explosion).
- **C2 — Deterministic execution model.** Fixed leaf trees with a single
  writer per cell; query-slab / batch-chunk / leaf-wave scheduling changes
  residency but never the arithmetic order (workspace invariance). No
  floating-point atomics anywhere.
- **C3 — Single-reconstruction backward.** One producer reconstructs
  {S, P, dP, dS} per (outer, tile) and feeds all four gradients; the one
  scattered gradient (dQ) flows through bounded BF16 key-block partials
  folded in a fixed order. A measured cost model: the recompute-vs-
  materialize inequality is decided by partial precision + transport
  pipelining + reduction implementation, not FLOPs. Includes the full
  failure→win ledger (FP32 partials lose 13.6%; BF16 + cp.async + vectorized
  reduce win 11–17%).
- **C4 — Delta contracts.** The backward's softmax-gradient row scalar
  ("delta") is a contract choice: classic dO·O_bf16 carries a coherent
  row-constant bias that appears on sharp (trained) attention;
  a *residual-corrected output delta* — the forward's BF16 rounding residual
  saved as compressed state — recovers most of the perfect-delta floor at
  +|O| bf16 memory; an FP32 shadow equals the perfect delta in practice.
  Ladder measured on synthetic sharp cases and real pretrained captures.
- **C5 — System & evaluation.** Fail-closed AOT binary contract; bitwise
  self-replay gates at every level; kernel/op/e2e evaluation against
  cuEquivariance, Triton, and stock PyTorch on real OpenFold3 training.

## Section plan

1. **Introduction** (1.25pp) — reproducibility motivation for scientific ML;
   the three-axis problem; contributions. [draft.md]
2. **Background** (1pp) — AF3/OF3 triangular attention; FA recompute
   paradigm; determinism state of practice. Receipts: AQLab Triton fp32
   `tl.atomic_add` (evoformer.py:597); atomic-twin replay divergence (up to
   11.8% of dBias elements); cuEq no replay contract.
3. **Deterministic execution model** (1.5pp) — fixed (o0+o1)+(o2+o3) leaf
   tree; global leaf order; planner; invariance statement + proof sketch;
   pass-invariance lemma (plane traffic independent of pass count).
4. **Single-reconstruction backward** (2pp) — ownership analysis; the SR
   leaf design (K-resident, 2 warpgroups, K/V/mask staged once, q/dO/bias
   double-buffered cp.async, dS shared round-trip, BF16 partials + fixed
   ascending reduce); cost model with measured constants; failure ledger.
5. **Delta contracts** (1.5pp) — taxonomy (classic / residual-corrected /
   fp32-shadow / recomputed-true / fp64-perfect); the coherent-bias
   mechanism; compressed-state argument; async-proxy fence contract note.
6. **Implementation** (1pp) — warpgroup/ss-wgmma encodings; register-cliff
   engineering; AOT fail-closed loader; env-gated contracts.
7. **Evaluation** (3pp) — see figures/tables below.
8. **Related work** (0.75pp) — DASH (ICLR'26: deterministic FA scheduling —
   they optimize *for* a fixed order; we make results order-invariant to
   scheduling), ELSA (associative monoid forward, exact in reals only; our
   claim is bitwise in fp), MegaFold, FastFold/ScaleFold, cuEq/DS4S, FA2/3.
9. **Discussion & limitations** — forward remains ~3% behind cuDNN kernel-
   only (this paper is about backward); trimul path is experimental-tier;
   D=128 keeps two-pass; single-GPU scope (world-size-invariant reduction =
   future work); BF16-partial accuracy factor 1.23×.

## Figures & tables (source status)

- **F1** Three-axis ownership diagram (drawn; NEW).
- **F2** Design-evolution ladder: double-recon → SR+fp32 (slower) →
  +bf16 → +cp.async → +vectorized reduce (wins). [HAVE: session ledger]
- **F3** Backward time by crop {384,640,768} × {a94b, two-pass-wg, SR}:
  7.84/7.07/5.84 @640 etc. [HAVE]
- **F4** Delta ladder bars, synthetic sharp + 8 real captures (classic /
  shadow / fp32-shadow / perfect). [HAVE: sr_ladder outputs; MORE COLUMNS
  PENDING: D=32 sites, 384/768 crops, OpenBind ckpt]
- **F5** e2e ABBA per-block walls + per-step time distributions.
  [PENDING: sr_stats_blocks campaign, running]
- **F6** Same-seed loss overlay cueq vs SR, scratch + pretrained.
  [PENDING: same campaign, csvlogger]
- **T1** Determinism cost: fixed-tree vs atomicAdd twin (+10–12% kernel /
  ~4.8% step; twin replay divergence up to 11.8% of elements).
  [HAVE (v8-era); RE-RUN on current kernels recommended]
- **T2** Kernel NCU profile: two-pass dq 6.7% HMMA / 217GB/s (6.5% DRAM)
  → the "I/O-optimal ≠ time-optimal" quantification. [HAVE]
- **T3** Contract matrix: gates per lane (bitwise battery / replay /
  sanitizers / dV invariant). [HAVE]
- **T4** Memory: workspace + partials + shadow state per crop. [HAVE calc]
- **T5** e2e stats: n-block mean/CI, both orders, bootstrap. [PENDING F5 data]

## Claims discipline (agreed wording)

- "residual-corrected output delta", NOT "probability-consistent/oracle"
  (equivalence is measured, not constructed).
- Determinism = same-contract bitwise replay; a new contract (SR/shadow) is
  a *different* deterministic contract, not a violation.
- "learn the same thing"-class claims only with the loss/optimizer evidence
  in F6, scoped to horizon actually measured.
- Trimul: "performance-validated, engineering contract below the attention
  path" — experimental tier.

## Evidence still owed before submission

1. sr_stats_blocks campaign (running): 4 scratch blocks + pretrained block,
   per-step times + losses → F5/F6/T5.
2. Capture matrix columns: deeper layers (skip-20/60), 384 crop (running in
   the same chain); OpenBind ckpt column; D=32-site hunt.
3. Atomic-twin re-run on current kernels (T1 refresh).
4. Optional: workspace 64↔512MB trainer-level bitwise θ_t demo (cheap,
   uniquely ours — strengthens C2).
5. Pin the artifact commit; wheel + receipts bundle for the SR branch.
