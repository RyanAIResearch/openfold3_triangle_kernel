# StableFold: Single-Reconstruction, Workspace-Invariant Backward for Triangular Attention

Working title. Alternatives:
- "One Reconstruction Is Enough: Deterministic Backward Execution for Triangular Attention"
- "StableFold: Deterministic Triangle Kernels with a Single-Reconstruction Backward"
Target: ICLR 2027 (abstract Sep 18, full paper Sep 25, 2026; 9-page main text, double-blind — no website links). Fallback: ICML 2027.

## One-sentence thesis

FlashAttention-style backward resolves conflicting gradient ownership through
repeated probability reconstruction; for triangular attention — which adds a
third outer reduction axis and a shared pair bias — we show that a
single-reconstruction reduction DAG with bounded BF16 partials and pipelined
transport outperforms an aggressively optimized double-reconstruction
baseline, while keeping all StableFold-produced gradients bitwise replayable under
the pinned binary, inputs, device architecture, and execution contract,
and the reduction tree invariant to workspace scheduling.

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
  row-scalar delta error that induces structured rowwise errors in dQ
  and dBias, appearing on sharp (trained) attention;
  a *residual-corrected output delta* — the forward's BF16 rounding residual
  saved as compressed state — recovers most of the perfect-delta floor at
  +|O| bf16 memory; an FP32-output shadow reaches the same observed
  dQ/dBias error floor as the perfect-delta oracle across measured
  captures (direct delta-vs-delta comparison still owed).
  Ladder measured on synthetic sharp cases and real pretrained captures.
  (The fail-closed AOT binary contract and replay gates fold into C4 and
  the implementation section — four major contributions, not five.)

## Section plan

1. **Introduction** (1.25pp) — reproducibility motivation for scientific ML;
   the three-axis problem; contributions. [draft.md]
2. **Background** (1pp) — AF3/OF3 triangular attention; FA recompute
   paradigm; determinism state of practice. Receipts: vendored Triton fp32
   `tl.atomic_add` (evoformer.py:598 AT OUR PINNED COMMIT — upstream since
   revised; pin explicitly); atomic-twin replay divergence (up to 11.8% of
   dBias elements, the primary receipt); cuEq no replay contract.
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
8. **Related work** (0.75pp) — five paragraphs (survey verified 2026-08-22):
   (a) Efficient exact attention: FA2/FA3/FA4 + attention I/O complexity
   (ICML'24) — we do NOT challenge standard-attention bounds; different
   computational graph.
   (b) Biomolecular/bias-aware systems: MegaFold/EvoFlash-3D (CLOSEST
   competitor: knows the dBias outer reduction, two-phase backward,
   atomic dBias in reported version — our diff: single producer, fixed
   trees, replay contract), FlashBias (structured-bias cost reduction ≠
   exact backward ownership), FastFold/ScaleFold, PairMixer et al.
   ("they change the model to avoid the operator; we change the
   execution of the original operator").
   (c) Deterministic execution & contracts: DASH (schedules an existing
   deterministic backward; we move the recompute/materialize boundary),
   TBIK (unified trees across TP sizes — cite, don't claim
   first-decoupling), Kernel Contracts (bits-as-API framing precedes us;
   we are a concrete AF3 instance), ELSA (forward scan, reals-exact).
   (d) Finite-precision backward: Qiu & Yao ICLR'26 (THE delta-defect
   mechanism paper — we cite, characterize on AF3, add residual state +
   SR dependency link), SageAttention3.
   (e) Programmable attention: FlexAttention, Flashlight, TileLang,
   QiMeng-Attention — mechanisms vs our specialized algorithm/contract.
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
  [HAVE: sr_stats_blocks, numbers in draft Results notes; 400-step
  pretrained pair running for the long-horizon panel]
- **F6** Same-seed loss overlay cueq vs SR, scratch + pretrained.
  [HAVE 30-step both regimes; 400-step pretrained running]
- **T1** Determinism cost: fixed-tree vs atomicAdd twin (+10–12% kernel /
  ~4.8% step; twin replay divergence up to 11.8% of elements).
  [HAVE (v8-era); RE-RUN on current kernels recommended]
- **T2** Kernel NCU profile: two-pass dq 6.7% HMMA / 217GB/s (6.5% DRAM)
  → the "I/O-optimal ≠ time-optimal" quantification. [HAVE]
- **T3** Contract matrix: gates per lane (bitwise battery / replay /
  sanitizers / dV invariant). [HAVE]
- **T4** Memory: workspace + partials + shadow state per crop. [HAVE calc]
- **T5** e2e stats: n-block mean/CI, both orders, bootstrap. [HAVE]

## Must-cite list (verified where marked)

FA2 (ICLR'24), FA3 (NeurIPS'24), FA4 (MLSys'26), Attention I/O complexity
(ICML'24), MegaFold/EvoFlash-3D (arXiv 2506.20686, VERIFIED), DASH
(arXiv 2601.21824, ICLR'26, VERIFIED), Qiu & Yao low-precision FA
(arXiv 2510.04212, ICLR'26, VERIFIED), FlashBias (NeurIPS'25), TBIK
(arXiv 2511.17826, verify at citation pass), Kernel Contracts (arXiv
2604.22032, verify), FastFold (PPoPP'24), ScaleFold (DAC'24).
Recommended: ELSA (2604.23798, VERIFIED), SageAttention3, DualKV,
FlexAttention, Flashlight, TileLang, PairMixer.

## Claims discipline (agreed wording)

- "residual-corrected output delta", NOT "probability-consistent/oracle"
  (equivalence is measured, not constructed).
- Determinism = same-contract bitwise replay; a new contract (SR/shadow) is
  a *different* deterministic contract, not a violation.
- "learn the same thing"-class claims only with the loss/optimizer evidence
  in F6, scoped to horizon actually measured.
- Trimul: "performance-validated, engineering contract below the attention
  path" — experimental tier.
- NEVER claim first: three-axis identification (MegaFold exposed it — we
  claim the unified single-producer FORMULATION), deterministic-DAG
  (DASH), fixed reduction trees / logical-physical decoupling (TBIK),
  bits-as-contract (Kernel Contracts), delta-defect discovery or
  FP32-output fix (Qiu & Yao). Safe formula: "to our knowledge, no prior
  work JOINTLY provides ..." over the full combination only.
- Do NOT claim FlashAttention I/O-suboptimality on standard attention;
  claim: FA's standard cost model does not cover the outer-shared bias
  backward graph.
- Triton atomic_add receipt must be COMMIT-PINNED (upstream removed it);
  prefer our atomic twin as the primary receipt.
- Generality wording: "StableFold uses fixed algorithmic tiles and
  capability-based dispatch, with no sequence-length-, crop-, or
  call-site-specific kernels; arbitrary supported shapes are handled by
  uniform padding, runtime guards, and a generic fallback." Never say
  "no edge-case handling" (guards/padding/fallbacks ARE edge handling);
  never claim the shape battery PROVES absence of specialization (it
  evidences the mechanisms; the static routing table is the proof).
- Statistics wording: the statistical unit is the PAIRED BLOCK (five
  paired comparisons, 4 scratch + 1 pretrained — never "ten arms");
  n=4 bootstrap is PRELIMINARY; per-step times are correlated within a
  run and are never independent samples.
- Defensible one-liner (the novelty sentence): "StableFold is a
  single-reconstruction triangular-attention backward in which each
  probability tile feeds all four gradient consumers, while bounded
  logical reduction trees make dQ and the outer-shared pair-bias gradient
  bitwise invariant to workspace scheduling; an optional BF16
  rounding-residual state improves the output-based delta without
  restoring a second reconstruction." 

## Arm naming (canonical, all headline numbers from one post-fence commit)

DR-Prod (a94b production) / DR-WG (modernized two-pass) / SR-C (classic
delta) / SR-R (residual-corrected). Historical intermediates appear only
in the F2 design-evolution ladder. Statistics discipline: the PAIRED
BLOCK is the statistical unit (never pool per-step samples as
independent); balance AB/BA; report order effect.

## Evidence status before submission

Completed: the order-balanced scratch/pretrained blocks, 400-step warmup
referee, shared-step1000 full-LR cueq-A/SR-R/cueq-B fork, skip20/60 and
N384/D16 capture columns, APB small-denominator forensic, post-fence atomic
twin, 64↔512 MiB trainer workspace sweep plus a 1/2/3/5/10-stripe bitwise
kernel holdout, the 26-capture direct-delta comparison, and the clean
`sm_90a` AOT wheel/attestation/no-JIT install gate. The workspace trainer
sweep is deliberately a negative systems result: it shows no whole-model
peak-memory reduction and no full-model bitwise trajectory, while the
separate kernel holdout proves the scoped arithmetic-invariance claim.

Still owed:

1. Finish the `.cu` terminology pass so every shadow path is called
   residual-corrected rather than exact/perfect.
2. Decide whether OpenBind and crop768 direct-delta columns are submission
   claims; run them only if retained.
3. Incremental related-work search in the week before submission (watch
   Aug–Sep 2026 arXiv and NeurIPS/EMNLP 2026 rollouts), including first-hand
   verification of TBIK, Kernel Contracts, FlashBias, and FA4.
4. Keep Nsys out of claims unless the host installation is upgraded: the
   installed 2024.6 agent hangs even on `/bin/true`; NCU mechanism evidence
   and CUDA-event performance evidence are complete.
5. Complete the planned contribution 5→4 merge and abstract trim during the
   full-paper pass.
