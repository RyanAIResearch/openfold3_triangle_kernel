# Draft — Abstract & Introduction

Status: first full draft 2026-08-22. Numbers cited from the session
receipts; bracketed TODOs mark pending campaign data.

---

## Abstract

Triangular attention, the dominant cost of AlphaFold3-class pair stacks,
complicates the FlashAttention backward recipe in a specific way: besides
the query- and key-owned gradients, it must produce the gradient of a pair
bias that the forward broadcasts across an entire outer axis of attention
planes. Production implementations resolve this third reduction axis with
unordered floating-point atomics — making training trajectories
unrepeatable — or by materializing per-plane gradient planes. We present
StableFold, a deterministic execution model for triangular attention built
on fixed reduction trees whose arithmetic order is a function of the
problem shape alone: query-slab, batch-chunk, and leaf-wave scheduling
change residency, never results. On this substrate we revisit
FlashAttention's central backward choice — reconstruct probabilities once
per consuming pass — and show that a *single* reconstruction feeding all
four gradients outperforms an aggressively optimized double-reconstruction
baseline, but only after the scattered gradient's partials are held in
BF16, staged transport is pipelined, and the fixed-order fold is
vectorized: the recompute-versus-materialize inequality is decided by
transport engineering, not FLOPs. We further treat the backward's softmax
row scalar as a contract choice: the known structured bias of the
output-based $\delta=dO\cdot O_{\mathrm{bf16}}$ surfaces strongly on
trained checkpoints, and a compressed rounding residual saved by the
forward corrects it without restoring a second reconstruction. On H100, StableFold's
backward runs 26.1–28.9% faster than cuEquivariance 0.10.0 across D=32
OpenFold3 crops, trains real OpenFold3 steps 3.5–4.5% faster (steady-state) than NVIDIA
cuEquivariance end to end — mean wall margin −9.5 s per 30 steps, bootstrap
95% CI [−11.5, −6.5] s over four order-balanced blocks, with the same-seed
loss gap inside cuEquivariance's own run-to-run scatter — and replays
bitwise.

---

## 1. Introduction

Biomolecular structure prediction has become an infrastructure workload.
AlphaFold3-class models are trained and fine-tuned continually — by
consortia that retrain the same architecture as new experimental data
arrives — and the dominant cost of these models is the pair stack, whose
triangular attention and triangular multiplicative update scale cubically
with sequence length. Two engineering communities have responded: fused
production kernels (NVIDIA cuEquivariance, DeepSpeed4Science, Triton
Evoformer kernels) and AF3-specific training systems (MegaFold) that
make the pair stack fast, and a reproducibility
community that increasingly asks that scientific training runs be exactly
repeatable. Today these goals conflict. The fastest triangular-attention
backwards resolve their gradient reductions with unordered floating-point
atomics: the vendored OpenFold3 Triton path we build against accumulates
the pair-bias gradient with `tl.atomic_add` (evoformer.py at our pinned
commit; upstream has since revised this path), and an otherwise-identical
atomic twin of our own kernel leaves up to 11.8% of pair-bias gradient
elements changed between identical replays. FlashAttention-3's deterministic mode shows the
accepted price: up to 37.9% backward throughput loss, and recent work
(DASH) treats recovering that loss as a scheduling problem. Determinism,
in current practice, is a tax.

This paper argues the tax is an artifact of treating standard attention's
execution model as if it fit triangular attention. Standard attention has
two gradient ownership axes: dQ reduces along keys, dK/dV along queries.
FlashAttention's backward resolves the conflict by *reconstructing* the
probability matrix from saved statistics once per consuming pass — a
choice justified when scores are cheap to recompute and HBM is the binding
resource. Triangular attention breaks both premises. It adds a third
reduction axis: the pair bias $B\in\mathbb{R}^{H\times N\times N}$ is
broadcast across $O$ outer attention planes in the forward, so its
gradient is a sum over planes that no query- or key-owned pass produces
naturally — broadcast in forward is reduction in backward. And its
regime is different: head dimensions are small (8–64), planes are many,
and we measure production backward kernels at 6–18% tensor-pipe
utilization with DRAM over 90% idle. The workload is latency- and
issue-bound; FlashAttention's recompute paradigm economizes a resource
that is free here and spends the one that is scarce.

We present StableFold, a deterministic execution model for triangular
attention on Hopper GPUs, and use it to re-decide FlashAttention's
central backward tradeoff under triangular attention's actual constraints.

First, we make determinism structural rather than modal. Every reduction
— the pair-bias gradient across outer planes, and later the scattered
query gradient across key blocks — flows through a fixed tree over
logical leaves: one writer per cell, a global fold order that is a
function of the problem shape alone. A planner chooses query slabs,
batch chunks, and resident leaf waves against a memory budget, and we
prove (and gate bitwise) that these choices alter residency and
parallelism but never the arithmetic history: changing the workspace
budget cannot change the operator's arithmetic output, bit for bit.

Second, on this substrate we build a single-reconstruction backward: one
producer reconstructs $\{S, P, dP, dS\}$ per (plane, tile) and feeds all
four gradients — dK/dV in registers with unchanged accumulation chains
(bit-identical to the two-pass baseline, which we gate), the pair-bias
tree unchanged, and dQ through bounded BF16 key-block partials folded in
fixed order. The result is not a free win, and we report the design
space honestly: with FP32 partials the single-reconstruction kernel is
13.6% *slower* than the two-pass baseline (2.1 GB of partial traffic per
launch); BF16 partials, cp.async double-buffered staging, and a
vectorized fixed-order fold are each necessary; together they win 19.4–23.2%
in backward across D=32 crops against a baseline that we first modernized ourselves
(warpgroup restructure, ss-WGMMA, −10% over the production kernel). The
measured conclusion generalizes the FlashAttention question: whether
recomputation beats materialization is decided by partial precision,
transport pipelining, and reduction implementation — jointly — not by
FLOP counting.

Third, we treat the backward's softmax row scalar
$\delta_q=\sum_k P_{qk}\,dP_{qk}$ as part of the numerical contract.
Recent analysis of low-precision FlashAttention training identified the
mechanism: evaluating $\delta=dO\cdot O$ from a low-precision saved
output injects structured, biased gradient error (Qiu & Yao, ICLR 2026).
We characterize this known defect where it matters for structure
prediction — real pretrained OpenFold3 activations, where it inflates
dQ error 2–3× at the trunk triangle sites that carry the gradient
signal (and, as a limit-case illustration, corrupts gradient
*direction* to cosine 0.78 at a saturated attention-pair-bias site
whose true gradient has already vanished to 10⁻⁷-norm) — and we
connect it to
single-reconstruction's dependency structure: the probability-derived
$\delta$ is unavailable until a full key scan completes, so repairing
$\delta$ by recomputation reintroduces the second reconstruction this
paper removes. Our contribution is the minimal state that breaks the
dependency: the forward's own BF16 rounding residual, saved as
compressed shadow state at one extra store per element, corrects the
output-based delta *before* the sweep begins. It recovers 75–80% of the
gap to a perfect FP64 delta in the original capture ladder. In a direct
26-capture comparison, BF16 shadow recovers 83.3% of the dQ gap on
average (88.1% median), while an uncompressed FP32 output shadow is not
bitwise perfect but reaches the same observed downstream dQ error floor:
its direct-delta relative L2 error is 4.22e-7 on average (7.84e-7 worst),
and its dQ-error ratio to the perfect-delta oracle is 1.0014 on average
(1.0323 worst). Delta choice becomes an explicit, priced state–accuracy
menu.

On an H100, across D=32 OpenFold3 crops 384/640/768, StableFold's
single-reconstruction backward runs 26.1–28.9% faster than
cuEquivariance 0.10.0 and 19.4–23.2% faster than the same-artifact
modernized two-pass. Forward plus backward is 21.3–23.5% faster than
cuEquivariance. This is an applicability region, not an all-dimension
claim: at N640/D64, SR-R is 41.9% slower than cuEquivariance in backward
under the same timing contract. In real
OpenFold3 training at crop 640 it wins every one of five paired
same-session comparisons (four scratch blocks and one pretrained): four
order-balanced scratch blocks give a mean margin of −9.5 s per 30 steps
(bootstrap 95% CI [−11.5, −6.5] s; steady-state 8.52 vs 8.83 s/step,
−3.5%), and a pretrained-weights block — within the official 1,000-step
learning-rate warmup, as the released checkpoint carries no optimizer
state — gives −7.8 s (8.56 vs 8.96 s/step, −4.5%). The same-seed loss gap between SR and cuEquivariance (mean
8.8e-4 scratch, 9.7e-4 pretrained) sits well inside cuEquivariance's own
four-run scatter (mean 3.6e-3), and SR tightens that model-level scatter
by 37% — the residual scatter belongs to the surrounding stock
operators, which our kernel contract does not cover.
Every configuration replays bitwise; an aggressive-atomics twin of our
own kernel — identical except the reduction mechanism — quantifies what
that guarantee costs (+10–12% per kernel, ~5% per step) and what it buys
(the twin corrupts replays). Artifacts ship under a fail-closed AOT
contract that verifies architecture, ABI, and binary hashes before

serving a single kernel.

**Contributions.**
1. A formalization of triangular attention's backward as a three-axis
   gradient-ownership problem with a shared-operand broadcast/reduction
   duality (§3).
2. A deterministic execution model — fixed leaf trees with
   workspace-invariant scheduling — that makes bitwise replay a
   structural property rather than a slow mode (§3).
3. A single-reconstruction backward and a measured cost model showing
   the recompute-vs-materialize decision is won by transport design;
   including the negative results that locate the crossover (§4).
4. A finite-precision state frontier for single-reconstruction
   backward: characterizing the known output-delta defect on pretrained
   AF3 activations — including its saturation limit, where relative
   corruption grows precisely as the true gradient vanishes — and a
   BF16 rounding-residual state that corrects delta without restoring
   a second reconstruction (§5).
5. An open, fail-closed, bitwise-replayable artifact evaluated on real
   OpenFold3 training against cuEquivariance, Triton, and stock PyTorch
   (§6–7).

---

## Results notes (data landed 2026-08-22, feeds F4/F5/F6/T5)

- LR-SCHEDULE CAVEAT (verified in code + checkpoint): released ckpt is a
  bare state_dict (no global_step/optimizer) and restore_lr_scheduler
  defaults false → every pretrained run before the shared-checkpoint fork
  sits inside the official 1,000-step warmup (step 30 ≈ 3% of peak LR,
  step 400 ≈ 40%). Loss statements from those runs are "warmup-horizon";
  the 400-step pair is the warmup-horizon referee. The post-warmup fork is
  now complete: a shared step-1000 checkpoint WITH optimizer state fed
  cueq-A / SR-R / cueq-B full-LR arms for 500 steps each. These instrumented
  arms are learning evidence, not speed evidence.
- F5/T5: scratch blocks cueq {736.5, 739.8, 739.3, 739.6} vs SR {731.6,
  728.1, 729.3, 728.4}; margins {−4.9, −11.7, −10.0, −11.3}; mean −9.46 s
  (−1.28%); bootstrap 95% CI [−11.46, −6.50]; steady-state per-step
  cueq 8.824–8.842 vs SR 8.517–8.540 (SR spread 23 ms across runs).
  Pretrained: 785.6 vs 777.8 (−7.8 s); steady 8.962 vs 8.556 (−4.5%).
- F6 (referee): cueq 4-run self-envelope mean 3.64e-3 / max 1.49e-2;
  SR 4-run envelope mean 2.29e-3 (−37%); SR-vs-cueq same-seed |dloss|
  mean 8.76e-4 scratch / 9.69e-4 pretrained — inside the envelope, both
  regimes. Full-model bitwise collapse is NOT claimed: surrounding stock
  ops are nondeterministic (known: official-arm self-replay fails ~half
  its steps); scoped claim = our kernels replay bitwise, model scatter
  tightens.
- F4 new columns: D=32 sites @640 (skip20/60): classic dq 0.0059–0.0195
  (brackets the archived 0.0121 pre-fix D32 baseline — second
  cross-validation), shadow 0.0035–0.0057, perfect 0.0033–0.0042.
  384/D=16: classic 0.0061–0.0092, shadow 0.0038–0.0085, perfect ~0.0035.
- Direct-delta gate (26 pretrained captures): FP32 shadow is not bitwise
  identical to the FP64-perfect delta cast to FP32 (0/26 exact), but its
  direct-delta relative L2 is 4.22e-7 mean / 7.84e-7 max and its downstream
  dQ error-floor ratio is 1.00136 mean / 1.03227 max. BF16 shadow recovers
  83.34% mean / 88.15% median of the classic-to-perfect dQ gap. Use "same
  observed downstream error floor," never "equals perfect delta."
- Workspace trainer gate (crop 640, mirrored 64/128/256/512 MiB order): all
  eight runs retained identical whole-model peak allocation/reservation;
  64 MiB was 1.62% slower than 256 MiB, while 128–512 MiB were within 0.2%.
  A separate N640/D16 holdout spanning 1/2/3/5/10 stripes passed bitwise for
  dQ/dK/dV/dBias and against the independent two-pass dK/dV oracle. Thus the
  planner-invariance claim is kernel-scoped; no trainer peak-memory win or
  full-model bitwise trajectory is claimed.
- Final public comparator gate: clean commit `acbc7a9ad` produced an
  `sm_90a` SASS-only wheel (SHA256 `4357b7f0a7ac0...`). Four paired ABBA
  blocks, each with 12 warmups and 30 CUDA-event samples, give SR-R
  backward speedups of 26.10/28.98/27.65% over cuEquivariance 0.10.0 at
  D32 N384/N640/N768; the descriptive block-level 95% intervals are
  [25.55,26.65]/[28.49,29.47]/[27.56,27.74]%. Forward+backward is
  21.28/23.45/22.81% faster. The paired block, not the 120 event samples,
  is the n=4 statistical unit; no samples were excluded. The six-backend
  matrix retains the negative N640/D64 boundary: SR-R backward is 41.90%
  slower than cuEquivariance.
- AOT release gate: the `acbc7a9ad` clean install disables compiler/JIT and
  passes the 7-shape DR-WG/SR-C/SR-R contract plus the 18-case public
  comparator correctness matrix. All public cases are finite with maximum
  relative L2 <= 0.006097 against the eager FP32-score reference; SR-R
  replays all five output/gradient tensors bitwise at D16/D32/D64. On the
  new N768 full-query/four-leaf-wave route, Compute Sanitizer reports zero
  memcheck errors, zero synccheck errors, and zero racecheck hazards or
  warnings.
- **APB-site forensic (CLOSED — small-denominator critique confirmed and
  absorbed):** case [1,O=1,H=16,N=640,D=24]. FP64 exact: dQ_ref l2 =
  2.45e-7 (trunk sites ~7e-4 — 3,000× larger), maxabs 5.1e-8, |dO| l2
  2.2e-6; softmax saturation mean max-prob 0.96, 78% of rows >0.99.
  Relative errors: classic 0.81 / shadow 0.17 / perfect 0.0035; cosine
  0.779 / 0.986 / 0.99999. VERDICT: absolute update-level effect ~2e-7 =
  negligible; the site is a MECHANISM ILLUSTRATION (relative corruption
  → large exactly as saturation drives the true gradient → 0), NOT harm
  evidence. The harm claim rests on trunk sites (2–3× relative on
  1e-3-norm gradients). Never headline the 0.81 without this framing.

TODO next: §2 Background, §3 model+invariance statement.
