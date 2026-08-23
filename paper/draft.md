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
transport engineering, not FLOPs. We further show the backward's softmax
row scalar is a numerical contract: the classic
$\delta=dO\cdot O_{\mathrm{bf16}}$ carries a coherent row bias that
surfaces on trained checkpoints, and a compressed rounding-residual saved
by the forward recovers the perfect-delta floor. On H100, StableFold's
backward runs 21–26% faster than our production baseline across
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
Evoformer kernels) that make the pair stack fast, and a reproducibility
community that increasingly asks that scientific training runs be exactly
repeatable. Today these goals conflict. The fastest triangular-attention
backwards resolve their gradient reductions with unordered floating-point
atomics: in the Triton production path, the pair-bias gradient is an
`atomic_add`; in our measurements, an atomic variant of our own kernel
leaves up to 11.8% of pair-bias gradient elements changed between
identical replays. FlashAttention-3's deterministic mode shows the
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
parallelism but never the arithmetic history. Changing the workspace
budget cannot change the model you train.

Second, on this substrate we build a single-reconstruction backward: one
producer reconstructs $\{S, P, dP, dS\}$ per (plane, tile) and feeds all
four gradients — dK/dV in registers with unchanged accumulation chains
(bit-identical to the two-pass baseline, which we gate), the pair-bias
tree unchanged, and dQ through bounded BF16 key-block partials folded in
fixed order. The result is not a free win, and we report the design
space honestly: with FP32 partials the single-reconstruction kernel is
13.6% *slower* than the two-pass baseline (2.1 GB of partial traffic per
launch); BF16 partials, cp.async double-buffered staging, and a
vectorized fixed-order fold are each necessary; together they win 11–17%
across crops against a baseline that we first modernized ourselves
(warpgroup restructure, ss-WGMMA, −10% over the production kernel). The
measured conclusion generalizes the FlashAttention question: whether
recomputation beats materialization is decided by partial precision,
transport pipelining, and reduction implementation — jointly — not by
FLOP counting.

Third, we identify the backward's softmax row scalar
$\delta_q=\sum_k P_{qk}\,dP_{qk}$ as a numerical contract with
consequences. The classic evaluation $\delta=dO\cdot O_{\mathrm{bf16}}$
— used industry-wide — injects a coherent row-constant bias into dQ and
dBias that is invisible on random weights and 2–3× on real pretrained
checkpoints; we cross-validate the effect against an independent FP64
oracle study to three significant figures. We then show the minimum
state that repairs it: the forward's own BF16 rounding residual, saved
as compressed shadow state at one extra store per element. A
residual-corrected delta recovers 75–80% of the gap to a perfect FP64
delta on real captures; an uncompressed FP32 output shadow equals the
perfect delta on every case we measure. Delta choice becomes an explicit,
priced menu rather than an implementation accident.

On an H100, across OpenFold3 crops 384/640/768, StableFold's
single-reconstruction backward runs 21–26% faster than our production
baseline and 11–17% faster than the modernized two-pass; in real
OpenFold3 training at crop 640 it beats NVIDIA cuEquivariance in every
one of ten paired same-session arms across three campaigns: four
order-balanced scratch blocks give a mean margin of −9.5 s per 30 steps
(bootstrap 95% CI [−11.5, −6.5] s; steady-state 8.52 vs 8.83 s/step,
−3.5%), and a pretrained block gives −7.8 s (8.56 vs 8.96 s/step,
−4.5%). The same-seed loss gap between SR and cuEquivariance (mean
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
4. A taxonomy and measurement of backward delta contracts, and a
   compressed shadow state that recovers perfect-delta accuracy at one
   store per element (§5).
5. An open, fail-closed, bitwise-replayable artifact evaluated on real
   OpenFold3 training against cuEquivariance, Triton, and stock PyTorch
   (§6–7).

---

## Results notes (data landed 2026-08-22, feeds F4/F5/F6/T5)

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
- **APB-site discovery** (case [1,O=1,H=16,N=640,D=24], skip60): classic
  delta corrupts dQ by 0.81 RELATIVE (dbias 0.12); bf16 shadow recovers
  to 0.17; fp32 shadow = perfect (0.0036). Mechanism: saturated softmax
  makes the true gradient tiny, so the coherent delta bias dominates
  relatively. Report relative AND absolute; this is the strongest case
  for the fp32-shadow tier and a new failure-mode datapoint for classic
  delta far beyond the known 1e-2 class.

TODO next: §2 Background, §3 model+invariance statement.
