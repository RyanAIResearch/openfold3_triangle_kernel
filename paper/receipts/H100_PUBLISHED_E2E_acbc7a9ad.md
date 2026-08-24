# H100 same-OpenFold3 published-backend end-to-end receipt

## Claim boundary

This campaign compares attention backends inside the same frozen OpenFold3
crop-768 training process on one H100. It does **not** copy headline numbers
from other papers, reproduce MegaFold's full native training system, or claim
a full TriFast replacement. The TriFast arm is explicitly a hybrid: public
TriFast handles triangle self-attention and cuEquivariance handles APB. All
four arms retain cuEquivariance triangle multiplicative update.

## Frozen identity

- Campaign: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/published_e2e_acbc7a9_20260824T064757Z`
- Project worktree: `/home/ubuntu/xzz/of3_sr_dq`
- AOT runtime commit: `acbc7a9adfe792f7e4a03ccff01ec5a28e18b1b4`
- Repository head before this documentation commit:
  `e9f5fbb0d26b548a875d8e9056eb0ff403fe8bb5`
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- Driver / CUDA / Torch / Python: 570.195.03 / 12.9 / 2.10.0 / 3.14.6
- MegaFold source: commit
  `845905e7c707cf4004404354156df97f2efc4fde`
- TriFast source: public 0.1.13 wheel/source. The only compatibility change
  removes the invalid `ensure_exists=False` keyword passed to
  `pathlib.Path`; the public wrapper and Triton kernel remain byte-identical.
  No shape, head-dimension, crop, or call-site special case was added.

## Arms

| Arm | Attention route | Triangle multiplicative update |
|---|---|---|
| cuEq | cuEquivariance | cuEquivariance |
| StableFold SR-R | single reconstruction + residual-corrected delta | cuEquivariance |
| MegaFold | public EvoFlash-3D on every observed cuEq-fused attention call | cuEquivariance |
| TriFast hybrid | public TriFast on triangle self-attention; cuEq on APB | cuEquivariance |

The adapter includes all public-API transpose, permute, mask conversion, and
contiguous-output costs in the measured training step. Its canonicalization
is rank/layout generic; TriFast dispatches by the semantic contract
`outer == sequence`, not by a named shape or dimension.

## Correctness and preflight

- Independent six-real-site adapter gate: every output and gradient is
  finite. MegaFold maximum relative L2 is at most 0.003822; TriFast triangle
  at most 0.004351; cuEq APB fallback at most 0.006582.
- MegaFold output/dQ/dK/dV replay bitwise; its triangle dBias retains the
  documented relaxed-atomic replay boundary. TriFast triangle and cuEq APB
  fallback replay bitwise in the tested 20-run cells.
- A four-arm, two-step full training preflight (forward, backward, optimizer)
  completes without OOM, exception, or silent fallback and reports identical
  aggregate train/loss across all four arms.
- The first crop-768 TriFast preflight exposed missing public Triton autotune
  records. A separate, non-formal cache-warm process populated the generic
  H100 keys. The cache was then made read-only; an independent verification
  process and every formal TriFast process emitted zero autotune or cache-
  write lines. Pre/post formal config hashes are identical.
- StableFold runtime JIT was forbidden; `jit_forbidden/` is empty.

## Formal design

- Crop 768, batch 1, BF16 mixed precision, zero validation batches.
- Eight independent seeds: 104729, 105738, 106747, 107756, 108765, 109774,
  110783, 111792.
- Four-arm orders are balanced: every arm occupies every process position
  exactly twice.
- Each independent process starts from the same checkpoint and uses the same
  data seed as the other arms in its block.
- Twelve complete training steps per process; discard steps 1-2 and average
  the ten synchronized steady steps.
- Primary statistical unit: one paired process-level steady mean per block,
  n=8. Intervals use a 100,000-replicate paired-block bootstrap. The 80
  within-arm steady steps are not independent replicates.
- Step time is CUDA-synchronized forward + backward + optimizer. Energy is
  the NVML total-energy-counter difference at the same synchronized batch
  boundaries. Outer wall is separate and includes startup/data preparation.

## Formal gates

- 32/32 processes exit zero; 384/384 synchronized traces are present.
- YAML configs are deeply equal except output path, independent block seed,
  and the SR selector.
- Requested and actual route keys match every arm. Within each block,
  cuEq calls = MegaFold calls = TriFast triangle calls + cuEq APB fallback
  calls. Call totals vary across seeds with the sampled structures, as
  expected.
- No traceback, OOM, fallback, formal autotune, failed cache write, cache
  mutation, or StableFold JIT artifact.
- Ninety-six step-indexed loss values per arm comparison are pairable.

## Results

| Arm | Full train step (s) | Energy/step (kJ) | Reserved peak (GiB, median) | Outer wall (s) |
|---|---:|---:|---:|---:|
| StableFold SR-R | 11.1708 | 5.943 | 52.32 | 298.7 |
| cuEq | 12.0285 | 6.435 | 52.84 | 308.7 |
| TriFast triangle + cuEq APB | 13.0616 | 6.909 | 51.72 | 321.4 |
| MegaFold EvoFlash-3D | 14.2753 | 7.810 | 52.32 | 335.5 |

Positive reduction means SR-R is faster/lower energy.

| Comparator | SR-R step reduction (95% bootstrap CI) | Throughput improvement | SR-R energy reduction (95% CI) | Wins | Exact sign p |
|---|---:|---:|---:|---:|---:|
| cuEq | 7.15% [6.78, 7.52] | 7.70% | 7.64% [7.53, 7.75] | 8/8 | 0.0078125 |
| TriFast hybrid | 14.52% [13.89, 15.16] | 16.99% | 14.04% [13.26, 14.87] | 8/8 | 0.0078125 |
| MegaFold | 21.75% [21.42, 22.09] | 27.81% | 23.89% [23.70, 24.12] | 8/8 | 0.0078125 |

Under the same integration contract, the TriFast hybrid is 8.63% slower
than cuEq and MegaFold is 18.67% slower than cuEq; both comparisons are
0/8 wins for the external backend. SR-R's paired outer-wall reductions are
3.28%, 7.17%, and 11.06% versus cuEq, TriFast hybrid, and MegaFold.

Median steady reserved memory differs by less than about 1.2 GiB across
arms. The high-memory final seed reaches roughly 66-68 GiB depending on
arm without OOM; this is not claimed as a memory result.

## Loss referee

Across 96 same-seed `train/loss` pairs, SR-R versus cuEq has mean absolute
gap 8.91e-5, maximum absolute gap 1.50e-3, and maximum relative gap
7.10e-5. Against MegaFold and the TriFast hybrid, mean absolute gaps are
about 2.14e-3 and the maximum relative gap is about 2.89% after sequential
optimizer updates. These are short-run numerical checks, not convergence,
validation-quality, or final-accuracy evidence.

## Artifact map and seal

- Main analysis: `analysis/published_e2e_b8_summary.json` and `.md`
- Flat tables: `analysis/published_e2e_b8_arms.csv` and
  `analysis/published_e2e_b8_comparisons.csv`
- Formal status/receipts: `e2e/formal_b8/`
- Adapter correctness: `correctness/`
- Cache receipts: `receipts/trifast_config_*`
- Formal campaign seal: `CAMPAIGN_SHA256SUMS`, SHA-256
  `fa339f455bc40fab956977ba90042030ac2b86dbd9e874abed4bd483bff7147c`
- Seal inventory: 3,699 listed files; campaign directory 283,321,867 bytes
  at seal time.

No Nsight timing claim is attached to this receipt. The host's installed
Nsight Systems 2024.6 hangs even on `/bin/true`; that negative environment
receipt remains separate, while this campaign uses synchronized wall time,
NVML hardware energy, and explicit route/cache/correctness gates.
