# H100 published-kernel comparator receipt (`acbc7a9ad` AOT runtime)

Date: 2026-08-24
Status: complete and sealed
Campaign: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/published_comparators_acbc7a9_20260824T052246Z`

## Question and scope

This campaign compares StableFold SR-R with the closest executable public
triangular/Evoformer attention kernels under the six exact OpenFold3 producer
layouts observed by the crop640 runtime census. It is an operator comparison,
not an end-to-end comparison with the full MegaFold or a TriFast-based system.
All conversions required by a public API are included in the CUDA-event timing.

- MegaFold/EvoFlash-3D: public repository
  `Supercomputing-System-AI-Lab/MegaFold`, commit
  `845905e7c707cf4004404354156df97f2efc4fde`; kernel
  `evoattention.py` SHA256
  `3899e574937e256e41de09ca4614ed65e068dc14a2cc50c68e392d2da8f1e759`.
- TriFast: public PyPI wheel 0.1.13, SHA256
  `e96fc0224b85e98802b04c495380198387110e3e712ce2ca7d2e672a27e10496`;
  official repository `latkins/trifast`, commit
  `2d5251135c774d0585e9bc042d7209c6b7c470cf`. TriFast implements triangle
  self-attention (`outer == sequence`) only; D24/D48 APB are recorded as
  `backend_unsupported`, not as failed or timed cells.
- StableFold: final read-only `sm_90a` AOT runtime built from
  `acbc7a9adfe792f7e4a03ccff01ec5a28e18b1b4`; runtime build-manifest SHA256
  `d09c6e2b1c2863d2a10a1ded9558306778f480b8b819b1fa970e9e554f6328d0`.
  `OPENFOLD3_TRIANGLE_ALLOW_JIT=0`; the forbidden extension directory is
  empty after the campaign.

## TriFast import compatibility receipt

The released wheel and official git HEAD both call
`pathlib.Path(..., ensure_exists=False)` and fail before any kernel call.
The campaign keeps a pristine wheel/source tree and a separate compatibility
copy. The only patch removes that invalid keyword; patch SHA256 is
`5680d8e21426afc4f57ce4c53b95203c8c5b4c314c9e071e8e822dbc6f34812b`.
The public kernel and autograd wrapper are byte-identical before/after:

- `trifast/triton.py`: `fb6e4385c1b345c39604416828cfb06007b286d12fba9a8da33d023e9e695743`
- `trifast/torch.py`: `1c818e876947681d53375d84c12992c79a9f3df2610e355f7fa182f6245b12b7`

This is a general import repair, independent of shape, D, crop, call site, or
GPU kernel. `jaxtyping==0.2.36` is target-installed inside the campaign rather
than into the OpenFold environment.

## Protocol

- Device: one NVIDIA H100 80GB HBM3, CC 9.0; Torch 2.10.0, CUDA 12.9,
  Triton 3.6.0, cuEquivariance 0.10.0.
- Input shape, stride, dtype, broadcast bias, mask view, and transposed-mask
  direction come from the real OpenFold3 producer census.
- N640: representative D16 start, D24 APB, D32 start, D48 diffusion sites.
- N768: all six sites, including both D16/D32 start and end directions.
- Each cell has 8 independent-seed process blocks. Within each block every
  backend receives the same tensor seed. Site and backend order are rotated
  and mirrored across blocks.
- Each process uses 12 warmups and 30 forward/backward CUDA-event samples;
  there is one device synchronization after the event batch. The process block,
  not the 30 samples, is the inferential unit. No timing result is excluded.
- TriFast is tuned on H100 before measurement with `TRIFAST_FORCE_TUNE=0`.
  The cache is then hash-frozen and read-only; its before/after hashes match.
- Time reduction is `(competitor - SR-R) / competitor`. Intervals are paired
  block-bootstrap 95% intervals. An 8/8 direction has two-sided exact sign
  `p=0.0078125` for that cell.

Performance harness SHA256:
`fb1bd834370b804d1dc42719569fff9469ea4afa3e9dc21da52a1a84c6406cc5`.

## Correctness and replay gates

- N128 versus the eager FP32-score reference: all supported MegaFold and
  TriFast output/dQ/dK/dV/dBias tensors are finite; maximum relative L2 is
  below 0.0044.
- N640 and N768 versus cuEquivariance at the timed shapes: all supported
  tensors are finite; maximum relative L2 is 0.004672.
- Every tested TriFast output/gradient tensor is bitwise stable across its
  20-run replays.
- MegaFold output/dQ/dK/dV are bitwise stable in the replay battery. Its
  triangle dBias uses relaxed floating-point atomic adds and may differ across
  replay; the observed outer=1 APB cases are stable. This is a may-diverge
  statement, not a claim that every triangle replay must diverge.

## N640 result (n=8)

Forward plus backward process-block means; memory is median incremental peak
and excludes already-resident input tensors.

| Site | Backend | F+B ms | Peak MiB | SR-R reduction, 95% CI | Direction |
|---|---:|---:|---:|---:|---:|
| D16 start | SR-R | 5.8191 | 1177.4 | reference | -- |
| D16 start | MegaFold | 16.3379 | 518.8 | 64.38% [64.29, 64.48] | 8/8 |
| D16 start | TriFast | 10.7604 | 319.5 | 45.92% [45.78, 46.07] | 8/8 |
| D24 APB | SR-R | 0.4821 | 59.5 | reference | -- |
| D24 APB | MegaFold | 0.6282 | 42.8 | 23.25% [19.28, 25.77] | 8/8 |
| D32 start | SR-R | 7.4725 | 2323.4 | reference | -- |
| D32 start | MegaFold | 13.5020 | 1018.8 | 44.66% [44.50, 44.80] | 8/8 |
| D32 start | TriFast | 12.9628 | 619.5 | 42.35% [42.19, 42.51] | 8/8 |
| D48 diffusion | SR-R | 0.5564 | 68.0 | reference | -- |
| D48 diffusion | MegaFold | 0.6278 | 47.0 | 11.36% [10.56, 12.15] | 8/8 |

## N768 six-site result (n=8)

| Site | Backend | F+B ms | Peak MiB | SR-R reduction, 95% CI | Direction |
|---|---:|---:|---:|---:|---:|
| D16 end | SR-R | 9.8192 | 1725.8 | reference | -- |
| D16 end | MegaFold | 34.8570 | 747.0 | 71.83% [71.83, 71.84] | 8/8 |
| D16 end | TriFast | 19.1378 | 460.1 | 48.69% [48.68, 48.70] | 8/8 |
| D16 start | SR-R | 9.8125 | 1725.8 | reference | -- |
| D16 start | MegaFold | 27.6875 | 747.0 | 64.56% [64.55, 64.57] | 8/8 |
| D16 start | TriFast | 18.4489 | 460.1 | 46.81% [46.80, 46.83] | 8/8 |
| D24 APB | SR-R | 0.4623 | 83.3 | reference | -- |
| D24 APB | MegaFold | 0.6538 | 59.2 | 29.28% [28.46, 30.23] | 8/8 |
| D32 end | SR-R | 12.7433 | 3165.8 | reference | -- |
| D32 end | MegaFold | 23.1001 | 1467.0 | 44.83% [44.83, 44.84] | 8/8 |
| D32 end | TriFast | 22.4948 | 892.1 | 43.35% [43.34, 43.36] | 8/8 |
| D32 start | SR-R | 12.7359 | 3165.8 | reference | -- |
| D32 start | MegaFold | 22.7472 | 1467.0 | 44.01% [44.00, 44.02] | 8/8 |
| D32 start | TriFast | 22.2402 | 892.1 | 42.73% [42.62, 42.83] | 8/8 |
| D48 diffusion | SR-R | 0.6892 | 95.5 | reference | -- |
| D48 diffusion | MegaFold | 0.6239 | 64.2 | **-10.47% [-10.86, -10.05]** | 0/8 |

The N768/D48 row is a real crossover and is retained as a negative boundary.
The triangle speedups also carry a substantial incremental-memory cost: at
N768/D32 SR-R uses 3165.8 MiB versus 1467.0 MiB for MegaFold and 892.1 MiB
for TriFast. Do not describe the published-kernel result as a memory win, and
do not conflate these operator peaks with the separate whole-model crop768
reserved-memory measurement.

## Seal and claim boundary

The final campaign contains 3,056 files / 216,471,521 bytes. SHA256 of
`CAMPAIGN_SHA256SUMS` is
`befd0336c67e193781cfb3f81b57e5fdbfa2fca1390b2082092b17005b5475ac`.

Defensible claim: under exact OpenFold3 producer layouts on one H100,
StableFold SR-R is faster than the pinned public MegaFold and TriFast kernels
throughout the measured D16/D32 triangle region, while using more incremental
operator memory; MegaFold wins the measured N768/D48 diffusion APB cell.
