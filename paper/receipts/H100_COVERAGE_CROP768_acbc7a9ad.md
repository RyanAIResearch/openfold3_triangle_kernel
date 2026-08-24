# H100 real-route coverage and crop-768 AOT receipt

Date: 2026-08-24 UTC

Artifact commit: `acbc7a9adfe792f7e4a03ccff01ec5a28e18b1b4`

Documentation base: `7742b79e98df27f1fd360b9d961b087fc24fc6fe`

Campaign: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/coverage_census_acbc7a9_20260824T024923Z`

## Immutable contract

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs.
- Python 3.14.6, PyTorch 2.10.0, CUDA 12.9, driver 570.195.03.
- Final read-only `runtime_acbc` install; `OPENFOLD3_TRIANGLE_ALLOW_JIT=0`.
- AOT wheel SHA-256:
  `4357b7f0a7ac0c71882ed7cb950dcd725cb6e4170f568ab09242c12a9c7ce7ad`.
- No project CUDA or Python source changed during this campaign.
- Verified campaign seal: 595 files, 5,071,492 bytes.
  `COVERAGE_SHA256SUMS` SHA-256 is
  `c911d360f3ac9a31780e2df0a3203dbaeaa8483453ea4de792eecc8b213f00fa`;
  seal JSON SHA-256 is
  `479965dfa0931a905cff8ec746ce801e3a491be71d4bcd56ce833154507e413f`.

## Runtime route census

Two real crop-640 training steps requested StableFold 2,184 times. Every
request reached the support predicate, every predicate returned supported,
and every call executed the facade: 2,184/2,184/2,184, with zero fallback
or hook error.

| D | Exact q shape | Executions | Execution share | Output-gradient hooks | Hook share |
|---:|:---|---:|---:|---:|---:|
| 16 | `1x1x640x4x640x16` | 112 | 5.13% | 32 | 8.89% |
| 24 | `1x1x16x640x24` | 352 | 16.12% | 104 | 28.89% |
| 32 | `1x640x4x640x32` | 760 | 34.80% | 224 | 62.22% |
| 48 | `1x1x16x640x48` | 960 | 43.96% | 0 | 0.00% |

D48 therefore has real forward execution coverage in this workload, but
no observed backward contribution. It is never given a synthetic backward
weight in the route-weighted or end-to-end claim.

## Exact-layout correctness

The final single-harness-SHA matrix covers D16/D32 triangle start and end
mask strides, D24 pairformer attention-pair-bias, and D48 diffusion
attention-pair-bias. At N=128, all six sites were tested against stock,
compiled stock, cuEq 0.10.0, Triton 3.6.0, DR-WG, and SR-R.

- 36/36 cells passed.
- Output, dQ, dK, dV, and dBias were finite in every cell.
- Every relative-L2 value was at most 0.01.
- SR-R worst case: 0.00346207 (D16-end dQ).
- DR-WG and SR-R: all 30 replay fields bitwise.
- cuEq and Triton replay divergences were recorded as comparator evidence,
  not excluded.

An attempt-1 harness bug is retained in the forensic receipts. The
transposed-mask generator masked a pre-transpose column, which became an
all-masked reference row. The general correction preserves the public
contract that the final logical dimension is the key dimension for either
physical stride. It changed no OpenFold3 or CUDA code. After correction,
the complete 36-cell matrix was rerun under one new harness SHA.

## Exact-layout performance

The N=640 screening matrix used a primary and a fully reversed mirror
order, 12 warmups and 30 CUDA-event samples per cell. All 72/72 cells passed
and their q/mask/pair-bias metadata matched the runtime census.

The confirmatory four-block design used AB/BA/BA/AB process order. One
process-level median per block is the statistical unit; adjacent CUDA
events are not treated as independent replicates.

| Site | Production metric | SR-R time reduction vs cuEq | 95% block-bootstrap CI | SR-R vs DR-WG |
|:---|:---|---:|:---|---:|
| D16 start | forward+backward | 38.19% | [37.91, 38.48] | 30.77% |
| D16 end | forward+backward | 38.76% | [38.63, 38.89] | 31.53% |
| D24 APB | forward+backward | 28.74% | [28.02, 29.46] | 4.20% |
| D32 start | forward+backward | 26.90% | [26.74, 27.10] | 16.88% |
| D32 end | forward+backward | 27.13% | [26.98, 27.29] | 16.98% |
| D48 diffusion | forward only | 40.85% | [40.60, 41.23] | -4.39% |

Weighting the measured forward medians by observed facade executions and
the backward medians by observed output-gradient hooks gives an estimated
attention GPU-time reduction of 22.65% vs cuEq (95% CI [22.57, 22.74]) and
9.30% vs DR-WG ([9.04, 9.63]). This is an attention-time estimate, not an
end-to-end step-time result.

## Crop-768 attention-only end-to-end ABBA

Both arms use the same seed, data order, BF16 precision, crop, and final AOT
runtime. Both retain cuEq triangle multiplicative update and disable the
StableFold multiplicative update. The only algorithm selector difference
is StableFold attention off vs on. The configuration diff also contains the
required arm-specific output directory.

Each run has 12 crop-768 training steps. The first two are discarded; the
ten remaining synchronized steps form one run-level mean. Four paired
blocks use AB/BA/BA/AB order. Step boundaries call CUDA synchronize, and
energy is the difference of the H100 NVML total-energy hardware counter
over the same interval.

- 8/8 runs exited zero; 96/96 step receipts present.
- All 40 paired steady steps were faster under SR-R.
- All 40 paired steady steps used less energy under SR-R.
- JIT output directory remained empty.
- No OOM, traceback, or StableFold fallback appeared.

| Block | cuEq s/step | SR-R s/step | Time reduction | cuEq J/step | SR-R J/step | Energy reduction | Full wall reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.4955 | 11.6523 | 6.748% | 6675.3 | 6176.8 | 7.468% | 2.739% |
| 2 | 12.5340 | 11.6778 | 6.831% | 6645.1 | 6167.5 | 7.187% | 2.645% |
| 3 | 12.5223 | 11.6773 | 6.748% | 6666.6 | 6177.4 | 7.338% | 2.660% |
| 4 | 12.5134 | 11.6620 | 6.804% | 6655.1 | 6163.4 | 7.388% | 2.352% |

Aggregate paired-block results:

- Synchronized steady step-time reduction: **6.783%**, 95% block-bootstrap
  CI **[6.748, 6.818]**; throughput speedup **7.277%**.
- Step-energy reduction: **7.345%**, CI **[7.238, 7.435]**.
- Full-process-wall reduction, including startup/data gaps: **2.599%**,
  CI **[2.429, 2.716]**.
- Whole-run peak reserved memory: cuEq 81.038 GB, SR-R 80.977 GB.

The short-run loss referee has 48 paired `train/loss` points: mean absolute
gap 7.250e-5 and maximum 4.902e-4. The four-run cuEq self-envelope has mean
1.925e-4 and maximum 4.807e-4. This supports numerical sanity over the
measured horizon; it is not a convergence or validation-quality claim.

## Claim boundaries

- Runtime execution counts are not timings; route weighting uses separately
  measured CUDA-event medians.
- Route-weighted attention GPU time, synchronized training-step time, and
  full-process wall are three distinct estimands.
- D48 backward timing is a robustness test only in this workload.
- At n=4, block-bootstrap intervals are descriptive/preliminary.
- The 40 within-run paired steps are a consistency diagnostic, not n=40
  independent inferential samples.
- No validation batches were run in the crop-768 speed experiment. Longer
  fork/validation evidence remains separate.
