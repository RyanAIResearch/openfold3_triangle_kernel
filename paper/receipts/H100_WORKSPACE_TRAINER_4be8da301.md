# H100 workspace planner trainer gate

Date: 2026-08-23 UTC

GPU: NVIDIA H100 80GB HBM3

Kernel commit: `4be8da30152591ff94c8b95cbe84dda0c3ebb37a`

Evidence root: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/workspace_trainer_4be8da301_20260823T231454Z`

## Design

The crop-640 SR-R trainer ran budgets 64/128/256/512 MiB in mirrored order
`64,128,256,512,512,256,128,64`. Each arm ran 12 steps; steps 1–2 were
discarded. The same YAML, checkpoint, seed, kernel commit, and experiment hook
were used throughout. The frozen harness and its SHA256 are stored in the
evidence root.

## Trainer result

| Budget | Mean s/step | Delta vs 256 MiB | Repeat drift |
|---:|---:|---:|---:|
| 64 MiB | 8.5413 | +1.615% | +0.119% |
| 128 MiB | 8.4024 | -0.100% | -0.014% |
| 256 MiB | 8.4103 | reference | -0.440% |
| 512 MiB | 8.3942 | -0.184% | -0.141% |

Every budget reported the same whole-model peak: 50,328.7 MiB allocated,
68,726 MiB reserved, and 69,779 MiB by `nvidia-smi`. This is a useful negative
result: changing this local scratch budget does not reduce trainer peak memory,
and 64 MiB pays a measurable multi-wave scheduling cost. Budgets 128–512 MiB
are effectively neutral at trainer scale.

The maximum cross-budget loss gap from the 256 MiB arm was 6.208e-4; the
maximum same-budget repeat gap was 3.467e-4. Therefore this trainer experiment
does not establish bitwise model trajectories. The surrounding stock model and
scheduling scatter remain outside the StableFold kernel contract.

## Kernel-scoped invariance holdout

A separate real N640/D16 holdout covered budgets 64/128/256/512 MiB and forced
query spans 64/128/256/full, producing 1/2/3/5/10-stripe schedules across the
matrix. All eight configurations matched bitwise for dQ, dK, dV, and dBias;
dK/dV also matched the independent two-pass oracle bitwise. This rejects a
single-shape or single-plan special fix and supports the scoped claim that
workspace planning changes residency, not arithmetic output.

Canonical summaries are `analysis/workspace_referee.md`,
`analysis/workspace_trainer_summary.json`, and
`analysis/n640_d16_workspace_holdout.jsonl`. The evidence manifest contains
121 files and verifies with `sha256sum -c`; its own SHA256 is
`ec7b4d026eddf98cae825d89aa47990a4fb6b82c59f3eba1ce7e670de92e8727`.
