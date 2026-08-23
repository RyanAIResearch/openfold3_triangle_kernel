# StableFold SR H100 campaign summary

Date: 2026-08-23 UTC
GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs
Kernel commit: `4be8da30152591ff94c8b95cbe84dda0c3ebb37a`
Branch: `sr/dq-warpgroup`
Evidence root: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/h100_4be8da301_20260823T205450Z`

## Outcome

The final H100 campaign passed the correctness, replay, workspace-plan, sanitizer,
kernel-matrix, same-commit training, and production-system comparison gates.
The campaign was executed without selecting or rerunning individual performance
arms.

### Correctness and safety

- The pre-campaign tip `377a177f9` exposed a real N=640 query-stripe bug: dK/dV
  retained only the final query stripe. The negative evidence is preserved under
  `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/h100_377a177f9_20260823T200651Z`.
- Commit `4d8fdcf6` adds exact FP32 dK/dV continuation state across query stripes.
- Commit `4be8da301` chooses the scratch-minimizing automatic query plan.
- Workspace/query-span/leaf-wave/batch-chunk gates passed bitwise across the
  tested N/D/budget matrix and against the independent two-pass dK/dV oracle.
- SR-C replay and FP64 referee gates passed. SR-R shadow gates passed.
- Compute Sanitizer memcheck, synccheck, and racecheck passed the actual
  query-striped target with zero errors/hazards.

### CUDA-event kernel matrix

Each arm has 60 backward samples in ABCCBA order. Negative deltas are speedups.

| Shape | Baseline ms | SR-C ms | SR-C delta | SR-R ms | SR-R delta |
|---|---:|---:|---:|---:|---:|
| N128 D32 | 0.2514 | 0.2302 | -8.43% | 0.2293 | -8.81% |
| N384 D32 | 1.7019 | 1.3476 | -20.82% | 1.3693 | -19.54% |
| N640 D32 | 7.0715 | 5.3864 | -23.83% | 5.4098 | -23.50% |
| N768 D32 | 12.0194 | 11.2798 | -6.15% | 11.3124 | -5.88% |
| N640 D16 | 6.9004 | 4.2627 | -38.23% | 4.2849 | -37.90% |
| N640 D64 (H2) | 7.6732 | 7.2692 | -5.27% | 7.2863 | -5.04% |
| outer24 N640 D32 | 0.3731 | 0.3207 | -14.05% | 0.3056 | -18.10% |

At N640 D32, automatic full-query planning measured 5.1447 ms and 1806.2 MiB
incremental workspace, versus forced span320 at 5.7043 ms and 1956.2 MiB.

### Same-commit causal training comparison

The only arm difference is `OPENFOLD3_SR_BACKWARD/SHADOW`. Four paired blocks
used AB/BA/BA/AB order; steps 1-2 were excluded by rule.

- Block deltas: -2.02%, -2.02%, -1.91%, -2.24%.
- Order-balanced mean: **-2.05%**.
- Descriptive block-bootstrap 95% interval: **[-2.18%, -1.94%]**.
- AB mean / BA mean: **-2.13% / -1.97%**.
- Maximum same-step absolute loss gap: 2.01e-4.

### Production cuEq system comparison

The production comparator is the archived cuEq wheel and extension. This is a
system comparison, not an isolated SR ablation, because the YAML selects both
triangle attention and multiplicative backends.

- Block deltas: -3.88%, -3.88%, -3.68%, -3.90%.
- Order-balanced mean: **-3.83%**.
- Descriptive block-bootstrap 95% interval: **[-3.89%, -3.73%]**.
- AB mean / BA mean: **-3.89% / -3.78%**.
- Mean peak-memory delta: **-1460 MiB** (71241 vs 69781 MiB in every block).
- Mean full-process energy delta: **-3.71%**. Energy includes initialization,
  training, and teardown and is therefore descriptive rather than per-step.
- Maximum same-step relative loss gap: 8.87e-5.

### Nsight Compute attribution

The process-scoped NVTX range covers one N640 D32 forward/backward invocation.
NCU replay duration is mechanism evidence; CUDA-event timing above is the
performance result.

- Baseline: 11 launches, 8.5865 ms replay-summed backward.
- SR-C: 8 launches, 6.6227 ms, -22.87%.
- SR-R: 8 launches, 6.6532 ms, -22.52%.
- The SR leaf is 5.17 ms, about 69% memory throughput, 128 registers/thread,
  and 111.9 KiB shared memory/block.
- The dQ reduction is 0.373 ms, about 92% DRAM throughput, 10.6% L2 hit rate,
  and 96.1% long-scoreboard PC samples. It is the clearest next optimization
  target.

Nsight Systems 2024.6 could not launch even a minimal target on this host and
timed out while starting its agent. The failure receipt is retained. Nsight
Compute 2025.1 succeeded; root was required solely because the driver restricts
GPU performance counters to administrators.

## Canonical receipts

- `receipts/kernel_matrix_summary.{json,md}`
- `receipts/abba_summary.{json,md}`
- `receipts/cueq_abba_summary.{json,md}`
- `receipts/ncu_summary.{json,md}`
- `correctness/` for gates and sanitizer logs
- `performance/` for raw samples, per-arm logs, GPU traces, and profiler reports
- `environment/` and `artifacts/` for software, GPU, source, and binary identity

No GitHub push is part of this campaign.
