# H100 AOT release gate

Date: 2026-08-24 UTC

GPU: NVIDIA H100 80GB HBM3

Artifact commit: `eb013dd0c6429bd6c82a5e47c1a0b65df6da098b`

Evidence root: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/aot_eb013dd0c_20260824T000736Z`

## Release result

A fresh detached clone at the exact clean commit built all four extensions for
`sm_90a` with SASS only and no embedded PTX. The wheel is
`openfold3-0.0.0-cp314-cp314-linux_x86_64.whl`, SHA256
`dfbc5cb1a0d4e5130b51687cc5e6227e133926baa524553afaa3a60cf36d5725`.
The attested ABI is Python 3.14.6, Torch 2.10.0, CUDA 12.9, and the C++11 ABI.

The embedded manifest records independent hashes for every extension and
explicitly contains these release-critical symbols:

- `stablefold_triangle_forward_shadow_out`
- `stablefold_triangle_delta_shadow_out`
- `stablefold_triangle_sr_backward_out`

The wheel was installed into a read-only isolated target with compiler/JIT
discovery disabled. Full forward+backward smoke tests passed for both the
modernized two-pass and SR-R arms, including contiguous and noncontiguous input
layouts. All four extension binaries were loaded from that installed target;
the final verdict is `PASS`.

Canonical artifacts are the wheel, `.whl.sha256`, `.whl.attestation.json`,
`aot_clean_install_smoke.json`, and `reports/build.log`. The evidence checksum
manifest verifies all five and has SHA256
`6412616ce00ad16ea776a9c275419f51ee9765cbd1f1385f393e8a8849c9d8ba`.

## Profiler note

The host only provides Nsight Systems 2024.6.2. Its agent-launch hang was
independently reproduced with a 30-second `/bin/true` smoke target, leaving the
same nested orphan agents as the earlier kernel attempt; those exact smoke
processes were terminated. No Nsys data is used in a paper claim. Nsight
Compute mechanism attribution, CUDA-event timing, and CUDA sanitizers all
completed independently. No system package was upgraded.
