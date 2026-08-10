# StableFold Triangle Attention for OpenFold3

This repository is a clean, runnable fork of
[AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3) containing the
current RyanAIResearch StableFold Triangle Attention production path for
NVIDIA H100 (`sm_90`). The upstream OpenFold3 model, PyTorch Lightning training
runner, data pipeline, and official recipe remain intact; the only model-level
addition is a default-off Triangle Attention backend.

> **Release status: production candidate, not GA.** The source is packaged,
> fail-closed, source-locked, and has passed the current unit/operator gates.
> The planned 1000-step training soak was not run in this release cycle;
> there is therefore no long-horizon convergence or validation-quality evidence.
> This repository does not claim production qualification.

## What is included

The runtime path is deliberately small:

```text
use_stable_triangle_attention
  -> TriangleAttention / Attention routing
  -> stablefold_triangle_attention.py
  -> stablefold_triangle_forward
  -> stablefold_triangle_delta
  -> stablefold_triangle_dkdv
  -> stablefold_triangle_dq_dbias
```

It provides:

- online-softmax BF16 forward on Hopper Tensor Cores;
- recompute backward with no floating-point atomics;
- fixed-order dBias reduction and bitwise self-replay in a fixed environment;
- bounded deterministic query-slab workspace planning (256 MiB target by
  default) without changing the global dBias reduction order;
- one CUDA batch for arbitrary positive input prefixes, including a fixed-order
  CUDA reduction when pair bias is broadcast;
- runtime sequence length and head count, with production `D in {16, 32}`;
- explicit tail guards for non-tile-aligned sequence lengths;
- defined zero-output/zero-gradient behavior for fully masked real `-inf` rows;
- fail-closed routing: unsupported inputs raise instead of silently falling
  back to PyTorch, AQLab Triton, or NVIDIA cuEquivariance;
- four independently named, wheel-owned `sm_90` CUDA extensions and public
  PyTorch autograd integration.

The exact runtime closure and numerical contract are documented in
[`PRODUCTION_MANIFEST.md`](openfold3/core/kernels/cuda/PRODUCTION_MANIFEST.md).

## Enable the backend

The backend is disabled by default. In the OpenFold3 mode configuration, set:

```yaml
use_stable_triangle_attention: true
use_triton_triangle_kernels: false
use_cueq_triangle_kernels: false
use_deepspeed_evo_attention: false
use_lma: false
chunk_size: null
```

Production imports the four prebuilt CUDA extensions directly from the AOT
wheel. It never invokes NVCC and never silently falls back to JIT. A source-tree
developer may explicitly opt into JIT with
`OPENFOLD3_TRIANGLE_ALLOW_JIT=1`; that switch is not a production fallback.

The canonical wheel is ABI-locked to CPython 3.14, Torch 2.10.0, Torch CUDA
12.9, and H100 `sm_90` SASS (no embedded PTX). Build it only from a clean,
committed checkout in the pinned H100 environment:

```bash
CUDA_HOME=/home/ubuntu/xzz/openfold3_workspace/src/.pixi/envs/openfold3-cuda12 \
CUDACXX=/home/ubuntu/xzz/openfold3_workspace/src/.pixi/envs/openfold3-cuda12/bin/nvcc \
tools/build_sm90_wheel.sh
```

The builder clones the exact commit into a fresh temporary directory, builds
without isolation, requires `nm` and `cuobjdump` attestation, installs the wheel
into a read-only compiler-free target, runs a public-facade full forward and
backward smoke, and copies the wheel plus hash receipts into `dist/`.

## Validated scope

- Device gate: H100 80 GB, compute capability 9.0.
- Tensor contract: BF16 Q/K/V and pair bias; FP32/BF16/FP16 additive mask.
- Clean source snapshot imported from
  `RyanAIResearch/hammer_inference_engine@6db02ff4b31e87bfced98331b22613e96e8ced37`.
- Formal operator shapes: `N={384,640,768}`, `H=4`, `D={16,32}`.
- Generic runtime paths exist for other positive `N/H` within CUDA grid and
  memory limits, but broad production qualification is still pending.
- A broadcast pair bias is accepted only when the target prefix rank is at
  most 16; a non-broadcast prefix is not subject to this bookkeeping limit.

The numerical formula matches Triangle Attention, but floating-point rounding
and the online-softmax reduction order intentionally differ from stock
PyTorch. The contract is deterministic self-replay plus bounded numerical and
training/quality gates—not raw-bit equality with another backend.

## Known release blockers

- the planned 1000-step training study was not run, so no long-horizon
  loss/optimizer/EMA trajectory or convergence evidence exists yet;
- validation quality and time-to-quality have not been established;
- the AOT wheel still needs its dynamic H100 build, binary attestation, and
  compiler-free clean-install receipt on the release commit;
- complete Compute Sanitizer and multi-GPU/DDP release evidence are incomplete.

These limitations are stated explicitly so development results are not
mistaken for a GA claim.

## Quick verification

Static/package tests:

```bash
python -m unittest \
  openfold3.tests.core.kernels.test_stablefold_triangle_aot_packaging \
  openfold3.tests.core.kernels.test_stablefold_triangle_packaging \
  openfold3.tests.core.kernels.test_stablefold_triangle_workspace \
  openfold3.tests.core.model.test_attention_stable_routing
```

One public-facade CUDA forward/backward smoke:

```bash
python scripts/benchmarks/stablefold_triangle_sanitizer_smoke.py \
  --sequence 65 --heads 4 --head-dim 32 --prefix 1
```

Run the same driver under `compute-sanitizer --tool memcheck`, `synccheck`, and
`racecheck` before a release build.

## Upstream and license

This is a derivative of AQLab OpenFold3. Upstream provenance is frozen in
[`UPSTREAM.md`](UPSTREAM.md), and the original project README is retained as
[`README_UPSTREAM.md`](README_UPSTREAM.md). The upstream and added code are
distributed under Apache-2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Please retain and cite the upstream OpenFold3 authors using the existing
[`CITATION.cff`](CITATION.cff).
