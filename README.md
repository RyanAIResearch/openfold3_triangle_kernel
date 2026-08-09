# Lambda Streaming Triangle Attention for OpenFold3

This repository is a clean, runnable fork of
[AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3) containing the
current RyanAIResearch Lambda Streaming Triangle Attention production path for
NVIDIA H100 (`sm_90`). The upstream OpenFold3 model, PyTorch Lightning training
runner, data pipeline, and official recipe remain intact; the only model-level
addition is a default-off Triangle Attention backend.

> **Release status: production candidate, not GA.** The source is packaged,
> fail-closed, source-locked, and has passed the current unit/operator gates.
> The three-backend 1000-step training soak and validation-quality gate are not
> yet complete, so this repository does not claim production qualification.

## What is included

The runtime path is deliberately small:

```text
use_stable_triangle_attention
  -> TriangleAttention / Attention routing
  -> streaming_triangle_attention.py
  -> streaming_triangle_forward_v1
  -> streaming_triangle_delta_v1
  -> stable_triangle_dkdv_v2
  -> fused_triangle_dq_dbias_v9
```

It provides:

- online-softmax BF16 forward on Hopper Tensor Cores;
- recompute backward with no floating-point atomics;
- fixed-order dBias reduction and bitwise self-replay in a fixed environment;
- runtime sequence length and head count, with `D in {16, 32, 64, 128}`;
- explicit tail guards for non-tile-aligned sequence lengths;
- fail-closed routing: unsupported inputs raise instead of silently falling
  back to PyTorch, AQLab Triton, or NVIDIA cuEquivariance;
- package-owned CUDA sources and public PyTorch autograd integration.

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

The current loader builds CUDA extensions on first use, so the runtime must
provide CUDA/NVCC, a compatible C++ compiler, Ninja, and a writable
`TORCH_EXTENSIONS_DIR`. A prebuilt AOT wheel is a remaining release task.

## Validated scope

- Device gate: Lambda H100 80 GB, compute capability 9.0.
- Tensor contract: BF16 Q/K/V and pair bias; FP32/BF16/FP16 additive mask.
- Clean source snapshot imported from
  `RyanAIResearch/hammer_inference_engine@6db02ff4b31e87bfced98331b22613e96e8ced37`.
- Formal operator shapes: `N={384,640,768}`, `H=4`, `D={16,32}`.
- Generic runtime paths exist for other positive `N/H` within CUDA grid and
  memory limits, but broad production qualification is still pending.

The numerical formula matches Triangle Attention, but floating-point rounding
and the online-softmax reduction order intentionally differ from stock
PyTorch. The contract is deterministic self-replay plus bounded numerical and
training/quality gates—not raw-bit equality with another backend.

## Known release blockers

- incomplete three-backend 1000-step stability and validation-quality study;
- runtime JIT rather than an AOT binary wheel;
- fully masked real `-inf` rows need an explicit backward contract;
- dBias fixed-tree workspace grows as `O(H*N^2)` with a large constant;
- batch prefixes are currently dispatched leaf-by-leaf in Python;
- clean sanitizer coverage and multi-GPU/DDP release evidence are incomplete.

These limitations are stated explicitly so development results are not
mistaken for a GA claim.

## Quick verification

Static/package tests:

```bash
python -m unittest \
  openfold3.tests.core.kernels.test_streaming_triangle_packaging \
  openfold3.tests.core.model.test_attention_stable_routing
```

One public-facade CUDA forward/backward smoke:

```bash
python scripts/benchmarks/streaming_triangle_sanitizer_smoke.py \
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
