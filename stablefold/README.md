# StableFold

Deterministic Hopper (`sm_90a`) CUDA kernels for the two O(N³) operators of
AlphaFold3-style pair stacks: Triangle Attention and the Triangle
Multiplicative Update (TriMul). StableFold is a standalone package; a host
model adopts it the same way it adopts cuEquivariance: one pip dependency, one
configuration flag per operator, and a stock fallback for unsupported shapes.

Measured on one H100 80 GB with the OpenFold3 training step at crop 384
(BF16 mixed precision, matched data and seeds), the two operators together
cut the step from 6.94 s (stock PyTorch) or 5.47 s (OpenFold3 Triton route)
to 4.47 s. Kernel outputs and gradients are bitwise reproducible against
themselves and their error against an FP64 oracle is at or below the stock
BF16 path.

## Requirements

- Linux x86-64, NVIDIA H100 80 GB (compute capability 9.0, `sm_90a`)
- The prebuilt wheel is ABI-locked: CPython 3.14, PyTorch 2.10.0, CUDA 12.9
- No CUDA toolkit is needed at runtime; the wheel ships SASS only, no PTX

## Install

```bash
uv pip install stablefold-0.1.0-cp314-cp314-linux_x86_64.whl
```

## Use from Python

```python
import stablefold

# q/k/v: [*, O, H, N, D] BF16 (raw, unscaled); mask: [*, O, 1, 1, N] additive;
# pair_bias: [*, 1, H, N, N] BF16 or FP32. Returns [*, O, N, H, D].
supported, reason = stablefold.is_triangle_attention_supported(q, k, v, mask, bias)
out = stablefold.triangle_attention(q, k, v, mask, bias)
```

The facades fail closed: an unsupported shape raises instead of silently
computing something else. Model-level routing with a stock fallback lives in
the host model (see `integration/openfold3/`).

Execution flags (read from the environment):

| Flag | Default | Meaning |
| --- | --- | --- |
| `STABLEFOLD_SR_BACKWARD=1` | recommended | single-reconstruction backward lane |
| `STABLEFOLD_SR_SHADOW=1` | recommended | store a BF16 rounding residual in the forward to sharpen the backward row scalar |
| `STABLEFOLD_ALLOW_JIT=1` | off | source-tree development only: compile the `.cu` files with nvcc on first use |

The `OPENFOLD3_*` spellings used by the original OpenFold3 fork are accepted as
aliases.

## Distributed training

`stablefold.distributed.StableFoldKernelDDPCallback` is a PyTorch Lightning
callback that, before the first DDP step, checks that every rank loads the
same attested binaries on a full H100 and writes a receipt to the run
directory. It is optional; see `integration/openfold3/` for how the OpenFold3
patch installs it.

## Build the wheel

```bash
CUDA_HOME=/path/to/cuda-12.9 tools/build_sm90_wheel.sh
```

## Tests

```bash
uv pip install -e ".[test]"
pytest tests
```

Kernel-level tests need an H100; `tests/host_openfold3/` additionally needs
an OpenFold3 installation and is skipped otherwise.

## Provenance

Derived from the StableFold integration in
[RyanAIResearch/openfold3_triangle_kernel](https://github.com/RyanAIResearch/openfold3_triangle_kernel),
itself derived from [AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3).
Apache-2.0; see `LICENSE` and `NOTICE`.
