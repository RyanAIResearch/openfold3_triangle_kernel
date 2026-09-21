# Protenix Attention + cuEq TriMul

`--backend protenix` selects the combination used in the matched crop
384/640/768 experiments: ByteDance Protenix Triangle Attention and NVIDIA
cuEquivariance 0.10.0 TriMul. The backend is optional and off by default.

## Install

Use the repository's Python 3.14 / PyTorch 2.10.0 CUDA 12.8 environment.
The tested kernel runtime uses Triton 3.6.0 and a single H100 (SM 9.0).
Install the optional runtime dependencies in that environment:

```bash
python -m pip install 'triton==3.6.0' 'cuequivariance==0.10.0' \
  'cuequivariance-torch==0.10.0' 'cuequivariance-ops-torch-cu12==0.10.0'
```

These dependencies are also declared in the source distribution's `protenix`
extra. The required Protenix kernel sources are included in an isolated
OpenFold3 namespace with their license and provenance. Installing the full
Protenix model package or changing a `protenix` installation is unnecessary.
The first selected GPU call compiles and autotunes Triton kernels.

## Select

Starting with a complete training YAML containing your existing data paths:

```bash
python tools/select_training_backend.py your_training.yml \
  --backend protenix --output training_protenix.yml
```

Run `training_protenix.yml` with the same OpenFold3 training command as before.
The selector preserves data, optimizer, model weights and schedule. Equivalent
settings under `model_update.custom.settings.memory.train` are:

```yaml
use_protenix_triangle_attention: true
use_cueq_triangle_kernels: true
use_stable_triangle_attention: false
use_stable_triangle_multiplicative: false
use_triton_triangle_kernels: false
use_deepspeed_evo_attention: false
use_lma: false
chunk_size: null
```

The `use_stable_*` keys above are existing main-branch configuration keys used
by TurboPairFormer; they must be disabled for this combination. The standalone
`turbopairformer` PyPI package is independent of this source integration.

## Scope

| Module | Route with the Protenix combination |
|---|---|
| BF16 template / MSA / main Pairformer triangle attention | Protenix |
| MSA / main Pairformer TriMul | Existing cuEq route |
| Confidence attention and TriMul | Stock OpenFold3 |
| Template TriMul | Stock OpenFold3 |
| Diffusion, attention pair bias, other attention | Stock OpenFold3 |

The standard experiment architecture has 108 targeted triangle attention and
104 targeted TriMul instances. Counts depend on the model configuration and
template participation. FP32 attention is left on the stock path; the adapter
does not downcast confidence or change projection precision. As in the official Protenix call sites, sequence lengths at most 16 use
stock attention, and unequal input/hidden TriMul channels use stock TriMul.
Existing cuEq shape/dtype fallback behavior also remains in effect.

This route is for BF16 mixed precision, unchunked, out-of-place execution.
Conflicting attention backends, chunked attention, inference offload and active
in-place execution are rejected. Keep the original model architecture and
projection weights. This integration does not certify multi-GPU training or
inference offload.

## Kernel contract and provenance

The kernel revision is
[`4c355be4553512f72453ecbfb65e69f4c35d1413`](https://github.com/bytedance/Protenix/tree/4c355be4553512f72453ecbfb65e69f4c35d1413/protenix/model/tri_attention).
All six vendored implementation files, including autotune, cache handling and
distributed utilities, match that revision byte for byte after reversing only
the private import namespace substitutions. Local host adapters provide layout
conversion and the input contract below.
The host scales Q once by `1/sqrt(head_dim)` before calling the native kernel.
The adapter preserves additive mask semantics, pair-bias gradients and logical
batch/template broadcasting, and makes output gradients contiguous for the
native backward. Additive masks must not require gradients.

Autotune follows the upstream implementation: JSON files are written under
`openfold3/core/kernels/protenix/_vendor/TriAttentionCache/<RANK>/`, with the
compute capability in each filename. That package directory must be writable.
Select the CUDA device before the first Protenix call; upstream captures the
current device when its native module is imported. Runtime caches are excluded
from source control and distributions. There is no custom cache override.
Runtime compilation and kernel failures propagate instead of silently selecting
another GPU attention backend.

The historical 70-step measurements used a frozen OpenFold3 experiment host;
they establish prior use of this operator combination, not a new performance
measurement of the current main branch. Run the checked-in regression tests
in the GPU environment:

```bash
python -m pytest openfold3/tests/test_protenix_gpu.py
```
