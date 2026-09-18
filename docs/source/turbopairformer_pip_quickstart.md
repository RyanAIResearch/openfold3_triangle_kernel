# TurboPairFormer pip package quickstart

The standalone package is `turbopairformer`. Its two operators can be called
without an OpenFold3 installation:

```python
import torch
from turbopairformer import turbo_attention, turbo_trimul

B, N, H, D = 1, 128, 4, 32
# q/k/v: [batch, outer, heads, sequence, head_dim]; here outer = N.
q, k, v = [
    torch.randn(B, N, H, N, D, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
    for _ in range(3)
]
# Pair bias is shared across the outer dimension.
pair_bias = torch.randn(B, 1, H, N, N, device="cuda", dtype=torch.bfloat16,
                        requires_grad=True)

attention_output = turbo_attention(q, k, v, pair_bias=pair_bias)
attention_output.float().square().mean().backward()
print(attention_output.shape)  # torch.Size([1, 128, 128, 4, 32])
```

All positions in this synthetic example are valid, so the mask is omitted.
Single-reconstruction backward and the saved forward rounding residual are
enabled by default; no environment switches are needed for this example.
For TriMul, pass your existing triangle-multiplication module and pair tensor:

```python
trimul_output = turbo_trimul(module, z)
```

Both masks default to `None` (all positions valid, noncausal). `pair_bias` is
required. For padded inputs, supply `attention_mask` and `pair_mask` explicitly.

See the [operator examples](../../packages/turbopairformer/README.md) for tensor
shapes and the TriMul weight interface.

## Install the standalone package

Install the PyPI package into a dedicated environment:

On Linux x86-64 with glibc 2.28+, CPython 3.14, and a Hopper GPU (H100/H200):

```bash
python3.14 -m venv .venv-turbopairformer
source .venv-turbopairformer/bin/activate
python -m pip install turbopairformer
```

The dedicated virtual environment keeps the pinned Torch dependency separate
from your other projects. The distribution name is case-insensitive; import it as `turbopairformer`.
The wheel pins PyPI Torch 2.10.0 with CUDA runtime 12.8. No CUDA compiler or
OpenFold3 installation is needed for direct operator calls.

## Install OpenFold3 and TurboPairFormer with Pixi

With repository access and Pixi 0.80 or newer, use a separate checkout so an
existing project's training environment is not updated:

```bash
git clone --branch feat/turbopairformer-pip-package --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git openfold3-turbopairformer
cd openfold3-turbopairformer
pixi install -e openfold3-cuda12
pixi run -e openfold3-cuda12 doctor-turbopairformer
```

Pixi installs into this checkout's `.pixi/envs/openfold3-cuda12`.
This installs the same repository-bundled TurboPairFormer wheel, its exact
PyPI Torch ABI,
Lightning 2.6.1, and the current OpenFold3 host as a non-editable package. It does
not compile the CUDA kernels. Every command in this Pixi environment explicitly
selects `OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer` and disables runtime JIT.
The doctor's success means that the installed package, manifest, GPU, ABI, and
five AOT modules load correctly; numerical execution is a separate release check.

For a prepared dataset YAML, run a single-GPU training smoke test:

```bash
pixi run -e openfold3-cuda12 select-training-backend \
  datasets/train_pdb_subset.yaml \
  --backend turbopairformer --provider turbopairformer --single-gpu-smoke \
  --output datasets/train_pdb_subset.turbopairformer.yaml
mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
pixi run -e openfold3-cuda12 train-single-gpu \
  --runner-yaml "$PWD/datasets/train_pdb_subset.turbopairformer.yaml" --seed 42
```

Use `pixi run -e openfold3-cuda12 prepare-pdb-subset -- --train-size 512 --seed 42`
if a small dataset has not been prepared. Downloading data and pretrained model
parameters is separate from package installation.

The package's host integration is limited to single-GPU launches for this
release. The historical in-tree distributed workflow is retained under
`openfold3-cuda12-intree`; see the
[bare-metal guide](bare_metal_distributed_quickstart.md). That environment has a
different Conda Torch / CUDA 12.9 ABI and does not use the standalone wheel.

## Build from source

The initial target is Linux x86_64, CPython 3.14, and Hopper compute capability
9.0 on H100 or H200 (the same `sm_90a` wheel), with PyTorch 2.10.0 and its
PyPI CUDA 12.8 runtime.
Install the exact PyTorch build required by the wheel. The AOT
loader checks Python, Torch, CUDA runtime, and C++ ABI against its embedded
manifest; an incompatible wheel fails before executing a kernel. A build
machine also needs a compatible CUDA toolkit and C++ compiler. A user installing
a prebuilt wheel does not need nvcc.

With a dedicated virtual environment containing the intended Torch build:

```bash
python3.14 -m venv /absolute/path/to/venv
/absolute/path/to/venv/bin/python -m pip install torch==2.10.0 setuptools wheel ninja patchelf
export TURBOPAIRFORMER_PYTHON=/absolute/path/to/venv/bin/python
export CUDA_HOME=/absolute/path/to/cuda-12.9
scripts/turbopairformer/install_turbopairformer.sh
```

The helper builds a wheel and installs it non-editably into that environment.
It uses `TURBOPAIRFORMER_BUILD_AOT=1`, `TORCH_CUDA_ARCH_LIST=9.0a`, and no build
isolation so it links against the selected Torch. It then checks the installed
package from a temporary working directory with JIT disabled. The build helper
requires `TURBOPAIRFORMER_PYTHON` explicitly and checks its runtime before building.
It does not use or upgrade the historical `openfold3-cuda12-intree` runtime.

To install an already built wheel into a matching environment:

```bash
python -m pip install /absolute/path/to/turbopairformer-<version>-<tags>.whl
```

See `packages/turbopairformer/README.md` for the supported runtime version and direct
operator examples. Build-toolkit and Torch CUDA-runtime versions are separate;
use the wheel manifest to identify the latter. H100 and H200 use the same
binary architecture target; retain GPU-specific validation receipts when
deploying to either model.

## Verify the installed provider

For a package-only environment or an OpenFold3 checkout with the provider
adapter:

```bash
OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer \
  /absolute/path/to/venv/bin/python /absolute/path/to/repo/tools/turbopairformer_doctor.py \
  --expect-provider turbopairformer --json
```

The doctor resolves the provider first, then checks that provider's imported
package, GPU capability, manifest, ABI, binary hashes, and five loaded AOT
extensions. `--allow-no-gpu` permits an **imports-only** check when CUDA is not
available; it does not validate GPU execution. A doctor pass checks loading,
not numerical correctness or performance.

## Use the package in this OpenFold3 fork

Existing stable-kernel model flags are unchanged. Provider selection is
explicit and memoized on first use:

| Setting | Serving implementation |
| --- | --- |
| unset or `OPENFOLD3_TRIANGLE_PROVIDER=intree` | Vendored OpenFold3 kernels |
| `OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer` | Installed standalone package |

The `openfold3-cuda12` Pixi environment sets the package provider automatically.
Installing TurboPairFormer alone does not change an existing OpenFold3 run. An
explicit package selection fails if the package is unavailable. The production
OpenFold3 wheel must include `openfold3/core/kernels/cuda/provider.py` to use
this switch; an older host wheel does not gain this adapter by installing
TurboPairFormer.

Prepare a one-step training configuration:

```bash
python -m tools.select_training_backend datasets/train_pdb_subset.yaml \
  --backend turbopairformer --provider turbopairformer --single-gpu-smoke \
  --output datasets/train_pdb_subset.turbopairformer.yaml
```

`--backend turbopairformer` selects the two model flags. `--provider turbopairformer`
prints the environment export needed to select the package; provider selection
is not stored in the YAML. The package launcher sets it explicitly:

```bash
export TURBOPAIRFORMER_PYTHON=/absolute/path/to/venv/bin/python
export OF3_RUN_CWD=/absolute/path/outside/repo/openfold3-run
export CUDA_VISIBLE_DEVICES=0
scripts/turbopairformer/run_single_gpu.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.turbopairformer.yaml" --seed 42
```

The environment must also contain this host revision of OpenFold3 and its
training dependencies. The launcher checks the provider and dataset paths
before training. It requires AOT binaries and disables runtime JIT. Set
`TURBOPAIRFORMER_PROVIDER=intree` to use an installed in-tree AOT build instead.

The standalone package's OpenFold3 integration currently supports single-GPU
launches. Its multi-GPU entry points stop with an explicit error because the
host DDP preflight has not been integrated and validated for this package. The
existing in-tree distributed workflow retains its separate certification and
artifact requirements.

## Compare four backends on one H100

Start with crop 384, using an already prepared full-crop dataset. From this
feature-branch checkout, run the following in Bash. Each backend starts from
scratch with seed 42, five warmup steps, and 20 measured training steps.

```bash
set -euo pipefail
mkdir -p ../openfold3-run
export OF3_RUN_CWD="$(cd ../openfold3-run && pwd)"
export CROP_DATA_ROOT="$OF3_RUN_CWD/pdb-full-crop-384-640-768"
export CUDA_VISIBLE_DEVICES=0
export RESULTS_ROOT="$OF3_RUN_CWD/backend-comparison-1gpu/crop384-$(date -u +%Y%m%dT%H%M%SZ)"
unset PYTHONPATH PYTHONHOME LD_PRELOAD

test -f "$CROP_DATA_ROOT/runner_crop_384.yaml"
test ! -e "$RESULTS_ROOT"
mkdir -p "$RESULTS_ROOT/logs"
FAILED_BACKENDS=()
for BACKEND in aqlab triton cueq turbopairformer; do
  if ! pixi run -e openfold3-cuda12 benchmark-single-node \
      --runner-yaml "$CROP_DATA_ROOT/runner_crop_384.yaml" \
      --backend "$BACKEND" \
      --output-root "$RESULTS_ROOT" \
      --warmup-steps 5 --measure-steps 20 \
      --devices 1 --num-workers 10 \
      --require-full-crop --use-tf32 --seed 42 \
      2>&1 | tee "$RESULTS_ROOT/logs/${BACKEND}.log"; then
    FAILED_BACKENDS+=("$BACKEND")
    continue
  fi
  if [[ ! -f "$RESULTS_ROOT/$BACKEND/summary.json" ]]; then
    FAILED_BACKENDS+=("$BACKEND")
  fi
done
if (( ${#FAILED_BACKENDS[@]} )); then
  echo "Incomplete: ${FAILED_BACKENDS[*]}; not plotting" >&2
  exit 1
fi
pixi run -e openfold3-cuda12 plot-backend-comparison \
  "$RESULTS_ROOT" --baseline triton
```

Set `CROP_DATA_ROOT` to the existing fixture directory if it lives elsewhere.
All four runs use BF16 mixed precision and the same explicit `--use-tf32`
setting for host FP32 matrix operations. The pinned cuEq 0.10.0 requires TF32
for its FP32 Q/K/V Triangle Attention backward; its BF16 path does not require
TF32. This setting does not change the arithmetic of TurboPairFormer's custom
CUDA kernels. A completed short run is
an integration and timing check; it does not establish long-run convergence
or reproduce a historical speedup by itself.

## Verify changes

```bash
python -m pytest packages/turbopairformer/tests --import-mode=importlib
python -m pytest openfold3/tests/test_stablefold_provider_equivalence.py \
  --import-mode=importlib
```

GPU tests require a compatible H100 or H200 runtime and built artifacts. CPU
tests or skips do not establish CUDA correctness. Compare installed artifacts in isolated
environments and retain the wheel manifest and test results. Historical
benchmarks from earlier artifacts are not validation for a rebuilt wheel.

For upstream AQLab OpenFold3, use the versioned patch and instructions in
`packages/turbopairformer/integration/openfold3/`; this fork already contains its own
provider adapter and should not apply that upstream patch.
