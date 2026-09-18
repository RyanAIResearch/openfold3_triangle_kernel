# TurboPairFormer pip package quickstart

The standalone package is `turbopairformer`. Its two operators can be called
without an OpenFold3 installation:

```python
from turbopairformer import turbo_attention, turbo_trimul

attention_output = turbo_attention(q, k, v, pair_bias=pair_bias)
trimul_output = turbo_trimul(module, z)
```

Both masks default to `None` (all positions valid, noncausal). `pair_bias` is
required. For padded inputs, supply `attention_mask` and `pair_mask` explicitly.

See the [operator examples](../../packages/turbopairformer/README.md) for tensor
shapes and the TriMul weight interface.

## Install the standalone package

PyPI publication is pending. The tested candidate is bundled in this private
repository; from a checkout with repository access, install it into a dedicated
environment below. After publication, use `python -m pip install turbopairformer`.

On Linux x86-64 with glibc 2.28+, CPython 3.14, and a Hopper GPU (H100/H200):

```bash
python3.14 -m venv .venv-turbopairformer
source .venv-turbopairformer/bin/activate
python -m pip install release_artifacts/turbopairformer/0.1.0/turbopairformer-0.1.0-cp314-cp314-manylinux_2_28_x86_64.whl
```

The dedicated virtual environment keeps the pinned Torch dependency separate
from your other projects. The distribution name is case-insensitive; import it as `turbopairformer`.
The wheel pins PyPI Torch 2.10.0 with CUDA runtime 12.8. No CUDA compiler or
OpenFold3 installation is needed for direct operator calls.

## Install OpenFold3 and TurboPairFormer with Pixi

With Pixi 0.80 or newer, use a separate checkout so an existing project's
training environment is not updated:

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
