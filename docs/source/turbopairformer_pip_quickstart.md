# TurboPairFormer pip package quickstart

The standalone package lives in `packages/turbopairformer`. Its public API is
`turbopairformer.triangle_attention`, `turbopairformer.is_triangle_attention_supported`,
`turbopairformer.run_trimul`, and `turbopairformer.check_trimul_support`. It does not
require an OpenFold3 installation for direct operator calls.

The current name is **TurboPairFormer**. Public index publication is a later release
step; these instructions build and install a local wheel. After publication,
`pip install turbopairformer` will download the published package. The retained `feat/trionce-pip-package` branch records the earlier package work;
new changes belong to `feat/turbopairformer-pip-package`.

## Build and install

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
It does not use or upgrade an older shared Pixi CUDA 12.9 runtime environment.

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
