# TurboPairFormer development

This guide covers building, testing and maintaining the standalone package.
For installation and examples, start with the [README](README.md). Operator
arguments, tensor layouts and gradients are in the
[API reference](turbopairformer/API_GUIDE.md).

- [Supported environment](#supported-environment)
- [Build and test](#build-and-test)
- [Runtime configuration](#runtime-configuration)
- [Current source synchronization](#current-source-synchronization)
- [OpenFold3 integration](#openfold3-integration)

All commands below start from the **repository root**, except where an explicit
`cd` changes the directory.

## Supported environment

| Component | Requirement |
| --- | --- |
| Platform | Linux x86-64, glibc 2.28 or newer |
| Python | CPython 3.14 |
| PyTorch | PyPI `torch==2.10.0`, CUDA runtime 12.8 |
| GPU | NVIDIA H100 or H200, Hopper compute capability 9.0 |
| CUDA target | `sm_90a` for both GPU models |
| Build toolkit | CUDA 12.9; unnecessary for an installed AOT wheel |

Use a dedicated environment with this Python/Torch combination. Mixing a Conda
Torch build or a wheel from another CUDA index with the PyPI dependencies is
unsupported. Other platforms and ABI combinations are not covered by this wheel.

AOT wheels contain all five compiled extensions. The loader checks ABI and
binary hashes before loading them. With the default configuration, a mismatch
raises an error instead of triggering a local build.

## Build and test

### Build a release candidate

Use a clean Git checkout on Linux x86-64 with Docker and GPU access. Choose
empty wheel-output and evidence directories outside the checkout. The builder
creates the pinned manylinux 2.28 build and runtime images for the current commit.

```bash
bash packages/turbopairformer/tools/build_manylinux_wheel.sh \
  /absolute/path/to/wheel-output
```

The builder compiles all five extensions, removes build-host RPATH/RUNPATH,
runs `auditwheel repair`, and records the repaired binary hashes and wheel
digest. It also checks that the installed extensions load without JIT or a
compiler. This loading check does not run the numerical test suite.

Next, test the final repaired wheel on a supported GPU from the same clean
commit, using the runtime image created by the builder:

```bash
bash scripts/turbopairformer/validate_release_gpu.sh \
  /absolute/path/to/wheel-output/turbopairformer-0.1.3-cp314-cp314-manylinux_2_28_x86_64.whl \
  /absolute/path/to/evidence
```

This installs the wheel in a separate environment without OpenFold3 or the CUDA
toolkit, disables JIT, makes the installed package read-only, and runs AOT,
forward/backward, CLI and standalone tests. It also runs `twine check --strict`.
Keep the resulting evidence with the exact wheel digest and source commit.
Changes to the package tree require a new build and validation before release;
receipts from earlier wheels do not certify a new artifact.
For release bundles and publication, follow the
[release procedure](../../docs/source/turbopairformer_release.md).

### Build a local AOT wheel

For development without Docker, use the supported Python/Torch environment,
a CUDA 12.9 toolkit and GPU access:

```bash
python -m pip install setuptools wheel patchelf pytest pytorch-lightning==2.6.1
CUDA_HOME=/absolute/path/to/cuda-12.9 \
  bash packages/turbopairformer/tools/build_sm90_wheel.sh
python -m pip install --force-reinstall --no-deps \
  packages/turbopairformer/dist/turbopairformer-0.1.3-cp314-cp314-linux_x86_64.whl

# Check the installed wheel from outside the source checkout.
unset PYTHONPATH
export TURBOPAIRFORMER_ALLOW_JIT=0
cd /tmp
python /absolute/path/to/repo/packages/turbopairformer/tools/smoke_installed.py
TURBOPAIRFORMER_TEST_WHEEL=/absolute/path/to/repo/packages/turbopairformer/dist/turbopairformer-0.1.3-cp314-cp314-linux_x86_64.whl \
  python -m pytest /absolute/path/to/repo/packages/turbopairformer/tests \
  --ignore=/absolute/path/to/repo/packages/turbopairformer/tests/host_openfold3
```

The builder uses an isolated source copy and leaves the source manifest
placeholder unchanged. It rejects uncommitted changes by default. For a local
experiment, `TURBOPAIRFORMER_ALLOW_DIRTY_BUILD=1` permits a build marked dirty;
such wheels are not release artifacts. The local `linux_x86_64` wheel is for
development; use the manylinux workflow above for a public wheel.

### Develop with editable source and JIT

For an edit/build/run loop, use a fresh checkout in the supported environment
with the CUDA toolkit available:

```bash
python -m pip install setuptools wheel ninja
TURBOPAIRFORMER_SOURCE_ONLY=1 python -m pip install \
  --no-build-isolation --no-deps -e packages/turbopairformer
export TURBOPAIRFORMER_ALLOW_JIT=1
export TORCH_CUDA_ARCH_LIST=9.0a
```

Source-only installation and runtime compilation require these separate opt-ins.
Without AOT building or the source-only opt-in, installation is rejected because
it would omit the binaries. JIT compilation occurs when an extension is first
needed and no usable AOT binary is available.

Editable AOT builds use an ignored `_build_manifest.local.json`; wheels embed
`_build_manifest.json`. Start JIT work without an old editable AOT manifest or
extension binaries, which would otherwise take precedence.

### What the checks establish

| Check | Coverage |
| --- | --- |
| `tools/smoke_installed.py` | Installed AOT loading, attention/TriMul forward and backward, fixed-input replay |
| `tests/test_numerics_vs_fp64.py` | Attention output and gradients against an FP64 reference |
| Other standalone tests | Public APIs, masks, runtime policy, TriMul contracts, packaging and distributed identity |
| `tests/host_openfold3/` | Host routing and integration; requires OpenFold3 and its dependencies |

These paths are relative to `packages/turbopairformer/`. The standalone tests
need `pytest` and `pytorch-lightning`; host tests are excluded from the
standalone installed-wheel gate.

The attention numerical bound remains
`relative_l2 <= 1.1 * stock_relative_l2 + 1e-6`. That FP64 test does not provide
a full-TriMul oracle or establish end-to-end model accuracy. Fixed-configuration
replay checks do not promise identical bits across GPU models. Record the GPU
used: H100 measurements are not H200 measurements. Full-model performance and
multi-GPU training require separate validation.

## Runtime configuration

The defaults already select single-reconstruction (SR) backward with the saved
forward rounding residual. Normal use needs no environment switches.

| Variable | Default | Behavior |
| --- | --- | --- |
| `TURBOPAIRFORMER_SR_BACKWARD` | `1` | Public attention requires SR; any other effective value raises before shape validation or execution. |
| `TURBOPAIRFORMER_SR_SHADOW` | `1` | Saves and uses the forward rounding residual. `0` disables that path while retaining SR backward. |
| `TURBOPAIRFORMER_ALLOW_JIT` | `0` | Local compilation is disabled unless explicitly enabled for development. |

Legacy aliases are `OPENFOLD3_SR_BACKWARD`, `OPENFOLD3_SR_SHADOW` and
`OPENFOLD3_TRIANGLE_ALLOW_JIT`, respectively. A canonical `TURBOPAIRFORMER_*`
value takes precedence over its alias. Host integrations can call
`turbopairformer.policy.validate_attention_policy()` before their support check
to reject an invalid setting before falling back to another backend.

The retained low-level split-backward functions support reference tests;
the public attention API cannot select them. AOT extension modules use
`_turbopairformer_*_sm90a` names; Torch custom operations use the
`turbopairformer` namespace. Keep Python modules and compiled extensions from
the same wheel.

For training, pass persistent TriMul weights using the
[functional or module API](turbopairformer/API_GUIDE.md#trimul---triangle-multiplicative-update).
Omitted normalization parameters become ones/zeros; omitted matrices are sampled
independently in FP32 on `z.device`, with standard deviation `1/sqrt(C)` and
without gradients. These defaults are recreated per call. Reusing parameters
or resetting the Torch seed is necessary for repeatable demonstrations.

The API reference is also available offline, without importing Torch or CUDA:

```bash
PYTHONPATH=packages/turbopairformer python -m turbopairformer --help
PYTHONPATH=packages/turbopairformer python -m turbopairformer api --all
```

## Current source synchronization

The five package CUDA files below match the host files at
[main commit `2842a19e`](https://github.com/RyanAIResearch/openfold3_triangle_kernel/commit/2842a19e41cbdbc7bc4c4ea2313c64e68e92f0c2)
byte for byte. Package files live in `packages/turbopairformer/turbopairformer/`;
host files live in `openfold3/core/kernels/turbopairformer/`.

| Package CUDA file | Host CUDA file |
| --- | --- |
| `triangle_forward.cu` | `turbopairformer_triangle_forward.cu` |
| `triangle_delta.cu` | `turbopairformer_triangle_delta.cu` |
| `triangle_dkdv.cu` | `turbopairformer_triangle_dkdv.cu` |
| `triangle_dq_dbias.cu` | `turbopairformer_triangle_dq_dbias.cu` |
| `trimul_cuda.cu` | `turbopairformer_trimul_cuda.cu` |

<details>
<summary>SHA256 values for the matching CUDA sources</summary>

```text
triangle_forward.cu   1dd3881e4ee05b451eb44e0bade615f1208f9bda72713f91dcb6ff34265f7965
triangle_delta.cu     657d3ec441d6c0906365b7c991356772af567e46964027b1a2b7d3e52220bb69
triangle_dkdv.cu      1e45df69f35a35debc85d64d9ef9fbdcb526b5032a563eee82276c362aea0f1c
triangle_dq_dbias.cu  94a621875e72833592e2990635725d01ccf334ca766980f4491ef2f759bf6db9
trimul_cuda.cu        3f9eb2165f3603d3a9569225b28cc7715a0f714499eb19845b2e5d7c6e543056
```

</details>

This synchronization carries the readability changes and host package rename,
while preserving computation, operation order, synchronization and numerical
policy. The Python layer retains the standalone optional-mask API, environment
aliases and AOT loader. Published 0.1.3 wheels predate this synchronization and
remain unchanged; the source update does not replace an existing wheel.

## OpenFold3 integration

The standalone package does not depend on OpenFold3 or change a host model's
backend when installed. This fork defaults to its in-tree kernels; select
`OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer` explicitly to use the package.

- [Fork integration and Pixi setup](../../docs/source/turbopairformer_pip_quickstart.md)
- [Adapter for upstream OpenFold3](integration/openfold3/README.md)
- [Single-GPU build, diagnosis and launch helpers](../../scripts/turbopairformer/README.md)

`turbopairformer.distributed.TurboPairFormerKernelDDPCallback` checks the
wheel/rank contract. Full multi-GPU OpenFold3 training with this standalone
package has not been qualified; use the supported single-GPU integration.
Historical host training results do not automatically apply to a new wheel.

## Attribution

See the [project team](README.md#project-team). This package derives from
[RyanAIResearch/openfold3_triangle_kernel](https://github.com/RyanAIResearch/openfold3_triangle_kernel)
and [AQLab OpenFold3](https://github.com/aqlaboratory/openfold-3).
It is licensed under Apache-2.0; retain [LICENSE](LICENSE) and [NOTICE](NOTICE).
