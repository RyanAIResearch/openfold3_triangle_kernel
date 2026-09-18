# TurboPairFormer 0.1.0

TurboPairFormer is a Python package providing two CUDA operators for AlphaFold3-style models:

- **TurboTriangleAttention** — Triangle Attention with autograd support.
- **TurboTriangleMultiplicative** — Triangle Multiplicative Update with autograd support.

These are the agreed public operator names. Their Python exports are pending; the attached candidate currently exports `triangle_attention` and `run_trimul`.

The standalone package can be used without installing OpenFold3. It includes precompiled Hopper kernels, so users do not need a CUDA compiler or a runtime JIT build.

## Developers

**Developer names and credit order are awaiting confirmation, including contributors from Lambda.**

The confirmed developer credits will appear here and in the package README and author metadata. GitHub and affiliation links will be included where provided.

## Installation

**pip** — in a fresh Python 3.14 virtual environment, after PyPI publication:

```bash
pip install TurboPairFormer
```

**Pixi / OpenFold3** — from a separate repository checkout:

```bash
pixi install -e openfold3-cuda12
```

[Environment setup and detailed instructions](https://github.com/RyanAIResearch/openfold3_triangle_kernel/blob/feat/turbopairformer-pip-package/docs/source/turbopairformer_pip_quickstart.md).

## Supported configuration

- Linux x86-64 with glibc 2.28 or newer.
- CPython 3.14.
- PyPI PyTorch 2.10.0 with CUDA runtime 12.8.
- Hopper compute capability 9.0 (`sm_90a`): H100 is tested; H200 hardware execution remains unverified.
- OpenFold3 integration supports the documented single-GPU launch path. Full multi-GPU host training is not yet qualified.

## Validation

The currently attached candidate passed **176 H100 tests with zero skips**. All five installed CUDA extensions were checked, including forward/backward execution, numerical checks and repeatability checks. Wheel portability and package metadata checks passed, and the exact Pixi installation command was tested in an isolated H100 environment.

The attached wheel, `release-validation.json`, `installed-aot.json` and `standalone-gpu.xml` identify the exact tested artifact. These checks do not establish a new performance result or qualify H200 and full multi-GPU host training.

## License and upstream acknowledgments

TurboPairFormer is distributed under Apache-2.0. It integrates with the AQLab OpenFold3 project; upstream copyright and license notices are retained in the source.

## Draft status

This release has not been published to PyPI. The two operator names above are the agreed public names; the currently attached candidate still exports `triangle_attention` and `run_trimul`. Before publication, finalize developer credits, synchronize the public function exports and package metadata, rebuild and validate the resulting wheel, and replace the draft's assets and evidence together. PyPI Trusted Publisher authorization is also pending.
