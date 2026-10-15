# TurboPairFormer development

Build and test the standalone package from this repository. For installation
and operator examples, see the [README](README.md).

## Build and verify

Use a Linux x86-64 machine with an H100 or H200 GPU, Docker, and NVIDIA
Container Toolkit. The build container supplies Python 3.14, PyPI PyTorch
2.10.0 with CUDA runtime 12.8, and CUDA toolkit 12.9.

Start from a clean, committed checkout. Run both commands from the **repository
root**, on the same machine and commit. Use empty output directories outside
the checkout.

Build the wheel:

```bash
bash packages/turbopairformer/tools/build_manylinux_wheel.sh \
  /absolute/path/to/wheel-output
```

Test the resulting wheel using the runtime image created by that build:

```bash
bash scripts/turbopairformer/validate_release_gpu.sh \
  /absolute/path/to/wheel-output/turbopairformer-0.1.3-cp314-cp314-manylinux_2_28_x86_64.whl \
  /absolute/path/to/test-results
```

The test installs the wheel with JIT disabled and checks AOT loading, numerical
accuracy, forward/backward execution and repeatability. Keep the test results
with the exact wheel and source commit. These are operator checks; full-model
and multi-GPU training need separate validation.

Normal use needs no environment switches: SR backward and the saved forward
rounding residual are enabled, and JIT is disabled.

## Current source synchronization

The CUDA files below match
[main commit `2842a19e`](https://github.com/RyanAIResearch/openfold3_triangle_kernel/commit/2842a19e41cbdbc7bc4c4ea2313c64e68e92f0c2)
byte for byte.

- Package directory: `packages/turbopairformer/turbopairformer/`
- Host directory: `openfold3/core/kernels/turbopairformer/`

| Package file | Host file |
| --- | --- |
| `triangle_forward.cu` | `turbopairformer_triangle_forward.cu` |
| `triangle_delta.cu` | `turbopairformer_triangle_delta.cu` |
| `triangle_dkdv.cu` | `turbopairformer_triangle_dkdv.cu` |
| `triangle_dq_dbias.cu` | `turbopairformer_triangle_dq_dbias.cu` |
| `trimul_cuda.cu` | `turbopairformer_trimul_cuda.cu` |

The synchronization preserves computation and the standalone Python APIs.
Published 0.1.3 wheels predate this source update and remain unchanged.

## Further reading

- [API reference](turbopairformer/API_GUIDE.md): arguments, shapes and gradients.
- [Source installation and OpenFold3 integration](../../docs/source/turbopairformer_pip_quickstart.md).
- [Release procedure](../../docs/source/turbopairformer_release.md): validation records and publishing.

Retain [LICENSE](LICENSE) and [NOTICE](NOTICE) when redistributing the package.
