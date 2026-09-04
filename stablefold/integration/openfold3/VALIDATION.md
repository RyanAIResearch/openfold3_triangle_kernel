# Validation record

Host: Lambda 1x H100 80 GB (driver 570.195.03), Python 3.14.7, PyTorch 2.10.0+cu12.9,
pytorch-lightning 2.6.1. Date: 2026-09-04.

## Package

| Check | Result |
| --- | --- |
| `tools/build_sm90_wheel.sh` | wheel built in 9.5 min; clean-install smoke (no compiler on PATH, JIT forbidden) passed |
| CPU tests: packaging, distributed contract, gloo preflight | 36 passed |
| GPU tests: attention facade, workspace planner, TriMul C++ contract | 64 passed |
| `tests/test_numerics_vs_fp64.py` | 10 passed: StableFold error at or below the stock BF16 path, bitwise replay, exact zeros on fully masked planes |

## Host patch on upstream `aqlaboratory/openfold-3` e47adbf

| Check | Result |
| --- | --- |
| `git apply` on a fresh checkout | clean, 16 files |
| Import without the package installed | OpenFold3 imports; `stablefold_aot_modules_from_config` resolves to `None` |
| `TriangleAttention` layer with `use_stable_triangle_attention=True` | routed to the package (invocation count +1) |
| `TriangleMultiplicationOutgoing` with `use_stable_triangle_multiplicative=True` | N=128 routed to the package; N=64 fell back to stock with one warning |
| Model config with both flags on | accepted by the upstream config validator |
| `run_openfold train`, one BF16 step, PDB subset, crop 384 | `max_steps=1` reached; 360 StableFold attention invocations in the step; zero fallback messages |

## Not yet validated

- Multi-GPU DDP through the patched upstream (the preflight callback is the
  same code that passed 2x H100 in the fork).
- AQLab's `openfold-3-freeze-confidence` branch: apply the patch with
  `--3way`; the hunks sit next to the `use_cueq_triangle_kernels` lines.
