# Current revision status

The host patch was updated on 2026-09-18 to use the current standalone package
module and function names. Its single-GPU route is retained and distributed
training now raises at startup pending host DDP integration. A fresh wheel and
GPU validation are required; the historical results below apply only to their
recorded artifacts, and the wheel hashes are not current release instructions.

Local structural checks: the updated patch applies cleanly to a fresh archive
of upstream `e47adbf` (`git apply --check` and `git apply`), and the modified
attention, TriMul, and experiment-runner modules parse successfully. These are
patch/API structure checks, not an end-to-end GPU result.

## Historical validation record

Host: Lambda 1x H100 80 GB (driver 570.195.03), Python 3.14.7, PyTorch 2.10.0+cu12.9,
pytorch-lightning 2.6.1. Date: 2026-09-04.

The package build and clean-install result below was obtained only after
`4098f42b` fixed source-tree shadowing in the smoke test. It must not be
attributed to the earlier `fe78ee99` documentation commit.

## Audited 0.1.0 artifact refresh

The checked-in wheel was rebuilt from clean source commit
`0b85b535f7ab81ee81c66848ca7bd5c9133879d2` after correcting packaged
README and NOTICE claims. The build directory-to-wheel interval was 8 min
15 s. The clean-install smoke passed with no compiler on `PATH` and JIT
forbidden; all 15 package-contract tests passed. The resulting wheel SHA256
is `1c7d63bf5ea11276f23bed42d2d12705f1e140bcf87c33dfdf3ca62bd2ea3ecf`.

No CUDA source or build flag changed in this refresh. Compared with the
previous checked-in wheel, `cuobjdump --dump-sass` output for all five
extensions was identical after normalizing only NVCC's generated anonymous-
namespace hash in function symbols. TriMul SASS was identical without that
normalization. This is an artifact-metadata refresh, not a kernel or numerical
change.

## Package

| Check | Result |
| --- | --- |
| `tools/build_sm90_wheel.sh` | wheel built in 9.5 min; clean-install smoke (no compiler on PATH, JIT forbidden) passed |
| CPU tests: packaging, distributed contract, gloo preflight | 36 passed |
| GPU tests: attention facade, workspace planner, TriMul C++ contract | 57 passed |
| `tests/test_numerics_vs_fp64.py` | 10 passed for Triangle Attention: FP64-oracle relative error within the test's 1.1x stock-BF16 envelope plus 1e-6, bitwise replay, exact zeros on fully masked planes; this test does not cover TriMul |

## Host patch on upstream `aqlaboratory/openfold-3` e47adbf

| Check | Result |
| --- | --- |
| `git apply` on a fresh checkout | clean, 16 files |
| Import without the package installed | OpenFold3 imports; `stablefold_aot_modules_from_config` resolves to `None` |
| `TriangleAttention` layer with `use_stable_triangle_attention=True` | routed to the package (invocation count +1) |
| `TriangleMultiplicationOutgoing` with `use_stable_triangle_multiplicative=True` | N=128 routed to the package; N=64 fell back to stock with one warning |
| Model config with both flags on | accepted by the upstream config validator |
| `run_openfold train`, one BF16 step, PDB subset, crop 384 | `max_steps=1` reached; 360 StableFold attention invocations in the step; zero fallback messages |
| `tests/host_openfold3/` (routing, TriMul custom op, support, offload, diffusion scope) against the patched upstream | 218 passed |

## Not yet validated

- Multi-GPU DDP through the patched upstream (the preflight callback is the
  same code that passed 2x H100 in the fork).
- AQLab's `openfold-3-freeze-confidence` branch: apply the patch with
  `--3way`; the hunks sit next to the `use_cueq_triangle_kernels` lines.
