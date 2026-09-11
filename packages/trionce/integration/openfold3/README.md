# Integrating StableFold into an OpenFold3 checkout

StableFold is adopted the same way OpenFold3 adopts cuEquivariance: one pip
dependency, one configuration flag per operator, and a stock fallback for
unsupported shapes. The host-side change is one patch of 16 files and about
420 added lines, most of which pass two boolean flags down the module tree
exactly where `use_cueq_triangle_kernels` already travels.

## 1. Install the package

```bash
uv pip install stablefold-0.1.0-cp314-cp314-linux_x86_64.whl
```

The wheel is ABI-locked to CPython 3.14, PyTorch 2.10.0, and CUDA 12.9,
the versions pinned by OpenFold3's `openfold3-cuda12` Pixi environment. In a
Pixi checkout add it as a PyPI dependency:

```toml
[feature.stablefold.pypi-dependencies]
stablefold = { path = "release_artifacts/stablefold-0.1.0-cp314-cp314-linux_x86_64.whl" }
```

## 2. Apply the host patch

`openfold3-main-e47adbf-stablefold.patch` is a `git diff` against
`aqlaboratory/openfold-3` commit `e47adbf`:

```bash
git apply --3way openfold3-main-e47adbf-stablefold.patch
```

What it changes:

| Area | Files | Change |
| --- | --- | --- |
| Config | `projects/of3_all_atom/config/model_config.py` | `use_stable_triangle_attention` and `use_stable_triangle_multiplicative`, default off, in `memory.train` and `memory.eval` |
| Attention primitive | `core/model/primitives/attention.py` | one branch calling `stablefold.triangle_attention`; shapes the facade rejects fall back to stock math and are logged once |
| TriMul layer | `core/model/layers/triangular_multiplicative_update.py` | one branch in both TriMul classes; `N <= 100` falls back to stock, matching the cuEq threshold |
| Flag plumbing | `model.py`, `base_blocks.py`, `base_stacks.py`, `pairformer.py`, `msa_module.py`, `template_module.py`, `evoformer.py`, `triangular_attention.py`, `attention_pair_bias.py`, `msa.py`, `head_modules.py`, `prediction_heads.py` | pass the two flags alongside `use_cueq_triangle_kernels`; validate mutual exclusivity with DeepSpeed, LMA, and chunked stacks |
| Runner | `entry_points/experiment_runner.py` | optional `stablefold.distributed` import; with the flags on and a distributed run, the DDP kernel preflight callback is added and DeepSpeed/ZeRO is rejected |

`stablefold` is an optional import: without the package installed, OpenFold3
behaves exactly as before unless a StableFold flag is set, in which case the
runner raises a clear error at startup. With the flags off, no code path
changes.

For branches that have drifted from `e47adbf`, apply the patch with
`--3way` and resolve conflicts hunk by hunk; every hunk is a self-contained
addition next to the corresponding `use_cueq_triangle_kernels` line.

## 3. Select the backend

In the training YAML:

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_stable_triangle_attention: true
          use_stable_triangle_multiplicative: true
          use_triton_triangle_kernels: false
          use_cueq_triangle_kernels: false
          use_deepspeed_evo_attention: false
```

Leave `memory.eval` on its existing route (Triton by default) unless the
validation path should also use StableFold. Keep `fuse_projection_weights:
false`, the OpenFold3 default, so the TriMul fast path applies. Export the
execution flags before launching:

```bash
export STABLEFOLD_SR_BACKWARD=1
export STABLEFOLD_SR_SHADOW=1
export STABLEFOLD_ALLOW_JIT=0
```

Nothing else in the training loop changes: DDP strategy, per-sample gradient
clipping, EMA, checkpoints, and logging stay as they are.

## 4. Verify

- Single GPU: run one training step; the log must not contain
  `stablefold fallback to stock` for the pair-stack shapes.
- Multi GPU: rank 0 writes `<log_dir>/stablefold_kernel_ddp.json` with
  `"status": "PASS"` and one entry per rank.
- Numerics: `tests/test_numerics_vs_fp64.py` in the package compares Triangle
  Attention against an FP64 oracle at training shapes. That test does not cover
  TriMul or end-to-end model accuracy.

Validation performed so far is recorded in `VALIDATION.md`.
