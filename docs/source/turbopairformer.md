# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides optional
triangle attention and triangle multiplicative update (TriMul) kernels for NVIDIA
Hopper GPUs. Follow {doc}`OpenFold3 Training <training>` for data preparation,
runner YAMLs, distributed training and checkpointing; the additions below cover
installation and kernel selection.

## Installation

Version 0.1.2 requires Linux x86-64 (glibc ≥ 2.28), Python 3.14, an H100 or H200,
and a driver that supports CUDA 12.8. Its prebuilt kernels require the PyPI
PyTorch 2.10.0 CUDA 12.8 build. Use a fresh environment so this requirement does
not conflict with an existing Conda or Pixi PyTorch installation. A host C
compiler (such as `gcc`) is needed by the default Triton validation kernels.

From an OpenFold3 checkout containing this integration, run:

```bash
python3.14 -m venv .venv-tpf
source .venv-tpf/bin/activate
python -m pip install -e . 'turbopairformer[lightning]==0.1.2' pytest
python -m pip check
```

The package pins PyTorch 2.10.0; its `lightning` extra pins PyTorch Lightning
2.6.1. Run the commands below in this activated environment.

## Enable TurboPairFormer

Merge these settings into the runner YAML from the training guide, preserving
its other settings, including `model_update.presets: [train]`:

```yaml
pl_trainer_args:
  precision: bf16-mixed

model_update:
  custom:
    settings:
      memory:
        train:
          use_turbopairformer_triangle_kernels: true
```

Launch training with the usual command:

```bash
run_openfold train --runner-yaml training.yaml --seed 42
```

The flag is off by default. Set it to `false` or remove it to disable the kernels.
Requesting TurboPairFormer without installing the package raises an error.
Each operator checks its inputs and falls back when they are unsupported:

- Eligible inputs are BF16; TriMul also requires FP32 parameters, as provided by
  `bf16-mixed`. FP32 inputs fall back.
- Leave triangle attention unchunked (the training default). Chunked attention
  falls back; this does not by itself disable TriMul.
- TriMul requires more than 100 tokens, equal input and hidden channel counts
  in {32, 64, 96, 128}, and `fuse_projection_weights: false` (the default).
  The training guide's 384-token crop budget allows this, but individual samples
  with 100 tokens or fewer still fall back.

The flag reaches triangle operations in the template, MSA, Pairformer and
confidence-head stacks. OpenFold3 keeps the confidence-head Pairformer in FP32,
so its operations fall back even with `bf16-mixed`. AttentionPairBias and the
diffusion transformer retain their configured backends.

At triangle sites TurboPairFormer takes precedence over cuEquivariance,
DeepSpeed and Triton. Unsupported attention inputs fall back to PyTorch;
unsupported TriMul inputs use the next enabled backend or PyTorch.
The `memory.train` setting applies to training. Validation and prediction use
`memory.eval`, where TurboPairFormer is also off by default.

## cuEquivariance comparison

To compare against {doc}`cuEquivariance <kernels>` in the same environment,
install the version compatible with PyTorch 2.10.0:

```bash
python -m pip install cuequivariance==0.10.0 cuequivariance-torch==0.10.0 cuequivariance-ops-torch-cu12==0.10.0
python -m pip check
```

Replace the training kernel settings with:

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_cueq_triangle_kernels: true
          use_turbopairformer_triangle_kernels: false
```

cuEquivariance 0.10 requires TF32 for backward through the confidence head's FP32
triangle attention, so launch training with:

```bash
run_openfold train --runner-yaml training.yaml --seed 42 --use_tf32 true
```

Without TF32, those FP32 backward calls fail with
`Full precision FP32 backward pass for triangle attention is not implemented yet`.
BF16 triangle attention itself does not require TF32.

| | TurboPairFormer | cuEquivariance 0.10 |
|---|---|---|
| Training flag | `use_turbopairformer_triangle_kernels` | `use_cueq_triangle_kernels` |
| Triangle kernels | Attention and TriMul | Attention and TriMul |
| Supported precision | BF16 inputs, FP32 TriMul parameters | BF16 and FP32; TF32 required for FP32 attention backward |
| Validation/prediction | Separate `memory.eval` flag, off by default | Separate `memory.eval` flag; see {doc}`kernels` |
| Targeted test selection | `-k turbopairformer` | `-k cueq` |

## Verify the installation

On a supported GPU, these comparisons check forward values and gradients and
fail if a tested TurboPairFormer operation silently falls back:

```bash
python -m pytest openfold3/tests/test_kernels.py -k turbopairformer
```

To exercise the training entry point, prepare the small PDB subset once and run
the training smoke test:

```bash
python scripts/datasets/generate_subset_cache.py
python scripts/datasets/download_subset.py
python -m pytest openfold3/tests/test_training_full.py -k turbopairformer
```

The kernel tests skip when the package or a Hopper GPU is unavailable. The
training test additionally skips when the local PDB subset is missing.
