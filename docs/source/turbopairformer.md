# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides optional
triangle attention and triangle multiplicative update (TriMul) kernels for training on
NVIDIA H100 and H200 GPUs. Everything else follows {doc}`OpenFold3 Training <training>`.

## Installation

TurboPairFormer 0.1.2 needs Linux x86-64, Python 3.14, a driver for CUDA 12.8, the
PyPI build of PyTorch 2.10.0 and a C compiler such as `gcc` (validation uses Triton
kernels). Install it in a fresh environment from this checkout:

```bash
python3.14 -m venv .venv-tpf
source .venv-tpf/bin/activate
python -m pip install -e . 'turbopairformer[lightning]==0.1.2' pytest
```

## Enable TurboPairFormer

Add this to the training runner YAML and launch training as usual:

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

The kernels run in the triangle operations of the template, MSA and Pairformer stacks
during training; validation and prediction keep their configured backends. Inputs
they do not support fall back to PyTorch (TriMul to the next enabled backend): FP32
inputs such as the confidence head, which OpenFold3 runs in FP32, chunked attention,
TriMul with 100 tokens or fewer or `c_z != c_hidden`, and `fuse_projection_weights: true`.

To compare with {doc}`cuEquivariance <kernels>`, install `cuequivariance==0.10.0`,
`cuequivariance-torch==0.10.0` and `cuequivariance-ops-torch-cu12==0.10.0`, set
`use_cueq_triangle_kernels: true` instead, and launch with `--use_tf32 true`, which
its FP32 triangle attention backward requires.

## Verify

On a Hopper GPU, these tests fail if a kernel falls back or differs from PyTorch:

```bash
python -m pytest openfold3/tests/test_kernels.py -k turbopairformer
```

`openfold3/tests/test_training_full.py -k turbopairformer` also trains the small PDB
subset from the training guide with the kernels enabled.
