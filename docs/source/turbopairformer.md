# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides optional
triangle attention and triangle multiplicative update (TriMul) kernels for training on
NVIDIA H100 and H200 GPUs. Training follows {doc}`OpenFold3 Training <training>`, with
the `openfold3` pixi environment in place of `openfold3-cuda12`.

## Installation

`openfold3` is `openfold3-cuda12-pypi` with TurboPairFormer 0.1.2 and the PyTorch build
it needs (Python 3.14, `torch==2.10.0`, CUDA 12.8). It requires Linux x86-64 and a
driver for CUDA 12.8. As in the training guide, generate the mini training set and run
the training test:

```bash
pixi run -e openfold3 setup-pdb-subset
pixi run -e openfold3 pytest openfold3/tests/test_training_full.py
```

The test includes a `turbopairformer` case that trains the mini set with the kernels
enabled.

## Enable TurboPairFormer

Add this to the training runner YAML and launch training as usual, for example with
`pixi run -e openfold3 run_openfold train --runner-yaml training.yaml --seed 42`:

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

The environment also includes {doc}`cuEquivariance <kernels>`. To compare, set
`use_cueq_triangle_kernels: true` instead and launch with `--use_tf32 true`, which its
FP32 triangle attention backward requires.

## Verify

On a Hopper GPU, these tests fail if a kernel falls back or differs from PyTorch:

```bash
pixi run -e openfold3 pytest openfold3/tests/test_kernels.py -k turbopairformer
```
