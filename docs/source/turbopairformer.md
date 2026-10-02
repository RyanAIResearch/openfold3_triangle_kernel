# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides triangle
attention and triangle multiplicative update kernels that speed up OpenFold3 training on
NVIDIA H100 and H200 GPUs. This guide follows [OpenFold3 Training](training.md); only the
pixi environment and one training setting differ.

## 1. Install the environment

You need Linux x86-64, an H100 or H200 GPU and a driver for CUDA 12.8. Install
[pixi](https://pixi.prefix.dev/latest/installation/), then clone this branch and install
the `openfold3` environment:

```bash
curl -fsSL https://pixi.sh/install.sh | sh  # once, then restart your shell
git clone -b feat/turbopairformer-aqlab-integration \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git
cd openfold3_triangle_kernel
pixi install -e openfold3
```

`openfold3` is the `openfold3-cuda12-pypi` environment with TurboPairFormer 0.1.2 and the
PyTorch build it needs (Python 3.14, `torch==2.10.0`, CUDA 12.8).

## 2. Download the dataset

To try training first, generate the mini PDB training set, with its MSAs, templates and
cache files, and run the training test (it includes a TurboPairFormer case):

```bash
pixi run -e openfold3 setup-pdb-subset
pixi run -e openfold3 pytest openfold3/tests/test_training_full.py
```

For full training, download the pre-processed PDB training set:

```bash
pixi run -e openfold3 aws s3 sync s3://openfold3-data/pdb_training_set/ \
  /shared/openfold3/pdb_training_set/ --no-sign-request
```

## 3. Train

Turn the kernels on in the training YAML:

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

For the mini set, the training YAML is `datasets/train_pdb_subset.yaml`. It already uses
`bf16-mixed`; add `use_turbopairformer_triangle_kernels: true` to its
`model_update.custom.settings.memory.train`. For the full set, write `training.yaml` as in
step 3 of [OpenFold3 Training](training.md) and add the lines above. Then launch training:

```bash
# Mini set
pixi run -e openfold3 run_openfold train --runner-yaml datasets/train_pdb_subset.yaml --seed 42
# Full PDB training set
pixi run -e openfold3 run_openfold train --runner-yaml training.yaml --seed 42
```

Multi-GPU training, monitoring, checkpointing and fine-tuning work as in the training
guide. Inputs the kernels do not support, such as the FP32 confidence head, fall back to
PyTorch. On a Hopper GPU, this test fails if the kernels fall back or differ from PyTorch:

```bash
pixi run -e openfold3 pytest openfold3/tests/test_kernels.py -k turbopairformer
```

To compare with cuEquivariance, which the environment also includes, set
`use_cueq_triangle_kernels: true` instead and add `--use_tf32 true`.
