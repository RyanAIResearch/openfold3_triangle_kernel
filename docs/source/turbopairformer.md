# TurboPairFormer Kernels

OF3 supports the [TurboPairFormer](https://pypi.org/project/turbopairformer/) triangle attention and triangle multiplicative update kernels, which speed up training on NVIDIA Hopper GPUs (H100 and H200).
Note: one flag enables both kernels, and it takes precedence over the other kernels at the triangle sites.
      Inputs a kernel does not support (non-BF16 tensors, chunked attention, fused TriMul projection weights, `c_z != c_hidden`, non-Hopper GPUs) fall back silently, as with cuEquivariance: attention to PyTorch, TriMul to the next enabled kernel or PyTorch.

## Installation

TurboPairFormer 0.1.2 supports NVIDIA H100 and H200.

Requirements: Linux x86-64 (glibc ≥ 2.28), Python 3.14, and PyTorch 2.10.0 with CUDA Runtime 12.8.

Clone our OpenFold3 integration branch, which includes the `openfold3-turbopairformer` environment and training integration. **This repository is private; your GitHub account needs access.**

```bash
git clone --branch feat/turbopairformer-aqlab-integration --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git openfold3-turbopairformer

cd openfold3-turbopairformer
```

Choose either Pixi or Conda below.

### Pixi

Install the environment from the repository's lock file:

```bash
pixi install --locked -e openfold3-turbopairformer
```

Check the package and OpenFold3 command-line interfaces:

```bash
pixi run --locked -e openfold3-turbopairformer python -m turbopairformer --help
pixi run --locked -e openfold3-turbopairformer run_openfold --help
```

If Pixi reports that the environment does not exist, check your branch:

```bash
git branch --show-current
```

It should print `feat/turbopairformer-aqlab-integration`.

### Conda

From the same repository checkout:

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

python -m pip install turbopairformer==0.1.2
python -m pip install .
```

## Usage

To enable TurboPairFormer for training, add this setting to your training runner YAML:

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_turbopairformer_triangle_kernels: true
```

The flag defaults to `false`.
It reaches the triangle attention and triangle multiplicative update layers of the template, MSA, Pairformer and confidence-head stacks; AttentionPairBias and the diffusion transformer keep their configured backends, as the kernels are specific to triangle shapes.
The TriMul kernel requires `fuse_projection_weights: false` (the default). The kernels have been validated for training; inference paths (`eval` settings, chunking, offloading) fall back to the existing backends.

After preparing your training data and `training.yaml` as described in {doc}`Training <training>`, launch with Pixi:

```bash
pixi run --locked -e openfold3-turbopairformer \
  run_openfold train --runner-yaml training.yaml --seed 42
```

Or, from the activated Conda environment:

```bash
run_openfold train --runner-yaml training.yaml --seed 42
```

## Demo: train on a PDB subset with each backend

One script prepares a small PDB training subset with OpenFold3's own dataset scripts (8 structures, 384-token crops, public S3 bucket, no AWS account), trains 20 steps with PyTorch, Triton, cuEquivariance and TurboPairFormer from the same runner YAML and seed, and writes the wall times to `runs/backend_comparison.png` as a bar chart.
Run it from the OpenFold3 checkout:

```bash
python scripts/compare_triangle_kernels.py                        # conda
pixi run -e openfold3-turbopairformer compare-triangle-kernels    # pixi
```

Every run builds the same model from the same seed on the same data with TF32 enabled for the remaining FP32 matmuls (cuEquivariance's attention backward requires it), so the wall-time differences come from the triangle kernels; `--steps`, `--backends`, `--seed` and `--no-tf32` adjust the comparison, and the chart needs `matplotlib` (included in the pixi environment; `pip install matplotlib` in the conda one).
The `cueq` run needs `cuequivariance==0.10.0`, `cuequivariance-torch==0.10.0` and `cuequivariance-ops-torch-cu12==0.10.0` installed next to `turbopairformer` (0.11 requires PyTorch 2.11); to record the loss, add a `logging_config.wandb_config` as described in the training documentation.
