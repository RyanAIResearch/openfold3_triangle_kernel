# TurboPairFormer Kernels

OF3 supports the [TurboPairFormer](https://pypi.org/project/turbopairformer/) triangle attention and triangle multiplicative update kernels, which speed up training on NVIDIA Hopper GPUs (H100).
Note: one flag enables both kernels, and it takes precedence over the other kernels at the triangle sites.
      Inputs a kernel does not support (non-BF16 tensors, chunked attention, fused TriMul projection weights, `c_z != c_hidden`, non-Hopper GPUs) fall back silently, as with cuEquivariance: attention to PyTorch, TriMul to the next enabled kernel or PyTorch.

## Installation

`turbopairformer==0.1.1.post1` requires Python 3.14, Linux x86-64 (glibc >= 2.28), PyTorch 2.10.0 from PyPI (CUDA 12.8) and a Hopper GPU. H100 is tested; H200 is not yet validated.
The wheel needs PyPI PyTorch 2.10.0 rather than the CUDA 12.9 builds of the `openfold3-cuda12*` Pixi environments, so it has its own environment, with conda or pixi.
Both start from a clone of OpenFold3:

```bash
git clone https://github.com/aqlaboratory/openfold-3.git
cd openfold-3
```

**conda**:

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
pip install turbopairformer
pip install .
```

**pixi**, with the `openfold3-turbopairformer` environment listed in {doc}`Installation <Installation>` (Linux x86-64), which is set up on first use:

```shell
pixi run -e openfold3-turbopairformer run_openfold train --runner-yaml training.yaml --seed 42
```

## Usage

To enable the kernels, add the following to the training runner.yaml:

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

## Demo: train on a PDB subset with each backend

One script prepares a small PDB training subset with OpenFold3's own dataset scripts (8 structures, 384-token crops, public S3 bucket, no AWS account), trains 20 steps with PyTorch, Triton, cuEquivariance and TurboPairFormer from the same runner YAML and seed, and writes the wall times to `runs/backend_comparison.png` as a bar chart.
Run it from the OpenFold3 checkout:

```bash
python scripts/compare_triangle_kernels.py                        # conda
pixi run -e openfold3-turbopairformer compare-triangle-kernels    # pixi
```

Every run builds the same model from the same seed on the same data with TF32 enabled for the remaining FP32 matmuls (cuEquivariance's attention backward requires it), so the wall-time differences come from the triangle kernels; `--steps`, `--backends`, `--seed` and `--no-tf32` adjust the comparison, and the chart needs `matplotlib` (included in the pixi environment; `pip install matplotlib` in the conda one).
The `cueq` run needs `cuequivariance==0.10.0`, `cuequivariance-torch==0.10.0` and `cuequivariance-ops-torch-cu12==0.10.0` installed next to `turbopairformer` (0.11 requires PyTorch 2.11); to record the loss, add a `logging_config.wandb_config` as described in the training documentation.
