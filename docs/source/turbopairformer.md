# TurboPairFormer Training

Welcome to the training documentation for OpenFold3 with the [TurboPairFormer](https://pypi.org/project/turbopairformer/) triangle attention and triangle multiplicative update kernels, which speed up training on NVIDIA Hopper GPUs (H100 and H200).
This guide follows {doc}`OpenFold3 Training <training>` step by step: every command is the same, with the `openfold3` pixi environment in place of `openfold3-cuda12`, and one setting in the training YAML turns the kernels on.

Note: one flag enables both kernels, and it takes precedence over the other kernels at the triangle sites.
      Inputs a kernel does not support (non-BF16 tensors, chunked attention, fused TriMul projection weights, `c_z != c_hidden`, non-Hopper GPUs) fall back silently, as with cuEquivariance: attention to PyTorch, TriMul to the next enabled kernel or PyTorch.

## 1. Prerequisites

- OpenFold3 environment. See {doc}`OpenFold3 Installation <Installation>` for instructions on how to build the environment. For TurboPairFormer, use the `openfold3` environment, defined on our OpenFold3 integration branch. **This repository is private; your GitHub account needs access.**
- An NVIDIA H100 or H200 GPU on Linux x86-64 (glibc ≥ 2.28), with a driver that supports CUDA 12.8.

```bash
git clone --branch feat/turbopairformer-aqlab-integration --single-branch \
  https://github.com/RyanAIResearch/openfold3_triangle_kernel.git openfold3-turbopairformer
cd openfold3-turbopairformer

pixi install --locked -e openfold3
```

`openfold3` is the `openfold3-cuda12` environment with TurboPairFormer added: the same OpenFold3 dependencies, DeepSpeed, cuEquivariance, CUDA toolkit and test tools.
Only PyTorch (`torch==2.10.0`, CUDA 12.8) and DeepSpeed come from PyPI instead of conda-forge, because the TurboPairFormer wheel only loads with the PyPI PyTorch build; the CUDA toolkit is pinned to 12.8 to match.
Pixi checks for an NVIDIA driver before installing; on a node without a GPU (for example a login node), prefix the install with `CONDA_OVERRIDE_CUDA=12.0`.

Optionally test your installation for training OpenFold3 using a small subset of the PDB dataset. The command `setup-pdb-subset`, comprised of `generate-pdb-subset` + `download-pdb-subset` will create a mini PDB training set complete with the MSAs, template files, and OpenFold cache files.

```
# Run once to generate a mini training set
pixi run -e openfold3 setup-pdb-subset

# Invoke the training test
pixi run -e openfold3 pytest openfold3/tests/test_training_full.py
```

On an H100 or H200, the training test includes a `turbopairformer` case that trains the mini set with the TurboPairFormer kernels enabled (add `-k turbopairformer` to run only that case).
`pixi run -e openfold3 pytest openfold3/tests/test_kernels.py -k turbopairformer` checks the kernels against the PyTorch reference.

The commands in the following sections run inside the environment: open it with `pixi shell -e openfold3`, or prefix each command with `pixi run -e openfold3`.

## 2. Download the Dataset

The pre-processed PDB training dataset is hosted on AWS S3. Download it using the AWS CLI:

```bash
aws s3 sync s3://openfold3-data/pdb_training_set/ /shared/openfold3/pdb_training_set/ --no-sign-request

```

## 3. Prepare the Training Config

The training configuration is stored in a YAML file that controls all aspects of training: model settings, dataset configuration, distributed training parameters, logging, and checkpointing.

**Note:** Make sure you update the paths to match your file locations. The examples below assume a `/shared/openfold3` directory that's accessible from all your training nodes.

Complete example YAML configurations for all stages of training are available in [examples/training_yamls/](https://github.com/aqlaboratory/openfold-3/tree/main/examples/training_yamls).

### 3.1 Basic Training Config

Here's a minimal configuration for single-GPU training, with the TurboPairFormer kernels enabled under `model_update.custom.settings`:

```yaml
experiment_settings:
  mode: train
  output_dir: ./test_train_output 
  restart_checkpoint_path: last

data_module_args:
  batch_size: 1
  num_workers: 1
  epoch_len: 500  # Ckpt every 500 steps (effective batch_size * # of steps)

logging_config:
  log_lr: false
  wandb_config: null

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: -1
  log_every_n_steps: 50

checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: -1

model_update:
  presets: 
    - train
  custom:
    settings:
      memory:
        train:
          use_turbopairformer_triangle_kernels: true
    architecture:
      shared:
        use_confidence_emb_prob: 0.8
        diffusion:
          use_conditioning_prob: 0.8

dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 1.0
      config:
        debug_mode: true
        template:
          n_templates: 4
          take_top_k: false
        crop:
          token_crop:
            enabled: true
            token_budget: 384
            crop_weights:
              contiguous: 0.2
              spatial: 0.4
              spatial_interface: 0.4
          chain_crop:
            enabled: true

  validation:
    val-weighted-pdb:
      dataset_class: ValidationPDBDataset
      config:
        debug_mode: true
        msa:
          subsample_main: false
        template:
          n_templates: 4
          take_top_k: true
        crop:
          token_crop:
            enabled: false

dataset_paths:
  weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_training_set/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_training_set/dataset_caches/training_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/reference_mols
    template_cache_directory: /shared/openfold3/pdb_training_set/templates/train_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_training_set/templates/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null

  val-weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_training_set/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_training_set/dataset_caches/validation_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/reference_mols
    template_cache_directory: /shared/openfold3/pdb_training_set/templates/val_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_training_set/templates/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null
```

`use_turbopairformer_triangle_kernels` defaults to `false`.
It reaches the triangle attention and triangle multiplicative update layers of the template, MSA, Pairformer and confidence-head stacks; AttentionPairBias and the diffusion transformer keep their configured backends, as the kernels are specific to triangle shapes.
The TriMul kernel requires `fuse_projection_weights: false` (the default). The kernels have been validated for training; inference paths (`eval` settings, chunking, offloading) fall back to the existing backends.

For example configurations for all stages of training, please see [examples/training_yamls/](https://github.com/aqlaboratory/openfold-3/tree/main/examples/training_yamls); add the same `settings` block to enable TurboPairFormer:
- `initial_training.yml`: Standard initial training configuration
- `finetune_1.yml`: Fine-tuning stage 1 configuration
- `finetune_2.yml`: Fine-tuning stage 2 configuration
- `finetune_3.yml`: Fine-tuning stage 3 configuration (used in OF3p)

## 4. Launch Training

### 4.1 Single-GPU Training

For testing or debugging:

```bash
run_openfold train --runner-yaml training.yaml --seed 42
```

### 4.2 Multi-GPU Training

To train on multiple GPUs within a single node, configure your YAML:

```yaml
pl_trainer_args:
  devices: 8      # GPUs per node
  num_nodes: 1
```

For multi-node distributed training, update your config as follows:

```yaml
pl_trainer_args:
  devices: 8       # GPUs per node
  num_nodes: 32    # Total number of nodes
```

Then launch training with:

```bash
run_openfold train --runner_yaml training.yaml --seed 42
```

## 5. Monitoring Training

Enable Weights & Biases logging by configuring `wandb_config` in your YAML:

```yaml
logging_config:
  log_lr: true
  wandb_config:
    project: openfold3-training
    entity: your-wandb-entity
    group: null
    id: null
    experiment_name: my-training-run
```

To use W&B, ensure you're logged in:

```bash
wandb login
```

## 6. Checkpointing and Resuming

### 6.1 Checkpoint Configuration

```yaml
checkpoint_config:
  every_n_epochs: 1              # Save checkpoint every N epochs
  auto_insert_metric_name: false # Don't add metric to filename
  save_last: true                # Always save 'last.ckpt'
  save_top_k: -1                 # Keep all checkpoints (-1) or top K
```

Checkpoints are saved to `{output_dir}/checkpoints/`.

### 6.2 Resuming Training

To resume from the last checkpoint:

```yaml
experiment_settings:
  restart_checkpoint_path: last
```

To resume from a specific checkpoint:

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/checkpoint.ckpt
```

## 7. Fine-tuning

### 7.1 Starting from a pre-trained checkpoint

To fine-tune from a pre-trained checkpoint, specify the checkpoint path and adjust training parameters as needed:

```yaml
experiment_settings:
  mode: train
  output_dir: ./finetune_output
  seed: 42
  restart_checkpoint_path: /path/to/pretrained.ckpt
```

### 7.2 Inferencing on fine-tuning checkpoints 

The checkpoints are generated in a format that's not compatible with torch.load. You'll see this error

> _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 

You can run a small utility script to get a functional checkpoint 

```bash
python scripts/dev/convert_ckpt_to_ema_only.py <wandb_id>/checkpoints/<epoch>-<step_num>.ckpt inference.ckpt
ls -l
-rw-rw-r-- 1 jandom jandom 2287987691 Jul  2 06:48 inference.ckpt.tmp
-rw-rw-r-- 1 jandom jandom 2287578277 Jul  2 06:48 inference.ckpt
```

This format of the checkpoint can be directly used with `run_openfold predict`

```bash
run_openfold predict \
    --inference-ckpt-path inference.ckpt \
    ... # other options
```

The TurboPairFormer setting only affects training; `run_openfold predict` uses the existing backends.

## 8. Compare the Triangle-Kernel Backends

The `compare-triangle-kernels` task prepares the same small PDB training subset (8 structures, 384-token crops, public S3 bucket, no AWS account), trains 20 steps with PyTorch, Triton, cuEquivariance and TurboPairFormer from the same runner YAML and seed, and writes the wall times to `runs/backend_comparison.png` as a bar chart:

```bash
pixi run -e openfold3 compare-triangle-kernels
```

Every run builds the same model from the same seed on the same data with TF32 enabled for the remaining FP32 matmuls (cuEquivariance's attention backward requires it), so the wall-time differences come from the triangle kernels.
Options are passed through to `scripts/compare_triangle_kernels.py`, for example:

```bash
pixi run -e openfold3 compare-triangle-kernels --steps 50 --backends pytorch turbopairformer
```

`--steps`, `--backends`, `--seed` and `--no-tf32` adjust the comparison.
The environment already includes `matplotlib` for the chart and `cuequivariance==0.10.0` for the `cueq` run (0.11 requires PyTorch 2.11); to record the loss, add a `logging_config.wandb_config` as described in section 5.

## 9. Without Pixi

From the same repository checkout, a Conda environment with the same PyTorch and TurboPairFormer versions can be built with `pip`:

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

python -m pip install turbopairformer==0.1.2
python -m pip install .
```

In the activated environment, sections 2–7 work as written.
The pixi tasks map to plain commands: `setup-pdb-subset` is `python scripts/datasets/generate_subset_cache.py` followed by `python scripts/datasets/download_subset.py`, and `compare-triangle-kernels` is `python scripts/compare_triangle_kernels.py`.
The training test additionally needs `pip install pytest`, and the comparison's `cueq` run and chart need `pip install cuequivariance==0.10.0 cuequivariance-torch==0.10.0 cuequivariance-ops-torch-cu12==0.10.0 matplotlib`.
