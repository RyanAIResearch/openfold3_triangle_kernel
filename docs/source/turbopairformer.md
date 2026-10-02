# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides triangle
attention and triangle multiplicative update kernels that speed up OpenFold3 training on
NVIDIA H100 and H200 GPUs. This guide covers how to train OpenFold3 with them on the PDB
dataset from scratch or fine-tune from an existing checkpoint.

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

For full training, download the pre-processed PDB training set from AWS S3:

```bash
pixi run -e openfold3 aws s3 sync s3://openfold3-data/pdb_training_set/ \
  /shared/openfold3/pdb_training_set/ --no-sign-request
```

## 3. Prepare the training config

The training configuration is stored in a YAML file that controls all aspects of
training: model settings, dataset configuration, distributed training parameters,
logging, and checkpointing. Update the paths to match your file locations; the example
below assumes a `/shared/openfold3` directory that's accessible from all your training
nodes.

Here's a minimal configuration for single-GPU training. `precision: bf16-mixed` and
`use_turbopairformer_triangle_kernels: true` turn the kernels on:

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

For the mini set, `setup-pdb-subset` already wrote a complete config,
`datasets/train_pdb_subset.yaml`, with `bf16-mixed`: add
`use_turbopairformer_triangle_kernels: true` to its
`model_update.custom.settings.memory.train`.

For example configurations for all stages of training, see
[examples/training_yamls/](https://github.com/aqlaboratory/openfold-3/tree/main/examples/training_yamls)
and add the same `settings` block to them:
- `initial_training.yml`: Standard initial training configuration
- `finetune_1.yml`: Fine-tuning stage 1 configuration
- `finetune_2.yml`: Fine-tuning stage 2 configuration
- `finetune_3.yml`: Fine-tuning stage 3 configuration (used in OF3p)

## 4. Launch training

### 4.1 Single-GPU training

```bash
# Mini set
pixi run -e openfold3 run_openfold train --runner-yaml datasets/train_pdb_subset.yaml --seed 42
# Full PDB training set
pixi run -e openfold3 run_openfold train --runner-yaml training.yaml --seed 42
```

### 4.2 Multi-GPU training

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
pixi run -e openfold3 run_openfold train --runner-yaml training.yaml --seed 42
```

## 5. Monitoring training

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
pixi run -e openfold3 wandb login
```

## 6. Checkpointing and resuming

### 6.1 Checkpoint configuration

```yaml
checkpoint_config:
  every_n_epochs: 1              # Save checkpoint every N epochs
  auto_insert_metric_name: false # Don't add metric to filename
  save_last: true                # Always save 'last.ckpt'
  save_top_k: -1                 # Keep all checkpoints (-1) or top K
```

Checkpoints are saved to `{output_dir}/checkpoints/`.

### 6.2 Resuming training

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

To fine-tune from a pre-trained checkpoint, specify the checkpoint path and adjust
training parameters as needed:

```yaml
experiment_settings:
  mode: train
  output_dir: ./finetune_output
  seed: 42
  restart_checkpoint_path: /path/to/pretrained.ckpt
```

### 7.2 Inferencing on fine-tuning checkpoints

The checkpoints are generated in a format that's not compatible with `torch.load`.
You'll see this error

> _pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.

You can run a small utility script to get a functional checkpoint

```bash
pixi run -e openfold3 python scripts/dev/convert_ckpt_to_ema_only.py <wandb_id>/checkpoints/<epoch>-<step_num>.ckpt inference.ckpt
```

This format of the checkpoint can be directly used with `run_openfold predict`

```bash
pixi run -e openfold3 run_openfold predict \
    --inference-ckpt-path inference.ckpt \
    ... # other options
```

## 8. Check the kernels

Inputs the kernels do not support, such as the FP32 confidence head, fall back to
PyTorch. On a Hopper GPU, this test fails if the kernels fall back or differ from PyTorch:

```bash
pixi run -e openfold3 pytest openfold3/tests/test_kernels.py -k turbopairformer
```

To compare with cuEquivariance, which the environment also includes, set
`use_cueq_triangle_kernels: true` instead and add `--use_tf32 true`.
