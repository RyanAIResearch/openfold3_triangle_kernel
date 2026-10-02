# TurboPairFormer training

[TurboPairFormer](https://pypi.org/project/turbopairformer/0.1.2/) provides triangle
attention and triangle multiplicative update kernels for training on H100 and H200
GPUs. Follow [OpenFold3 Training](training.md) with the `openfold3` pixi environment
(Linux x86-64, CUDA 12.8 driver) in place of `openfold3-cuda12`, and enable the kernels
in the training YAML:

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

```bash
pixi run -e openfold3 setup-pdb-subset
pixi run -e openfold3 pytest openfold3/tests/test_training_full.py
pixi run -e openfold3 pytest openfold3/tests/test_kernels.py -k turbopairformer
pixi run -e openfold3 run_openfold train --runner-yaml training.yaml --seed 42
```

Inputs the kernels do not support, such as the FP32 confidence head, fall back to
PyTorch. To compare with cuEquivariance, set `use_cueq_triangle_kernels: true` instead
and add `--use_tf32 true`.
