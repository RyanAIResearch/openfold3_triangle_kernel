# StableFold installation and single-GPU launch helpers

These scripts build and verify an installed StableFold wheel and run the
OpenFold3 host adapter on one GPU. See the
[package quickstart](../../docs/source/stablefold_pip_quickstart.md) for the full
installation and provider contract.

Use a dedicated environment containing the intended Torch build and, for
training, this version of OpenFold3:

```bash
python3.14 -m venv /absolute/path/to/venv
/absolute/path/to/venv/bin/python -m pip install torch==2.10.0 setuptools wheel ninja
export STABLEFOLD_PYTHON=/absolute/path/to/venv/bin/python
export CUDA_HOME=/absolute/path/to/cuda-12.9
scripts/stablefold/install_stablefold.sh
scripts/stablefold/doctor_stablefold.sh --expect-provider stablefold
```

Package build and launch helpers require `STABLEFOLD_PYTHON` explicitly. The
build helper checks CPython 3.14, Torch 2.10.0, and CUDA runtime 12.8 before
compiling with the CUDA 12.9 toolkit. It builds a wheel in a temporary directory,
installs non-editably, and checks installed modules outside the checkout. It
does not modify an older shared Pixi runtime or install the OpenFold3 host and
its training dependencies. For the `intree` provider, diagnostic and launch
helpers can still use this checkout's Pixi environment (override its name with
`STABLEFOLD_PIXI_ENV`).

Prepare and run a smoke configuration:

```bash
"$STABLEFOLD_PYTHON" -m tools.select_training_backend \
  datasets/train_pdb_subset.yaml --backend stablefold --provider stablefold \
  --single-gpu-smoke --output datasets/train_pdb_subset.stablefold.yaml
export OF3_RUN_CWD=/absolute/path/outside/repo/openfold3-run
export CUDA_VISIBLE_DEVICES=0
scripts/stablefold/run_single_gpu.sh \
  --runner-yaml "$PWD/datasets/train_pdb_subset.stablefold.yaml" --seed 42
```

The launcher explicitly selects `STABLEFOLD_PROVIDER=stablefold`, sets attention
execution flags, disables JIT, verifies the actual provider's AOT extensions,
and validates data paths before training. `STABLEFOLD_PROVIDER=intree` selects
the vendored kernels and requires their own installed AOT build. Outside these
package launchers, OpenFold3 defaults to `intree`.

The standalone package's host DDP integration is not yet validated.
`run_single_node.sh`, `run_multinode.sh`, and `sbatch_multinode.sh` reject the
`stablefold` provider before launching workers. The separately certified
in-tree distributed workflow is described in the repository's distributed
quickstart; installing this package does not alter its requirements.

`doctor_stablefold.sh --allow-no-gpu` can check imports on a CPU machine. It
explicitly reports that CUDA binaries were not checked. A GPU doctor pass
verifies AOT loading; numerical, gradient, and performance tests are separate.
