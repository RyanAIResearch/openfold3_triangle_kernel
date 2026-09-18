# Docker builds

OpenFold-3 builds its image with pixi:

- **`Dockerfile.pixi`** — uses [pixi](https://pixi.sh) to manage all dependencies including CUDA toolkit, cuDNN, CUTLASS, and build tools from conda-forge. No `nvidia/cuda` base image needed.

## Pixi-based builds

These images install the current checkout in editable mode for development and
tests. The default CUDA 12 environment is `openfold3-cuda12-dev`. The separate
`openfold3-cuda12` production environment installs the fixed, attested OpenFold3
wheel containing the in-tree TurboPairFormer kernels; this Dockerfile rejects that environment to avoid replacing its
wheel with an editable checkout. For production installation and verification, follow the
[Pixi production workflow](../README.md#pixi-production-workflow).

### Development image

```bash
docker build \
    -f docker/Dockerfile.pixi \
    --target devel \
    -t openfold-docker:pixi-devel .
```

### Test image

```bash
docker build \
    -f docker/Dockerfile.pixi \
    --target test \
    -t openfold-docker:pixi-test .
```

### Running tests

```bash
docker run \
    --rm \
    --gpus all \
    -v $(pwd -P):/opt/openfold3 \
    -t openfold-docker:pixi-test \
    pytest openfold3/tests -vvv
```

### CUDA 13 builds and tests

```bash
docker build \
    -f docker/Dockerfile.pixi \
    --build-arg PIXI_ENV=openfold3-cuda13 \
    --target test \
    -t openfold-docker:pixi-test-cuda13 .
```

```bash
docker run \
    --rm \
    --gpus all \
    -v $(pwd -P):/opt/openfold3 \
    -t openfold-docker:pixi-test-cuda13 \
    pytest openfold3/tests -vvv
```

### Build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `ubuntu:22.04` | Base Docker image |
| `PIXI_VERSION` | `v0.80.0` | Pixi version to install |
| `PIXI_ENV` | `openfold3-cuda12-dev` | Pixi development environment name (`openfold3-cuda12-dev` or `openfold3-cuda13`) |

### cuEquivariance

The `openfold3-cuda12-dev` and `openfold3-cuda13` pixi environments include cuEquivariance by default. No additional build argument is needed. See the [cuEquivariance.yml example](../examples/example_runner_yamls/cuequivariance.yml) and the [kernels documentation](https://openfold-3.readthedocs.io/en/latest/kernels.html) for usage details.

### Notes

- Uses `ubuntu:22.04` as the base image; CUDA comes from conda-forge rather than an `nvidia/cuda` base
- `pixi.lock` is the lock file
- Environment variables are set automatically via `pixi shell-hook` from `pixi.toml` activation sections

### Exporting conda-compatible environments

`pixi.lock` is the source of truth, but conda-consumable files can be exported
from it for anyone not using pixi:

```bash
# conda environment YAML, per environment and platform
pixi run export-conda openfold3-cuda12-dev linux-64
pixi run export-conda-all

# conda-lock file, per environment
pixi run export-conda-lock openfold3-cuda12-dev
```

These write into `environments/`. Note that `export-conda-lock` emits
placeholder platform names (`p1`, `p2`, ...) for pixi's custom CUDA platforms
such as `linux-64-cuda12`, since they are not conda platform names; standard
platforms (`linux-64`, `osx-arm64`, ...) come through unchanged.
