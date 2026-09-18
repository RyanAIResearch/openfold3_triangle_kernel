# Integrating StableFold into upstream OpenFold3

The standalone package exposes triangle attention and TriMul without requiring
OpenFold3. This optional host patch adds model flags and routing to an upstream
AQLab checkout. The RyanAIResearch fork already has a provider adapter and
should not apply this patch.

Install a freshly built StableFold wheel into the same environment as
OpenFold3. Match Python, Torch, CUDA runtime and C++ ABI to its build manifest:

```bash
python -m pip install /absolute/path/to/stablefold-<version>-<tags>.whl
```

Public index publication is a separate release step. No old checked-in wheel
is certified for the current package revision.

## Apply the versioned patch

`openfold3-main-e47adbf-stablefold.patch` targets
`aqlaboratory/openfold-3` commit `e47adbf`. In that upstream checkout:

```bash
git apply --check /absolute/path/to/openfold3-main-e47adbf-stablefold.patch
git apply /absolute/path/to/openfold3-main-e47adbf-stablefold.patch
```

On a different revision, port and review the hunks before using them. The patch
adds `use_stable_triangle_attention` and `use_stable_triangle_multiplicative`
flags, passes them through the model, calls the current package API, and leaves
unsupported shapes on the host stock route. Kernel import and execution errors
propagate instead of being treated as unsupported shapes.

## Select a single-GPU training run

```yaml
model_update:
  custom:
    settings:
      memory:
        train:
          use_stable_triangle_attention: true
          use_stable_triangle_multiplicative: true
          use_triton_triangle_kernels: false
          use_cueq_triangle_kernels: false
          use_deepspeed_evo_attention: false
pl_trainer_args:
  devices: 1
  num_nodes: 1
```

Keep `fuse_projection_weights: false` for the TriMul fast path. Leave evaluation
on the existing configuration unless intentionally testing its StableFold
routes. Export these flags before starting the host:

```bash
export STABLEFOLD_SR_BACKWARD=1
export STABLEFOLD_SR_SHADOW=1
export STABLEFOLD_ALLOW_JIT=0
```

The patch imports `stablefold` optionally. Enabling its flags without the
package raises at startup. Distributed StableFold training also raises at
startup: this versioned host patch currently supports a single GPU, and its
DDP integration needs a separate implementation and validation pass.

## Verify the current artifact and patch

First run the package's numerical and gradient tests in the installed wheel
environment. Then run a one-step upstream training smoke and the host routing
tests, checking invocation counts and supported-shape fallback messages. The
[validation record](VALIDATION.md) distinguishes historical results from the
current revision; earlier numbers do not certify a rebuilt package or edited
patch.
