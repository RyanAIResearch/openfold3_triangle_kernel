# Integrating TurboPairFormer 0.1.3 into upstream OpenFold3

The standalone package exposes triangle attention and TriMul without requiring
OpenFold3. This optional host patch adds model flags and routing to an upstream
AQLab checkout. The RyanAIResearch fork already has a provider adapter and
should not apply this patch.

Install TurboPairFormer **0.1.3** into the same environment as
OpenFold3. Match Python, Torch, CUDA runtime and C++ ABI to its build manifest:

```bash
python -m pip install turbopairformer==0.1.3
```

For a locally rebuilt 0.1.3 wheel, install its full path instead. Validation
applies to the exact wheel used, not just its version number.

## Apply the versioned patch

`openfold3-main-e47adbf-turbopairformer.patch` targets
`aqlaboratory/openfold-3` commit `e47adbf`. In that upstream checkout:

```bash
git apply --check /absolute/path/to/openfold3-main-e47adbf-turbopairformer.patch
git apply /absolute/path/to/openfold3-main-e47adbf-turbopairformer.patch
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
on the existing configuration unless intentionally testing its TurboPairFormer
routes. Export these flags before starting the host:

```bash
export TURBOPAIRFORMER_SR_BACKWARD=1
export TURBOPAIRFORMER_SR_SHADOW=1
export TURBOPAIRFORMER_ALLOW_JIT=0
```

The patch imports `turbopairformer` optionally. Enabling its flags without the
package raises at startup. Distributed TurboPairFormer training also raises at
startup: this versioned host patch currently supports a single GPU, and its
DDP integration needs a separate implementation and validation pass.

## Verify the current artifact and patch

First run the package's numerical and gradient tests in the installed wheel
environment. Then run a one-step upstream training smoke and the host routing
tests, checking invocation counts and supported-shape fallback messages. The
[validation record](VALIDATION.md) distinguishes historical results from the
current revision; earlier numbers do not certify a rebuilt package or edited
patch.
