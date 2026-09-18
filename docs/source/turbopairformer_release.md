# Publishing TurboPairFormer

The release workflow lives on `feat/turbopairformer-pip-package`. A GitHub
`release: published` event runs the workflow from the tagged commit, so it does
not require changing production `main`. A GitHub prerelease validates the exact
release assets and skips PyPI; a stable release uploads the validated wheel.

The first binary targets Linux x86_64, CPython 3.14, PyTorch 2.10.0/CUDA 12.8 and
Hopper `sm_90a`. GPU evidence records the device actually tested. H100 validation
does not establish H200 execution results. The public command after publication
is `pip install turbopairformer`; Python imports use `turbopairformer`.

## Build and validate the final wheel

Use a clean committed checkout on a Linux/Hopper host with Docker and NVIDIA
Container Toolkit. The container builder produces the manylinux wheel and its
matching runtime image without changing the host's Python environment:

```bash
bash packages/turbopairformer/tools/build_manylinux_wheel.sh \
  /absolute/path/to/new-wheel-directory
```

On the same host, validate the final wheel using that runtime image:

```bash
bash scripts/turbopairformer/validate_release_gpu.sh \
  /absolute/path/to/final-wheel.whl /absolute/path/to/new-validation-directory
```

The driver installs the **final repaired wheel** into an isolated environment,
runs outside the source checkout, disables JIT, and checks that OpenFold3 is not
required. It records the installed smoke test, doctor, standalone GPU tests,
and environment information in the validation directory. After it succeeds,
create the release bundle using the separate command below.

For a manual equivalent, run outside the source checkout with `PYTHONPATH`
unset, JIT disabled and no compiler available. Produce these two machine-readable
GPU receipts from the installed wheel:

```bash
OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer python \
  "$REPOSITORY/tools/turbopairformer_doctor.py" \
  --expect-provider turbopairformer --json > installed-aot.json
TURBOPAIRFORMER_TEST_WHEEL="$WHEEL" python -m pytest \
  "$REPOSITORY/packages/turbopairformer/tests" \
  --ignore="$REPOSITORY/packages/turbopairformer/tests/host_openfold3" \
  --junitxml=standalone-gpu.xml
```

Run each command successfully before continuing. The bundle checker requires
successful, non-skipped attention support, attention FP64 accuracy, TriMul
execution and TriMul numerical tests. The doctor must load all five extensions
from site-packages, with a manifest digest matching the wheel. Keep host-provider
and pixi integration results alongside the release notes; these are separate
from standalone kernel validation.

From the repository root, bundle those files without rebuilding the wheel:

```bash
python scripts/turbopairformer/release_bundle.py create release-bundle \
  --wheel "$WHEEL" --doctor "$RECEIPTS/installed-aot.json" \
  --junit "$RECEIPTS/standalone-gpu.xml" --expected-commit "$COMMIT" \
  --doctor-command 'tools/turbopairformer_doctor.py --expect-provider turbopairformer --json' \
  --gpu-command 'python -m pytest packages/turbopairformer/tests --ignore=packages/turbopairformer/tests/host_openfold3 --junitxml=standalone-gpu.xml'
```

`COMMIT` is the full source commit embedded in the wheel. Preserve that build
commit in the wheel, doctor report and release receipt. A release tag may point
to that commit or a descendant whose complete tracked `packages/turbopairformer`
Git tree is identical. This allows later host, pixi or workflow changes without
rebuilding unchanged package code. Any package-tree change requires a new build
and GPU validation. The workflow checks ancestry and tree identity with full
Git history and reports both build and release commits.

`release-bundle` is a new directory containing the wheel,
`release-validation.json`, `installed-aot.json`, and `standalone-gpu.xml`.

## Validate a GitHub prerelease

For a version `0.1.0` wheel, create candidate tag `turbopairformer-v0.1.0-rc.1`
at a qualifying release commit described above. Attach all four bundle files to
a **draft** prerelease, then publish the prerelease. Uploading assets before
publication prevents workflow races.
The workflow verifies source identity, wheel and receipt digests, GPU test
results, wheel metadata, and the actual ELF portability audit. It stores a
validated wheel artifact. Its PyPI job is skipped for prereleases.

GitHub release assets follow the repository's access permissions. In a private
repository, publishing a GitHub prerelease does not make its wheel anonymously
downloadable; PyPI publication is a separate step.

Use a new `-rc.N` candidate when changing source or artifacts. The candidate tag
suffix identifies a release rehearsal; the wheel itself remains version 0.1.0
so the tested bytes can be published unchanged.

## Configure PyPI and publish

In PyPI account publishing settings, register a pending trusted publisher:

| Field | Value |
| --- | --- |
| Project | `turbopairformer` |
| GitHub owner | `RyanAIResearch` |
| Repository | `openfold3_triangle_kernel` |
| Workflow | `turbopairformer-publish.yml` |
| Environment | `pypi` |

Account recovery codes are for sign-in recovery; this publishing flow uses
GitHub OIDC and needs no upload token. A pending publisher does not reserve the
project name.

After the candidate succeeds and PyPI is configured, create stable tag
`turbopairformer-v0.1.0` at the validated release commit. Attach the exact same four
files to a draft release, then publish it. **Publishing this stable GitHub release starts
the PyPI upload.** The workflow never rebuilds the wheel. Keep the version and
assets unchanged during a retry; do not replace an already published PyPI file.

After a successful upload, verify in a new environment on a supported
Linux/Hopper host:

```bash
python3.14 -m venv /absolute/path/to/new-pypi-validation-venv
/absolute/path/to/new-pypi-validation-venv/bin/python -m pip install \
  --only-binary=:all: turbopairformer==0.1.0
/absolute/path/to/new-pypi-validation-venv/bin/python -m pip check
```

Use that environment's Python to run the installed doctor and smoke test again
against the PyPI download.

References: [GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release),
[PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
