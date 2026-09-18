# TurboPairFormer 0.1.0 release wheel

This is the tested standalone wheel built from commit
`59439005a05c638b53377b18d07269232008ace8`, retained in this repository so
`pixi install -e openfold3-cuda12` does not need a second download authenticated
to private GitHub Releases. The same wheel is distributed on PyPI.

The wheel targets Linux x86-64 / glibc 2.28+, CPython 3.14, PyPI Torch
2.10.0 / CUDA 12.8, and Hopper CC 9.0 (H100/H200). Its final H100 numerical
suite passed 194 tests with zero skips; H200 hardware execution is not yet verified. The
OpenFold3 host integration supports single-GPU launch only.

`SHA256SUMS` pins the whole wheel. `release-validation.json` binds that digest
to the included installed-AOT doctor and standalone GPU test receipts.
The wheel contains its own source, ABI, and per-extension hash manifest.

The public calls are `from turbopairformer import turbo_attention, turbo_trimul`.
Both are exercised by the isolated installed-wheel GPU smoke; Attention is also
called through `turbo_attention` in the FP64 numerical tests.

Both masks default to `None` (all input positions valid, noncausal). `pair_bias`
is required. Five GPU cases verify omitted masks against explicit masks for
outputs and gradients, including padded lengths and batch/bias broadcasting.
