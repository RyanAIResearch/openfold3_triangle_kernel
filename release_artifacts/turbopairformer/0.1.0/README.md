# TurboPairFormer 0.1.0 candidate wheel

This is the tested standalone wheel built from commit
`d77e8d06cba4918fa551ce1f4ec41cbb21d055af`, retained in this repository so
`pixi install -e openfold3-cuda12` does not need a second download authenticated
to private GitHub Releases. PyPI publication is a separate pending step.

The wheel targets Linux x86-64 / glibc 2.28+, CPython 3.14, PyPI Torch
2.10.0 / CUDA 12.8, and Hopper CC 9.0 (H100/H200). Its final H100 numerical
suite passed 176 tests; H200 hardware execution is not yet verified. The
OpenFold3 host integration supports single-GPU launch only.

`SHA256SUMS` pins the whole wheel. `release-validation.json` binds that digest
to the included installed-AOT doctor and standalone GPU test receipts.
The wheel contains its own source, ABI, and per-extension hash manifest.
