# TurboPairFormer H100 kernel artifact

Build source: `84f96a31e96f27324a93df7e0485a8326a15fd40`.

This OpenFold3 wheel contains five renamed AOT CUDA extensions. It targets CPython 3.14, Conda Torch 2.10.0 / CUDA 12.9, and H100 `sm_90a`.

Fresh installed-wheel validation passed: 258 tests (0 skipped), nine Attention FP64 reference cases with bitwise replay, four TriMul FP32 fold cases, and a clean-install forward/backward smoke without a compiler or JIT.

The full TriMul smoke checks execution and eager/compiled consistency; the FP32 reference here covers weight-gradient folding. This artifact has no new multi-GPU, end-to-end training, performance, checkpoint/resume, or Compute Sanitizer qualification.

`RECEIPT.json` states the scope. `SHA256SUMS` and the artifact manager pin every evidence file and the wheel. The public standalone Python package is released separately.
