# Protenix attention provider

This optional provider vendors ByteDance Protenix triangle attention at commit
`4c355be4553512f72453ecbfb65e69f4c35d1413`, under its included Apache 2.0
[license](LICENSE). [PROVENANCE.json](PROVENANCE.json) records original upstream
SHA256/git-blob hashes, adapted file hashes, and the exact infrastructure changes.
The source revision and host normalization match the frozen September 20, 2026
three-crop training experiment.

`attention_scaled(q, k, v, mask, bias)` consumes Q already scaled once by
`1/sqrt(D)`. Q/K/V and pair bias are BF16 on the same CUDA device. The additive
mask is BF16 or FP32 and cannot require gradients. Q/K/V must have identical
shapes; D is a power of two at least 16, as required by these Triton kernels.
Inputs use `[..., rows, heads, sequence, D]`; output uses
`[..., rows, sequence, heads, D]`. The frozen layout normalization supports:

- 4D unbatched inputs with one shared pair bias, or flattened chunks with an
  explicit pair bias for each row;
- 5D batch/row inputs, with mask and pair-bias broadcasting;
- 6D batch/template/row inputs, expanding logical broadcasting before merging
  batch and template dimensions, including gradient reductions.

The native `deterministic=False` behavior and contiguous output-gradient hook
match the experiment. There is no precision cast, extra Q scaling, or runtime
fallback. Kernel compilation/execution failures propagate to the caller. This
provider does not claim deterministic gradients or parity with every hardware
and shape configuration; the checked-in GPU tests cover the supported route.

The vendored forward/backward arithmetic, autograd function bodies, and five
autotune configuration lists are unchanged. Adaptations namespace imports under
`openfold3.core.kernels.protenix._vendor` and defer dependency/device work until
execution. The upstream public fallback wrapper and distributed utility are
omitted. No top-level `protenix` package is imported or injected.

Autotune JSON files are written below
`$XDG_CACHE_HOME/openfold3/protenix` (default `~/.cache/openfold3/protenix`), or
`$OPENFOLD3_PROTENIX_CACHE_DIR` when set. Subdirectories separate upstream
revision, actual CUDA device, PyTorch/Triton runtime, and distributed rank. Rank
comes from initialized `torch.distributed`, otherwise `RANK` (default 0). Device
metadata is cached; kernel selection and autotune candidates are preserved.
The installed source directory stays read-only, and experiment-specific tuning
JSON files are not shipped. First use of each shape may autotune.
