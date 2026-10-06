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

The six vendored implementation files (op, forward, backward, autotune,
autotune_helpers, distributed) are exactly the pinned upstream bytes after
reversing only the private import namespace substitutions recorded in
PROVENANCE.json. No local type annotations, tuning, cache policy, or arithmetic
changes are applied to these files. The optional host adapter loads them on
first use and does not import or inject a top-level `protenix` package.

Autotune follows upstream: files live in `_vendor/TriAttentionCache/<RANK>/`
with compute capability in their filenames. The package directory must be
writable, and the current CUDA device must be selected before the first native
import. Rank comes from the upstream distributed utility's environment rules.
The upstream rank-0 cache-copy behavior is retained. Runtime tuning JSON files
are excluded from source control and distribution artifacts; no experiment
precache is shipped. First use of each shape may autotune.

This is the native-kernel route used by the frozen experiments. The optional
upstream SDPA fallback wrapper is not used. The host's BF16 validation, 4D/5D/6D
layout conversion, pre-scaled Q and contiguous output-gradient boundary remain
explicit integration requirements; this does not replace the host model with
the full Protenix model or claim identical model-level routing.
