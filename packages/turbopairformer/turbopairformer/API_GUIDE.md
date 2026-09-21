# TurboPairFormer offline API guide

This guide describes the source shipped with this package. Use
`turbopairformer --version` to identify the installed distribution.
`turbopairformer api` lists all top-level exports; `api --all` also includes
signatures and docstrings read from the installed Python modules without
importing Torch, Lightning, OpenFold3 or a CUDA extension. Examples below that
execute operators require the supported GPU runtime.

## attention - Triangle Attention

```python
from turbopairformer import turbo_attention

output = turbo_attention(q, k, v, res_mask=None, pair_bias=pair_bias)
```

`TurboTriangleAttention` and `triangle_attention` are aliases of
`turbo_attention`. All three are functions with the same arguments, not
`torch.nn.Module` constructors.

Arguments and return:

| Argument | Contract |
| --- | --- |
| `q`, `k`, `v` | BF16 tensors of identical shape `[*B, O, H, N, D]` on one Hopper CUDA device. At least one positive batch dimension; `O,N,H >= 1`, `1 <= D <= 64`. Q is raw/unscaled: the operator applies `1/sqrt(D)`. |
| `res_mask` | Optional floating **additive** mask, FP32/BF16/FP16, tail `[O, 1, 1, N]`. Its batch prefix broadcasts exactly to `*B`. Zero keeps keys; negative infinity excludes keys. No gradients. |
| `pair_bias` | **Required**, even though the Python signature defaults to `None`. BF16/FP32, tail `[1, H, N, N]`, with batch prefix broadcasting exactly to `*B`. FP32 is cast to BF16 internally; its gradient returns through the cast to the FP32 source. Pass explicit zeros for a model without bias. |
| return | Contiguous BF16 `[*B, O, N, H, D]`. The sequence and head axes change order relative to Q/K/V. |

`res_mask=None` means every key is valid and attention is noncausal. The facade
packs/pads supported inputs internally; the public tensors need not already
use the kernel's prepared layout. Launch-grid and broadcast-rank limits also
apply; `is_triangle_attention_supported` returns the precise reason when
metadata falls outside those limits.

Complete forward/backward example:

```python
import torch
from turbopairformer import turbo_attention

B, O, H, N, D = 1, 128, 4, 128, 32
options = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = [torch.randn(B, O, H, N, D, **options) for _ in range(3)]
pair_bias = torch.randn(B, 1, H, N, N, **options)
output = turbo_attention(q, k, v, pair_bias=pair_bias)
output.float().square().mean().backward()
print(output.shape)  # [1, 128, 128, 4, 32]
print(q.grad.shape, pair_bias.grad.shape)
```

For padding, construct an additive mask (not a boolean mask):

```python
res_mask = torch.zeros(B, O, 1, 1, N, device=q.device, dtype=torch.float32)
res_mask[..., -8:] = float("-inf")
output = turbo_attention(q, k, v, res_mask=res_mask, pair_bias=pair_bias)
# The original five-positional-argument call is also accepted:
output = turbo_attention(q, k, v, res_mask, pair_bias)
```

Backward provides gradients for Q/K/V and pair bias. The mask is not
differentiable. Unsupported metadata raises `ValueError` before kernel
allocation; an incompatible or missing binary raises a runtime loader error.
Model fallback belongs to the host (see `api support`).

## trimul - Triangle Multiplicative Update

The functional calling convention is:

```python
from turbopairformer import turbo_trimul

output = turbo_trimul(
    z,
    direction="outgoing",
    mask=None,
    norm_in_weight=None,
    norm_in_bias=None,
    p_in_weight=None,
    g_in_weight=None,
    norm_out_weight=None,
    norm_out_bias=None,
    p_out_weight=None,
    g_out_weight=None,
    eps=1e-5,
)
```

`TurboTriangleMultiplicative` and `run_trimul` are function aliases of
`turbo_trimul`. `z=z` is also accepted. Only `z`, `direction` and `mask` may be
positional in functional calls; weights and `eps` are keyword-only.

| Argument | Contract |
| --- | --- |
| `z` | BF16 CUDA pair tensor `[*B, N, N, C]`. Empty batch prefix is allowed. `N > 100`; `C` is one of `32,64,96,128`. Hopper CC 9.0. |
| `direction` | `"outgoing"` (default) or `"incoming"`. |
| `mask` | Optional **multiplicative** pair mask broadcastable exactly to `z.shape[:-1]`; bool/BF16/FP32 on the same device, no gradients. One keeps pairs and zero excludes them. `None` means all pairs valid. |
| `norm_in_weight`, `norm_in_bias` | Each `[C]`, contiguous FP32 on `z.device`. |
| `p_in_weight`, `g_in_weight` | Each `[2*C, C]`, contiguous FP32. Rows contain branch a followed by branch b. |
| `norm_out_weight`, `norm_out_bias` | Each `[C]`, contiguous FP32. |
| `p_out_weight`, `g_out_weight` | Each `[C, C]`, contiguous FP32. |
| `eps` | Must equal `1e-5` exactly. Linear projection/gate biases are unsupported. |
| return | Same shape and dtype as `z`. First-order autograd for `z` and caller-supplied parameters; no mask gradients or higher-order gradient support. |

Quick demonstration:

```python
import torch
from turbopairformer import turbo_trimul

z = torch.randn(
    1, 128, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
output = turbo_trimul(z, direction="outgoing")
output.float().square().mean().backward()
```

Missing or `None` normalization weights/biases become ones/zeros. Missing
matrices are independently sampled **on every call** from a normal distribution
with standard deviation `1/sqrt(C)`, using Torch RNG on the CUDA device. These
demonstration tensors have no `requires_grad` and are not retained as model
state. Supply and reuse all eight parameters for training or repeated evaluation.

Complete example with persistent trainable parameters:

```python
import torch
from turbopairformer import turbo_trimul

C, N = 128, 128
options = dict(device="cuda", dtype=torch.float32)
weights = torch.nn.ParameterDict(
    {
        "norm_in_weight": torch.nn.Parameter(torch.ones(C, **options)),
        "norm_in_bias": torch.nn.Parameter(torch.zeros(C, **options)),
        "p_in_weight": torch.nn.Parameter(torch.randn(2 * C, C, **options) / C**0.5),
        "g_in_weight": torch.nn.Parameter(torch.randn(2 * C, C, **options) / C**0.5),
        "norm_out_weight": torch.nn.Parameter(torch.ones(C, **options)),
        "norm_out_bias": torch.nn.Parameter(torch.zeros(C, **options)),
        "p_out_weight": torch.nn.Parameter(torch.randn(C, C, **options) / C**0.5),
        "g_out_weight": torch.nn.Parameter(torch.randn(C, C, **options) / C**0.5),
    }
)
optimizer = torch.optim.Adam(weights.parameters(), lr=1e-3)
z = torch.randn(1, N, N, C, device="cuda", dtype=torch.bfloat16, requires_grad=True)
mask = torch.ones(1, N, N, device=z.device, dtype=torch.bool)
optimizer.zero_grad()
output = turbo_trimul(z, direction="incoming", mask=mask, **weights)
output.float().square().mean().backward()
optimizer.step()
# Keep weights (e.g. as a submodule); reuse them on subsequent calls.
```

Existing module integrations remain accepted:

```python
output = turbo_trimul(module, z, mask=None)
output = turbo_trimul(module=module, z=z, mask=mask)
```

The host-owned module supplies integer `c_z == c_hidden == C`, boolean
`_outgoing`, and these bias-free FP32 linear layers: `linear_a_p`,
`linear_a_g`, `linear_b_p`, `linear_b_g`, `linear_g`, `linear_z`. Each weight
is contiguous `[C,C]` on the input device. It also supplies FP32
`layer_norm_in` / `layer_norm_out`, each with `[C]` weight/bias and epsilon
`1e-5`. The package does not require importing OpenFold3. Fused `linear_ab_p/g`
modules are outside this adapter's contract. Module calls cannot mix in
functional weights, `direction` or `eps`.

Unsupported inputs raise `ValueError`; ambiguous/mixed argument forms raise
`TypeError`. The actual Python signature uses an internal `_NOT_GIVEN` sentinel
to distinguish omitted arguments across the two calling conventions.

## support - Check inputs and route in the host

```python
from turbopairformer import (
    is_triangle_attention_supported,
    check_trimul_support,
    evaluate_trimul_support,
    SupportResult,
    TrimulSupportMetadata,
    TRIMUL_STOCK_FALLBACK_THRESHOLD,
    triangle_attention_invocation_count,
)
```

`is_triangle_attention_supported(q, k, v, res_mask=None, pair_bias=None)`
returns `(supported: bool, reason: str)`. Unpack the tuple; a nonempty tuple is
truthy even when its first element is `False`.

```python
# Integration fragment: q/k/v/bias and stock_attention belong to your host.
ok, reason = is_triangle_attention_supported(q, k, v, pair_bias=pair_bias)
if ok:
    output = turbo_attention(q, k, v, pair_bias=pair_bias)
else:
    print(reason)
    output = stock_attention(q, k, v, pair_bias=pair_bias)
```

`check_trimul_support(module, z, mask=None)` checks the module adapter.
For functional weights use the module-level helper:

```python
from turbopairformer.trimul_support import check_trimul_tensor_support

# z/weights/mask are the persistent tensors from the TriMul example.
support = check_trimul_tensor_support(z, direction="incoming", mask=mask, **weights)
if support:
    output = turbo_trimul(z, direction="incoming", mask=mask, **weights)
else:
    print(support.code, support.reason)  # the host chooses its fallback
```

The functional check accepts the same eight optional parameter keywords and
`eps=1e-5` as the operator and validates before any demonstration allocation.
`check_trimul_tensor_support` is accessed through `turbopairformer.trimul_support`,
not the top-level package.

`SupportResult(supported, code, reason, batch=None, sequence=None, channels=None)`
is an immutable verdict. `bool(result)` is its `supported` value; inspect
`code` and `reason` for diagnostics. `TRIMUL_STOCK_FALLBACK_THRESHOLD` is `100`:
the CUDA path starts at N=101. The package rejects unsupported shapes; it does
not automatically execute stock PyTorch.

Live tensor support checks do not load extensions or launch kernels, but may
query CUDA device metadata. `evaluate_trimul_support(metadata)` performs only
metadata evaluation and can be used without a CUDA device. Construct its
immutable records using:

```python
from turbopairformer.trimul_support import (
    TensorMetadata,
    NamedTensorMetadata,
    TrimulSupportMetadata,
    evaluate_trimul_support,
)

# Record signatures (dtype is a torch.dtype):
# TensorMetadata(shape, dtype, device_type, device_index, contiguous, requires_grad)
# NamedTensorMetadata(name, tensor)
# TrimulSupportMetadata(z, mask, device_name, capability, multiprocessor_count,
#     c_z, c_hidden, outgoing, weights, linear_bias_is_none, layer_norm_eps)
```

`TensorMetadata.device_key` returns `(device_type, device_index)`. In
`TrimulSupportMetadata`, `weights` is a tuple of named records for the six
linear weights and four normalization tensors described in `api trimul`.
`linear_bias_is_none` contains `(linear_name, True)` pairs;
`layer_norm_eps` contains `("layer_norm_in", 1e-5)` and
`("layer_norm_out", 1e-5)`. Supply actual device capability and SM count when
describing a real target. A metadata verdict does not verify an installed AOT
binary or certify GPU execution.

Other `trimul_support` module constants: `SUPPORTED_CHANNELS`,
`SUPPORTED_MASK_DTYPES`, `MAX_FLATTENED_BATCH`, `MAX_GRID_X`, `MAX_GRID_YZ`.
The channels are `{32,64,96,128}`, masks are bool/BF16/FP32, flattened batch
and grid Y/Z limits are 65535, and grid X is 2147483647. Query any qualified
constant (e.g. `api trimul_support.SUPPORTED_CHANNELS`) for its source declaration.
Use `api trimul_support.TrimulSupportMetadata` for the exact annotated fields.

`triangle_attention_invocation_count() -> int` is a process-local counter of
accepted attention facade calls, useful for checking host routing. It is not
a GPU kernel count, completion check, elapsed time, or a cross-rank total.

## distributed - Optional Lightning kernel preflight

Install the optional dependency in the package environment:

```bash
python -m pip install 'turbopairformer[lightning]'
```

`turbopairformer.distributed` requires PyTorch Lightning 2.6.1. Its public APIs:

```python
from turbopairformer.distributed import (
    TurboPairFormerKernelDDPCallback,
    turbopairformer_aot_modules_from_config,
    required_turbopairformer_aot_modules,
    installed_distribution_identity,
)
from turbopairformer import DistributedContractError
```

- `turbopairformer_aot_modules_from_config(config)` returns a tuple of selected
  AOT module keys. It reads `config.settings.memory.train/eval` and the package
  integration's `use_stable_triangle_attention` /
  `use_stable_triangle_multiplicative` flags. Missing settings yield `()`.
- `required_turbopairformer_aot_modules(pl_module)` derives the same union from
  `pl_module.config`.
- `installed_distribution_identity()` verifies the installed wheel RECORD and
  returns an identity dictionary. It rejects missing/edited files, editable
  installs, and imports outside the verified wheel.

Callback construction (host integration fragment; `model` is your configured
Lightning module and the receipt location is shared by all ranks):

```python
from pathlib import Path
import pytorch_lightning as pl

callback = TurboPairFormerKernelDDPCallback(
    expected_world_size=2,
    expected_num_nodes=1,
    expected_devices_per_node=2,
    expected_aot_modules=required_turbopairformer_aot_modules(model),
    receipt_path=Path("/shared/run/turbopairformer-ddp.json"),
)
trainer = pl.Trainer(
    accelerator="gpu", devices=2, num_nodes=1, strategy="ddp", callbacks=[callback]
)
# trainer.fit(model, ...) with your host data and training configuration
```

Install the callback into the trainer; do not call its lifecycle hooks manually.
It preloads and compares kernel/wheel identities across ranks before the first
fit forward, freezes effective execution flags, and validates receipt readback.
The current contract requires initialized NCCL with Lightning `DDPStrategy`,
distinct full Hopper devices (MIG disabled), at least 75 GiB and 100 SMs per
device, matching identities/flags, JIT disabled, and both SR flags enabled
when attention is selected. Violations raise `DistributedContractError`, a
`RuntimeError` subclass also exported at package top level.

The module reexports `distributed_execution_flag`, `distributed_strict_enabled`
and `validate_distributed_environment` (see `api environment`). This callback
checks kernel identity; it is not full multi-GPU model-training certification.

Installing the package alone does not change a host model's backend. The fork's
package integration selects `OPENFOLD3_TRIANGLE_PROVIDER=turbopairformer` with
the relevant model settings. Upstream integration branches can use different
configuration names; follow the guide for the exact host checkout. Repository
`scripts/`, `tools/` and `integration/` commands are source tools and are not
automatically installed as package CLI commands.

## environment - Runtime, version and execution flags

```bash
turbopairformer --version
python -m turbopairformer --help
```

`from turbopairformer import __version__` reports installed distribution metadata.
A source checkout without distribution metadata reports `0.0.0+source`.
The help command is offline and works without importing Torch. Operator
execution uses the package's supported wheel: Linux x86-64 (glibc >=2.28),
CPython 3.14, PyTorch 2.10.0 / CUDA runtime 12.8, Hopper CC 9.0 (`sm_90a`).
H100 has execution receipts; H200 hardware eligibility is not H200 validation.
A precompiled wheel needs no local CUDA toolkit. This CLI does not probe or
modify the GPU, install packages, launch training, or rebuild a wheel.

| Canonical flag | Default | Meaning |
| --- | --- | --- |
| `TURBOPAIRFORMER_SR_BACKWARD` | `1` | Single-reconstruction attention backward. |
| `TURBOPAIRFORMER_SR_SHADOW` | `1` | Save/use forward rounding residual with SR. |
| `TURBOPAIRFORMER_ALLOW_JIT` | `0` | Explicit development JIT opt-in; normal wheels use AOT. |

The package resolves canonical names before `STABLEFOLD_*`, then `OPENFOLD3_*`
legacy names. JIT's OpenFold spelling is `OPENFOLD3_TRIANGLE_ALLOW_JIT`;
the others are `OPENFOLD3_SR_BACKWARD` and `OPENFOLD3_SR_SHADOW`.
Only the string `"1"` is true for boolean policy reads. Set flags before model
execution; changing them after DDP preflight violates the frozen policy.

```python
from turbopairformer import distributed_execution_flag, distributed_strict_enabled
from turbopairformer.policy import (
    SR_BACKWARD,
    SR_SHADOW,
    ALLOW_JIT,
    EXECUTION_ENV_KEYS,
    LEGACY_ENV_NAMES,
    canonical_flag_name,
    read_execution_flag,
    execution_environment,
    validate_distributed_environment,
)

print(execution_environment())  # effective strings, including defaults
print(distributed_execution_flag(SR_BACKWARD))  # bool; frozen value under DDP
print(distributed_strict_enabled())  # whether strict DDP routing is armed
print(read_execution_flag(ALLOW_JIT))  # effective string, without mutation
print(canonical_flag_name("OPENFOLD3_SR_BACKWARD"))
validate_distributed_environment()  # raises on post-preflight flag changes
```

`canonical_flag_name(name)` raises `KeyError` for unknown flag names.
`EXECUTION_ENV_KEYS` lists canonical flags; `LEGACY_ENV_NAMES` maps them to
accepted aliases. `DistributedContractError` is the policy violation exception.
There is no public command to enable strict routing by itself; the callback
arms it only after successful preflight.

## internals - Advanced Python helpers and native ABI

The reference appended by this command comes from the Python source shipped
with this package. It includes every non-underscore top-level function/class,
constant and package reexport
in operator, support, policy, distributed, workspace and loader modules.
No module is executed to inspect its signatures. Exact qualified private
functions can also be inspected, for example
`turbopairformer api _trimul._run_supported_trimul`.

Use `turbo_attention` / `turbo_trimul` for normal model autograd. Internal
interfaces can change and raw CUDA wrappers require prepared layouts, dtype,
alignment and allocation contracts from the matching source. The internal
`stablefold_*` names are retained ABI identifiers in this package branch.

Workspace planning example (no kernel launch; importing the module needs Torch):

```python
from turbopairformer.triangle_dq_dbias import workspace_plan, sr_workspace_plan

plan = workspace_plan(
    128, 4, outer=128, batch=1, workspace_budget_bytes=256 * 1024 * 1024
)
print(plan.elements, plan.elements * 2)  # BF16 slab elements and bytes
sr_plan = sr_workspace_plan(128, 4, 32, outer=128, batch=1)
```

`WorkspacePlan` exposes query span, resident leaves, batch chunks, launch
counts and BF16 slab size. `workspace_elements(...)` returns the slab element
count and has no `outer` argument. The default budget is 268435456 bytes; it
is a target for the slab, not a hard bound on total scratch memory. A minimum
tile can exceed it and SR continuation buffers are additional allocations.
`outer_leaf_for(sequence)` returns the fixed leaf size (4) for positive inputs.
`delta_true_partial_elements(sequence, heads, batch, outer=None)` counts FP32
row/key-block partials for optional diagnostic delta recomputation.

Raw backward wrappers (all under `torch.no_grad`, no public autograd boundary):

- `triangle_dkdv.stablefold_triangle_dkdv` returns `(dk,dv)`, or
  `(dk,dv,delta_true)` with `emit_delta_true=True`.
- `triangle_dq_dbias.stablefold_triangle_dq_dbias` returns `(dq,dbias)`.
- `triangle_dq_dbias.stablefold_triangle_sr_backward` returns
  `(dq,dk,dv,dbias)` for prepared exact-tile inputs (`N % 64 == 0`, `D <= 64`).
- `triangle_dq_dbias.stablefold_reduce_broadcast_dbias` folds flattened batch
  contributions into the original bias prefix, returning BF16.

Q/K/V and returned dQ/dK/dV use internal `[B,O,H,N,D]` layout; `d_output`
uses `[B,O,N,H,D]`, the same axis order as the public attention output.
Inputs still need the matching flattened/padded dimensions, contiguous buffers
and saved logsumexp/delta, mask and bias from the corresponding forward.
Optional output
and workspace arguments let internal callers reuse matching buffers. Public
facades own packing, padding, saved-state handling and full support validation.

Each of `triangle_forward`, `triangle_delta`, `triangle_dkdv`,
`triangle_dq_dbias`, `trimul_cuda` provides `source_sha256()` for source identity
and `load_extension(verbose=None)` for lazy AOT loading. Loading validates ABI
and binary hashes and can raise `_aot_runtime.AOTExtensionError`. Calling it
does load a native extension; displaying its help does not.

`_aot_runtime.load_extension_module`, `distributed_aot_identity`, and
`source_sha256_for` implement that loader/identity contract. The private
`_trimul.TurboPairFormerTrimul(module)` callable adapter accepts `(z, mask)`;
it is not a top-level `nn.Module`. Private Torch custom operations
`torch.ops.turbopairformer._trimul_fwd_v1` / `_trimul_bwd_v1` and the native
C++ bindings are implementation details; use the public facade for training.

The following source-derived reference supplies exact signatures, record
fields, constructor parameters, and any docstrings provided by this version.
