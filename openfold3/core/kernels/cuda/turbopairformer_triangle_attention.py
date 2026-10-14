# SPDX-License-Identifier: Apache-2.0

"""TurboPairFormer Triangle Attention facade — deterministic production path.

Online-softmax forward plus a single-reconstruction backward: each probability
tile is reconstructed once and reused for dQ, dK, dV, and dBias, with fixed
reductions and no floating-point atomics. Sequence length, head count, and batch
prefix are runtime dimensions; D is selected from the compiled Hopper
specializations. By default, the forward saves an output rounding residual
that backward uses to correct delta. It is bitwise reproducible against itself,
but intentionally does not claim bitwise alignment with the stock PyTorch
floating-point path.

Contract: q is RAW (unscaled); the kernels apply 1/sqrt(D) internally.
The public wrapper validates inputs, pads supported dimensions, packs Q/K/V,
calls autograd, then removes padding. Q/K/V use [*, O, H, N, D]; the forward
kernel writes [*, O, N, H, D] directly, and the wrapper returns it contiguously.
Here * is the batch prefix, O the outer index, H the attention heads, N the
query/key sequence, and D the channels per head.

Reading order: start at ``turbopairformer_triangle_attention`` for preparation
and output layout, then ``_TurboPairFormerTriangleAttention`` for the tensors
saved between forward and backward. ``_PackQKV`` handles the fused input copy.
"""

from __future__ import annotations

from typing import Any

import torch

from openfold3.core.utils.distributed_contract import (
    turbopairformer_distributed_execution_flag,
    validate_turbopairformer_attention_policy,
)

_PAD_HEAD_LANES = (16, 32, 64)  # Compiled storage widths; public D <= 64.
_CAPABILITY_CACHE: dict = {}
_MAX_GRID_X = 2_147_483_647
_MAX_GRID_YZ = 65_535
_SMALLEST_QUERY_BLOCK = 32
# LCM of every kernel tile axis (forward 64x64 blocks, dK/dV 64-key blocks,
# dQ/dBias 32-row query blocks and 4-outer leaves): padded sequences hit
# the exact-tiles lanes in all three kernels.
_PAD_QUANTUM = 64
_MAX_BROADCAST_PREFIX_RANK = 16

_FORWARD: Any = None
_DELTA: Any = None
_INVOCATIONS = 0
# Support verdicts depend on shape, dtype, device, and requires_grad.
# Strides are not part of this key: the facade packs supported strided inputs.
_SUPPORT_CACHE: dict = {}
_SUPPORT_CACHE_LIMIT = 4096


def invocation_count() -> int:
    """Number of TurboPairFormer facade calls (routing regression lock)."""
    return _INVOCATIONS


def _forward_extension() -> Any:
    """Load the forward/packing extension on first use and retain the module."""
    global _FORWARD
    if _FORWARD is None:
        from openfold3.core.kernels.cuda.turbopairformer_triangle_forward import (
            load_extension,
        )

        _FORWARD = load_extension()
    return _FORWARD


def _delta_extension() -> Any:
    """Load the delta extension when backward first needs it."""
    global _DELTA
    if _DELTA is None:
        from openfold3.core.kernels.cuda.turbopairformer_triangle_delta import (
            load_extension,
        )

        _DELTA = load_extension()
    return _DELTA


def is_turbopairformer_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    """Return a cached support verdict without reading tensor values.

    Unsupported cases return ``(False, reason)``; the public wrapper converts
    that verdict into an exception before preparing or launching the kernels.
    """

    try:
        key = tuple(
            (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.requires_grad)
            for tensor in (q, k, v, res_mask, pair_bias)
        )
    except AttributeError:
        key = None
    if key is not None:
        cached = _SUPPORT_CACHE.get(key)
        if cached is not None:
            return cached
    verdict = _is_turbopairformer_supported_uncached(q, k, v, res_mask, pair_bias)
    if key is not None:
        if len(_SUPPORT_CACHE) >= _SUPPORT_CACHE_LIMIT:
            _SUPPORT_CACHE.clear()
        _SUPPORT_CACHE[key] = verdict
    return verdict


def _is_turbopairformer_supported_uncached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    """Check hardware, dimensions, dtypes, and broadcasting in that order."""

    # Check tensor metadata and the Hopper device before interpreting shapes.
    named_inputs = {
        "q": q,
        "k": k,
        "v": v,
        "res_mask": res_mask,
        "pair_bias": pair_bias,
    }
    if any(not isinstance(tensor, torch.Tensor) for tensor in named_inputs.values()):
        return False, "all inputs must be torch.Tensor instances"
    if not q.is_cuda:
        return False, "q/k/v, res_mask, and pair_bias must be CUDA tensors"
    capability = _CAPABILITY_CACHE.get(q.device.index)
    if capability is None:
        capability = torch.cuda.get_device_capability(q.device)
        _CAPABILITY_CACHE[q.device.index] = capability
    if capability != (9, 0):
        return False, "TurboPairFormer Triangle Attention requires Hopper sm_90a"
    if q.ndim < 5:
        return False, "q must have shape [*B,O,H,N,D]"
    # Flatten the batch prefix only for launch-bound checks; O and N may differ.
    prefix = tuple(q.shape[:-4])
    outer, heads, sequence, head_dim = q.shape[-4:]
    if not prefix or any(dimension <= 0 for dimension in prefix):
        return False, "requires a positive batch prefix"
    batch = 1
    for dimension in prefix:
        batch *= int(dimension)
    if outer <= 0 or sequence <= 0:
        return False, "requires positive outer and key sequence dimensions"
    query_blocks = (int(sequence) + _SMALLEST_QUERY_BLOCK - 1) // (
        _SMALLEST_QUERY_BLOCK
    )
    if query_blocks > _MAX_GRID_YZ:
        return False, "sequence requires too many CUDA query blocks"
    if heads > _MAX_GRID_YZ:
        return False, "head count exceeds Hopper CUDA grid-y limit (65535)"
    if batch * sequence > _MAX_GRID_X or batch * outer > _MAX_GRID_X:
        return False, "flattened batch*outer/sequence exceeds CUDA grid-x limit"
    if heads < 1 or head_dim < 1 or head_dim > _PAD_HEAD_LANES[-1]:
        return False, "requires H>=1 and D<=64 (pads to lanes {16,32,64})"
    # Q/K/V share a shape; mask and bias have distinct broadcastable layouts.
    for name, tensor in {"k": k, "v": v}.items():
        if tensor.shape != q.shape:
            return False, f"{name} shape must match q"
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        return False, "q/k/v must be BF16"
    if res_mask.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False, "res_mask must be a floating additive mask"
    if tuple(res_mask.shape[-4:]) != (outer, 1, 1, sequence):
        return False, "res_mask tail must be (N,1,1,N)"
    if tuple(pair_bias.shape[-4:]) != (1, heads, sequence, sequence):
        return False, "pair_bias tail must be (1,H,N,N)"
    if pair_bias.dtype not in (torch.bfloat16, torch.float32):
        return False, "pair_bias must be BF16 or FP32"
    devices = {tensor.device for tensor in (q, k, v, res_mask, pair_bias)}
    if len(devices) != 1:
        return False, "q/k/v, res_mask, and pair_bias must share one device"
    for name, tensor in {"res_mask": res_mask, "pair_bias": pair_bias}.items():
        try:
            broadcast_prefix = torch.broadcast_shapes(prefix, tuple(tensor.shape[:-4]))
        except RuntimeError:
            return False, f"{name} prefix is not broadcastable to q prefix"
        if tuple(broadcast_prefix) != prefix:
            return False, f"{name} prefix must broadcast exactly to q prefix"
    bias_prefix = tuple(pair_bias.shape[:-4])
    aligned_bias_prefix = (1,) * (len(prefix) - len(bias_prefix)) + bias_prefix
    bias_requires_broadcast = any(
        source != target
        for source, target in zip(aligned_bias_prefix, prefix, strict=True)
    )
    if bias_requires_broadcast and len(prefix) > _MAX_BROADCAST_PREFIX_RANK:
        return (
            False,
            "pair_bias broadcast supports at most 16 target-prefix dimensions",
        )
    if res_mask.requires_grad:
        return False, "res_mask must not require gradients"
    return True, "supported"


class _PackQKV(torch.autograd.Function):
    """Fused q/k/v gather into kernel layout.

    Forward reads the [.., N, H, D]-major producer views in one launch and
    emits three contiguous [.., H, N, D] tensors. The transpose before
    flattening is essential: flattening the public [.., H, N, D] view directly
    would materialize logical H-major order and then permute it a second time
    in CUDA. Backward is the same logical identity as ``contiguous()``; the
    producer transpose owns the inverse layout mapping.
    """

    @staticmethod
    def forward(ctx, q, k, v, bo, heads, seq, dim):
        ctx.view_shapes = (q.shape, k.shape, v.shape)
        packed = _forward_extension().turbopairformer_pack_qkv(
            q.transpose(-2, -3).reshape(-1),
            k.transpose(-2, -3).reshape(-1),
            v.transpose(-2, -3).reshape(-1),
            bo,
            heads,
            seq,
            dim,
        )
        return packed[0], packed[1], packed[2]

    @staticmethod
    def backward(ctx, dq, dk, dv):
        # Same contract as .contiguous(): identity pass-through. The upstream
        # permute's own backward inverts the layout symbolically.
        q_shape, k_shape, v_shape = ctx.view_shapes
        return (
            dq.reshape(q_shape),
            dk.reshape(k_shape),
            dv.reshape(v_shape),
            None,
            None,
            None,
            None,
        )


class _TurboPairFormerTriangleAttention(torch.autograd.Function):
    """Run the CUDA kernels and own the saved state for their backward pass.

    Kernel inputs have flattened batches and padded, contiguous storage.
    ``expanded_bias`` is detached; ``source_bias`` keeps the autograd edge and
    unexpanded bias shape, after any dtype conversion or sequence padding.
    """

    @staticmethod
    def forward(  # noqa: D102
        ctx, q, k, v, mask, expanded_bias, source_bias, target_prefix, bias_prefix
    ):
        with torch.cuda.device(q.device):
            sequence = q.shape[-2]
            # The kernel writes the PUBLIC [B, O, N, H, D] layout directly;
            # no transpose/contiguous pass on either side of autograd.
            output = torch.empty(
                (q.shape[0], q.shape[1], q.shape[3], q.shape[2], q.shape[4]),
                device=q.device,
                dtype=q.dtype,
            )
            logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
            shadow_residual = None
            if turbopairformer_distributed_execution_flag("OPENFOLD3_SR_SHADOW"):
                # Save the output's rounding residual in BF16. Backward uses
                # it alongside output and d_output when calculating delta.
                shadow_residual = torch.empty_like(output)
                _forward_extension().turbopairformer_triangle_forward_shadow_out(
                    q, k, v, mask, expanded_bias, output, logsumexp, shadow_residual
                )
            else:
                _forward_extension().turbopairformer_triangle_forward_out(
                    q, k, v, mask, expanded_bias, output, logsumexp
                )
        if shadow_residual is None:
            ctx.save_for_backward(q, k, v, mask, expanded_bias, output, logsumexp)
        else:
            ctx.save_for_backward(
                q, k, v, mask, expanded_bias, output, logsumexp, shadow_residual
            )
        ctx.has_shadow = shadow_residual is not None
        # The saved shadow fixes the delta path for this autograd context;
        # backward never rereads the environment to choose numerical behavior.
        ctx.sequence = int(sequence)
        ctx.target_prefix = tuple(int(value) for value in target_prefix)
        ctx.bias_prefix = tuple(int(value) for value in bias_prefix)
        ctx.bias_shape = tuple(source_bias.shape)
        return output

    @staticmethod
    def backward(ctx, d_output):  # noqa: D102
        # Import backward helpers when backward is first needed. These local
        # imports reuse Python's module cache; their CUDA extensions load on
        # the first kernel call.
        from openfold3.core.kernels.cuda.turbopairformer_triangle_dq_dbias import (
            turbopairformer_reduce_broadcast_dbias,
            turbopairformer_triangle_sr_backward,
        )

        saved_tensors = ctx.saved_tensors
        q, k, v, mask, expanded_bias, output, logsumexp = saved_tensors[:7]
        shadow_residual = (
            saved_tensors[7] if getattr(ctx, "has_shadow", False) else None
        )
        with torch.cuda.device(q.device):
            # First form delta, the per-query reduction of d_output * output.
            # The optional residual corrects for the rounded BF16 output.
            d_output = d_output.contiguous()
            delta = torch.empty_like(logsumexp)
            if shadow_residual is not None:
                _delta_extension().turbopairformer_triangle_delta_shadow_out(
                    d_output, output, shadow_residual, delta
                )
            else:
                _delta_extension().turbopairformer_triangle_delta_out(
                    d_output, output, delta
                )
            # One reconstruction of each P/dS tile supplies all four
            # gradients. Shadow changes delta accuracy, not reconstruction count.
            d_q, d_k, d_v, per_batch_dbias = turbopairformer_triangle_sr_backward(
                q, k, v, d_output, logsumexp, delta, mask, expanded_bias
            )
            if ctx.bias_prefix == ctx.target_prefix:
                # No batch-prefix reduction is needed when bias was not
                # broadcast. Both branches return its BF16 source shape.
                d_bias = per_batch_dbias.reshape(ctx.bias_shape).to(
                    dtype=torch.bfloat16
                )
            else:
                d_bias = turbopairformer_reduce_broadcast_dbias(
                    per_batch_dbias,
                    target_prefix=ctx.target_prefix,
                    bias_prefix=ctx.bias_prefix,
                )
            if tuple(d_bias.shape) != ctx.bias_shape:
                raise RuntimeError("fixed dBias reduction returned the wrong shape")
        return d_q, d_k, d_v, None, None, d_bias, None, None


def turbopairformer_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Validate, prepare, and run Triangle Attention, then remove padding.

    Q/K/V have shape ``[*, O, H, N, D]`` with unscaled Q; output has shape
    ``[*, O, N, H, D]``. Backward reconstructs each probability tile once for
    all four gradients. The mask is additive and must not require gradients.
    """

    # 1. Validate before allocating padded or packed copies.
    validate_turbopairformer_attention_policy()
    supported, reason = is_turbopairformer_supported(q, k, v, res_mask, pair_bias)
    if not supported:
        raise ValueError(f"unsupported TurboPairFormer case: {reason}")
    global _INVOCATIONS
    _INVOCATIONS += 1
    # 2. Convert bias and pad the head/sequence dimensions used by the kernels.
    # Hopper kernels store the bias tile as BF16. Accept the model's native
    # FP32 pair-bias contract and make the precision boundary explicit here;
    # autograd owns the inverse cast, so an FP32 source receives an FP32
    # gradient.
    if pair_bias.dtype == torch.float32:
        pair_bias = pair_bias.to(torch.bfloat16)
    batch_prefix = q.shape[:-4]
    outer = int(q.shape[-4])
    sequence = int(q.shape[-2])
    head_dim = int(q.shape[-1])

    # Head dimensions divisible by 8 run without tensor-level padding; the
    # kernels zero-extend them to a compiled storage lane during staging.
    # Other dimensions are zero-padded to the next multiple of 8, with Q
    # rescaled to preserve the intended softmax scale in exact arithmetic.
    # That BF16 rescaling can introduce rounding. Autograd tracks the
    # rescaling, padding and output slicing.
    # Only sequence is padded to a multiple of 64 below; outer is unchanged.
    # Padded keys receive -inf mask values and padded output rows are removed.
    if head_dim % 8 == 0:
        padded_head_dim = head_dim
    else:
        padded_head_dim = head_dim + (-head_dim) % 8
    head_padding = padded_head_dim - head_dim
    if head_padding:
        # Compensate for the kernel's 1/sqrt(padded_head_dim) scale to preserve
        # 1/sqrt(head_dim) in exact arithmetic, before BF16 rounding of Q.
        q = q * float(padded_head_dim / head_dim) ** 0.5
        q = torch.nn.functional.pad(q, (0, head_padding))
        k = torch.nn.functional.pad(k, (0, head_padding))
        v = torch.nn.functional.pad(v, (0, head_padding))
    # outer is a pure plane count for every kernel (decoupled from the key
    # sequence); only the key/query sequence needs tile-quantum padding.
    sequence_padding = (-sequence) % _PAD_QUANTUM
    if sequence_padding:
        pad_qkv = (0, 0, 0, sequence_padding)
        q = torch.nn.functional.pad(q, pad_qkv)
        k = torch.nn.functional.pad(k, pad_qkv)
        v = torch.nn.functional.pad(v, pad_qkv)
        pair_bias = torch.nn.functional.pad(
            pair_bias, (0, sequence_padding, 0, sequence_padding)
        )
        padded_sequence = sequence + sequence_padding
        padded_mask = res_mask.new_full(
            tuple(res_mask.shape[:-4]) + (outer, 1, 1, padded_sequence), float("-inf")
        )
        padded_mask[..., :outer, :, :, :sequence] = res_mask
        res_mask = padded_mask

    # 3. Flatten the batch prefix and pack each input into its kernel layout.
    kernel_tail_shape = tuple(q.shape[-4:])
    batch = 1
    for batch_dimension in batch_prefix:
        batch *= int(batch_dimension)

    def _is_swapped_hn_view(tensor):
        # producer layout [.., N, H, D] viewed as [.., H, N, D]: one fused
        # gather replaces three separate copy kernels.
        outer_count, num_heads, sequence_length, head_channels = tensor.shape[-4:]
        strides = tensor.stride()
        return (
            tensor.dtype == torch.bfloat16
            and head_channels % 2 == 0
            and strides[-1] == 1
            and strides[-2] == num_heads * head_channels
            and strides[-3] == head_channels
        )

    if _is_swapped_hn_view(q) and _is_swapped_hn_view(k) and _is_swapped_hn_view(v):
        packed_outer, packed_heads, packed_sequence, packed_head_dim = kernel_tail_shape
        q_flat, k_flat, v_flat = _PackQKV.apply(
            q,
            k,
            v,
            batch * packed_outer,
            packed_heads,
            packed_sequence,
            packed_head_dim,
        )
        q_flat = q_flat.view((batch,) + kernel_tail_shape)
        k_flat = k_flat.view((batch,) + kernel_tail_shape)
        v_flat = v_flat.view((batch,) + kernel_tail_shape)
    else:
        q_flat = q.reshape((batch,) + kernel_tail_shape).contiguous()
        k_flat = k.reshape((batch,) + kernel_tail_shape).contiguous()
        v_flat = v.reshape((batch,) + kernel_tail_shape).contiguous()
    # Materialize the mask and bias over the flattened batch prefix.
    mask_flat = (
        res_mask.expand(batch_prefix + tuple(res_mask.shape[-4:]))
        .reshape((batch,) + tuple(res_mask.shape[-4:]))
        .contiguous()
        .float()
    )
    # Detach this expanded copy: custom backward returns the bias gradient
    # through pair_bias below, with the CUDA extension's fixed reduction order.
    bias_flat = (
        pair_bias.expand(batch_prefix + tuple(pair_bias.shape[-4:]))
        .reshape((batch,) + tuple(pair_bias.shape[-4:]))
        .contiguous()
        .detach()
    )
    # 4. Run custom autograd, then crop padding and restore the batch prefix.
    flat_batch_output = _TurboPairFormerTriangleAttention.apply(
        q_flat,
        k_flat,
        v_flat,
        mask_flat,
        bias_flat,
        pair_bias,
        tuple(batch_prefix),
        tuple(pair_bias.shape[:-4]),
    )
    if sequence_padding:
        flat_batch_output = flat_batch_output[:, :, :sequence, :, :]
    if head_padding:
        flat_batch_output = flat_batch_output[..., :head_dim]
    output = (
        flat_batch_output
        if flat_batch_output.is_contiguous()
        else flat_batch_output.contiguous()
    )
    return output.reshape(batch_prefix + output.shape[1:])


__all__ = [
    "invocation_count",
    "is_turbopairformer_supported",
    "turbopairformer_triangle_attention",
]
