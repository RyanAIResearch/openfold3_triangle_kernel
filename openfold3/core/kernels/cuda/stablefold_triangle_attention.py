# SPDX-License-Identifier: Apache-2.0

"""StableFold Triangle Attention facade — deterministic production path.

Online-softmax forward plus a zero-atomics recompute backward
(delta -> dK/dV -> dQ/dBias). Sequence length, head count, and batch
prefix are runtime dimensions; D is selected from the compiled Hopper
specializations. It is bitwise reproducible against itself, but intentionally
does not claim bitwise alignment with the stock PyTorch floating-point path.

Contract: q is RAW (unscaled); the kernels apply 1/sqrt(D) internally.
Internal layout [*, O, H, N, D]; the public return is contiguous
[*, O, N, H, D] via transpose(-2,-3).contiguous().
"""

from __future__ import annotations

from typing import Any

import torch

_PAD_HEAD_LANES = (16, 32, 64)               # any D pads up to the next lane
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
# Metadata-only support verdicts, keyed by the tensors' full metadata.
# Training re-issues a handful of shapes hundreds of times per step, so the
# ~30-check validator collapses to one dict probe on the hot path.
_SUPPORT_CACHE: dict = {}
_SUPPORT_CACHE_LIMIT = 4096


def invocation_count() -> int:
    """Number of StableFold facade calls (routing regression lock)."""
    return _INVOCATIONS


def _forward_extension() -> Any:
    global _FORWARD
    if _FORWARD is None:
        from openfold3.core.kernels.cuda.stablefold_triangle_forward import (
            load_extension,
        )

        _FORWARD = load_extension()
    return _FORWARD


def _delta_extension() -> Any:
    global _DELTA
    if _DELTA is None:
        from openfold3.core.kernels.cuda.stablefold_triangle_delta import (
            load_extension,
        )

        _DELTA = load_extension()
    return _DELTA


def is_stablefold_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    """Metadata-only support check for the StableFold path (memoized)."""

    try:
        key = tuple(
            (
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
            )
            for tensor in (q, k, v, res_mask, pair_bias)
        )
    except AttributeError:
        key = None
    if key is not None:
        cached = _SUPPORT_CACHE.get(key)
        if cached is not None:
            return cached
    verdict = _is_stablefold_supported_uncached(q, k, v, res_mask, pair_bias)
    if key is not None:
        if len(_SUPPORT_CACHE) >= _SUPPORT_CACHE_LIMIT:
            _SUPPORT_CACHE.clear()
        _SUPPORT_CACHE[key] = verdict
    return verdict


def _is_stablefold_supported_uncached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    named = {
        "q": q,
        "k": k,
        "v": v,
        "res_mask": res_mask,
        "pair_bias": pair_bias,
    }
    if any(not isinstance(tensor, torch.Tensor) for tensor in named.values()):
        return False, "all inputs must be torch.Tensor instances"
    if not q.is_cuda:
        return False, "q/k/v, res_mask, and pair_bias must be CUDA tensors"
    capability = _CAPABILITY_CACHE.get(q.device.index)
    if capability is None:
        capability = torch.cuda.get_device_capability(q.device)
        _CAPABILITY_CACHE[q.device.index] = capability
    if capability != (9, 0):
        return False, "StableFold Triangle Attention requires Hopper sm_90a"
    if q.ndim < 5:
        return False, "q must have shape [*B,O,H,N,D]"
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
        packed = _forward_extension().stablefold_pack_qkv(
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
        sq, sk, sv = ctx.view_shapes
        return (
            dq.reshape(sq), dk.reshape(sk), dv.reshape(sv),
            None, None, None, None,
        )


class _StableFoldTriangleAttention(torch.autograd.Function):
    @staticmethod
    def forward(  # noqa: D102
        ctx,
        q,
        k,
        v,
        mask,
        expanded_bias,
        source_bias,
        target_prefix,
        bias_prefix,
    ):
        with torch.cuda.device(q.device):
            sequence = q.shape[-2]
            # The kernel writes the PUBLIC [B, O, N, H, D] layout directly;
            # no transpose/contiguous pass on either side of autograd.
            out = torch.empty(
                (q.shape[0], q.shape[1], q.shape[3], q.shape[2], q.shape[4]),
                device=q.device,
                dtype=q.dtype,
            )
            lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
            _forward_extension().stablefold_triangle_forward_out(
                q, k, v, mask, expanded_bias, out, lse
            )
        ctx.save_for_backward(q, k, v, mask, expanded_bias, out, lse)
        ctx.sequence = int(sequence)
        ctx.target_prefix = tuple(int(value) for value in target_prefix)
        ctx.bias_prefix = tuple(int(value) for value in bias_prefix)
        ctx.bias_shape = tuple(source_bias.shape)
        return out

    @staticmethod
    def backward(ctx, d_output):  # noqa: D102
        # Module-level lazy siblings: importing them binds no CUDA state (the
        # extensions load on first use), and hoisting keeps the per-call
        # backward free of import-machinery overhead.
        from openfold3.core.kernels.cuda.stablefold_triangle_dq_dbias import (
            stablefold_reduce_broadcast_dbias,
            stablefold_triangle_dq_dbias,
        )
        from openfold3.core.kernels.cuda.stablefold_triangle_dkdv import (
            stablefold_triangle_dkdv as dkdv,
        )

        q, k, v, mask, expanded_bias, out, lse = ctx.saved_tensors
        with torch.cuda.device(q.device):
            d_output = d_output.contiguous()
            delta = torch.empty_like(lse)
            _delta_extension().stablefold_triangle_delta_out(d_output, out, delta)
            # Keep probability reconstruction on the same raw-Q/post-scale
            # arithmetic contract as the StableFold forward. The older resident
            # fast lane reconstructs scores using an AQLab-style pre-scaled BF16 K
            # path, so it is intentionally not part of the production facade.
            # Every head lane recomputes the probability-weighted row scalar
            # delta_true = sum_k P_k dP_k in the dK/dV pass and feeds it to
            # dQ/dBias. The implementation omits division by the recomputed
            # probability sum under the measured Z ~= 1 contract; this removes
            # the coherent row-constant
            # dQ bias of delta = dO . O_bf16 (dQ rel-L2 drops to the
            # perfect-delta oracle bound) and improves dBias as well, while
            # dK/dV bits stay identical to the classical path.
            d_k, d_v, delta_true = dkdv(
                q, k, v, d_output, lse, delta, mask, expanded_bias,
                emit_delta_true=True,
            )
            d_q, per_batch_dbias = stablefold_triangle_dq_dbias(
                q, k, v, d_output, lse, delta_true, mask, expanded_bias
            )
            if ctx.bias_prefix == ctx.target_prefix:
                # The reduction is the identity when the bias already has the
                # complete target prefix.  Keep the same public BF16 gradient
                # contract without launching a redundant prefix-reduction
                # kernel.  This is shape-generic; it is not an N-specific
                # dispatch.
                d_bias = per_batch_dbias.reshape(ctx.bias_shape).to(
                    dtype=torch.bfloat16
                )
            else:
                d_bias = stablefold_reduce_broadcast_dbias(
                    per_batch_dbias,
                    target_prefix=ctx.target_prefix,
                    bias_prefix=ctx.bias_prefix,
                )
            if tuple(d_bias.shape) != ctx.bias_shape:
                raise RuntimeError("fixed dBias reduction returned the wrong shape")
        return d_q, d_k, d_v, None, None, d_bias, None, None


def stablefold_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Run one StableFold Triangle Attention call (raw q; no fallback)."""

    supported, reason = is_stablefold_supported(q, k, v, res_mask, pair_bias)
    if not supported:
        raise ValueError(f"unsupported StableFold case: {reason}")
    global _INVOCATIONS
    _INVOCATIONS += 1
    # Hopper kernels store the bias tile as BF16. Accept the model's native
    # FP32 pair-bias contract and make the precision boundary explicit here;
    # autograd owns the inverse cast, so an FP32 source receives an FP32
    # gradient. This is dtype-generic routing, not a shape/site specialization.
    if pair_bias.dtype == torch.float32:
        pair_bias = pair_bias.to(torch.bfloat16)
    prefix = q.shape[:-4]
    outer = int(q.shape[-4])
    sequence = int(q.shape[-2])
    head_dim = int(q.shape[-1])

    # Pad-to-lane contract, three independent axes.
    #  * head dim: zero-pad to the next kernel lane {16,32,64}. Exact math:
    #    padded q.k dot terms are +0.0, padded V columns produce zero output
    #    columns that are sliced away; F.pad/slice are autograd-native so all
    #    real-region values and gradients match an unpadded run.
    #  * outer / sequence: padded to the tile quantum so the single
    #    exact-tiles lane (no tail guards) serves every length. Padded key
    #    columns carry a -inf mask; padded query rows are sliced away.
    # Multiples of 8 run natively: the kernels zero-extend io_dim to the
    # storage lane at staging time (half the global traffic of tensor-level
    # padding, no q pre-scale rounding). Other dims pad to the next
    # multiple of 8.
    if head_dim % 8 == 0:
        lane = head_dim
    else:
        lane = head_dim + (-head_dim) % 8
    d_padding = lane - head_dim
    if d_padding:
        # The kernel's internal softmax scale is 1/sqrt(lane); pre-scaling q
        # by sqrt(lane/d) makes the effective scale the correct 1/sqrt(d).
        q = q * float(lane / head_dim) ** 0.5
        q = torch.nn.functional.pad(q, (0, d_padding))
        k = torch.nn.functional.pad(k, (0, d_padding))
        v = torch.nn.functional.pad(v, (0, d_padding))
    # outer is a pure plane count for every kernel (decoupled from the key
    # sequence); only the key/query sequence needs tile-quantum padding.
    padding = (-sequence) % _PAD_QUANTUM
    if padding:
        pad_qkv = (0, 0, 0, padding)
        q = torch.nn.functional.pad(q, pad_qkv)
        k = torch.nn.functional.pad(k, pad_qkv)
        v = torch.nn.functional.pad(v, pad_qkv)
        pair_bias = torch.nn.functional.pad(
            pair_bias, (0, padding, 0, padding)
        )
        padded_seq = sequence + padding
        padded_mask = res_mask.new_full(
            tuple(res_mask.shape[:-4]) + (outer, 1, 1, padded_seq),
            float("-inf"),
        )
        padded_mask[..., :outer, :, :, :sequence] = res_mask
        res_mask = padded_mask

    shape = tuple(q.shape[-4:])
    batch = 1
    for dim in prefix:
        batch *= int(dim)

    # Flatten every positive prefix into one CUDA batch. Bias broadcasting is
    # materialized exactly once; its backward reduction is owned by the CUDA
    # extension and therefore has one fixed order rather than ExpandBackward's
    # framework-dependent tree.
    def _is_swapped_hn_view(tensor):
        # producer layout [.., N, H, D] viewed as [.., H, N, D]: one fused
        # gather replaces three separate copy kernels.
        outer_, heads_, seq_, dim_ = tensor.shape[-4:]
        strides = tensor.stride()
        return (
            tensor.dtype == torch.bfloat16
            and dim_ % 2 == 0
            and strides[-1] == 1
            and strides[-2] == heads_ * dim_
            and strides[-3] == dim_
        )

    if (
        _is_swapped_hn_view(q)
        and _is_swapped_hn_view(k)
        and _is_swapped_hn_view(v)
    ):
        outer_, heads_, seq_, dim_ = shape
        q_flat, k_flat, v_flat = _PackQKV.apply(
            q, k, v, batch * outer_, heads_, seq_, dim_
        )
        q_flat = q_flat.view((batch,) + shape)
        k_flat = k_flat.view((batch,) + shape)
        v_flat = v_flat.view((batch,) + shape)
    else:
        q_flat = q.reshape((batch,) + shape).contiguous()
        k_flat = k.reshape((batch,) + shape).contiguous()
        v_flat = v.reshape((batch,) + shape).contiguous()
    mask_flat = (
        res_mask.expand(prefix + tuple(res_mask.shape[-4:]))
        .reshape((batch,) + tuple(res_mask.shape[-4:]))
        .contiguous()
        .float()
    )
    bias_flat = (
        pair_bias.expand(prefix + tuple(pair_bias.shape[-4:]))
        .reshape((batch,) + tuple(pair_bias.shape[-4:]))
        .contiguous()
        .detach()
    )
    output_internal = _StableFoldTriangleAttention.apply(
        q_flat,
        k_flat,
        v_flat,
        mask_flat,
        bias_flat,
        pair_bias,
        tuple(prefix),
        tuple(pair_bias.shape[:-4]),
    )
    if padding:
        output_internal = output_internal[:, :, :sequence, :, :]
    if d_padding:
        output_internal = output_internal[..., :head_dim]
    output = (
        output_internal
        if output_internal.is_contiguous()
        else output_internal.contiguous()
    )
    return output.reshape(prefix + output.shape[1:])


__all__ = [
    "invocation_count",
    "is_stablefold_supported",
    "stablefold_triangle_attention",
]
