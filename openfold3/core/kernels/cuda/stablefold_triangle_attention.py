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

_SUPPORTED_HEAD_DIMS = frozenset((16, 32))
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
    """Metadata-only support check for the StableFold path."""

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
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return False, "StableFold Triangle Attention requires Hopper sm_90"
    if q.ndim < 5:
        return False, "q must have shape [*B,O,H,N,D]"
    prefix = tuple(q.shape[:-4])
    outer, heads, sequence, head_dim = q.shape[-4:]
    if not prefix or any(dimension <= 0 for dimension in prefix):
        return False, "requires a positive batch prefix"
    batch = 1
    for dimension in prefix:
        batch *= int(dimension)
    if outer != sequence or sequence <= 0:
        return False, "requires equal positive outer/key sequence dimensions"
    query_blocks = (int(sequence) + _SMALLEST_QUERY_BLOCK - 1) // (
        _SMALLEST_QUERY_BLOCK
    )
    if query_blocks > _MAX_GRID_YZ:
        return False, "sequence requires too many CUDA query blocks"
    if heads > _MAX_GRID_YZ:
        return False, "head count exceeds Hopper CUDA grid-y limit (65535)"
    if batch * sequence > _MAX_GRID_X:
        return False, "flattened batch*sequence exceeds CUDA grid-x limit"
    if heads < 1 or head_dim not in _SUPPORTED_HEAD_DIMS:
        return False, "requires H>=1 and production D in {16,32}"
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
    if pair_bias.dtype != torch.bfloat16:
        return False, "pair_bias must be BF16"
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
            out = torch.empty_like(q)
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
        from openfold3.core.kernels.cuda.stablefold_triangle_dq_dbias import (
            stablefold_reduce_broadcast_dbias,
            stablefold_triangle_dq_dbias,
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
            from openfold3.core.kernels.cuda.stablefold_triangle_dkdv import (
                stablefold_triangle_dkdv as dkdv,
            )

            # Both head dims recover the mathematically consistent
            # delta_true = sum_k P_k dP_k in the dK/dV pass and feed it to
            # the dQ/dBias recompute: this removes the coherent row-constant
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
    prefix = q.shape[:-4]
    sequence = int(q.shape[-2])

    # Pad-to-tile contract: sequences are padded to the kernel tile quantum
    # so the single exact-tiles lane (no tail guards) serves every length.
    # Padded key columns carry a -inf mask, so every real-region value and
    # gradient is bit-identical to an unpadded computation; padded rows are
    # sliced away below and their gradients are exact zeros. F.pad/slice are
    # autograd-native, so q/k/v/bias gradients flow back unpadded.
    padding = (-sequence) % _PAD_QUANTUM
    if padding:
        pad_qkv = (0, 0, 0, padding, 0, 0, 0, padding)
        q = torch.nn.functional.pad(q, pad_qkv)
        k = torch.nn.functional.pad(k, pad_qkv)
        v = torch.nn.functional.pad(v, pad_qkv)
        pair_bias = torch.nn.functional.pad(
            pair_bias, (0, padding, 0, padding)
        )
        padded = sequence + padding
        padded_mask = res_mask.new_full(
            tuple(res_mask.shape[:-4]) + (padded, 1, 1, padded),
            float("-inf"),
        )
        padded_mask[..., :sequence, :, :, :sequence] = res_mask
        res_mask = padded_mask

    shape = tuple(q.shape[-4:])
    batch = 1
    for dim in prefix:
        batch *= int(dim)

    # Flatten every positive prefix into one CUDA batch. Bias broadcasting is
    # materialized exactly once; its backward reduction is owned by the CUDA
    # extension and therefore has one fixed order rather than ExpandBackward's
    # framework-dependent tree.
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
        output_internal = output_internal[:, :sequence, :, :sequence, :]
    output = output_internal.transpose(-2, -3).contiguous()
    return output.reshape(prefix + output.shape[1:])


__all__ = [
    "invocation_count",
    "is_stablefold_supported",
    "stablefold_triangle_attention",
]
