# SPDX-License-Identifier: Apache-2.0

"""Streaming Triangle Attention facade — deterministic research candidate.

Online-softmax forward plus a zero-atomics recompute backward
(delta -> v2 dK/dV -> v9 dQ/dBias). Sequence length, head count, and batch
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

_SUPPORTED_HEAD_DIMS = frozenset((16, 32, 64, 128))
_MAX_GRID_YZ = 65_535

_FORWARD: Any = None
_DELTA: Any = None
_INVOCATIONS = 0


def invocation_count() -> int:
    """Number of streaming forward launches (routing regression lock)."""
    return _INVOCATIONS


def _forward_extension() -> Any:
    global _FORWARD
    if _FORWARD is None:
        from openfold3.core.kernels.cuda.streaming_triangle_forward_v1 import (
            load_extension,
        )

        _FORWARD = load_extension()
    return _FORWARD


def _delta_extension() -> Any:
    global _DELTA
    if _DELTA is None:
        from openfold3.core.kernels.cuda.streaming_triangle_delta_v1 import (
            load_extension,
        )

        _DELTA = load_extension()
    return _DELTA


def is_streaming_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> tuple[bool, str]:
    """Metadata-only support check for the streaming path."""

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
        return False, "streaming Triangle Attention requires Hopper sm_90"
    if q.ndim < 5:
        return False, "q must have shape [*B,O,H,N,D]"
    prefix = tuple(q.shape[:-4])
    outer, heads, sequence, head_dim = q.shape[-4:]
    if not prefix or any(dimension <= 0 for dimension in prefix):
        return False, "requires a positive batch prefix"
    if outer != sequence or sequence <= 0:
        return False, "requires equal positive outer/key sequence dimensions"
    if sequence > _MAX_GRID_YZ:
        return False, "sequence exceeds Hopper CUDA grid-z limit (65535)"
    if heads > _MAX_GRID_YZ:
        return False, "head count exceeds Hopper CUDA grid-y limit (65535)"
    if heads < 1 or head_dim not in _SUPPORTED_HEAD_DIMS:
        return False, "requires H>=1 and D in {16,32,64,128}"
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
    if res_mask.requires_grad:
        return False, "res_mask must not require gradients"
    return True, "supported"


class _StreamingTriangleAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, bias):  # noqa: D102
        with torch.cuda.device(q.device):
            sequence = q.shape[-2]
            out = torch.empty_like(q)
            lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
            _forward_extension().streaming_triangle_forward_out(
                q, k, v, mask, bias, out, lse
            )
        ctx.save_for_backward(q, k, v, mask, bias, out, lse)
        ctx.sequence = int(sequence)
        return out

    @staticmethod
    def backward(ctx, d_output):  # noqa: D102
        from openfold3.core.kernels.cuda.fused_triangle_dq_dbias_v9 import (
            fused_triangle_dq_dbias_v9,
        )

        q, k, v, mask, bias, out, lse = ctx.saved_tensors
        with torch.cuda.device(q.device):
            d_output = d_output.contiguous()
            delta = torch.empty_like(lse)
            _delta_extension().streaming_triangle_delta_out(d_output, out, delta)
            # Keep probability reconstruction on the same raw-Q/post-scale
            # arithmetic contract as the streaming forward.  The older resident
            # fast lane reconstructs scores using an AQLab-style pre-scaled BF16 K
            # path, so it is intentionally not part of the production facade.
            from openfold3.core.kernels.cuda.stable_triangle_dkdv_v2 import (
                stable_triangle_dkdv_v2 as dkdv,
            )

            d_k, d_v = dkdv(q, k, v, d_output, lse, delta, mask, bias)
            d_q, d_bias = fused_triangle_dq_dbias_v9(
                q, k, v, d_output, lse, delta, mask, bias
            )
        return d_q, d_k, d_v, None, d_bias.to(bias.dtype)


def streaming_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    res_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Run one streaming Triangle Attention call (raw q; no fallback)."""

    supported, reason = is_streaming_supported(q, k, v, res_mask, pair_bias)
    if not supported:
        raise ValueError(f"unsupported streaming case: {reason}")
    global _INVOCATIONS
    _INVOCATIONS += 1
    prefix = q.shape[:-4]
    shape = tuple(q.shape[-4:])
    batch = 1
    for dim in prefix:
        batch *= int(dim)

    def one_leaf(qb, kb, vb, mb, bb):
        return _StreamingTriangleAttention.apply(
            qb.reshape((1,) + shape).contiguous(),
            kb.reshape((1,) + shape).contiguous(),
            vb.reshape((1,) + shape).contiguous(),
            mb.reshape((1,) + tuple(res_mask.shape[-4:])).contiguous().float(),
            bb.reshape((1,) + tuple(pair_bias.shape[-4:])).contiguous(),
        )

    if batch == 1:
        output_internal = one_leaf(q, k, v, res_mask, pair_bias)
    else:
        # Batched prefixes run leaf-by-leaf (each leaf saturates the GPU);
        # torch.cat keeps the whole thing differentiable.
        q_flat = q.reshape((batch,) + shape)
        k_flat = k.reshape((batch,) + shape)
        v_flat = v.reshape((batch,) + shape)
        mask_flat = res_mask.expand(prefix + tuple(res_mask.shape[-4:])).reshape(
            (batch,) + tuple(res_mask.shape[-4:])
        )
        bias_flat = pair_bias.expand(prefix + tuple(pair_bias.shape[-4:])).reshape(
            (batch,) + tuple(pair_bias.shape[-4:])
        )
        output_internal = torch.cat(
            [
                one_leaf(
                    q_flat[i],
                    k_flat[i],
                    v_flat[i],
                    mask_flat[i],
                    bias_flat[i],
                )
                for i in range(batch)
            ],
            dim=0,
        )
    output = output_internal.transpose(-2, -3).contiguous()
    return output.reshape(prefix + output.shape[1:])


__all__ = [
    "invocation_count",
    "is_streaming_supported",
    "streaming_triangle_attention",
]
