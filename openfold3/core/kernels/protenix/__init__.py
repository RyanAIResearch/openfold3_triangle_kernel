# SPDX-License-Identifier: Apache-2.0
"""Optional Protenix attention with the OpenFold host layout and Q scaling."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

from openfold3.core.kernels.protenix._layout import _normalize_inputs

if TYPE_CHECKING:
    import torch

__all__ = ["attention_scaled"]


@lru_cache(maxsize=1)
def _native_function() -> Any:
    try:
        module = import_module("openfold3.core.kernels.protenix._vendor.op")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "triton"}:
            raise ImportError(
                "Protenix attention requires CUDA PyTorch and Triton. Install the "
                "project's GPU dependencies (including triton) before selecting "
                "use_protenix_triangle_attention."
            ) from exc
        raise
    return module.TriAttentionFunction


def _validate_inputs(q: Any, k: Any, v: Any, mask: Any, bias: Any) -> None:
    import torch

    tensors = {"q": q, "k": k, "v": v, "mask": mask, "bias": bias}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Protenix {name} must be a torch.Tensor")
    if q.ndim not in (4, 5, 6):
        raise ValueError("Protenix Q must have 4, 5 or 6 dimensions")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Protenix Q, K and V must have identical shapes")
    if any(dimension <= 0 for dimension in q.shape):
        raise ValueError("Protenix attention requires nonempty Q, K and V")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, bias)):
        raise TypeError("Protenix Q, K, V and pair bias must all be bfloat16")
    if mask.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Protenix additive mask must be bfloat16 or float32")
    if mask.requires_grad:
        raise ValueError("Protenix does not support a trainable additive mask")
    head_dim = q.shape[-1]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("Protenix head dimension must be a power of two >= 16")
    if any(t.device != q.device for t in tensors.values()):
        raise ValueError("Protenix inputs must all be on the same CUDA device")
    if q.device.type != "cuda":
        raise ValueError("Protenix attention requires CUDA tensors")


def _contiguous_dout(grad: torch.Tensor) -> torch.Tensor:
    return grad.contiguous()


def attention_scaled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run the pinned Protenix kernel with Q already scaled by ``1/sqrt(D)``.

    Q/K/V use ``[..., rows, heads, sequence, D]`` and must be CUDA BF16.
    The non-trainable additive mask broadcasts to ``[B, rows, 1, 1, S]``;
    BF16 pair bias broadcasts to ``[B, 1, H, S, S]``. Unbatched 4D,
    batched 5D, and template-batched 6D inputs retain the experiment's
    layout mapping. The output is ``[..., rows, sequence, heads, D]``.

    Kernel failures propagate. No cast, second Q scaling, or fallback is
    applied. The native deterministic flag remains False, as in the frozen
    experiments; this provider does not promise deterministic gradients.
    """
    _validate_inputs(q, k, v, mask, bias)
    q, k, v, mask, bias, shape = _normalize_inputs(q, k, v, mask, bias)
    output = _native_function().apply(
        q.transpose(-2, -3),
        k.transpose(-2, -3),
        v.transpose(-2, -3),
        mask,
        bias,
        False,
    )
    if output.requires_grad:
        output.register_hook(_contiguous_dout)
    return output.reshape(*shape[:-3], shape[-2], shape[-3], shape[-1])
