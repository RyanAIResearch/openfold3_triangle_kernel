# SPDX-License-Identifier: Apache-2.0
"""TurboPairFormer triangle multiplicative update: fused-kernel CUDA implementation.

Both public calling forms validate their inputs, then enter
``_run_supported_trimul_tensors`` and the ``_trimul_fwd_v1`` custom op.
Autograd calls ``_trimul_bwd_v1`` through ``_trimul_fwd_v1_backward``.
The implementation helpers below supply both custom ops.

Dimensions: B is the flattened batch prefix, N the pair-axis length, C the
input/output channel count, and H the hidden channel count (not attention heads).

Forward has three computation stages:
  z --[trimul_in_fwd: LN(fp32 stats) + 5-way projection GEMM (mma.sync,
       persistent col-sliced CTAs, weights resident in smem) + sigmoid-gate
       split + channel-major write]--> a, b ([B*H,N,N]), raw gate
  a,b --[cuBLAS bmm (op-flag views, no copies)]--> x
  x --[trimul_out_fwd: transpose-LN + out-projection GEMM + sigmoid gate mul,
       cp.async double-buffered tile pipeline]--> out
The fused stages avoid materializing the full five-projection tensor and the
ungated output. When backward state is requested, the forward also saves
normalized inputs, gates, pair-product intermediates and layer-normalization
statistics.

Backward: trimul_gate_split_bwd recomputes proj tiles on tensor cores from
the saved z_ln with the forward accumulation order; sig_mul_bwd_out uses
the saved gated output to evaluate d_gate = dy*out*(1-sig), without
reconstructing the ungated output. Remaining GEMMs are cuBLAS; LN backwards are
the fixed-order kernels below. dW5 uses K=16384 BF16-input bmm chunks with FP32
partials under a fixed 48 MiB partial cap, followed by one globally ascending
FP32 fold.

Custom reduction kernels use fixed accumulation orders without floating-point
atomics; repeatability assumes fixed inputs, execution settings and runtime.
Contract (explicit reject, no silent specialization): c_hidden == c_z,
C in {32, 64, 96, 128}. Both public calling forms require N > 100 and check
CUDA grid, dtype and layout limits before dispatch.

The private, versioned ``torch.library`` boundary below is first-order-only.
Its backward remains an opaque custom op; higher-order differentiation is not
part of the production contract.
"""

from typing import NamedTuple

import torch

_ext = None

_WGRAD_K_CHUNK = 16384
_WGRAD_PARTIAL_CAP_BYTES = 48 * 1024 * 1024
_WGRAD_FP32_BYTES = 4


class _WgradWave(NamedTuple):
    first_chunk: int
    full_chunks: int
    tail_rows: int


class _WgradPlan(NamedTuple):
    partial_bytes_per_chunk: int
    chunks_per_wave: int
    waves: tuple[_WgradWave, ...]


def _get_ext():
    global _ext
    if _ext is None:
        from turbopairformer.trimul_cuda import (
            load_extension,
        )

        _ext = load_extension()
    return _ext


def _plan_wgrad_fixed_fold(rows: int, hidden: int, channels: int) -> _WgradPlan:
    """Plan bounded split-K partials in one global ascending chunk order."""
    if rows <= 0:
        raise ValueError(f"wgrad rows must be positive, got {rows}")
    if hidden <= 0 or channels <= 0:
        raise ValueError(
            "wgrad hidden/channels must be positive, got "
            f"hidden={hidden}, channels={channels}"
        )
    partial_bytes = 5 * hidden * channels * _WGRAD_FP32_BYTES
    if partial_bytes > _WGRAD_PARTIAL_CAP_BYTES:
        raise ValueError(
            f"one wgrad partial exceeds the fixed 48 MiB cap: {partial_bytes} bytes"
        )
    chunks_per_wave = _WGRAD_PARTIAL_CAP_BYTES // partial_bytes
    full_chunks, tail_rows = divmod(rows, _WGRAD_K_CHUNK)
    logical_chunks = full_chunks + int(tail_rows != 0)
    waves = []
    first_chunk = 0
    while first_chunk < logical_chunks:
        wave_chunks = min(chunks_per_wave, logical_chunks - first_chunk)
        wave_full_chunks = min(
            wave_chunks,
            max(0, full_chunks - first_chunk),
        )
        wave_tail_rows = tail_rows if wave_full_chunks < wave_chunks else 0
        waves.append(_WgradWave(first_chunk, wave_full_chunks, wave_tail_rows))
        first_chunk += wave_chunks
    return _WgradPlan(partial_bytes, chunks_per_wave, tuple(waves))


def _wgrad_full_chunk_views(d_proj5, z_ln, first_chunk: int, full_chunks: int):
    """Return transpose-compatible aliases for complete K=16384 chunks."""
    row_start = first_chunk * _WGRAD_K_CHUNK
    wave_rows = full_chunks * _WGRAD_K_CHUNK
    five_hidden = d_proj5.shape[1]
    channels = z_ln.shape[1]
    projection = d_proj5.narrow(0, row_start, wave_rows).view(
        full_chunks, _WGRAD_K_CHUNK, five_hidden
    )
    normalized = z_ln.narrow(0, row_start, wave_rows).view(
        full_chunks, _WGRAD_K_CHUNK, channels
    )
    return projection.transpose(1, 2), normalized


def _trimul_wgrad_fixed(ext, d_proj5, z_ln):
    """Compute dW5 from BF16 split-K inputs, FP32 partials and a fixed fold."""
    rows, five_hidden = d_proj5.shape
    z_rows, channels = z_ln.shape
    if rows != z_rows:
        raise ValueError(f"wgrad operands disagree on rows: {rows} versus {z_rows}")
    if five_hidden % 5 != 0:
        raise ValueError(
            f"wgrad projection width must be divisible by five, got {five_hidden}"
        )
    hidden = five_hidden // 5
    plan = _plan_wgrad_fixed_fold(rows, hidden, channels)
    accumulator = torch.empty(
        (five_hidden, channels),
        dtype=torch.float32,
        device=d_proj5.device,
    )
    initialize = True
    for wave in plan.waves:
        if wave.full_chunks:
            projection_t, normalized = _wgrad_full_chunk_views(
                d_proj5, z_ln, wave.first_chunk, wave.full_chunks
            )
            partials = torch.bmm(
                projection_t,
                normalized,
                out_dtype=torch.float32,
            )
            ext.trimul_wgrad_fixed_fold(partials, accumulator, initialize)
            initialize = False
            del partials, projection_t, normalized

        if wave.tail_rows:
            tail_start = (wave.first_chunk + wave.full_chunks) * _WGRAD_K_CHUNK
            projection_tail = d_proj5.new_zeros((1, _WGRAD_K_CHUNK, five_hidden))
            normalized_tail = z_ln.new_zeros((1, _WGRAD_K_CHUNK, channels))
            projection_tail[0, : wave.tail_rows].copy_(
                d_proj5.narrow(0, tail_start, wave.tail_rows)
            )
            normalized_tail[0, : wave.tail_rows].copy_(
                z_ln.narrow(0, tail_start, wave.tail_rows)
            )
            partials = torch.bmm(
                projection_tail.transpose(1, 2),
                normalized_tail,
                out_dtype=torch.float32,
            )
            ext.trimul_wgrad_fixed_fold(partials, accumulator, initialize)
            initialize = False
            del partials, projection_tail, normalized_tail

    return accumulator


def _trimul_forward_impl(
    ext,
    z,
    mask,
    w_ap,
    w_ag,
    w_bp,
    w_bg,
    w_g,
    w_z,
    g_in,
    b_in,
    g_out,
    b_out,
    outgoing,
    training,
):
    orig_shape = z.shape
    C = z.shape[-1]
    N = z.shape[-2]
    if z.shape[-3] != N:
        raise ValueError(f"expected square pair input, got {tuple(z.shape)}")
    B = z.numel() // (N * N * C)
    H = w_ap.shape[0]
    if H != C:
        raise ValueError("stable trimul requires c_hidden == c_z")
    z_bf = z.reshape(B, N, N, C).to(torch.bfloat16).contiguous()
    mask_f = (
        mask.expand(*orig_shape[:-1]).reshape(B, N, N).to(torch.float32).contiguous()
    )
    rows = B * N * N
    NN = N * N
    mask_flat = mask_f.view(-1)

    a, b, gate, mean_in, rstd_in, z_ln = ext.trimul_in_fwd(
        z_bf.view(rows, C),
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        mask_flat,
        g_in,
        b_in,
        B,
        NN,
        C,
        H,
        training,
    )
    av = a.view(B * H, N, N)
    bv = b.view(B * H, N, N)
    if outgoing:
        x = torch.bmm(av, bv.transpose(1, 2))
    else:
        x = torch.bmm(av.transpose(1, 2), bv)
    out, xlnT, mean_out, rstd_out = ext.trimul_out_fwd(
        x.view(B * H, NN), w_z, gate, g_out, b_out, B, NN, C, H, training
    )

    # Shared backward-state order: input/mask; six intermediates; four LN
    # statistics; six projection/gate weights; two LN scales; final output.
    # The custom-op setup_context below stores the same order for autograd.
    saved_tensors = (
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
    )
    dims = (B, N, C, H, outgoing)
    return out.view(orig_shape).to(z.dtype), saved_tensors, dims, orig_shape, z.dtype


def _trimul_backward_combined_impl(
    ext,
    grad_out,
    saved_tensors,
    dims,
    orig_shape,
    z_dtype,
):
    (
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
    ) = saved_tensors
    B, N, C, H, outgoing = dims
    rows = B * N * N
    NN = N * N
    go = grad_out.to(torch.bfloat16).contiguous().view(rows, C)

    # Gate slot [4H:5H) of d_proj5 and d_raw, using the saved gated output.
    d_proj5 = torch.empty(rows, 5 * H, dtype=torch.bfloat16, device=z_bf.device)
    d_raw = ext.sig_mul_bwd_out(
        out.view(rows, C), gate, go, d_proj5.view(rows, 5 * H)[:, 4 * H :], H, 5 * H
    )
    # out GEMM backward (w_z is [C, H]; no transposes materialized)
    d_xln = (d_raw @ w_z.to(torch.bfloat16)).contiguous()
    d_wz = torch.mm(d_raw.t(), xlnT.t(), out_dtype=torch.float32)
    # LN out backward (fused transpose variant, dgb folded)
    d_x, d_gout, d_bout = ext.transpose_ln_bwd(
        x.view(B * H, NN), d_xln, N, H, B, g_out, mean_out, rstd_out
    )
    # bmm backward via cublas op views
    av = a.view(B * H, N, N)
    bv = b.view(B * H, N, N)
    if outgoing:
        d_a = torch.bmm(d_x, bv)
        d_b = torch.bmm(d_x.transpose(1, 2), av)
    else:
        d_a = torch.bmm(bv, d_x.transpose(1, 2))
        d_b = torch.bmm(av, d_x)
    # recompute-proj gate-split backward -> slots [0:4H)
    ext.trimul_gate_split_bwd(
        z_ln,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        mask_f.view(-1),
        d_a.view(B * H, NN),
        d_b.view(B * H, NN),
        d_proj5,
        B,
        NN,
        C,
        H,
    )
    # in GEMM backward: W5 [5H, C] built once, only here
    w5 = torch.cat([w_ap, w_ag, w_bp, w_bg, w_g], 0).to(torch.bfloat16)
    d_zln = (d_proj5 @ w5).contiguous()
    d_w5 = _trimul_wgrad_fixed(ext, d_proj5, z_ln)
    # LN in backward
    d_z = ext.ln_bwd_dx(z_bf.view(rows, C), d_zln, rows, C, 1, g_in, mean_in, rstd_in)
    d_gin, d_bin = ext.ln_bwd_dgb(
        z_bf.view(rows, C), d_zln, rows, C, 1, C, mean_in, rstd_in
    )
    return (
        d_z.view(orig_shape).to(z_dtype),
        d_w5,
        d_wz,
        d_gin,
        d_bin,
        d_gout,
        d_bout,
    )


def _trimul_backward_impl(
    ext,
    grad_out,
    saved_tensors,
    dims,
    orig_shape,
    z_dtype,
):
    d_z, d_w5, d_wz, d_gin, d_bin, d_gout, d_bout = _trimul_backward_combined_impl(
        ext,
        grad_out,
        saved_tensors,
        dims,
        orig_shape,
        z_dtype,
    )
    H = dims[3]
    return (
        d_z,
        None,
        d_w5[0 * H : 1 * H],
        d_w5[1 * H : 2 * H],
        d_w5[2 * H : 3 * H],
        d_w5[3 * H : 4 * H],
        d_w5[4 * H : 5 * H],
        d_wz,
        d_gin,
        d_bin,
        d_gout,
        d_bout,
        None,
    )


@torch.library.custom_op(
    "turbopairformer::_trimul_fwd_v1",
    mutates_args=(),
    device_types="cuda",
)
def _trimul_fwd_v1(
    z_bf: torch.Tensor,
    mask_f: torch.Tensor,
    w_ap: torch.Tensor,
    w_ag: torch.Tensor,
    w_bp: torch.Tensor,
    w_bg: torch.Tensor,
    w_g: torch.Tensor,
    w_z: torch.Tensor,
    g_in: torch.Tensor,
    b_in: torch.Tensor,
    g_out: torch.Tensor,
    b_out: torch.Tensor,
    outgoing: bool,
    save_state: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the complete forward while exposing only fresh functional outputs."""
    ext = _get_ext()
    output, saved_tensors, _, _, _ = _trimul_forward_impl(
        ext,
        z_bf,
        mask_f,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        b_in,
        g_out,
        b_out,
        outgoing,
        save_state,
    )
    (
        _saved_z_bf,
        _saved_mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        _w_ap,
        _w_ag,
        _w_bp,
        _w_bg,
        _w_g,
        _w_z,
        _g_in,
        _g_out,
        _out,
    ) = saved_tensors
    return (
        output,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
    )


@torch.library.register_fake(_trimul_fwd_v1)
def _trimul_fwd_v1_fake(
    z_bf: torch.Tensor,
    mask_f: torch.Tensor,
    w_ap: torch.Tensor,
    w_ag: torch.Tensor,
    w_bp: torch.Tensor,
    w_bg: torch.Tensor,
    w_g: torch.Tensor,
    w_z: torch.Tensor,
    g_in: torch.Tensor,
    b_in: torch.Tensor,
    g_out: torch.Tensor,
    b_out: torch.Tensor,
    outgoing: bool,
    save_state: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, N, _, C = z_bf.shape
    H = w_ap.shape[0]
    rows = B * N * N
    nn = N * N
    saved_rows = rows if save_state else 0
    return (
        z_bf.new_empty((B, N, N, C)),
        z_bf.new_empty((saved_rows, C)),
        z_bf.new_empty((rows, H)),
        z_bf.new_empty((B * H, nn)),
        z_bf.new_empty((B * H, nn)),
        z_bf.new_empty((B * H, N, N)),
        z_bf.new_empty((H if save_state else 0, rows)),
        g_in.new_empty((rows,)),
        g_in.new_empty((rows,)),
        g_out.new_empty((saved_rows,)),
        g_out.new_empty((saved_rows,)),
    )


@torch.library.custom_op(
    "turbopairformer::_trimul_bwd_v1",
    mutates_args=(),
    device_types="cuda",
)
def _trimul_bwd_v1(
    grad_out: torch.Tensor,
    z_bf: torch.Tensor,
    mask_f: torch.Tensor,
    z_ln: torch.Tensor,
    gate: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    xlnT: torch.Tensor,
    mean_in: torch.Tensor,
    rstd_in: torch.Tensor,
    mean_out: torch.Tensor,
    rstd_out: torch.Tensor,
    w_ap: torch.Tensor,
    w_ag: torch.Tensor,
    w_bp: torch.Tensor,
    w_bg: torch.Tensor,
    w_g: torch.Tensor,
    w_z: torch.Tensor,
    g_in: torch.Tensor,
    g_out: torch.Tensor,
    out: torch.Tensor,
    outgoing: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the complete non-traceable backward as one functional custom op."""
    ext = _get_ext()
    B, N, _, C = z_bf.shape
    H = w_ap.shape[0]
    saved_tensors = (
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
    )
    return _trimul_backward_combined_impl(
        ext,
        grad_out,
        saved_tensors,
        (B, N, C, H, outgoing),
        z_bf.shape,
        z_bf.dtype,
    )


@torch.library.register_fake(_trimul_bwd_v1)
def _trimul_bwd_v1_fake(
    grad_out: torch.Tensor,
    z_bf: torch.Tensor,
    mask_f: torch.Tensor,
    z_ln: torch.Tensor,
    gate: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    xlnT: torch.Tensor,
    mean_in: torch.Tensor,
    rstd_in: torch.Tensor,
    mean_out: torch.Tensor,
    rstd_out: torch.Tensor,
    w_ap: torch.Tensor,
    w_ag: torch.Tensor,
    w_bp: torch.Tensor,
    w_bg: torch.Tensor,
    w_g: torch.Tensor,
    w_z: torch.Tensor,
    g_in: torch.Tensor,
    g_out: torch.Tensor,
    out: torch.Tensor,
    outgoing: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    H, C = w_ap.shape
    return (
        torch.empty_like(z_bf, memory_format=torch.contiguous_format),
        w_ap.new_empty((5 * H, C)),
        torch.empty_like(w_z, memory_format=torch.contiguous_format),
        torch.empty_like(g_in, memory_format=torch.contiguous_format),
        torch.empty_like(g_in, memory_format=torch.contiguous_format),
        torch.empty_like(g_out, memory_format=torch.contiguous_format),
        torch.empty_like(g_out, memory_format=torch.contiguous_format),
    )


def _trimul_fwd_v1_setup_context(ctx, inputs, output):
    """Save the shared backward-state layout from _trimul_forward_impl."""
    (
        z_bf,
        mask_f,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        _b_in,
        g_out,
        _b_out,
        outgoing,
        _save_state,
    ) = inputs
    if not _save_state and any(ctx.needs_input_grad):
        raise RuntimeError(
            "TurboPairFormer TriMul requires save_state=True when gradients are enabled"
        )
    (
        out,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
    ) = output
    ctx.save_for_backward(
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
    )
    ctx.outgoing = outgoing
    ctx.mark_non_differentiable(
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
    )
    ctx.set_materialize_grads(False)


@torch.autograd.function.once_differentiable
def _trimul_fwd_v1_backward(
    ctx,
    grad_out,
    grad_z_ln,
    grad_gate,
    grad_a,
    grad_b,
    grad_x,
    grad_xlnT,
    grad_mean_in,
    grad_rstd_in,
    grad_mean_out,
    grad_rstd_out,
):
    """Compute first-order gradients; auxiliary outputs never accept grads."""
    aux_grads = (
        grad_z_ln,
        grad_gate,
        grad_a,
        grad_b,
        grad_x,
        grad_xlnT,
        grad_mean_in,
        grad_rstd_in,
        grad_mean_out,
        grad_rstd_out,
    )
    if any(grad is not None for grad in aux_grads):
        raise RuntimeError(
            "TurboPairFormer TriMul auxiliary outputs are non-differentiable"
        )
    (
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
    ) = ctx.saved_tensors
    # This order matches setup_context and the shared implementation helper;
    # the custom-op boundary passes tensors positionally rather than by name.
    d_z, d_w5, d_wz, d_gin, d_bin, d_gout, d_bout = _trimul_bwd_v1(
        grad_out,
        z_bf,
        mask_f,
        z_ln,
        gate,
        a,
        b,
        x,
        xlnT,
        mean_in,
        rstd_in,
        mean_out,
        rstd_out,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        g_out,
        out,
        ctx.outgoing,
    )
    H = w_ap.shape[0]
    return (
        d_z,
        None,
        d_w5[0 * H : 1 * H],
        d_w5[1 * H : 2 * H],
        d_w5[2 * H : 3 * H],
        d_w5[3 * H : 4 * H],
        d_w5[4 * H : 5 * H],
        d_wz,
        d_gin,
        d_bin,
        d_gout,
        d_bout,
        None,
        None,
    )


torch.library.register_autograd(
    _trimul_fwd_v1,
    _trimul_fwd_v1_backward,
    setup_context=_trimul_fwd_v1_setup_context,
)


class _TurboPairFormerTrimulFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        mask,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        b_in,
        g_out,
        b_out,
        outgoing,
    ):
        ext = _get_ext()
        output, saved_tensors, dims, orig_shape, z_dtype = _trimul_forward_impl(
            ext,
            z,
            mask,
            w_ap,
            w_ag,
            w_bp,
            w_bg,
            w_g,
            w_z,
            g_in,
            b_in,
            g_out,
            b_out,
            outgoing,
            any(ctx.needs_input_grad),
        )
        ctx.save_for_backward(*saved_tensors)
        ctx.dims = dims
        ctx.orig_shape = orig_shape
        ctx.z_dtype = z_dtype
        return output

    @staticmethod
    def backward(ctx, grad_out):
        ext = _get_ext()
        return _trimul_backward_impl(
            ext,
            grad_out,
            ctx.saved_tensors,
            ctx.dims,
            ctx.orig_shape,
            ctx.z_dtype,
        )


def _run_supported_trimul_tensors(
    z, mask, w_ap, w_ag, w_bp, w_bg, w_g, w_z, g_in, b_in, g_out, b_out, outgoing
):
    """Run validated tensor parameters through the shared autograd boundary.

    The helper names linear_a_p/linear_b_p weights w_ap/w_bp,
    linear_a_g/linear_b_g weights w_ag/w_bg, the final gate weight w_g, and
    the normalized pair-product projection weight w_z. g_in/b_in and
    g_out/b_out are layer-normalization scales/biases, not gate matrices.
    """
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    orig_shape = z.shape
    C = z.shape[-1]
    N = z.shape[-2]
    B = z.numel() // (N * N * C)
    z_bf = z.reshape(B, N, N, C).to(torch.bfloat16).contiguous()
    mask_f = (
        mask.expand(*orig_shape[:-1]).reshape(B, N, N).to(torch.float32).contiguous()
    )
    tensor_inputs = (
        z_bf,
        mask_f,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        w_z,
        g_in,
        b_in,
        g_out,
        b_out,
    )
    save_state = any(tensor.requires_grad for tensor in tensor_inputs)
    output, *_ = _trimul_fwd_v1(
        *tensor_inputs,
        outgoing,
        save_state,
    )
    return output.view(orig_shape).to(z.dtype)


def _run_supported_trimul(module, z, mask):
    """Preserve the module integration adapter after its support check."""
    return _run_supported_trimul_tensors(
        z,
        mask,
        module.linear_a_p.weight,
        module.linear_a_g.weight,
        module.linear_b_p.weight,
        module.linear_b_g.weight,
        module.linear_g.weight,
        module.linear_z.weight,
        module.layer_norm_in.weight.float(),
        module.layer_norm_in.bias.float(),
        module.layer_norm_out.weight.float(),
        module.layer_norm_out.bias.float(),
        module._outgoing,
    )


def _run_functional_trimul(z, direction, mask, parameters, eps):
    from turbopairformer.trimul_support import (
        check_trimul_tensor_support,
    )

    support = check_trimul_tensor_support(z, direction, mask, **parameters, eps=eps)
    if not support:
        raise ValueError(
            f"TurboPairFormer TriMul unsupported [{support.code}]: {support.reason}"
        )
    channels = z.shape[-1]
    # Optional matrices are fresh demonstration parameters for this call. They
    # are neither retained by the package nor registered as trainable state.
    options = {"device": z.device, "dtype": torch.float32}
    for name in ("norm_in_weight", "norm_out_weight"):
        if parameters[name] is None:
            parameters[name] = torch.ones(channels, **options)
    for name in ("norm_in_bias", "norm_out_bias"):
        if parameters[name] is None:
            parameters[name] = torch.zeros(channels, **options)
    for name in ("p_in_weight", "g_in_weight", "p_out_weight", "g_out_weight"):
        if parameters[name] is None:
            rows = 2 * channels if name in ("p_in_weight", "g_in_weight") else channels
            parameters[name] = torch.randn(rows, channels, **options) * channels**-0.5
    p_in = parameters["p_in_weight"]
    g_in = parameters["g_in_weight"]
    # Here g_in is the public packed gate-projection matrix. The shared
    # helper's g_in argument instead names norm_in_weight (the LN scale).
    # Keep these as autograd-connected views so gradients reach caller-owned
    # packed matrices in their original [a; b] row order.
    return _run_supported_trimul_tensors(
        z,
        mask,
        p_in[:channels],
        g_in[:channels],
        p_in[channels:],
        g_in[channels:],
        parameters["g_out_weight"],
        parameters["p_out_weight"],
        parameters["norm_in_weight"],
        parameters["norm_in_bias"],
        parameters["norm_out_weight"],
        parameters["norm_out_bias"],
        direction == "outgoing",
    )


_NOT_GIVEN = object()


def turbo_trimul(
    x=_NOT_GIVEN,
    *args,
    z=_NOT_GIVEN,
    direction=_NOT_GIVEN,
    mask=_NOT_GIVEN,
    norm_in_weight=_NOT_GIVEN,
    norm_in_bias=_NOT_GIVEN,
    p_in_weight=_NOT_GIVEN,
    g_in_weight=_NOT_GIVEN,
    norm_out_weight=_NOT_GIVEN,
    norm_out_bias=_NOT_GIVEN,
    p_out_weight=_NOT_GIVEN,
    g_out_weight=_NOT_GIVEN,
    eps=_NOT_GIVEN,
    module=_NOT_GIVEN,
):
    """Run functional TriMul on ``z[..., N, N, C]`` with optional parameters.

    Use ``turbo_trimul(z, direction="outgoing", mask=None, ...)`` or ``z=z``.
    ``direction`` is ``"outgoing"`` or ``"incoming"``; the mask defaults to all
    pairs valid. Parameters are contiguous FP32 tensors on the same device as
    BF16 ``z``. ``p_in_weight`` and ``g_in_weight`` have shape ``[2C, C]`` with
    a-branch rows followed by b-branch rows. ``p_out_weight`` and
    ``g_out_weight`` have shape ``[C, C]``. Both LayerNorm weight/bias pairs
    have shape ``[C]``, and ``eps`` must be exactly ``1e-5``.

    An omitted or ``None`` matrix is independently sampled each call from a
    normal distribution with standard deviation ``1 / sqrt(C)``. Missing norm
    weights/biases are ones/zeros. These defaults are demonstration tensors,
    not persistent trainable state. Supply and reuse all eight parameters for
    training or repeated evaluation; their gradients flow to the supplied
    tensors. ``mask`` is not differentiable. Only first-order gradients are
    supported. Validation runs before demonstration allocation or CUDA loading.

    Existing ``turbo_trimul(module, z, mask=None)`` and keyword ``module=``
    integrations remain supported, and cannot mix in functional parameters.
    Only z, direction and mask may be positional in functional calls.
    """
    parameters = {
        "norm_in_weight": norm_in_weight,
        "norm_in_bias": norm_in_bias,
        "p_in_weight": p_in_weight,
        "g_in_weight": g_in_weight,
        "norm_out_weight": norm_out_weight,
        "norm_out_bias": norm_out_bias,
        "p_out_weight": p_out_weight,
        "g_out_weight": g_out_weight,
    }
    if x is not _NOT_GIVEN:
        if module is not _NOT_GIVEN:
            raise TypeError("turbo_trimul cannot combine a first argument with module=")
        if isinstance(x, torch.Tensor):
            if z is not _NOT_GIVEN:
                raise TypeError("turbo_trimul received z more than once")
            z = x
            if len(args) > 2:
                raise TypeError(
                    "functional turbo_trimul accepts at most 3 positional arguments"
                )
            if args:
                if direction is not _NOT_GIVEN:
                    raise TypeError("turbo_trimul received direction more than once")
                direction = args[0]
            if len(args) == 2:
                if mask is not _NOT_GIVEN:
                    raise TypeError("turbo_trimul received mask more than once")
                mask = args[1]
        else:
            module = x
            if len(args) > 2:
                raise TypeError(
                    "module turbo_trimul accepts at most 3 positional arguments"
                )
            if args:
                if z is not _NOT_GIVEN:
                    raise TypeError("turbo_trimul received z more than once")
                z = args[0]
            if len(args) == 2:
                if mask is not _NOT_GIVEN:
                    raise TypeError("turbo_trimul received mask more than once")
                mask = args[1]
    if z is _NOT_GIVEN:
        raise TypeError("turbo_trimul requires z")
    if mask is _NOT_GIVEN:
        mask = None
    if module is not _NOT_GIVEN:
        if (
            direction is not _NOT_GIVEN
            or eps is not _NOT_GIVEN
            or any(value is not _NOT_GIVEN for value in parameters.values())
        ):
            raise TypeError(
                "module turbo_trimul cannot be mixed with functional parameters"
            )
        from turbopairformer.trimul_support import check_trimul_support

        support = check_trimul_support(module, z, mask)
        if not support:
            raise ValueError(
                f"TurboPairFormer TriMul unsupported [{support.code}]: {support.reason}"
            )
        return _run_supported_trimul(module, z, mask)
    return _run_functional_trimul(
        z,
        "outgoing" if direction is _NOT_GIVEN else direction,
        mask,
        {
            name: None if value is _NOT_GIVEN else value
            for name, value in parameters.items()
        },
        1e-5 if eps is _NOT_GIVEN else eps,
    )


# Compatibility with integrations using the original descriptive name.
run_trimul = turbo_trimul


class TurboPairFormerTrimul:
    """Callable mirroring the official module's numerics with fused passes."""

    def __init__(self, module):
        self.module = module

    def __call__(self, z, mask):
        return turbo_trimul(self.module, z, mask)
