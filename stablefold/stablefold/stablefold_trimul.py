# SPDX-License-Identifier: Apache-2.0
"""StableFold triangle multiplicative update: fused-kernel CUDA implementation.

Forward (3 launches):
  z --[trimul_in_fwd: LN(fp32 stats) + 5-way projection GEMM (mma.sync,
       persistent col-sliced CTAs, weights resident in smem) + sigmoid-gate
       split + channel-major write]--> a, b ([B*H,N,N]), raw gate
  a,b --[cuBLAS bmm (op-flag views, no copies)]--> x
  x --[trimul_out_fwd: transpose-LN + out-projection GEMM + sigmoid gate mul,
       cp.async double-buffered tile pipeline]--> out
z_ln / proj / out_raw never round-trip HBM on the forward; training saves
z_ln, gate, x_ln^T and LN stats only (activation ~630MB/call lighter at
N=640 than materializing proj).

Backward: trimul_gate_split_bwd recomputes proj tiles on tensor cores from
the saved z_ln with the exact forward accumulation order; sig_mul_bwd_out
reconstructs out_raw from the forward output (r*sig = out identity, so
d_gate = dy*out*(1-sig)).  Remaining GEMMs are cuBLAS; LN backwards are the
fixed-order kernels below.  dW5 uses K=16384 BF16 bmm partials under a fixed
48 MiB partial cap, followed by one globally ascending FP32 fold.

All reductions fixed-order, zero atomics, bitwise replay-deterministic.
Contract (explicit reject, no silent specialization): c_hidden == c_z,
C in {32, 64, 96, 128}, arbitrary N and batch within the validated CUDA grid
and memory limits.

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
        from stablefold.stablefold_trimul_cuda import (
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
    """Compute dW5 with bounded BF16 split-K GEMMs and a fixed FP32 fold."""
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

    # gate slot [4H:5H) of d_proj5 + d_raw, reconstructing out_raw from out
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
    "stablefold::_trimul_fwd_v1",
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
    "stablefold::_trimul_bwd_v1",
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
            "StableFold TriMul requires save_state=True when gradients are enabled"
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
        raise RuntimeError("StableFold TriMul auxiliary outputs are non-differentiable")
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


class _StableTrimulFn(torch.autograd.Function):
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


def _run_supported_stable_trimul(module, z, mask):
    """Run a call that the shared metadata contract already accepted."""
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
    )
    save_state = any(tensor.requires_grad for tensor in tensor_inputs)
    output, *_ = _trimul_fwd_v1(
        *tensor_inputs,
        module._outgoing,
        save_state,
    )
    return output.view(orig_shape).to(z.dtype)


def run_stable_trimul(module, z, mask):
    """Strict differentiable entry point bound to an official module."""
    from stablefold.stablefold_trimul_support import (
        check_stable_trimul_support,
    )

    support = check_stable_trimul_support(module, z, mask)
    if not support:
        raise ValueError(
            f"StableFold TriMul unsupported [{support.code}]: {support.reason}"
        )
    return _run_supported_stable_trimul(module, z, mask)


class StableTrimul:
    """Callable mirroring the official module's numerics with fused passes."""

    def __init__(self, module):
        self.module = module

    def __call__(self, z, mask):
        return run_stable_trimul(self.module, z, mask)
