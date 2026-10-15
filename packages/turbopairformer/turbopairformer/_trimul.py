# SPDX-License-Identifier: Apache-2.0
"""Run TurboPairFormer triangle multiplicative update and its backward pass.

Both public calling forms validate their inputs, then enter
``_run_supported_trimul_tensors`` and the ``_trimul_fwd_v1`` custom op.
Autograd calls ``_trimul_bwd_v1`` through ``_trimul_fwd_v1_backward``.
The implementation helpers below supply both custom ops.

Dimensions: B is the flattened batch prefix, N the pair-axis length, C the
input/output channel count, and H the hidden channel count (not attention heads).

Forward follows three stages:
  1. Normalize z, form five projections, and apply the two branch gates/mask.
     ``trimul_in_fwd`` writes a and b in channel-major layout [B*H, N, N]
     and saves the raw output gate in row-major layout [B*N*N, H].
  2. Multiply the pair matrices with cuBLAS: outgoing uses a @ b.T;
     incoming uses a.T @ b. Transposes are views of the same storage.
  3. Normalize the pair product, project back to C channels, and apply the
     output gate in ``trimul_out_fwd``.
The fused kernels do not store the full five-projection tensor or the ungated
output. The backward recomputes the first four projections from saved z_ln;
the output-gate gradient uses the saved gated output as
``d_gate = d_out * out * (1 - sigmoid(gate))``.

Backward reverses these stages. Layer-normalization reductions use the CUDA
kernels in ``turbopairformer_trimul_cuda.cu``. The five input weight gradients
share a BF16-input GEMM split into K=16384 row chunks. Each chunk produces an
FP32 partial; waves cap these partials at 48 MiB, and a fixed FP32 fold visits
chunks in global ascending order. This cap excludes operands and the final
accumulator.

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
    """Consecutive full chunks, optionally followed by the one padded tail."""

    first_chunk: int
    full_chunks: int
    tail_rows: int


class _WgradPlan(NamedTuple):
    """Bound the FP32 partial storage without changing the chunk fold order."""

    partial_bytes_per_chunk: int
    chunks_per_wave: int
    waves: tuple[_WgradWave, ...]


def _get_ext():
    """Load the CUDA extension on first execution, then reuse it."""
    global _ext
    if _ext is None:
        from turbopairformer.trimul_cuda import (
            load_extension,
        )

        _ext = load_extension()
    return _ext


def _plan_wgrad_fixed_fold(rows: int, hidden: int, channels: int) -> _WgradPlan:
    """Group K=16384 row chunks into waves of at most 48 MiB of FP32 partials.

    Each partial has shape [5 * hidden, channels]. The final short chunk,
    if any, occupies one logical slot and is computed separately with padding.
    Waves affect temporary storage, not the ascending order of the final fold.
    """
    if rows <= 0:
        raise ValueError(f"wgrad rows must be positive, got {rows}")
    if hidden <= 0 or channels <= 0:
        raise ValueError(
            "wgrad hidden/channels must be positive, got "
            f"hidden={hidden}, channels={channels}"
        )
    partial_bytes_per_chunk = 5 * hidden * channels * _WGRAD_FP32_BYTES
    if partial_bytes_per_chunk > _WGRAD_PARTIAL_CAP_BYTES:
        raise ValueError(
            "one wgrad partial exceeds the fixed 48 MiB cap: "
            f"{partial_bytes_per_chunk} bytes"
        )
    chunks_per_wave = _WGRAD_PARTIAL_CAP_BYTES // partial_bytes_per_chunk
    full_chunks, tail_rows = divmod(rows, _WGRAD_K_CHUNK)
    total_chunks = full_chunks + int(tail_rows != 0)
    waves = []
    first_chunk = 0
    while first_chunk < total_chunks:
        wave_chunks = min(chunks_per_wave, total_chunks - first_chunk)
        wave_full_chunks = min(wave_chunks, max(0, full_chunks - first_chunk))
        wave_tail_rows = tail_rows if wave_full_chunks < wave_chunks else 0
        waves.append(_WgradWave(first_chunk, wave_full_chunks, wave_tail_rows))
        first_chunk += wave_chunks
    return _WgradPlan(partial_bytes_per_chunk, chunks_per_wave, tuple(waves))


def _wgrad_full_chunk_views(d_proj5, z_ln, first_chunk: int, full_chunks: int):
    """View a wave as [chunks, 5H, K] and [chunks, K, C], without copying."""
    row_start = first_chunk * _WGRAD_K_CHUNK
    wave_rows = full_chunks * _WGRAD_K_CHUNK
    five_hidden = d_proj5.shape[1]
    channels = z_ln.shape[1]
    projection_chunks = d_proj5.narrow(0, row_start, wave_rows).view(
        full_chunks, _WGRAD_K_CHUNK, five_hidden
    )
    normalized_chunks = z_ln.narrow(0, row_start, wave_rows).view(
        full_chunks, _WGRAD_K_CHUNK, channels
    )
    return projection_chunks.transpose(1, 2), normalized_chunks


def _trimul_wgrad_fixed(ext, d_proj5, z_ln):
    """Compute dW5 = d_proj5.T @ z_ln with a fixed chunk accumulation order."""
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
        (five_hidden, channels), dtype=torch.float32, device=d_proj5.device
    )
    # The first fold initializes this uninitialized allocation; later folds
    # continue it. Keep the same accumulator across all waves and the tail.
    initialize = True
    for wave in plan.waves:
        if wave.full_chunks:
            # Full chunks alias the inputs. Each GEMM produces one FP32 partial.
            projection_t, normalized = _wgrad_full_chunk_views(
                d_proj5, z_ln, wave.first_chunk, wave.full_chunks
            )
            partials = torch.bmm(projection_t, normalized, out_dtype=torch.float32)
            ext.trimul_wgrad_fixed_fold(partials, accumulator, initialize)
            initialize = False
            del partials, projection_t, normalized

        if wave.tail_rows:
            # Only the last short chunk is padded to K=16384. Fold it after all
            # preceding full chunks, then release its temporary operands.
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
    """Run the three forward stages and pack their shared backward state."""
    # Flatten any batch prefix; rows enumerate all (batch, pair-row, pair-col).
    orig_shape = z.shape
    channels = z.shape[-1]
    sequence = z.shape[-2]
    if z.shape[-3] != sequence:
        raise ValueError(f"expected square pair input, got {tuple(z.shape)}")
    batch = z.numel() // (sequence * sequence * channels)
    hidden = w_ap.shape[0]
    if hidden != channels:
        raise ValueError("stable trimul requires c_hidden == c_z")
    z_bf = (
        z.reshape(batch, sequence, sequence, channels).to(torch.bfloat16).contiguous()
    )
    mask_f = (
        mask.expand(*orig_shape[:-1])
        .reshape(batch, sequence, sequence)
        .to(torch.float32)
        .contiguous()
    )
    rows = batch * sequence * sequence
    pair_elements = sequence * sequence
    mask_flat = mask_f.view(-1)

    # 1. Input LN, five projections, then branch gates and mask.
    # a/b use [batch * hidden, pair_elements]; gate uses [rows, hidden].
    a, b, gate, mean_in, rstd_in, z_ln = ext.trimul_in_fwd(
        z_bf.view(rows, channels),
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        w_g,
        mask_flat,
        g_in,
        b_in,
        batch,
        pair_elements,
        channels,
        hidden,
        training,
    )
    # 2. One pair-matrix product per batch item and hidden channel.
    a_matrices = a.view(batch * hidden, sequence, sequence)
    b_matrices = b.view(batch * hidden, sequence, sequence)
    if outgoing:
        x = torch.bmm(a_matrices, b_matrices.transpose(1, 2))
    else:
        x = torch.bmm(a_matrices.transpose(1, 2), b_matrices)
    # 3. Output LN, output projection and sigmoid gate.
    # When training, xlnT saves normalized x as [hidden, rows] for d_wz.
    out, xlnT, mean_out, rstd_out = ext.trimul_out_fwd(
        x.view(batch * hidden, pair_elements),
        w_z,
        gate,
        g_out,
        b_out,
        batch,
        pair_elements,
        channels,
        hidden,
        training,
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
    dims = (batch, sequence, channels, hidden, outgoing)
    return out.view(orig_shape).to(z.dtype), saved_tensors, dims, orig_shape, z.dtype


def _trimul_backward_combined_impl(
    ext, grad_out, saved_tensors, dims, orig_shape, z_dtype
):
    """Reverse the forward stages; return all five input weight grads together."""
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
    batch, sequence, channels, hidden, outgoing = dims
    rows = batch * sequence * sequence
    pair_elements = sequence * sequence
    grad_output_rows = grad_out.to(torch.bfloat16).contiguous().view(rows, channels)

    # 1. Differentiate the output gate using the saved gated output.
    # d_proj5 holds [a projection, a gate, b projection, b gate, output gate].
    # This first call fills only the final H channels; d_raw feeds output GEMM.
    d_proj5 = torch.empty(rows, 5 * hidden, dtype=torch.bfloat16, device=z_bf.device)
    d_raw = ext.sig_mul_bwd_out(
        out.view(rows, channels),
        gate,
        grad_output_rows,
        d_proj5.view(rows, 5 * hidden)[:, 4 * hidden :],
        hidden,
        5 * hidden,
    )
    # 2. Output projection and output LN backward. w_z is [C, H]; xlnT
    # is [H, rows]. The transposes below are views, not materialized copies.
    d_xln = (d_raw @ w_z.to(torch.bfloat16)).contiguous()
    d_wz = torch.mm(d_raw.t(), xlnT.t(), out_dtype=torch.float32)
    # Restore channel-major d_x for the pair products; fold LN parameter grads.
    d_x, d_gout, d_bout = ext.transpose_ln_bwd(
        x.view(batch * hidden, pair_elements),
        d_xln,
        sequence,
        hidden,
        batch,
        g_out,
        mean_out,
        rstd_out,
    )
    # 3. Differentiate a @ b.T (outgoing) or a.T @ b (incoming).
    a_matrices = a.view(batch * hidden, sequence, sequence)
    b_matrices = b.view(batch * hidden, sequence, sequence)
    if outgoing:
        d_a = torch.bmm(d_x, b_matrices)
        d_b = torch.bmm(d_x.transpose(1, 2), a_matrices)
    else:
        d_a = torch.bmm(b_matrices, d_x.transpose(1, 2))
        d_b = torch.bmm(a_matrices, d_x)
    # 4. Recompute the branch projections and differentiate their gates/mask.
    # Fill d_proj5's first four H-channel slots, preserving its output-gate slot.
    ext.trimul_gate_split_bwd(
        z_ln,
        w_ap,
        w_ag,
        w_bp,
        w_bg,
        mask_f.view(-1),
        d_a.view(batch * hidden, pair_elements),
        d_b.view(batch * hidden, pair_elements),
        d_proj5,
        batch,
        pair_elements,
        channels,
        hidden,
    )
    # 5. Backpropagate all five input projections, then input LN.
    # Concatenate weights in the same order as d_proj5. Only d_w5 uses the
    # bounded partial-wave plan; d_wz above remains one FP32-output GEMM.
    w5 = torch.cat([w_ap, w_ag, w_bp, w_bg, w_g], 0).to(torch.bfloat16)
    d_zln = (d_proj5 @ w5).contiguous()
    d_w5 = _trimul_wgrad_fixed(ext, d_proj5, z_ln)
    d_z = ext.ln_bwd_dx(
        z_bf.view(rows, channels), d_zln, rows, channels, 1, g_in, mean_in, rstd_in
    )
    d_gin, d_bin = ext.ln_bwd_dgb(
        z_bf.view(rows, channels), d_zln, rows, channels, 1, channels, mean_in, rstd_in
    )
    return (d_z.view(orig_shape).to(z_dtype), d_w5, d_wz, d_gin, d_bin, d_gout, d_bout)


def _trimul_backward_impl(ext, grad_out, saved_tensors, dims, orig_shape, z_dtype):
    """Split combined dW5 into the legacy autograd entry point's input order."""
    d_z, d_w5, d_wz, d_gin, d_bin, d_gout, d_bout = _trimul_backward_combined_impl(
        ext, grad_out, saved_tensors, dims, orig_shape, z_dtype
    )
    hidden = dims[3]
    return (
        d_z,
        None,
        d_w5[0 * hidden : 1 * hidden],
        d_w5[1 * hidden : 2 * hidden],
        d_w5[2 * hidden : 3 * hidden],
        d_w5[3 * hidden : 4 * hidden],
        d_w5[4 * hidden : 5 * hidden],
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
    """Expose the output plus ten backward-state tensors as fresh custom-op results.

    Returning the auxiliary tensors lets setup_context save them for autograd.
    They are not independently differentiable outputs.
    """
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
    return (output, z_ln, gate, a, b, x, xlnT, mean_in, rstd_in, mean_out, rstd_out)


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
    """Describe forward output shapes/dtypes without loading CUDA kernels."""
    batch, sequence, _, channels = z_bf.shape
    hidden = w_ap.shape[0]
    rows = batch * sequence * sequence
    pair_elements = sequence * sequence
    # Input LN statistics and pair products always exist. save_state controls
    # saved z_ln, xlnT, and output LN statistics; inference returns empty storage
    # for those optional tensors while preserving the eleven-output schema.
    saved_rows = rows if save_state else 0
    return (
        z_bf.new_empty((batch, sequence, sequence, channels)),
        z_bf.new_empty((saved_rows, channels)),
        z_bf.new_empty((rows, hidden)),
        z_bf.new_empty((batch * hidden, pair_elements)),
        z_bf.new_empty((batch * hidden, pair_elements)),
        z_bf.new_empty((batch * hidden, sequence, sequence)),
        z_bf.new_empty((hidden if save_state else 0, rows)),
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
    """Keep backward inside one custom op, returning dW5 as a combined tensor."""
    ext = _get_ext()
    batch, sequence, _, channels = z_bf.shape
    hidden = w_ap.shape[0]
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
        (batch, sequence, channels, hidden, outgoing),
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
    """Describe seven fresh gradients: input, two weight groups, and four LN grads."""
    hidden, channels = w_ap.shape
    return (
        torch.empty_like(z_bf, memory_format=torch.contiguous_format),
        w_ap.new_empty((5 * hidden, channels)),
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
    (out, z_ln, gate, a, b, x, xlnT, mean_in, rstd_in, mean_out, rstd_out) = output
    # Keep the exact shared 21-tensor order used by the implementation helpers.
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
    # Only the first custom-op output participates in differentiation. Leave
    # unused auxiliary gradients as None instead of allocating zero tensors.
    ctx.mark_non_differentiable(
        z_ln, gate, a, b, x, xlnT, mean_in, rstd_in, mean_out, rstd_out
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
    hidden = w_ap.shape[0]
    # Return one slot per forward input: z, mask, six weights, four LN tensors,
    # outgoing, save_state. The mask and both booleans have no gradients.
    return (
        d_z,
        None,
        d_w5[0 * hidden : 1 * hidden],
        d_w5[1 * hidden : 2 * hidden],
        d_w5[2 * hidden : 3 * hidden],
        d_w5[3 * hidden : 4 * hidden],
        d_w5[4 * hidden : 5 * hidden],
        d_wz,
        d_gin,
        d_bin,
        d_gout,
        d_bout,
        None,
        None,
    )


torch.library.register_autograd(
    _trimul_fwd_v1, _trimul_fwd_v1_backward, setup_context=_trimul_fwd_v1_setup_context
)


class _TurboPairFormerTrimulFn(torch.autograd.Function):
    """Legacy autograd adapter over the same forward/backward implementation."""

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
            ext, grad_out, ctx.saved_tensors, ctx.dims, ctx.orig_shape, ctx.z_dtype
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
    # Normalize layouts at the custom-op boundary. Validation accepts a batch
    # prefix and a broadcastable mask; kernels consume flattened, dense inputs.
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    orig_shape = z.shape
    channels = z.shape[-1]
    sequence = z.shape[-2]
    batch = z.numel() // (sequence * sequence * channels)
    z_bf = (
        z.reshape(batch, sequence, sequence, channels).to(torch.bfloat16).contiguous()
    )
    mask_f = (
        mask.expand(*orig_shape[:-1])
        .reshape(batch, sequence, sequence)
        .to(torch.float32)
        .contiguous()
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
    # Save backward state whenever a prepared input requires gradients.
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
