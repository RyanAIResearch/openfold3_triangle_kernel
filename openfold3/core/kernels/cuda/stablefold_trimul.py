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
fixed-order kernels below.

All reductions fixed-order, zero atomics, bitwise replay-deterministic.
Contract (explicit reject, no silent specialization): c_hidden == c_z,
C % 32 == 0, C <= 128, arbitrary N and batch.  Legacy pack/gather helpers
remain for benchmark compatibility; production attention owns its fused QKV
pack in the forward AOT extension and does not import this extension module.
"""
import torch

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        from openfold3.core.kernels.cuda.stablefold_trimul_cuda import (
            load_extension,
        )

        _ext = load_extension()
    return _ext


class _StableTrimulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, w_ap, w_ag, w_bp, w_bg, w_g, w_z,
                g_in, b_in, g_out, b_out, outgoing):
        ext = _get_ext()
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
        mask_f = (mask.expand(*orig_shape[:-1]).reshape(B, N, N)
                  .to(torch.float32).contiguous())
        rows = B * N * N
        NN = N * N
        training = any(ctx.needs_input_grad)
        mask_flat = mask_f.view(-1)

        a, b, gate, mean_in, rstd_in, z_ln = ext.trimul_in_fwd(
            z_bf.view(rows, C), w_ap, w_ag, w_bp, w_bg, w_g,
            mask_flat, g_in, b_in, B, NN, C, H, training)
        av = a.view(B * H, N, N)
        bv = b.view(B * H, N, N)
        if outgoing:
            x = torch.bmm(av, bv.transpose(1, 2))
        else:
            x = torch.bmm(av.transpose(1, 2), bv)
        out, xlnT, mean_out, rstd_out = ext.trimul_out_fwd(
            x.view(B * H, NN), w_z, gate, g_out, b_out, B, NN, C, H, training)

        ctx.save_for_backward(z_bf, mask_f, z_ln, gate, a, b, x, xlnT,
                              mean_in, rstd_in, mean_out, rstd_out,
                              w_ap, w_ag, w_bp, w_bg, w_g, w_z,
                              g_in, g_out, out)
        ctx.dims = (B, N, C, H, outgoing)
        ctx.orig_shape = orig_shape
        ctx.z_dtype = z.dtype
        return out.view(orig_shape).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _get_ext()
        (z_bf, mask_f, z_ln, gate, a, b, x, xlnT, mean_in, rstd_in,
         mean_out, rstd_out, w_ap, w_ag, w_bp, w_bg, w_g, w_z,
         g_in, g_out, out) = ctx.saved_tensors
        B, N, C, H, outgoing = ctx.dims
        rows = B * N * N
        NN = N * N
        go = grad_out.to(torch.bfloat16).contiguous().view(rows, C)

        # gate slot [4H:5H) of d_proj5 + d_raw, reconstructing out_raw from out
        d_proj5 = torch.empty(rows, 5 * H, dtype=torch.bfloat16,
                              device=z_bf.device)
        d_raw = ext.sig_mul_bwd_out(
            out.view(rows, C), gate, go,
            d_proj5.view(rows, 5 * H)[:, 4 * H:], H, 5 * H)
        # out GEMM backward (w_z is [C, H]; no transposes materialized)
        d_xln = (d_raw @ w_z.to(torch.bfloat16)).contiguous()
        d_wz = torch.mm(d_raw.t(), xlnT.t(), out_dtype=torch.float32)
        # LN out backward (fused transpose variant, dgb folded)
        d_x, d_gout, d_bout = ext.transpose_ln_bwd(
            x.view(B * H, NN), d_xln, N, H, B, g_out, mean_out, rstd_out)
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
        ext.trimul_gate_split_bwd(z_ln, w_ap, w_ag, w_bp, w_bg,
                                  mask_f.view(-1),
                                  d_a.view(B * H, NN), d_b.view(B * H, NN),
                                  d_proj5, B, NN, C, H)
        # in GEMM backward: W5 [5H, C] built once, only here
        w5 = torch.cat([w_ap, w_ag, w_bp, w_bg, w_g], 0).to(torch.bfloat16)
        d_zln = (d_proj5 @ w5).contiguous()
        d_w5 = torch.mm(d_proj5.t(), z_ln, out_dtype=torch.float32)
        # LN in backward
        d_z = ext.ln_bwd_dx(z_bf.view(rows, C), d_zln, rows, C, 1,
                            g_in, mean_in, rstd_in)
        d_gin, d_bin = ext.ln_bwd_dgb(z_bf.view(rows, C), d_zln, rows, C, 1,
                                      C, mean_in, rstd_in)
        return (d_z.view(ctx.orig_shape).to(ctx.z_dtype), None,
                d_w5[0 * H:1 * H], d_w5[1 * H:2 * H], d_w5[2 * H:3 * H],
                d_w5[3 * H:4 * H], d_w5[4 * H:5 * H], d_wz,
                d_gin, d_bin, d_gout, d_bout, None)


def _run_supported_stable_trimul(module, z, mask):
    """Run a call that the shared metadata contract already accepted."""
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    return _StableTrimulFn.apply(
        z, mask,
        module.linear_a_p.weight, module.linear_a_g.weight,
        module.linear_b_p.weight, module.linear_b_g.weight,
        module.linear_g.weight, module.linear_z.weight,
        module.layer_norm_in.weight.float(),
        module.layer_norm_in.bias.float(),
        module.layer_norm_out.weight.float(),
        module.layer_norm_out.bias.float(),
        module._outgoing,
    )


def run_stable_trimul(module, z, mask):
    """Strict differentiable entry point bound to an official module."""
    from openfold3.core.kernels.cuda.stablefold_trimul_support import (
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
