"""Full delta-contract ladder on the SR substrate.

Arms (all through the SAME SR backward, only delta differs, plus the
two-pass reference):
  two-pass      facade, delta_true (accuracy champion reference)
  sr-classic    delta = dO.O_bf16            (kernel delta)
  sr-shadow     delta = dO.(O_bf16 + R_bf16) (kernel shadow delta)
  sr-fp32shadow delta = dO.O* with O* = fp32 torch forward   [oracle-ish]
  sr-perfect    delta = sum_k P_fp64 dP_fp64 cast to fp32    [absolute floor]

If shadow ~= fp32shadow ~= perfect, the BF16 residual loses nothing and
int8 compression is purely a memory play.  Errors vs an FP64 reference.
Cases can come from synthetic CASES or from captured real tensors
(--capture dir with case_*.pt files holding q,k,v,mask,bias,grad_out).
"""

import glob
import os
import sys
import zlib

ROOT = os.environ.get("SR_ROOT", "/home/ubuntu/xzz/of3_sr_dq")
sys.path.insert(0, ROOT)
os.environ["OPENFOLD3_TRIANGLE_ALLOW_JIT"] = "1"
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    "/tmp/claude-1000/-home-ubuntu-xzz/7e522781-b3b4-47bf-a008-e0fc688a239d/"
    "scratchpad/torch_ext_sr",
)

import torch  # noqa: E402

from openfold3.core.kernels.cuda import (  # noqa: E402
    turbopairformer_triangle_attention as facade,
)
from openfold3.core.kernels.cuda.turbopairformer_triangle_dq_dbias import (  # noqa: E402
    turbopairformer_triangle_sr_backward,
)

CASES = {
    "mild_n192": (192, 1.0, 0.1),
    "sharp_n256": (256, 2.6, 1.0),
}


def synthetic_case(name):
    seq, qk_sigma, bias_sigma = CASES[name]
    torch.manual_seed(zlib.crc32(name.encode()) % (2**31))
    heads, dim = 4, 32
    shape = (1, seq, heads, seq, dim)
    q = (qk_sigma * torch.randn(shape, device="cuda")).bfloat16()
    k = (qk_sigma * torch.randn(shape, device="cuda")).bfloat16()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, seq, 1, 1, seq, device="cuda", dtype=torch.float32)
    bias = (
        bias_sigma * torch.randn(1, 1, heads, seq, seq, device="cuda")
    ).bfloat16()
    torch.manual_seed(7)
    grad_out = torch.randn(
        (1, seq, seq, heads, dim), device="cuda", dtype=torch.bfloat16
    )
    return q, k, v, mask, bias, grad_out


def load_capture(path):
    blob = torch.load(path, weights_only=True, map_location="cuda")
    def flat(t):
        return t.reshape(-1, *t.shape[-4:])

    q = flat(blob["q"]).bfloat16().contiguous()
    k = flat(blob["k"]).bfloat16().contiguous()
    v = flat(blob["v"]).bfloat16().contiguous()
    mask = flat(blob["mask"]).float().contiguous()
    bias = flat(blob["bias"]).bfloat16().contiguous()
    grad_out = flat(blob["grad_out"]).bfloat16().contiguous()
    return q, k, v, mask, bias, grad_out


def kernel_forward(q, k, v, mask, bias):
    ext = facade._forward_extension()
    batch, outer, heads, seq, dim = q.shape
    out = torch.empty(
        (batch, outer, seq, heads, dim), device=q.device, dtype=torch.bfloat16
    )
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    shadow = torch.empty_like(out)
    ext.turbopairformer_triangle_forward_shadow_out(
        q, k, v, mask, bias, out, lse, shadow
    )
    return out, lse, shadow


def fp64_pieces(q, k, v, mask, bias, grad_out, chunk=32):
    """Chunked-over-outer FP64 forward/backward reference + delta pieces."""
    dim = q.shape[-1]
    scale = 1.0 / (dim ** 0.5)
    refs = {"dq": [], "dk": [], "dv": [], "delta": []}
    dbias_acc = None
    o_fp32_rows = []
    outer = q.shape[1]
    for start in range(0, outer, chunk):
        stop = min(start + chunk, outer)
        q64 = q[:, start:stop].double().requires_grad_()
        k64 = k[:, start:stop].double().requires_grad_()
        v64 = v[:, start:stop].double().requires_grad_()
        b64 = bias.double().requires_grad_()
        m64 = mask[:, start:stop].double().permute(0, 1, 3, 2, 4)
        scores = (
            torch.einsum("bohqd,bohkd->bohqk", q64, k64) * scale + m64 + b64
        )
        probs = torch.softmax(scores, dim=-1)
        out_public = torch.einsum("bohqk,bohkd->bohqd", probs, v64).permute(
            0, 1, 3, 2, 4
        )
        go = grad_out[:, start:stop].double()
        grads = torch.autograd.grad(
            out_public, (q64, k64, v64, b64), grad_outputs=go
        )
        refs["dq"].append(grads[0])
        refs["dk"].append(grads[1])
        refs["dv"].append(grads[2])
        dbias_acc = grads[3] if dbias_acc is None else dbias_acc + grads[3]
        # perfect delta: sum_k P dP with dP = dO @ V^T (fp64)
        dp64 = torch.einsum(
            "bonhd,bohkd->bohnk", go, v64
        )
        refs["delta"].append((probs * dp64).sum(-1))
        # fp32 shadow output (torch fp32 forward path)
        scores32 = (
            torch.einsum(
                "bohqd,bohkd->bohqk", q64.detach().float(), k64.detach().float()
            )
            * scale
            + m64.float()
            + b64.detach().float()
        )
        probs32 = torch.softmax(scores32, dim=-1)
        o32 = torch.einsum(
            "bohqk,bohkd->bohqd", probs32, v64.detach().float()
        ).permute(0, 1, 3, 2, 4)
        o_fp32_rows.append(o32)
        del q64, k64, v64, b64, scores, probs, out_public, scores32, probs32
        torch.cuda.empty_cache()
    return {
        "dq": torch.cat(refs["dq"], dim=1),
        "dk": torch.cat(refs["dk"], dim=1),
        "dv": torch.cat(refs["dv"], dim=1),
        "dbias": dbias_acc,
        "delta_perfect": torch.cat(refs["delta"], dim=1),
        "o_fp32": torch.cat(o_fp32_rows, dim=1),
    }


def rel_l2(candidate, reference):
    denominator = reference.norm().item()
    return (
        (candidate.double() - reference.double()).norm().item() / denominator
        if denominator
        else 0.0
    )


def delta_from_out(grad_out, out_like):
    # delta rows [B,O,H,N] from public-layout [B,O,N,H,D] tensors
    return (
        (grad_out.float() * out_like.float()).sum(-1).permute(0, 1, 3, 2)
    ).contiguous()


def run_case(name, q, k, v, mask, bias, grad_out):
    out, lse, shadow = kernel_forward(q, k, v, mask, bias)
    ref = fp64_pieces(q, k, v, mask, bias, grad_out)
    arms = {}
    arms["sr-classic"] = delta_from_out(grad_out, out)
    arms["sr-shadow"] = delta_from_out(
        grad_out, out.float() + shadow.float()
    )
    arms["sr-fp32shadow"] = delta_from_out(grad_out, ref["o_fp32"])
    arms["sr-perfect"] = ref["delta_perfect"].float().contiguous()
    lines = [name]
    for arm, delta in arms.items():
        dq, dk, dv, dbias = turbopairformer_triangle_sr_backward(
            q, k, v, grad_out, lse, delta, mask, bias
        )
        e_dq = rel_l2(dq, ref["dq"])
        e_db = rel_l2(dbias, ref["dbias"])
        lines.append(f"{arm}: dq {e_dq:.5f} dbias {e_db:.5f}")
    print(" | ".join(lines), flush=True)


def main():
    capture_dir = None
    if len(sys.argv) > 2 and sys.argv[1] == "--capture":
        capture_dir = sys.argv[2]
    if capture_dir:
        for path in sorted(glob.glob(os.path.join(capture_dir, "case_*.pt"))):
            q, k, v, mask, bias, grad_out = load_capture(path)
            run_case(os.path.basename(path), q, k, v, mask, bias, grad_out)
    else:
        for name in CASES:
            run_case(name, *synthetic_case(name))
    print("LADDER_DONE")


if __name__ == "__main__":
    main()
