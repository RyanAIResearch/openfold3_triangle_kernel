"""SR backward gate: bitwise dK/dV + replay, fp64-referenced dQ/dBias.

Baseline = the two-pass backward (env flag off); SR = single-reconstruction
(OPENFOLD3_SR_BACKWARD=1).  Expected: dK/dV bit-identical to baseline (same
chains, same classic delta), dQ/dBias replay-bitwise but NOT baseline-equal
(classic vs true delta by design) -- their quality is judged against an
FP64 torch reference on the same inputs.
"""

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

from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (  # noqa: E402
    turbopairformer_triangle_attention,
)

SHAPES = {
    "n192_d32": (1, 192, 4, 192, 32, False),
    "n128_d16": (1, 128, 4, 128, 16, False),
    "n128_d64": (1, 128, 2, 128, 64, False),
    "outer24_n128_d32": (1, 24, 4, 128, 32, False),
    "b2_bias_bcast_n128_d32": (2, 128, 2, 128, 32, True),
    "n130_d32_oddouter": (1, 130, 2, 192, 32, False),
}


def make_case(name):
    batch, outer, heads, seq, dim, bias_one = SHAPES[name]
    torch.manual_seed(zlib.crc32(name.encode()) % (2**31))
    dev = "cuda"
    shape = (batch, outer, heads, seq, dim)
    q = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(batch, outer, 1, 1, seq, device=dev, dtype=torch.float32)
    mask[:, :, :, :, -3:] = float("-inf")
    bias_batch = 1 if bias_one else batch
    bias = (
        0.1
        * torch.randn(
            bias_batch, 1, heads, seq, seq, device=dev, dtype=torch.float32
        )
    ).requires_grad_()
    return q, k, v, mask, bias


def run_grads(name, sr):
    os.environ["OPENFOLD3_SR_BACKWARD"] = "1" if sr else "0"
    q, k, v, mask, bias = make_case(name)
    out = turbopairformer_triangle_attention(q, k, v, mask, bias)
    torch.manual_seed(99)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v, bias), grad_outputs=grad_out)
    torch.cuda.synchronize()
    return {"dq": grads[0], "dk": grads[1], "dv": grads[2], "dbias": grads[3]}


def fp64_reference(name):
    q, k, v, mask, bias = make_case(name)
    q64 = q.detach().double().requires_grad_()
    k64 = k.detach().double().requires_grad_()
    v64 = v.detach().double().requires_grad_()
    b64 = bias.detach().double().requires_grad_()
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (
        torch.einsum("bohqd,bohkd->bohqk", q64, k64) * scale
        + mask.double().permute(0, 1, 3, 2, 4)
        + b64
    )
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bohqk,bohkd->bohqd", probs, v64)
    out_public = out.permute(0, 1, 3, 2, 4)
    torch.manual_seed(99)
    grad_out = torch.randn(
        out_public.shape, device=out.device, dtype=torch.bfloat16
    ).double()
    grads = torch.autograd.grad(
        out_public, (q64, k64, v64, b64), grad_outputs=grad_out
    )
    return {"dq": grads[0], "dk": grads[1], "dv": grads[2], "dbias": grads[3]}


def rel_l2(candidate, reference):
    c = candidate.double()
    r = reference
    denominator = r.norm().item()
    return ((c - r).norm().item() / denominator) if denominator else 0.0


def main():
    failures = 0
    for name in SHAPES:
        base = run_grads(name, sr=False)
        sr_first = run_grads(name, sr=True)
        sr_second = run_grads(name, sr=True)
        for key in ("dq", "dk", "dv", "dbias"):
            if not torch.equal(sr_first[key], sr_second[key]):
                print(f"REPLAY-FAIL {name}:{key}")
                failures += 1
        for key in ("dk", "dv"):
            if not torch.equal(sr_first[key], base[key]):
                diff = (sr_first[key].float() - base[key].float()).abs()
                print(
                    f"DKDV-BITWISE-FAIL {name}:{key} "
                    f"maxabs={diff.max().item():.3e} "
                    f"n={(diff > 0).sum().item()}/{diff.numel()}"
                )
                failures += 1
        ref = fp64_reference(name)
        report = []
        for key in ("dq", "dbias"):
            base_err = rel_l2(base[key], ref[key])
            sr_err = rel_l2(sr_first[key], ref[key])
            report.append(f"{key}: base {base_err:.5f} sr {sr_err:.5f}")
            if sr_err > max(8.0 * base_err, 0.05):
                print(f"QUALITY-FAIL {name}:{key} sr={sr_err} base={base_err}")
                failures += 1
        print(f"{name}: dk/dv bitwise+replay OK | {' | '.join(report)}")
    print("SR_GATE_PASS" if failures == 0 else f"SR_GATE_FAILURES={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
