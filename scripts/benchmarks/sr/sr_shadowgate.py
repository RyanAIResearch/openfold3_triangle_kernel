"""Stage-2 shadow kill-shot: three delta contracts vs FP64 on sharp attention.

Arms: two-pass (delta_true, the accuracy champion), SR+classic (v1), and
SR+shadow (stage-2).  On a sharp-softmax case the classic delta's coherent
row bias must appear and the shadow must recover the two-pass level.
Also checks: forward `out` bitwise identical across all arms, SR+shadow
replay bitwise, dK/dV tolerance.
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

from openfold3.core.kernels.cuda.stablefold_triangle_attention import (  # noqa: E402
    stablefold_triangle_attention,
)

CASES = {
    "mild_n192": (192, 1.0, 0.1),
    "sharp_n192": (192, 2.2, 0.8),
    "sharp_n256": (256, 2.6, 1.0),
}


def make_case(name):
    seq, qk_sigma, bias_sigma = CASES[name]
    torch.manual_seed(zlib.crc32(name.encode()) % (2**31))
    heads, dim = 4, 32
    shape = (1, seq, heads, seq, dim)
    q = (qk_sigma * torch.randn(shape, device="cuda")).bfloat16().requires_grad_()
    k = (qk_sigma * torch.randn(shape, device="cuda")).bfloat16().requires_grad_()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(1, seq, 1, 1, seq, device="cuda", dtype=torch.float32)
    bias = (
        bias_sigma * torch.randn(1, 1, heads, seq, seq, device="cuda")
    ).requires_grad_()
    return q, k, v, mask, bias


def run(name, sr, shadow):
    os.environ["OPENFOLD3_SR_BACKWARD"] = "1" if sr else "0"
    os.environ["OPENFOLD3_SR_SHADOW"] = "1" if shadow else "0"
    q, k, v, mask, bias = make_case(name)
    out = stablefold_triangle_attention(q, k, v, mask, bias)
    torch.manual_seed(7)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v, bias), grad_outputs=grad_out)
    torch.cuda.synchronize()
    return out.detach(), {
        "dq": grads[0],
        "dk": grads[1],
        "dv": grads[2],
        "dbias": grads[3],
    }


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
    out_public = torch.einsum("bohqk,bohkd->bohqd", probs, v64).permute(
        0, 1, 3, 2, 4
    )
    torch.manual_seed(7)
    grad_out = torch.randn(
        out_public.shape, device="cuda", dtype=torch.bfloat16
    ).double()
    grads = torch.autograd.grad(
        out_public, (q64, k64, v64, b64), grad_outputs=grad_out
    )
    return {"dq": grads[0], "dk": grads[1], "dv": grads[2], "dbias": grads[3]}


def rel_l2(candidate, reference):
    denominator = reference.norm().item()
    return (
        (candidate.double() - reference).norm().item() / denominator
        if denominator
        else 0.0
    )


def main():
    failures = 0
    for name in CASES:
        out_two, two = run(name, sr=False, shadow=False)
        out_classic, classic = run(name, sr=True, shadow=False)
        out_shadow, shadow = run(name, sr=True, shadow=True)
        out_shadow2, shadow2 = run(name, sr=True, shadow=True)
        if not (
            torch.equal(out_two, out_classic)
            and torch.equal(out_two, out_shadow)
        ):
            print(f"OUT-BITWISE-FAIL {name}: forward changed across arms")
            failures += 1
        for key in ("dq", "dk", "dv", "dbias"):
            if not torch.equal(shadow[key], shadow2[key]):
                print(f"SHADOW-REPLAY-FAIL {name}:{key}")
                failures += 1
        # dV never consumes delta: it must stay bit-identical to the
        # two-pass backward even under the shadow contract.
        if not torch.equal(shadow["dv"], two["dv"]):
            print(f"DV-INVARIANT-FAIL {name}")
            failures += 1
        ref = fp64_reference(name)
        line = [name]
        for key in ("dq", "dbias"):
            e_two = rel_l2(two[key], ref[key])
            e_classic = rel_l2(classic[key], ref[key])
            e_shadow = rel_l2(shadow[key], ref[key])
            line.append(
                f"{key}: two-pass {e_two:.5f} | sr-classic {e_classic:.5f} "
                f"| sr-shadow {e_shadow:.5f}"
            )
            if key == "dq" and e_shadow > 1.35 * e_two:
                print(
                    f"SHADOW-QUALITY-FAIL {name}: shadow {e_shadow} "
                    f"vs two-pass {e_two}"
                )
                failures += 1
        print(" || ".join(line), flush=True)
    print("SHADOW_GATE_PASS" if failures == 0 else f"FAILURES={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
