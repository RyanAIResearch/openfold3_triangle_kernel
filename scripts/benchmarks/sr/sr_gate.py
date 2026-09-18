"""Golden bitwise gate for the SR dq_dbias restructure.

Usage:
  python sr_gate.py baseline   # capture golden tensors from the current build
  python sr_gate.py compare    # rebuild-and-compare bitwise vs the golden set

Runs the public facade (fwd + autograd.grad) over a shape battery spanning
trunk crops, odd-N padding, outer!=sequence, batch>1 with bias broadcast,
io<storage, and D=16/64 lanes. Also asserts self-replay bitwise per shape.
"""

import os
import sys

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

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_golden.pt")

# name -> (batch, outer, heads, sequence, dim, bias_prefix_is_one)
SHAPES = {
    "n640_d32": (1, 640, 4, 640, 32, False),
    "n384_d32": (1, 384, 4, 384, 32, False),
    "n253_d32": (1, 253, 4, 253, 32, False),
    "n256_d16": (1, 256, 4, 256, 16, False),
    "n128_d64": (1, 128, 2, 128, 64, False),
    "outer24_n128_d32": (1, 24, 4, 128, 32, False),
    "b2_bias_bcast_n96_d32": (2, 96, 4, 96, 32, True),
    "n128_d24_iolane": (1, 128, 4, 128, 24, False),
    # user-law battery extension: crops {597, 100, 96}, D in {8, 48},
    # batch>1 combined with outer != sequence
    "n597_d32": (1, 597, 4, 597, 32, False),
    "n100_d8_msarow": (1, 100, 4, 100, 8, False),
    "n64_d48_dit": (1, 64, 2, 64, 48, False),
    "b2_outer12_n128_d32": (2, 12, 4, 128, 32, False),
}


def make_case(name):
    import zlib

    batch, outer, heads, seq, dim, bias_one = SHAPES[name]
    torch.manual_seed(zlib.crc32(name.encode()) % (2**31))
    dev = "cuda"
    shape = (batch, outer, heads, seq, dim)
    q = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(batch, outer, 1, 1, seq, device=dev, dtype=torch.float32)
    if name == "n253_d32":
        mask[:, :, :, :, -5:] = float("-inf")  # real masked tail keys
    bias_batch = 1 if bias_one else batch
    bias = (
        0.1
        * torch.randn(
            bias_batch, 1, heads, seq, seq, device=dev, dtype=torch.float32
        )
    ).requires_grad_()
    return q, k, v, mask, bias


def run_case(name):
    q, k, v, mask, bias = make_case(name)
    out = turbopairformer_triangle_attention(q, k, v, mask, bias)
    torch.manual_seed(1234)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v, bias), grad_outputs=grad_out)
    torch.cuda.synchronize()
    return {
        "out": out.detach().cpu(),
        "dq": grads[0].cpu(),
        "dk": grads[1].cpu(),
        "dv": grads[2].cpu(),
        "dbias": grads[3].cpu(),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    results = {}
    for name in SHAPES:
        first = run_case(name)
        second = run_case(name)
        for key in first:
            assert torch.equal(first[key], second[key]), (
                f"REPLAY MISMATCH {name}:{key}"
            )
        results[name] = first
        print(f"{name}: replay bitwise OK", flush=True)

    if mode == "baseline":
        torch.save(results, GOLDEN)
        print(f"golden saved -> {GOLDEN}")
        return

    golden = torch.load(GOLDEN, weights_only=True)
    failures = 0
    for name, tensors in golden.items():
        for key, reference in tensors.items():
            candidate = results[name][key]
            if torch.equal(candidate, reference):
                continue
            failures += 1
            diff = (candidate.float() - reference.float()).abs()
            print(
                f"MISMATCH {name}:{key} maxabs={diff.max().item():.3e} "
                f"nmismatch={(diff > 0).sum().item()}/{diff.numel()}"
            )
    if failures == 0:
        print("ALL_BITWISE_PASS")
    else:
        print(f"FAILURES={failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
