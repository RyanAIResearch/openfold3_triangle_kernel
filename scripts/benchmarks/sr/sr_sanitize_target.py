"""Minimal fwd+bwd shapes for compute-sanitizer (racecheck is ~100x slow)."""

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

for batch, outer, heads, seq, dim in (
    (1, 130, 2, 130, 32),  # pads to 192; odd outer tail (130 % 4 != 0)
    (1, 128, 2, 128, 64),
    (1, 128, 2, 128, 16),
):
    torch.manual_seed(7)
    shape = (batch, outer, heads, seq, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(batch, outer, 1, 1, seq, device="cuda", dtype=torch.float32)
    bias = (
        0.1
        * torch.randn(batch, 1, heads, seq, seq, device="cuda", dtype=torch.float32)
    ).requires_grad_()
    out = turbopairformer_triangle_attention(q, k, v, mask, bias)
    torch.autograd.grad(out, (q, k, v, bias), grad_outputs=torch.ones_like(out))
    torch.cuda.synchronize()
    print(f"shape {shape} done", flush=True)
print("SANITIZE_TARGET_DONE")
