"""Quick A/B timing: backward wall + per-kernel CUDA times at trunk 640."""

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

REPS = 60


def bench(n, heads=4, dim=32):
    torch.manual_seed(0)
    shape = (1, n, heads, n, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(1, n, 1, 1, n, device="cuda", dtype=torch.float32)
    bias = (
        0.1 * torch.randn(1, 1, heads, n, n, device="cuda", dtype=torch.float32)
    ).requires_grad_()
    out = turbopairformer_triangle_attention(q, k, v, mask, bias)
    grad_out = torch.randn_like(out)

    def one():
        result = turbopairformer_triangle_attention(q, k, v, mask, bias)
        torch.autograd.grad(result, (q, k, v, bias), grad_outputs=grad_out)

    for _ in range(12):
        one()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fwd, bwd = [], []
    for _ in range(REPS):
        start.record()
        result = turbopairformer_triangle_attention(q, k, v, mask, bias)
        mid.record()
        torch.autograd.grad(result, (q, k, v, bias), grad_outputs=grad_out)
        end.record()
        torch.cuda.synchronize()
        fwd.append(start.elapsed_time(mid))
        bwd.append(mid.elapsed_time(end))
    fwd.sort()
    bwd.sort()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(20):
            one()
        torch.cuda.synchronize()
    dq_ms = dkdv_ms = 0.0
    for item in prof.key_averages():
        cuda_us = getattr(item, "self_device_time_total", 0) or 0
        if "dq_dbias_leaf" in item.key or "dq_dbias_wg" in item.key:
            dq_ms += cuda_us / 1000.0 / 20
        elif "dkdv" in item.key:
            dkdv_ms += cuda_us / 1000.0 / 20
    print(
        f"N={n}: fwd {fwd[REPS // 2]:.3f}  bwd {bwd[REPS // 2]:.3f}  "
        f"dq_kernel {dq_ms:.3f}  dkdv {dkdv_ms:.3f}  (ms, median/{REPS})",
        flush=True,
    )


for n in (640, 384, 768):
    bench(n)
