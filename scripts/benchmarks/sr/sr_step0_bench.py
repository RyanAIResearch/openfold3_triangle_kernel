"""SR Step-0: per-kernel decomposition of the StableFold triangle backward.

Runs the production facade (wheel_env_a94b AOT binaries) at trunk shape
B=1, O=N, H=4, D=32 and reports per-CUDA-kernel time from torch.profiler,
plus event-timed forward / backward walls. No kernel edits; measurement only.
"""

import sys

WHEEL = "/home/ubuntu/xzz/of3_perf_runs/crop640_c527e41bc.XAkp3Y/wheel_env_a94b"
sys.path.insert(0, WHEEL)

import torch  # noqa: E402

from openfold3.core.kernels.cuda.stablefold_triangle_attention import (  # noqa: E402
    stablefold_triangle_attention,
)

REPS_PROFILE = 30
REPS_EVENTS = 50
WARMUP = 12


def make_inputs(n: int, heads: int = 4, dim: int = 32):
    torch.manual_seed(0)
    dev = "cuda"
    shape = (1, n, heads, n, dim)
    q = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device=dev, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.zeros(1, n, 1, 1, n, device=dev, dtype=torch.float32)
    bias = (
        0.1 * torch.randn(1, 1, heads, n, n, device=dev, dtype=torch.float32)
    ).requires_grad_()
    return q, k, v, mask, bias


def one_iter(q, k, v, mask, bias, grad_out):
    out = stablefold_triangle_attention(q, k, v, mask, bias)
    torch.autograd.grad(out, (q, k, v, bias), grad_outputs=grad_out)


def event_walls(q, k, v, mask, bias, grad_out):
    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fwd_ms, bwd_ms = [], []
    for _ in range(REPS_EVENTS):
        start.record()
        out = stablefold_triangle_attention(q, k, v, mask, bias)
        mid.record()
        torch.autograd.grad(out, (q, k, v, bias), grad_outputs=grad_out)
        end.record()
        torch.cuda.synchronize()
        fwd_ms.append(start.elapsed_time(mid))
        bwd_ms.append(mid.elapsed_time(end))
    fwd_ms.sort()
    bwd_ms.sort()
    med = REPS_EVENTS // 2
    return fwd_ms[med], bwd_ms[med]


def profile_kernels(q, k, v, mask, bias, grad_out):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(REPS_PROFILE):
            one_iter(q, k, v, mask, bias, grad_out)
        torch.cuda.synchronize()
    rows = []
    for item in prof.key_averages():
        cuda_us = getattr(item, "self_device_time_total", 0) or 0
        if item.device_type != torch.autograd.DeviceType.CUDA and cuda_us == 0:
            continue
        if cuda_us > 0:
            rows.append((cuda_us, item.count, item.key))
    rows.sort(reverse=True)
    return rows


def main() -> None:
    for n in (640, 384, 768):
        q, k, v, mask, bias = make_inputs(n)
        grad_out = None
        out = stablefold_triangle_attention(q, k, v, mask, bias)
        grad_out = torch.randn_like(out)
        for _ in range(WARMUP):
            one_iter(q, k, v, mask, bias, grad_out)
        torch.cuda.synchronize()
        fwd_ms, bwd_ms = event_walls(q, k, v, mask, bias, grad_out)
        rows = profile_kernels(q, k, v, mask, bias, grad_out)
        total_us = sum(r[0] for r in rows)
        print(f"\n===== N={n}  fwd {fwd_ms:.3f} ms  bwd {bwd_ms:.3f} ms "
              f"(facade walls, median of {REPS_EVENTS}) =====")
        print(f"{'cuda total ms':>14} {'ms/iter':>9} {'calls':>6}  kernel")
        for cuda_us, count, key in rows:
            if cuda_us / 1000.0 / REPS_PROFILE < 0.005:
                continue
            print(
                f"{cuda_us / 1000.0:14.3f} {cuda_us / 1000.0 / REPS_PROFILE:9.4f} "
                f"{count:6d}  {key[:100]}"
            )
        print(f"{'-' * 14}")
        print(f"{total_us / 1000.0:14.3f} {total_us / 1000.0 / REPS_PROFILE:9.4f}"
              f"         ALL CUDA")


if __name__ == "__main__":
    main()
