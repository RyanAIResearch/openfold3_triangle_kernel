"""APB-site forensic: absolute norms, cosine, and cross-site share for the
three delta contracts — guards the 0.81 relative-error headline against the
small-denominator critique."""
import glob, os, sys
sys.path.insert(0, "/home/ubuntu/xzz/of3_sr_dq")
os.environ["OPENFOLD3_TRIANGLE_ALLOW_JIT"] = "1"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.getcwd() + "/torch_ext_sr")
import torch
from sr_ladder import load_capture, kernel_forward, fp64_pieces, delta_from_out
from openfold3.core.kernels.cuda.turbopairformer_triangle_dq_dbias import (
    turbopairformer_triangle_sr_backward,
)

CASES = {
    "APB_O1_H16_D24": "/home/ubuntu/xzz/of3_perf_runs/sr_capture_matrix/c640_skip60/case_02_1x1x16x640x24.pt",
    "trunk_D32_same_batch": "/home/ubuntu/xzz/of3_perf_runs/sr_capture_matrix/c640_skip60/case_00_1x640x4x640x32.pt",
}

def stats(candidate, reference):
    c, r = candidate.double().flatten(), reference.double().flatten()
    err = c - r
    cos = torch.nn.functional.cosine_similarity(c, r, dim=0).item()
    return {
        "ref_l2": r.norm().item(),
        "abs_err_l2": err.norm().item(),
        "rel_l2": (err.norm() / r.norm()).item(),
        "max_abs_err": err.abs().max().item(),
        "cosine": cos,
    }

rows = {}
for name, path in CASES.items():
    q, k, v, mask, bias, grad_out = load_capture(path)
    out, lse, shadow = kernel_forward(q, k, v, mask, bias)
    ref = fp64_pieces(q, k, v, mask, bias, grad_out)
    arms = {
        "classic": delta_from_out(grad_out, out),
        "shadow": delta_from_out(grad_out, out.float() + shadow.float()),
        "perfect": ref["delta_perfect"].float().contiguous(),
    }
    rows[name] = {"dq_ref_l2": ref["dq"].norm().item(),
                  "dq_ref_maxabs": ref["dq"].abs().max().item(),
                  "numel": ref["dq"].numel()}
    for arm, delta in arms.items():
        dq, dk, dv, dbias = turbopairformer_triangle_sr_backward(
            q, k, v, grad_out, lse, delta, mask, bias)
        rows[name][arm] = stats(dq, ref["dq"])
    del ref
    torch.cuda.empty_cache()

print(f"{'site':<22}{'arm':<9}{'||ref||':>10}{'||err||':>10}{'rel':>8}"
      f"{'maxabs':>10}{'cos':>9}")
for name, data in rows.items():
    for arm in ("classic", "shadow", "perfect"):
        s = data[arm]
        print(f"{name:<22}{arm:<9}{s['ref_l2']:>11.3e}{s['abs_err_l2']:>11.3e}"
              f"{s['rel_l2']:>8.4f}{s['max_abs_err']:>11.3e}{s['cosine']:>9.5f}")
apb, trunk = rows["APB_O1_H16_D24"], rows["trunk_D32_same_batch"]
print(f"\nAPB   dq ref l2 {apb['dq_ref_l2']:.3e} maxabs {apb['dq_ref_maxabs']:.3e} numel {apb['numel']}")
print(f"trunk dq ref l2 {trunk['dq_ref_l2']:.3e} maxabs {trunk['dq_ref_maxabs']:.3e} numel {trunk['numel']}")
print(f"APB classic ABSOLUTE err vs trunk classic ABSOLUTE err: "
      f"{apb['classic']['abs_err_l2']:.3e} vs {trunk['classic']['abs_err_l2']:.3e}")
print("FORENSIC_DONE")
