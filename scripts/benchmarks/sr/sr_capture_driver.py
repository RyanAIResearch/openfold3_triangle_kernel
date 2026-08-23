"""Run `run_openfold train` with a capture patch on the StableFold facade.

Saves the first CAPTURE_LIMIT triangle-attention calls (inputs + the
gradient that later arrives at the output) from a real pretrained
training step to CAPTURE_DIR/case_XX_<tag>.pt for the delta ladder.
Patches before any model import; everything else is the stock trainer.
"""

import os
import sys

CAPTURE_DIR = os.environ["SR_CAPTURE_DIR"]
CAPTURE_LIMIT = int(os.environ.get("SR_CAPTURE_LIMIT", "8"))
CAPTURE_SKIP = int(os.environ.get("SR_CAPTURE_SKIP", "0"))

os.makedirs(CAPTURE_DIR, exist_ok=True)

import torch  # noqa: E402

from openfold3.core.kernels.cuda import (  # noqa: E402
    stablefold_triangle_attention as facade_module,
)

_original = facade_module.stablefold_triangle_attention
_state = {"captured": 0, "pending": 0, "seen": 0}


def _capturing(q, k, v, res_mask, pair_bias):
    out = _original(q, k, v, res_mask, pair_bias)
    if torch.is_grad_enabled() and out.requires_grad:
        _state["seen"] += 1
    if (
        torch.is_grad_enabled()
        and out.requires_grad
        and _state["seen"] > CAPTURE_SKIP
        and _state["captured"] + _state["pending"] < CAPTURE_LIMIT
    ):
        index = _state["captured"] + _state["pending"]
        _state["pending"] += 1
        snapshot = {
            "q": q.detach().to("cpu", copy=True),
            "k": k.detach().to("cpu", copy=True),
            "v": v.detach().to("cpu", copy=True),
            "mask": res_mask.detach().to("cpu", copy=True),
            "bias": pair_bias.detach().to("cpu", copy=True),
        }
        shape_tag = "x".join(str(int(value)) for value in q.shape)

        def _save_grad(grad_out, snapshot=snapshot, index=index,
                       shape_tag=shape_tag):
            snapshot["grad_out"] = grad_out.detach().to("cpu", copy=True)
            path = os.path.join(
                CAPTURE_DIR, f"case_{index:02d}_{shape_tag}.pt"
            )
            torch.save(snapshot, path)
            _state["pending"] -= 1
            _state["captured"] += 1
            print(f"[sr-capture] saved {path}", file=sys.stderr, flush=True)
            return None

        out.register_hook(_save_grad)
    return out


facade_module.stablefold_triangle_attention = _capturing

# The model's attention primitive may have imported the symbol by value;
# patch every already-imported alias too (none exist yet at this point,
# since we run before model imports), then hand off to the stock CLI.
from openfold3.run_openfold import cli  # noqa: E402

sys.exit(cli())
