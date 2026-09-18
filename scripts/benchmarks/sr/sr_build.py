"""Build (dev-JIT) all four TurboPairFormer extensions from the SR worktree."""

import os
import sys
import time

ROOT = os.environ.get("SR_ROOT", "/home/ubuntu/xzz/of3_sr_dq")
sys.path.insert(0, ROOT)
os.environ["OPENFOLD3_TRIANGLE_ALLOW_JIT"] = "1"
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    "/tmp/claude-1000/-home-ubuntu-xzz/7e522781-b3b4-47bf-a008-e0fc688a239d/"
    "scratchpad/torch_ext_sr",
)

t0 = time.time()
from openfold3.core.kernels.cuda import (  # noqa: E402
    turbopairformer_triangle_attention as facade,
)

for name in ("_forward_extension", "_delta_extension"):
    getattr(facade, name)()
    print(f"{name} ready at {time.time() - t0:.0f}s", flush=True)

from openfold3.core.kernels.cuda import turbopairformer_triangle_dkdv as dkdv_mod  # noqa: E402
from openfold3.core.kernels.cuda import (  # noqa: E402
    turbopairformer_triangle_dq_dbias as dq_mod,
)

dkdv_mod.load_extension()
print(f"dkdv ready at {time.time() - t0:.0f}s", flush=True)
dq_mod.load_extension()
print(f"dq_dbias ready at {time.time() - t0:.0f}s", flush=True)
print("ALL_BUILT", flush=True)
