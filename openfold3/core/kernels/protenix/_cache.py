# SPDX-License-Identifier: Apache-2.0
"""Writable, revision/device/rank-scoped cache for vendored autotune results."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

UPSTREAM_REVISION = "4c355be4553512f72453ecbfb65e69f4c35d1413"


def get_config_dir(device: Any = None) -> Path:
    """Resolve the runtime cache without writing into an installed package."""
    import torch

    override = os.environ.get("OPENFOLD3_PROTENIX_CACHE_DIR")
    root = (
        Path(override).expanduser()
        if override
        else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "openfold3"
        / "protenix"
    )
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    device = torch.cuda.current_device() if device is None else device
    index = torch.device(device).index if not isinstance(device, int) else device
    if index is None:
        index = torch.cuda.current_device()
    return _device_cache_path(root, rank, index)


@lru_cache(maxsize=128)
def _device_cache_path(root: Path, rank: int, index: int) -> Path:
    import torch
    import triton

    capability = torch.cuda.get_device_capability(index)
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", torch.cuda.get_device_name(index))
    gpu = f"{name}-sm{capability[0]}{capability[1]}-device{index}"
    # A Triton upgrade must not inherit stale serialized configuration choices.
    runtime = f"torch-{torch.__version__}-triton-{triton.__version__}"
    runtime = re.sub(r"[^A-Za-z0-9_.-]", "-", runtime)
    return root / UPSTREAM_REVISION / gpu / runtime / f"rank-{rank}"
