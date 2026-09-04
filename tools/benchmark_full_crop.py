"""CPU-side no-padding validation for backend training benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial, partialmethod
from typing import Any

import torch


def require_full_crop_collate(
    samples: list[Any],
    *,
    collator: Any,
    expected_tokens: int,
) -> Any:
    """Collate on CPU and reject any sample that would pad the token crop."""

    batch = collator(samples)
    if not isinstance(batch, Mapping) or not isinstance(
        batch.get("token_mask"), torch.Tensor
    ):
        raise RuntimeError("collated training batch has no tensor token_mask")
    token_mask = batch["token_mask"]
    if token_mask.device.type != "cpu":
        raise RuntimeError(
            "full-crop validation must run on CPU before device transfer"
        )
    if token_mask.shape[-1] != expected_tokens:
        raise RuntimeError(
            "token dimension does not match configured crop: "
            f"shape={tuple(token_mask.shape)}, expected={expected_tokens}"
        )
    token_counts = token_mask.to(dtype=torch.int64).sum(dim=-1)
    if not bool(torch.all(token_counts == expected_tokens)):
        pdb_ids = batch.get("pdb_id", "unknown")
        raise RuntimeError(
            "benchmark sample would use token padding: "
            f"pdb_id={pdb_ids}, real_token_counts={token_counts.tolist()}, "
            f"expected={expected_tokens}"
        )
    return batch


def generate_dataloader_with_full_crop_gate(
    data_module_instance: Any,
    mode: Any,
    sampler: Any = None,
    *,
    original_generate_dataloader: Any,
    expected_tokens: int,
) -> Any:
    """Build the official DataLoader, then wrap its still-canonical collator."""

    dataloader = original_generate_dataloader(data_module_instance, mode, sampler)
    dataloader.collate_fn = partial(
        require_full_crop_collate,
        collator=dataloader.collate_fn,
        expected_tokens=expected_tokens,
    )
    return dataloader


def install_full_crop_collator(expected_tokens: int) -> None:
    """Install a forkserver-pickleable no-padding gate in OpenFold3's collator."""

    from openfold3.core.data.framework import data_module

    data_module_class: Any = data_module.DataModule
    original_generate_dataloader = data_module_class.generate_dataloader
    data_module_class.generate_dataloader = partialmethod(
        generate_dataloader_with_full_crop_gate,
        original_generate_dataloader=original_generate_dataloader,
        expected_tokens=expected_tokens,
    )
