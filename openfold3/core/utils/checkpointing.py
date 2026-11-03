# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch
import torch.utils.checkpoint

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed
    from deepspeed.runtime.activation_checkpointing.checkpointing import (
        non_reentrant_checkpoint as ds_non_reentrant_checkpoint,
    )


BLOCK_ARG = Any
BLOCK_ARGS = Sequence[BLOCK_ARG]

_CHECKPOINT_ACTIVE: ContextVar[bool] = ContextVar(
    "openfold3_checkpoint_active", default=False
)


def is_inference_without_checkpoint(
    module: torch.nn.Module, *inputs: torch.Tensor | None
) -> bool:
    """Whether automatic inference-only kernels may replace trainable math.

    Reentrant checkpointing runs its first forward under no_grad, including
    when the module is in eval mode. Our checkpoint wrapper marks that entire
    call, since intermediate projections can erase the inputs' requires_grad.
    The input check also covers direct torch checkpoint calls at this boundary.
    External checkpoint calls around larger eval-mode functions should use
    get_checkpoint_fn so their nested modules see the same context.

    Ordinary module inference requires eval and no_grad (or inference_mode).
    Direct native inference APIs retain their explicit no_grad-only contract.
    """
    return (
        not module.training
        and not torch.is_grad_enabled()
        and not _CHECKPOINT_ACTIVE.get()
        and not any(value is not None and value.requires_grad for value in inputs)
    )


def is_deepspeed_configured() -> bool:
    return deepspeed_is_installed and deepspeed.checkpointing.is_configured()


def get_checkpoint_fn(use_reentrant: bool | None = None):
    if is_deepspeed_configured():
        if use_reentrant is False:
            checkpoint = ds_non_reentrant_checkpoint
        else:
            checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    is_native_checkpoint = checkpoint is torch.utils.checkpoint.checkpoint

    def checkpoint_with_context(function, *args, **kwargs):
        # Bind the factory's default even when callers omit the keyword. An
        # explicit native override still determines whether a marker is needed.
        # DeepSpeed selects a separate function and does not accept this option.
        if is_native_checkpoint and use_reentrant is not None:
            kwargs.setdefault("use_reentrant", use_reentrant)
        actual_reentrant = (
            kwargs.get("use_reentrant") if is_native_checkpoint else use_reentrant
        )
        # Non-reentrant checkpointing keeps autograd enabled in the original
        # forward and recomputation, so inference dispatch is already excluded.
        if actual_reentrant is False:
            return checkpoint(function, *args, **kwargs)
        if is_native_checkpoint and torch.compiler.is_compiling():
            # Dynamo lowers native checkpointing to a non-reentrant graph,
            # whose forward keeps autograd enabled. A Python ContextVar write
            # would prevent that graph from being captured. DeepSpeed still
            # needs the eager marker and never takes this compiler shortcut.
            return checkpoint(function, *args, **kwargs)

        @wraps(function)
        def run_function(*inputs, **input_kwargs):
            # Mark both the first forward and every recomputation. ContextVar
            # makes nested calls and independent Python threads restore safely,
            # including when checkpoint's early-stop unwinds the recomputation.
            token = _CHECKPOINT_ACTIVE.set(True)
            try:
                return function(*inputs, **input_kwargs)
            finally:
                _CHECKPOINT_ACTIVE.reset(token)

        return checkpoint(run_function, *args, **kwargs)

    return checkpoint_with_context


@torch.jit.ignore
def checkpoint_blocks(
    blocks: list[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: int | None,
    use_reentrant: bool | None = None,
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
        use_reentrant:
            Whether to use reentrant checkpointing. If set, torch checkpointing
            will be used.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn(use_reentrant=use_reentrant)

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt

        if is_deepspeed_configured():
            args = checkpoint(chunker(s, e), *args)
        else:
            args = checkpoint(chunker(s, e), *args, use_reentrant=use_reentrant)

        args = wrap(args)

    return args


@torch.jit.ignore
def checkpoint_section(
    fn: Callable,
    args: BLOCK_ARGS,
    apply_ckpt: bool | None = True,
    use_reentrant: bool | None = None,
):
    """
    Apply checkpointing to a single function.

    Args:
        fn:
            Function to checkpoint
        args:
            Tuple of arguments for the first block.
        apply_ckpt:
            Whether to apply checkpointing. If None, no checkpointing
            is performed.
        use_reentrant:
            Whether to use reentrant checkpointing. If set, torch checkpointing
            will be used.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(fn, a):
        return fn(*a)

    # Avoids mishaps when the function takes just one argument
    args = wrap(args)

    if not apply_ckpt or not torch.is_grad_enabled():
        return exec(fn, args)

    checkpoint = get_checkpoint_fn(use_reentrant=use_reentrant)

    if is_deepspeed_configured():
        args = checkpoint(fn, *args)
    else:
        args = checkpoint(fn, *args, use_reentrant=use_reentrant)

    return args
