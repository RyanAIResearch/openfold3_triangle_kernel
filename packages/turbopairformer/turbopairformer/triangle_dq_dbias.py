# SPDX-License-Identifier: Apache-2.0

"""Plan storage and launch TurboPairFormer Triangle Attention backward.

The split path computes dQ and dBias; single-reconstruction (SR) backward
computes dQ, dK, dV, and dBias together. GPU computation lives in the sibling
``.cu`` file; tensor preparation and autograd live in
``_triangle_attention.py``.

Both paths use the same dBias reduction tree. Each leaf combines four outer
rows into a BF16 plane with one writer per cell. The finalizer accumulates
these planes in ascending leaf order into FP32 dBias, without floating-point atomics.
Query slabs, batch chunks, and leaf waves control temporary storage without
changing that reduction order. See the CUDA source for the numerical contract.

The binary is installed under ``turbopairformer._ext.*``;
``_aot_runtime`` owns its module name, required exports, and loading policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

# Keep leaf membership fixed even when the workspace budget changes.
_OUTER_LEAF = 4
_QUERY_BLOCK = 64  # Automatic query slabs contain whole 64-row launch blocks.
_MAX_GRID_YZ = 65_535
_MAX_INT32 = (1 << 31) - 1
_MAX_INT64 = (1 << 63) - 1
_DEFAULT_WORKSPACE_BUDGET_BYTES = 256 * 1024 * 1024
# Load on first use, then reuse the successfully loaded extension module.
_EXTENSION: Any = None


@dataclass(frozen=True)
class WorkspacePlan:
    """Describe the reusable BF16 workspace and the launches that fill it.

    Its layout is ``[resident_leaves, batch_chunk, heads, query_span, keys]``.
    A query slab contains up to ``query_span`` query rows; a leaf wave contains
    up to ``resident_leaves`` consecutive leaves. Query slabs and batch chunks
    partition independent output cells. Leaf waves preserve the global leaf
    order, so none of these capacity choices changes the reduction tree.
    """

    # Workspace capacities; the final slab, chunk, or wave may use less.
    query_span: int
    resident_leaves: int
    batch_chunk: int
    # Total leaves and the number of iterations along each scheduling axis.
    leaves: int
    query_stripes: int
    leaf_waves: int
    batch_chunks: int
    # Each pair is one leaf-kernel launch followed by its dBias finalizer.
    launch_pairs: int
    # BF16 element count, followed by the requested planning policy.
    elements: int
    workspace_budget_bytes: int
    debug_full_plane: bool


def source_sha256() -> dict[str, str]:
    """Return ``{source filename: SHA256}`` without loading the extension.

    See ``_aot_runtime.source_sha256_for`` for manifest precedence, local-file
    fallback, and errors. A recorded hash need not match the current source.
    """
    from turbopairformer._aot_runtime import source_sha256_for

    source = Path(__file__).with_name("turbopairformer_triangle_dq_dbias.cu")
    return source_sha256_for("turbopairformer_dq_dbias", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    """Load once using the shared AOT/JIT policy; return the cached module.

    See ``_aot_runtime.load_extension_module`` for validation and fallback.
    ``verbose`` controls JIT logging; ``None`` means ``False``.
    """
    global _EXTENSION
    if _EXTENSION is None:
        from turbopairformer._aot_runtime import (
            _PRODUCTION_MODULE_CONTRACT,
            load_extension_module,
        )

        module_key = "turbopairformer_dq_dbias"
        aot_module, expected_symbols = _PRODUCTION_MODULE_CONTRACT[module_key]
        _EXTENSION = load_extension_module(
            module_key=module_key,
            aot_module=aot_module,
            source=Path(__file__).with_name("turbopairformer_triangle_dq_dbias.cu"),
            expected_symbols=expected_symbols,
            verbose=bool(verbose),
        )
    return _EXTENSION


def outer_leaf_for(sequence: int) -> int:
    """Return the canonical number of outer rows owned by one leaf.

    This is an algorithmic tile, not a sequence-length specialization. A
    constant leaf makes the reduction tree independent of the workspace
    budget; waves only decide how many already-defined leaves coexist.
    """

    if sequence <= 0:
        raise ValueError("sequence must be positive")
    return _OUTER_LEAF


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _batch_chunk_candidates(batch: int, capacity: int) -> tuple[int, ...]:
    """Return every objective breakpoint without scanning a huge batch.

    For fixed query span, the score changes only when either
    ``capacity // batch_chunk`` or ``ceil(batch / batch_chunk)`` changes.
    Evaluating the endpoints of those joint intervals is therefore exact.
    """

    max_batch_chunk = max(1, min(batch, capacity))
    candidates: set[int] = {1, max_batch_chunk}
    candidate_batch_chunk = 1
    while candidate_batch_chunk <= max_batch_chunk:
        leaf_capacity = capacity // candidate_batch_chunk
        leaf_capacity_interval_end = capacity // leaf_capacity
        batch_chunks = _ceil_div(batch, candidate_batch_chunk)
        batch_count_interval_end = (
            max_batch_chunk if batch_chunks == 1 else (batch - 1) // (batch_chunks - 1)
        )
        interval_end = min(
            max_batch_chunk, leaf_capacity_interval_end, batch_count_interval_end
        )
        candidates.add(candidate_batch_chunk)
        candidates.add(interval_end)
        candidate_batch_chunk = interval_end + 1
    return tuple(sorted(candidates))


@lru_cache(maxsize=1024)
def workspace_plan(
    sequence: int,
    heads: int,
    *,
    outer: int | None = None,
    batch: int = 1,
    workspace_budget_bytes: int = _DEFAULT_WORKSPACE_BUDGET_BYTES,
    query_span: int | None = None,
    batch_chunk: int | None = None,
    debug_full_plane: bool = False,
) -> WorkspacePlan:
    """Choose query, batch, and leaf capacities for the reusable dBias workspace.

    The planner minimizes leaf/finalizer launch pairs first, then minimizes
    the number of root passes.  This gives query slabs only when they reduce
    root traffic without increasing the best available launch count.  It
    enumerates query spans and batch/leaf tilings from the runtime dimensions;
    there is no sequence-length lookup table.

    With automatic sizing, the budget is a target with a minimum allocation
    of one batch item, one 64-row query block, and one leaf across all heads
    and keys. That minimum is ``O(H*N*query_block)`` and can exceed the target.
    Forced query spans, forced batch chunks, and ``debug_full_plane`` can also
    require more than the target. The budget covers the BF16 dBias workspace,
    not output tensors or the other temporary tensors used by SR backward.
    """

    # 1. Validate dimensions and optional forced capacities before planning.
    if outer is None:
        outer = sequence
    if sequence <= 0 or heads <= 0 or batch <= 0 or outer <= 0:
        raise ValueError("sequence, outer, heads, and batch must be positive")
    if (
        sequence > _MAX_INT32
        or heads > _MAX_GRID_YZ
        or batch > _MAX_INT32
        or batch * sequence > _MAX_INT32
        or batch * outer > _MAX_INT32
        or _ceil_div(sequence, _QUERY_BLOCK) > _MAX_GRID_YZ
    ):
        raise ValueError("batch/sequence/head count exceeds the CUDA grid contract")
    if workspace_budget_bytes <= 0:
        raise ValueError("workspace_budget_bytes must be positive")
    if query_span is not None and (
        query_span <= 0 or query_span > _MAX_INT32 or query_span % _QUERY_BLOCK != 0
    ):
        raise ValueError("query_span must be an int32 positive multiple of 64")
    if batch_chunk is not None and not 1 <= batch_chunk <= batch:
        raise ValueError("batch_chunk must lie in [1, batch]")
    if debug_full_plane and (query_span is not None or batch_chunk is not None):
        raise ValueError("debug_full_plane cannot be combined with a forced slab shape")

    leaves = _ceil_div(outer, outer_leaf_for(outer))
    budget_elements = workspace_budget_bytes // 2  # BF16 elements

    # Diagnostic path: keep the entire batch and query plane in each leaf.
    if debug_full_plane:
        plane_elements = batch * heads * sequence * sequence
        resident_leaves = min(
            leaves,
            max(1, budget_elements // plane_elements),
        )
        elements = resident_leaves * plane_elements
        if elements > _MAX_INT64:
            raise ValueError("full-plane workspace element count overflows int64")
        return WorkspacePlan(
            query_span=sequence,
            resident_leaves=resident_leaves,
            batch_chunk=batch,
            leaves=leaves,
            query_stripes=1,
            leaf_waves=_ceil_div(leaves, resident_leaves),
            batch_chunks=1,
            launch_pairs=_ceil_div(leaves, resident_leaves),
            elements=elements,
            workspace_budget_bytes=workspace_budget_bytes,
            debug_full_plane=True,
        )

    # 2. Try each query span and the batch capacities where the score changes.
    query_blocks = _ceil_div(sequence, _QUERY_BLOCK)
    query_candidates = (
        (query_span,)
        if query_span is not None
        else tuple(
            block_count * _QUERY_BLOCK for block_count in range(1, query_blocks + 1)
        )
    )
    best_score: tuple[int, ...] | None = None
    best_plan: WorkspacePlan | None = None
    for candidate_query_span in query_candidates:
        elements_per_batch_leaf = heads * candidate_query_span * sequence
        if (
            query_span is None
            and candidate_query_span != _QUERY_BLOCK
            and elements_per_batch_leaf > budget_elements
        ):
            # Only the irreducible one-query-block tile may exceed the target.
            # Otherwise a large query span could win on launch count by
            # turning the budget back into an O(N^2) full-plane allocation.
            continue
        # Capacity is measured in (batch slot, leaf) pairs.  One pair is the
        # irreducible, query-tiled workspace even when it exceeds the target.
        batch_leaf_capacity = max(1, budget_elements // elements_per_batch_leaf)
        chunk_candidates = (
            (batch_chunk,)
            if batch_chunk is not None
            else _batch_chunk_candidates(batch, batch_leaf_capacity)
        )
        for candidate_batch_chunk in chunk_candidates:
            assert candidate_batch_chunk is not None
            resident_leaves = min(
                leaves,
                max(1, batch_leaf_capacity // candidate_batch_chunk),
            )
            query_stripes = _ceil_div(sequence, candidate_query_span)
            leaf_waves = _ceil_div(leaves, resident_leaves)
            batch_chunks = _ceil_div(batch, candidate_batch_chunk)
            launch_pairs = query_stripes * leaf_waves * batch_chunks
            elements = resident_leaves * candidate_batch_chunk * elements_per_batch_leaf
            if elements > _MAX_INT64:
                continue
            # 3. Minimize launch count, then repeated FP32 root scans.
            # Remaining terms make ties stable and prefer fuller workspace
            # tiles. The tuple order is part of the planning policy.
            candidate_score = (
                launch_pairs,
                leaf_waves,
                batch_chunks * query_stripes,
                -elements,
                -candidate_batch_chunk,
                candidate_query_span,
            )
            if best_score is None or candidate_score < best_score:
                best_score = candidate_score
                best_plan = WorkspacePlan(
                    query_span=candidate_query_span,
                    resident_leaves=resident_leaves,
                    batch_chunk=candidate_batch_chunk,
                    leaves=leaves,
                    query_stripes=query_stripes,
                    leaf_waves=leaf_waves,
                    batch_chunks=batch_chunks,
                    launch_pairs=launch_pairs,
                    elements=elements,
                    workspace_budget_bytes=workspace_budget_bytes,
                    debug_full_plane=False,
                )
    if best_plan is None:
        raise ValueError("no query-slab plan fits the int64 workspace contract")
    return best_plan


def workspace_elements(
    sequence: int,
    heads: int,
    *,
    batch: int = 1,
    workspace_budget_bytes: int = _DEFAULT_WORKSPACE_BUDGET_BYTES,
    query_span: int | None = None,
    batch_chunk: int | None = None,
    debug_full_plane: bool = False,
) -> int:
    """Return the BF16 element count with the default ``outer=sequence``."""

    return workspace_plan(
        sequence,
        heads,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
        debug_full_plane=debug_full_plane,
    ).elements


def turbopairformer_reduce_broadcast_dbias(
    per_batch_dbias: torch.Tensor,
    *,
    target_prefix: tuple[int, ...],
    bias_prefix: tuple[int, ...],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a flattened batch dBias with one fixed CUDA-owned order.

    The kernel visits flattened target batches in ascending order and rounds
    every public per-batch BF16 contribution before the FP32 accumulation.
    This is the same result obtained by summing independent public calls, but
    it does not delegate the training trajectory to ``ExpandBackward``.
    """

    # Match the original bias batch dimensions against the expanded input.
    if len(bias_prefix) > len(target_prefix):
        raise ValueError("bias prefix rank cannot exceed target prefix rank")
    aligned_bias_prefix = (1,) * (len(target_prefix) - len(bias_prefix)) + tuple(
        bias_prefix
    )
    for target_extent, bias_extent in zip(
        target_prefix, aligned_bias_prefix, strict=True
    ):
        if (
            target_extent <= 0
            or bias_extent <= 0
            or bias_extent not in (1, target_extent)
        ):
            raise ValueError("bias prefix must broadcast exactly to target prefix")
    expected_batch = 1
    for dimension in target_prefix:
        expected_batch *= int(dimension)
    if per_batch_dbias.shape[0] != expected_batch:
        raise ValueError("per_batch_dbias does not match flattened target prefix")
    # The CUDA reduction rounds each batch contribution to BF16 before the
    # fixed FP32 sum, then writes the public BF16 bias gradient.
    if out is None:
        out = torch.empty(
            tuple(bias_prefix) + tuple(per_batch_dbias.shape[1:]),
            device=per_batch_dbias.device,
            dtype=torch.bfloat16,
        )
    with torch.cuda.device(per_batch_dbias.device), torch.no_grad():
        load_extension().turbopairformer_reduce_broadcast_dbias_out(
            per_batch_dbias,
            out,
            list(target_prefix),
            list(bias_prefix),
        )
    return out


def turbopairformer_triangle_dq_dbias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_output: torch.Tensor,
    logsumexp: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    workspace: torch.Tensor | None = None,
    workspace_budget_bytes: int = _DEFAULT_WORKSPACE_BUDGET_BYTES,
    query_span: int | None = None,
    batch_chunk: int | None = None,
    debug_full_plane: bool = False,
    out_dq: torch.Tensor | None = None,
    out_dbias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run split backward and return BF16 dQ and per-batch FP32 dBias.

    Q/K/V use ``[B, O, H, N, D]``; dOutput uses ``[B, O, N, H, D]``.
    ``logsumexp`` is the forward FP32 log-sum-exp; ``delta`` is the
    caller-supplied FP32 softmax-backward correction for each query row.
    The returned dBias has shape ``[B, 1, H, N, N]``; a broadcast bias input
    still needs :func:`turbopairformer_reduce_broadcast_dbias` afterward.

    Reuse supplied workspace and output tensors, or allocate them here.
    The CUDA entry point runs all query slabs, batch chunks, and leaf waves.
    This path computes no dK/dV; those gradients use the separate split kernel.
    """

    # 1. Plan workspace using the kernel's explicit batch and outer axes.
    outer = int(q.shape[1])
    heads = int(q.shape[2])
    sequence = int(q.shape[3])
    batch = int(q.shape[0])
    plan = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
        debug_full_plane=debug_full_plane,
    )
    # 2. Allocate reusable BF16 leaf storage and the final gradient tensors.
    if workspace is None:
        workspace = torch.empty(
            plan.elements,
            device=q.device,
            dtype=torch.bfloat16,
        )
    elif workspace.numel() < plan.elements:
        raise ValueError(
            f"workspace has {workspace.numel()} elements, "
            f"but this plan requires {plan.elements}"
        )
    if out_dq is None:
        out_dq = torch.empty_like(q)
    if out_dbias is None:
        out_dbias = torch.empty(
            (batch, 1, heads, sequence, sequence),
            device=q.device,
            dtype=torch.float32,
        )
    # 3. Launch on q's device with flat, contiguous FP32 row state.
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().turbopairformer_triangle_dq_dbias_out(
            q,
            k,
            v,
            d_output,
            logsumexp.reshape(-1).contiguous(),
            delta.reshape(-1).contiguous(),
            mask,
            pair_bias,
            out_dq,
            out_dbias,
            workspace,
            plan.query_span,
            plan.resident_leaves,
            plan.batch_chunk,
        )
    return out_dq, out_dbias


def turbopairformer_triangle_sr_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_output: torch.Tensor,
    logsumexp: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    workspace_budget_bytes: int = _DEFAULT_WORKSPACE_BUDGET_BYTES,
    query_span: int | None = None,
    batch_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct each probability tile once for dQ, dK, dV, and dBias.

    Input layouts and the per-batch FP32 dBias result match the split path.
    dQ/dK/dV outputs are BF16. The caller-supplied FP32 delta is used to form
    dS; the public facade computes delta using the forward rounding residual
    by default. Requires ``sequence % 64 == 0``, ``D <= 64``, and ``D % 8 == 0``.

    The dBias budget does not cover BF16 dQ partials or FP32 dK/dV continuation
    storage. :func:`sr_workspace_plan` accounts for the continuation cost when
    comparing alternative dBias plans; dQ partials have the same size for all
    of them. CUDA folds those partials in ascending key-block order.
    """

    # 1. Check the SR tile constraints and choose a dBias workspace plan.
    outer = int(q.shape[1])
    heads = int(q.shape[2])
    sequence = int(q.shape[3])
    head_dim = int(q.shape[4])
    batch = int(q.shape[0])
    if sequence % 64 != 0 or head_dim > 64 or head_dim % 8 != 0:
        raise ValueError("SR backward requires sequence % 64 == 0 and D <= 64")
    plan = sr_workspace_plan(
        sequence,
        heads,
        head_dim,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
    )
    # 2. Allocate the reusable dBias slab and one dQ partial per key block.
    workspace = torch.empty(plan.elements, device=q.device, dtype=torch.bfloat16)
    key_blocks = sequence // 64
    dq_partials = torch.empty(
        (key_blocks, batch * outer, heads, sequence, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    # 3. Keep dK/dV partial sums in FP32 across query-slab launches.
    # dK/dV are reduced across the query axis. When the bounded dBias
    # workspace forces multiple query stripes, preserve their FP32 MMA
    # accumulators between launches so a launch boundary remains a pure
    # store/reload and does not change (or discard) arithmetic.
    continuation_elements = q.numel() if plan.query_stripes > 1 else 0
    dk_continuation = torch.empty(
        continuation_elements, device=q.device, dtype=torch.float32
    )
    dv_continuation = torch.empty_like(dk_continuation)
    # 4. Allocate the final gradients, then launch all waves and reductions.
    out_dq = torch.empty_like(q)
    out_dk = torch.empty_like(q)
    out_dv = torch.empty_like(q)
    out_dbias = torch.empty(
        (batch, 1, heads, sequence, sequence),
        device=q.device,
        dtype=torch.float32,
    )
    with torch.cuda.device(q.device), torch.no_grad():
        load_extension().turbopairformer_triangle_sr_backward_out(
            q,
            k,
            v,
            d_output,
            logsumexp.reshape(-1).contiguous(),
            delta.reshape(-1).contiguous(),
            mask,
            pair_bias,
            out_dq,
            out_dk,
            out_dv,
            out_dbias,
            workspace,
            dq_partials,
            dk_continuation,
            dv_continuation,
            plan.query_span,
            plan.resident_leaves,
            plan.batch_chunk,
        )
    return out_dq, out_dk, out_dv, out_dbias


def sr_workspace_plan(
    sequence: int,
    heads: int,
    dim: int,
    *,
    outer: int | None = None,
    batch: int = 1,
    workspace_budget_bytes: int = _DEFAULT_WORKSPACE_BUDGET_BYTES,
    query_span: int | None = None,
    batch_chunk: int | None = None,
) -> WorkspacePlan:
    """Choose an SR dBias plan while accounting for dK/dV continuation storage.

    ``workspace_budget_bytes`` is the preferred BF16 dBias-slab size.
    Automatic SR planning compares the generic slab against two alternatives
    that span all queries: leaf waves planned with the requested budget, and
    all leaves resident at once. These forced full-query shapes may exceed
    the dBias budget, but are eligible only if they need no more launch pairs
    and no more slab-plus-continuation bytes than the generic plan.

    Eligible plans minimize launch pairs first, then slab-plus-continuation
    bytes. The BF16 dQ partials are identical in size for all plans and do not
    enter this comparison. Explicit query spans or batch chunks bypass the
    comparison and are honored for diagnostics and controlled experiments.
    """

    if dim <= 0:
        raise ValueError("dim must be positive")
    if outer is None:
        outer = sequence
    # Start with the generic dBias plan; explicit capacities stop here.
    slab_plan = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
    )
    if query_span is not None or batch_chunk is not None:
        return slab_plan

    continuation_bytes = (
        2 * batch * outer * heads * sequence * dim * 4
        if slab_plan.query_stripes > 1
        else 0
    )
    slab_total_bytes = slab_plan.elements * 2 + continuation_bytes
    candidates = [(slab_plan, slab_total_bytes)]

    # Spanning all queries removes the two Q-shaped FP32 continuation tensors.
    # Try the requested budget first; at least one full-query leaf must fit
    # in the allocation, even if that minimum exceeds the requested target.
    full_query_plan = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=sequence,
    )
    full_query_total_bytes = full_query_plan.elements * 2
    if (
        full_query_plan.launch_pairs <= slab_plan.launch_pairs
        and full_query_total_bytes <= slab_total_bytes
    ):
        candidates.append((full_query_plan, full_query_total_bytes))

    # Also try keeping every leaf and batch item resident, accepting it only
    # when the original slab-plus-continuation footprint and launch count allow.
    all_resident_workspace_bytes = (
        batch * slab_plan.leaves * heads * sequence * sequence * 2
    )
    all_resident_plan = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=max(
            workspace_budget_bytes, all_resident_workspace_bytes
        ),
        query_span=sequence,
        batch_chunk=batch,
    )
    all_resident_total_bytes = all_resident_plan.elements * 2
    if (
        all_resident_plan.launch_pairs <= slab_plan.launch_pairs
        and all_resident_total_bytes <= slab_total_bytes
    ):
        candidates.append((all_resident_plan, all_resident_total_bytes))

    return min(
        candidates,
        key=lambda candidate: (
            candidate[0].launch_pairs,
            candidate[1],
            candidate[0].leaf_waves,
            candidate[0].query_span,
        ),
    )[0]


__all__ = [
    "WorkspacePlan",
    "load_extension",
    "source_sha256",
    "sr_workspace_plan",
    "turbopairformer_reduce_broadcast_dbias",
    "turbopairformer_triangle_dq_dbias",
    "turbopairformer_triangle_sr_backward",
    "workspace_elements",
    "workspace_plan",
]
