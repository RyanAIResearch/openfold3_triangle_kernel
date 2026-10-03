# SPDX-License-Identifier: Apache-2.0

"""TurboPairFormer fused dQ + dBias with a bounded fixed dBias tree.

Each leaf plane has a single writer and the finalizer uses one global fixed
order. Leaf planes are processed in deterministic waves so temporary storage
is capped instead of growing with the number of outer leaves. The
implementation uses no floating-point atomics. See the CUDA source for the
numerical contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

_OUTER_LEAF = 4
_QUERY_BLOCK = 64
_MAX_GRID_YZ = 65_535
_MAX_INT32 = (1 << 31) - 1
_MAX_INT64 = (1 << 63) - 1
_DEFAULT_WORKSPACE_BUDGET_BYTES = 256 * 1024 * 1024
_EXTENSION: Any = None


@dataclass(frozen=True)
class WorkspacePlan:
    """Shape-generic launch/workspace plan for the fixed dBias tree.

    ``query_span`` and ``batch_chunk`` partition independent output cells;
    ``resident_leaves`` partitions the one globally ordered leaf axis.  None
    of the three values changes the reduction tree.
    """

    query_span: int
    resident_leaves: int
    batch_chunk: int
    leaves: int
    query_stripes: int
    leaf_waves: int
    batch_chunks: int
    launch_pairs: int
    elements: int
    workspace_budget_bytes: int
    debug_full_plane: bool


def source_sha256() -> dict[str, str]:
    from openfold3.core.kernels.cuda._aot_runtime import source_sha256_for

    source = Path(__file__).with_suffix(".cu")
    return source_sha256_for("turbopairformer_dq_dbias", source)


def load_extension(*, verbose: bool | None = None) -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from openfold3.core.kernels.cuda._aot_runtime import load_extension_module

        _EXTENSION = load_extension_module(
            module_key="turbopairformer_dq_dbias",
            aot_module=(
                "openfold3.core.kernels.cuda._ext._turbopairformer_dq_dbias_sm90a"
            ),
            source=Path(__file__).with_suffix(".cu"),
            expected_symbols=(
                "turbopairformer_triangle_dq_dbias_out",
                "turbopairformer_reduce_broadcast_dbias_out",
                "turbopairformer_triangle_sr_backward_out",
            ),
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

    maximum = max(1, min(batch, capacity))
    candidates: set[int] = {1, maximum}
    cursor = 1
    while cursor <= maximum:
        resident_capacity = capacity // cursor
        capacity_end = capacity // resident_capacity
        chunks = _ceil_div(batch, cursor)
        batch_end = (
            maximum if chunks == 1 else (batch - 1) // (chunks - 1)
        )
        interval_end = min(maximum, capacity_end, batch_end)
        candidates.add(cursor)
        candidates.add(interval_end)
        cursor = interval_end + 1
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
    """Plan deterministic two-dimensional query-slab waves.

    The planner minimizes leaf/finalizer launch pairs first, then minimizes
    the number of root passes.  This gives query slabs only when they reduce
    root traffic without increasing the best available launch count.  It
    enumerates query spans and batch/leaf tilings from the runtime dimensions;
    there is no sequence-length lookup table.

    The budget is a target.  If even one ``batch x query-block x leaf`` tile
    exceeds it, that single tile is the shape-generic lower bound.  Unlike the
    old full-plane path, the lower bound is ``O(H*N*query_block)``, not
    ``O(B*H*N^2)``.
    """

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
        query_span <= 0
        or query_span > _MAX_INT32
        or query_span % _QUERY_BLOCK != 0
    ):
        raise ValueError("query_span must be an int32 positive multiple of 64")
    if batch_chunk is not None and not 1 <= batch_chunk <= batch:
        raise ValueError("batch_chunk must lie in [1, batch]")
    if debug_full_plane and (query_span is not None or batch_chunk is not None):
        raise ValueError(
            "debug_full_plane cannot be combined with a forced slab shape"
        )

    leaves = _ceil_div(outer, outer_leaf_for(outer))
    budget_elements = workspace_budget_bytes // 2  # BF16 elements

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

    query_blocks = _ceil_div(sequence, _QUERY_BLOCK)
    query_candidates = (
        (query_span,)
        if query_span is not None
        else tuple(
            block_count * _QUERY_BLOCK
            for block_count in range(1, query_blocks + 1)
        )
    )
    best_score: tuple[int, ...] | None = None
    best_plan: WorkspacePlan | None = None
    for candidate_query_span in query_candidates:
        unit_elements = heads * candidate_query_span * sequence
        if (
            query_span is None
            and candidate_query_span != _QUERY_BLOCK
            and unit_elements > budget_elements
        ):
            # Only the irreducible one-query-block tile may exceed the target.
            # Otherwise a large Q could win on launch count by silently
            # turning the budget back into an O(N^2) full-plane allocation.
            continue
        # Capacity is measured in (batch slot, leaf) pairs.  One pair is the
        # irreducible, query-tiled workspace even when it exceeds the target.
        capacity = max(1, budget_elements // unit_elements)
        chunk_candidates = (
            (batch_chunk,)
            if batch_chunk is not None
            else _batch_chunk_candidates(batch, capacity)
        )
        for candidate_batch_chunk in chunk_candidates:
            assert candidate_batch_chunk is not None
            resident_leaves = min(
                leaves,
                max(1, capacity // candidate_batch_chunk),
            )
            query_stripes = _ceil_div(sequence, candidate_query_span)
            leaf_waves = _ceil_div(leaves, resident_leaves)
            batch_chunks = _ceil_div(batch, candidate_batch_chunk)
            launch_pairs = query_stripes * leaf_waves * batch_chunks
            elements = (
                resident_leaves
                * candidate_batch_chunk
                * unit_elements
            )
            if elements > _MAX_INT64:
                continue
            # First preserve or improve launch count.  Among equal launch
            # counts, fewer waves remove repeated FP32 root scans.  Remaining
            # terms make the plan stable and prefer fuller workspace tiles.
            score = (
                launch_pairs,
                leaf_waves,
                batch_chunks * query_stripes,
                -elements,
                -candidate_batch_chunk,
                candidate_query_span,
            )
            if best_score is None or score < best_score:
                best_score = score
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
    """Return the BF16 element count for :func:`workspace_plan`."""

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

    if len(bias_prefix) > len(target_prefix):
        raise ValueError("bias prefix rank cannot exceed target prefix rank")
    aligned_bias_prefix = (1,) * (
        len(target_prefix) - len(bias_prefix)
    ) + tuple(bias_prefix)
    for target, bias in zip(target_prefix, aligned_bias_prefix, strict=True):
        if target <= 0 or bias <= 0 or bias not in (1, target):
            raise ValueError("bias prefix must broadcast exactly to target prefix")
    expected_batch = 1
    for dimension in target_prefix:
        expected_batch *= int(dimension)
    if per_batch_dbias.shape[0] != expected_batch:
        raise ValueError("per_batch_dbias does not match flattened target prefix")
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
    """Single-reconstruction backward: dQ, dK, dV, per-batch dBias.

    Uses the caller-supplied FP32 delta when forming dS. The public
    facade uses the forward rounding residual to compute delta by default.
    Flat BF16 dQ partials of [key_blocks, B*O, H, N, D] are folded in fixed
    ascending key-block order. Requires sequence % 64 == 0 and D <= 64 with
    D % 8 == 0.
    """

    outer = int(q.shape[1])
    heads = int(q.shape[2])
    sequence = int(q.shape[3])
    dim = int(q.shape[4])
    batch = int(q.shape[0])
    if sequence % 64 != 0 or dim > 64 or dim % 8 != 0:
        raise ValueError("SR backward requires sequence % 64 == 0 and D <= 64")
    plan = sr_workspace_plan(
        sequence,
        heads,
        dim,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
    )
    workspace = torch.empty(
        plan.elements, device=q.device, dtype=torch.bfloat16
    )
    key_blocks = sequence // 64
    dq_partials = torch.empty(
        (key_blocks, batch * outer, heads, sequence, dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    # dK/dV are reduced across the query axis.  When the bounded dBias
    # workspace forces multiple query stripes, preserve their FP32 MMA
    # accumulators between launches so a launch boundary remains a pure
    # store/reload and does not change (or discard) arithmetic.
    continuation_elements = q.numel() if plan.query_stripes > 1 else 0
    dk_continuation = torch.empty(
        continuation_elements, device=q.device, dtype=torch.float32
    )
    dv_continuation = torch.empty_like(dk_continuation)
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
    """Plan SR scratch while accounting for striped dK/dV continuation.

    ``workspace_budget_bytes`` is the preferred dBias-slab cap.  Automatic SR
    planning compares the generic slab against two full-query alternatives:
    a budget-respecting leaf-wave plan and an all-resident plan.  A candidate
    is eligible only when it uses no more total scratch (including the slab
    plan's two q-shaped FP32 dK/dV continuation tensors) and no more launch
    pairs.  Eligible candidates minimize launch pairs first, then total
    scratch.  Explicit query spans or batch chunks are honored exactly for
    diagnostics and controlled experiments.
    """

    if dim <= 0:
        raise ValueError("dim must be positive")
    if outer is None:
        outer = sequence
    plan = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=query_span,
        batch_chunk=batch_chunk,
    )
    if query_span is not None or batch_chunk is not None:
        return plan

    continuation_bytes = (
        2 * batch * outer * heads * sequence * dim * 4
        if plan.query_stripes > 1
        else 0
    )
    plan_total_bytes = plan.elements * 2 + continuation_bytes
    candidates = [(plan, plan_total_bytes)]

    # Query stripes require q-shaped FP32 dK/dV continuation tensors.  The
    # generic dBias planner does not know about that SR-only cost and can
    # prefer query stripes over an equal-launch full-query leaf-wave plan.
    # Evaluate that missing Pareto candidate without exceeding the requested
    # workspace budget.
    full_span = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=workspace_budget_bytes,
        query_span=sequence,
    )
    full_span_total_bytes = full_span.elements * 2
    if (
        full_span.launch_pairs <= plan.launch_pairs
        and full_span_total_bytes <= plan_total_bytes
    ):
        candidates.append((full_span, full_span_total_bytes))

    # Retain the existing speed-first promotion when all leaves can reside at
    # once without exceeding the original slab-plus-continuation footprint.
    full_workspace_bytes = batch * plan.leaves * heads * sequence * sequence * 2
    full_resident = workspace_plan(
        sequence,
        heads,
        outer=outer,
        batch=batch,
        workspace_budget_bytes=max(workspace_budget_bytes, full_workspace_bytes),
        query_span=sequence,
        batch_chunk=batch,
    )
    full_resident_total_bytes = full_resident.elements * 2
    if (
        full_resident.launch_pairs <= plan.launch_pairs
        and full_resident_total_bytes <= plan_total_bytes
    ):
        candidates.append((full_resident, full_resident_total_bytes))

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
    "turbopairformer_triangle_dq_dbias",
    "turbopairformer_triangle_sr_backward",
    "load_extension",
    "turbopairformer_reduce_broadcast_dbias",
    "source_sha256",
    "WorkspacePlan",
    "sr_workspace_plan",
    "workspace_elements",
    "workspace_plan",
]
