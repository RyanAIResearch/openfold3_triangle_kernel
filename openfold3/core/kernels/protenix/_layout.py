# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Frozen host batch/layout normalization; no precision changes."""


def _normalize_inputs(q, k, v, mask, pair_bias):
    """Resolve explicit host batch layouts without losing distinct pair biases."""
    original_shape = tuple(q.shape)
    if q.ndim == 6:
        batch, templates, rows, heads, sequence, _ = q.shape
        if mask.ndim != 6 or pair_bias.ndim != 6:
            raise ValueError("6D Q requires 6D mask and pair bias")
        # Expand logical batch broadcasting before flattening B*T. Autograd
        # retains reductions back to the original trainable pair-bias shape.
        mask = mask.expand(batch, templates, rows, 1, 1, sequence)
        pair_bias = pair_bias.expand(batch, templates, 1, heads, sequence, sequence)
        q, k, v = [x.reshape(batch * templates, *x.shape[2:]) for x in (q, k, v)]
        mask = mask.reshape(batch * templates, *mask.shape[2:])
        pair_bias = pair_bias.reshape(batch * templates, *pair_bias.shape[2:])
    elif q.ndim == 4:
        if mask.ndim != 4 or pair_bias.ndim != 4:
            raise ValueError("4D Q requires 4D mask and pair bias")
        if pair_bias.shape[0] == 1:
            # Unbatched rows with one shared trainable pair bias.
            q, k, v, mask, pair_bias = [
                x.unsqueeze(0) for x in (q, k, v, mask, pair_bias)
            ]
        elif pair_bias.shape[0] == q.shape[0]:
            # chunk_layer flattened rows, each with an explicit pair bias.
            q, k, v, mask, pair_bias = [
                x.unsqueeze(1) for x in (q, k, v, mask, pair_bias)
            ]
        else:
            raise ValueError("Unsupported 4D pair-bias batch mapping")
    elif q.ndim != 5:
        raise ValueError(f"Expected 4D, 5D or 6D Q, got {q.ndim}D")
    batch, rows, heads, sequence, _ = q.shape
    expected_mask = (batch, rows, 1, 1, sequence)
    expected_bias = (batch, 1, heads, sequence, sequence)
    mask = mask.expand(expected_mask)
    pair_bias = pair_bias.expand(expected_bias)
    if tuple(mask.shape) != expected_mask:
        raise ValueError(f"Mask shape {tuple(mask.shape)} != {expected_mask}")
    if tuple(pair_bias.shape) != expected_bias:
        raise ValueError(f"Pair-bias shape {tuple(pair_bias.shape)} != {expected_bias}")
    return q, k, v, mask, pair_bias, original_shape
