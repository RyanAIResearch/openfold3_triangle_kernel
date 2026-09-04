# SPDX-License-Identifier: Apache-2.0

"""Regression tests for StableFold TriMul inference-offload propagation."""

from unittest.mock import patch

import torch
from openfold3.core.model.latent.base_stacks import MSAStack


class _EmptyMSAStack(MSAStack):
    def __init__(self):
        super().__init__(blocks_per_ckpt=None)


def test_msa_offload_propagates_stable_trimul_flag():
    stack = _EmptyMSAStack().eval()
    m = torch.randn((1, 2, 3, 4))
    z = torch.randn((1, 3, 3, 8))
    msa_mask = torch.ones((1, 2, 3))
    pair_mask = torch.ones((1, 3, 3))

    with (
        patch.object(stack, "_prep_blocks", wraps=stack._prep_blocks) as prep,
        torch.no_grad(),
    ):
        output_m, output_z = stack.forward_offload(
            [m, z],
            msa_mask,
            pair_mask,
            use_stable_triangle_multiplicative=True,
        )

    assert output_m is m
    assert output_z is z
    assert prep.call_args.kwargs["use_stable_triangle_multiplicative"] is True
