from __future__ import annotations

import unittest
from unittest import mock

import torch

from openfold3.core.kernels.cuda import streaming_triangle_attention as streaming
from openfold3.core.model.primitives.attention import Attention


class AttentionStableRoutingTest(unittest.TestCase):
    def test_unsupported_streaming_case_fails_closed(self):
        module = Attention(
            c_q=128,
            c_k=128,
            c_v=128,
            c_hidden=32,
            no_heads=4,
            gating=False,
        )
        q = torch.ones((1, 2, 4, 2, 32), dtype=torch.bfloat16)
        k = torch.ones_like(q)
        v = torch.ones_like(q)
        q_x = torch.zeros((1, 2, 128), dtype=torch.bfloat16)
        mask = torch.zeros((1, 2, 1, 1, 2), dtype=torch.bfloat16)
        bias = torch.zeros((1, 1, 4, 2, 2), dtype=torch.bfloat16)
        with (
            mock.patch.object(module, "_prep_qkv", return_value=(q, k, v)) as prep,
            mock.patch.object(
                streaming, "is_streaming_supported", return_value=(False, "test")
            ),
            self.assertRaisesRegex(ValueError, "unsupported.*test"),
        ):
            module(
                q_x,
                q_x,
                biases=[mask, bias],
                use_stable_triangle_attention=True,
            )

        prep.assert_called_once_with(q_x, q_x, apply_scale=False)


if __name__ == "__main__":
    unittest.main()
