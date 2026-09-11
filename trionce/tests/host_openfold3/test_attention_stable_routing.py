from __future__ import annotations

import unittest
from unittest import mock

import torch
from openfold3.core.model.primitives import attention as attention_module
from openfold3.core.model.primitives.attention import Attention

from trionce.distributed import (
    _set_distributed_strict,
)


class AttentionStableRoutingTest(unittest.TestCase):
    def test_unsupported_model_site_falls_back_to_stock(self):
        module = Attention(
            c_q=128,
            c_k=128,
            c_v=128,
            c_hidden=32,
            no_heads=4,
            gating=False,
        ).to(dtype=torch.bfloat16)
        q = torch.ones((1, 2, 4, 2, 32), dtype=torch.bfloat16)
        k = torch.ones_like(q)
        v = torch.ones_like(q)
        q_x = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
        mask = torch.zeros((1, 2, 1, 1, 2), dtype=torch.bfloat16)
        bias = torch.zeros((1, 1, 4, 2, 2), dtype=torch.bfloat16)
        stock_output = torch.zeros_like(q)
        with (
            mock.patch.object(module, "_prep_qkv", return_value=(q, k, v)) as prep,
            mock.patch.object(
                attention_module, "_attention", return_value=stock_output
            ) as stock,
        ):
            output = module(
                q_x,
                q_x,
                biases=[mask, bias],
                use_stable_triangle_attention=True,
            )

        # CPU is rejected by the model-level pre-gate before QKV preparation,
        # so the stock path owns the scale exactly once.
        prep.assert_called_once_with(q_x, q_x, apply_scale=True)
        stock.assert_called_once()
        self.assertEqual(tuple(output.shape), tuple(q_x.shape))

    def test_unsupported_model_site_is_fatal_in_distributed_strict_mode(self):
        module = Attention(
            c_q=128,
            c_k=128,
            c_v=128,
            c_hidden=32,
            no_heads=4,
            gating=False,
        ).to(dtype=torch.bfloat16)
        q_x = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
        mask = torch.zeros((1, 2, 1, 1, 2), dtype=torch.bfloat16)
        bias = torch.zeros((1, 1, 4, 2, 2), dtype=torch.bfloat16)
        _set_distributed_strict(
            True,
            {
                "OPENFOLD3_SR_BACKWARD": "1",
                "OPENFOLD3_SR_SHADOW": "1",
                "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
            },
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "strict route rejected"):
                module(
                    q_x,
                    q_x,
                    biases=[mask, bias],
                    use_stable_triangle_attention=True,
                )
        finally:
            _set_distributed_strict(False)


if __name__ == "__main__":
    unittest.main()
