from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from openfold3.core.model.primitives import attention as attention_module
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.utils.distributed_contract import (
    _set_turbopairformer_distributed_strict,
)


class AttentionStableRoutingTest(unittest.TestCase):
    def test_disabled_sr_raises_before_host_stock_fallback(self):
        module = Attention(
            c_q=128, c_k=128, c_v=128, c_hidden=32, no_heads=4, gating=False
        )
        q_x = torch.zeros((1, 2, 2, 128))
        with (
            mock.patch.dict(os.environ, {"OPENFOLD3_SR_BACKWARD": "0"}),
            mock.patch.object(module, "_prep_qkv") as prep,
            mock.patch.object(attention_module, "_attention") as stock,
            self.assertRaisesRegex(
                RuntimeError, "requires single-reconstruction backward"
            ),
        ):
            module(q_x, q_x, use_stable_triangle_attention=True)
        prep.assert_not_called()
        stock.assert_not_called()

    def test_disabled_sr_does_not_affect_default_stock_attention(self):
        module = Attention(
            c_q=128, c_k=128, c_v=128, c_hidden=32, no_heads=4, gating=False
        )
        q_x = torch.zeros((1, 2, 2, 128))
        with (
            mock.patch.dict(os.environ, {"OPENFOLD3_SR_BACKWARD": "0"}),
            mock.patch.object(
                attention_module, "_attention", wraps=attention_module._attention
            ) as stock,
        ):
            output = module(q_x, q_x)
        stock.assert_called_once()
        self.assertEqual(tuple(output.shape), tuple(q_x.shape))
        self.assertTrue(torch.isfinite(output).all())

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
        _set_turbopairformer_distributed_strict(
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
            _set_turbopairformer_distributed_strict(False)

    def _fp32_training_call(self):
        """One FP32 training-mode call, as at a confidence-head site."""
        module = Attention(
            c_q=128, c_k=128, c_v=128, c_hidden=32, no_heads=4, gating=False
        ).train()
        q_x = torch.zeros((1, 2, 2, 128), requires_grad=True)
        mask = torch.zeros((1, 2, 1, 1, 2))
        bias = torch.zeros((1, 1, 4, 2, 2))
        return module(q_x, q_x, biases=[mask, bias], use_stable_triangle_attention=True)

    def _strict(self, enabled):
        _set_turbopairformer_distributed_strict(
            enabled,
            {
                "OPENFOLD3_SR_BACKWARD": "1",
                "OPENFOLD3_SR_SHADOW": "1",
                "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
            },
        )

    def test_fp32_training_inputs_keep_fp32_stock_attention(self):
        # The stock FP32 route is the intended choice for FP32 training inputs,
        # not a coverage fallback, so the distributed strict mode (which rejects
        # fallbacks) accepts it, and the stock math receives FP32 operands.
        self._strict(True)
        try:
            with mock.patch.object(
                attention_module, "_attention", wraps=attention_module._attention
            ) as stock:
                output = self._fp32_training_call()
        finally:
            self._strict(False)
        stock.assert_called_once()
        self.assertEqual(stock.call_args.args[0].dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)

    def test_bf16_cast_option_selects_the_turbopairformer_route(self):
        # With the BF16 cast the call enters the TurboPairFormer route,
        # whose coverage gate rejects CPU inputs; strict mode turns that
        # fallback into an error, which shows the route that was taken.
        attention_module.set_tpf_bf16_cast_for_fp32_training(True)
        self._strict(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "strict route rejected"):
                self._fp32_training_call()
        finally:
            self._strict(False)
            attention_module.set_tpf_bf16_cast_for_fp32_training(False)


if __name__ == "__main__":
    unittest.main()
