from __future__ import annotations

import unittest

import torch

from openfold3.core.kernels.cuda.turbopairformer_triangle_attention import (
    is_turbopairformer_supported,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Hopper sm_90a CUDA is required",
)
class TurboPairFormerTriangleAttentionSupportTest(unittest.TestCase):
    def setUp(self):
        shape = (1, 384, 4, 384, 32)
        # Expanded views exercise the facade's real non-contiguous input
        # contract without allocating three full synthetic pair tensors.
        self.q = torch.empty(
            (1, 1, 4, 1, 32), device="cuda", dtype=torch.bfloat16
        ).expand(shape)
        self.k = torch.empty_like(self.q)
        self.v = torch.empty_like(self.q)
        self.mask = torch.zeros(
            (1, 384, 1, 1, 384), device="cuda", dtype=torch.bfloat16
        )
        self.bias = torch.zeros(
            (1, 1, 4, 384, 384), device="cuda", dtype=torch.bfloat16
        )

    def support(self, *, q=None, k=None, v=None, mask=None, bias=None):
        return is_turbopairformer_supported(
            self.q if q is None else q,
            self.k if k is None else k,
            self.v if v is None else v,
            self.mask if mask is None else mask,
            self.bias if bias is None else bias,
        )

    def test_accepts_the_declared_bf16_contract(self):
        supported, reason = self.support()
        self.assertTrue(supported, reason)

    def test_accepts_template_d16_with_singleton_template_prefix(self):
        shape = (1, 1, 384, 4, 384, 16)
        q = torch.empty(
            (1, 1, 1, 4, 1, 16), device="cuda", dtype=torch.bfloat16
        ).expand(shape)
        mask = torch.zeros(
            (1, 1, 384, 1, 1, 384), device="cuda", dtype=torch.bfloat16
        )
        bias = torch.zeros(
            (1, 1, 1, 4, 384, 384), device="cuda", dtype=torch.bfloat16
        )
        supported, reason = self.support(
            q=q, k=q, v=q, mask=mask, bias=bias
        )
        self.assertTrue(supported, reason)

    def test_accepts_runtime_sequence_length(self):
        shape = (1, 512, 4, 512, 32)
        q = torch.empty((1, 1, 4, 1, 32), device="cuda", dtype=torch.bfloat16).expand(
            shape
        )
        mask = torch.zeros(
            (1, 512, 1, 1, 512), device="cuda", dtype=torch.bfloat16
        )
        bias = torch.zeros(
            (1, 1, 4, 512, 512), device="cuda", dtype=torch.bfloat16
        )
        supported, reason = self.support(
            q=q, k=q, v=q, mask=mask, bias=bias
        )
        self.assertTrue(supported, reason)

    def test_accepts_runtime_head_count_and_compiled_d16(self):
        shape = (1, 384, 8, 384, 16)
        q = torch.empty((1, 1, 8, 1, 16), device="cuda", dtype=torch.bfloat16).expand(
            shape
        )
        bias = torch.zeros((1, 1, 8, 384, 384), device="cuda", dtype=torch.bfloat16)
        supported, reason = self.support(q=q, k=q, v=q, bias=bias)
        self.assertTrue(supported, reason)

    def test_accepts_all_representative_head_dimensions(self):
        for head_dim in (1, 7, 8, 16, 24, 31, 32, 48, 64):
            with self.subTest(head_dim=head_dim):
                shape = (1, 64, 2, 64, head_dim)
                q = torch.empty(
                    (1, 1, 2, 1, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                ).expand(shape)
                mask = torch.zeros(
                    (1, 64, 1, 1, 64), device="cuda", dtype=torch.bfloat16
                )
                bias = torch.zeros(
                    (1, 1, 2, 64, 64), device="cuda", dtype=torch.bfloat16
                )
                supported, reason = self.support(
                    q=q, k=q, v=q, mask=mask, bias=bias
                )
                self.assertTrue(supported, reason)

    def test_accepts_batched_prefix(self):
        q = self.q.expand((2,) + tuple(self.q.shape[1:]))
        mask = self.mask.expand((2,) + tuple(self.mask.shape[1:]))
        bias = self.bias.expand((2,) + tuple(self.bias.shape[1:]))
        supported, reason = self.support(q=q, k=q, v=q, mask=mask, bias=bias)
        self.assertTrue(supported, reason)

    def test_rejects_head_dimensions_above_the_runtime_contract(self):
        for head_dim in (65, 128):
            with self.subTest(head_dim=head_dim):
                shape = (1, 384, 4, 384, head_dim)
                q = torch.empty(
                    (1, 1, 4, 1, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                ).expand(shape)
                supported, reason = self.support(q=q, k=q, v=q)
                self.assertFalse(supported)
                self.assertIn("D<=64", reason)

    def _ranked_prefix_case(self, prefix, bias_prefix):
        sequence, heads, head_dim = 1, 1, 16
        q_shape = prefix + (sequence, heads, sequence, head_dim)
        q = torch.empty(
            (1,) * len(prefix) + (1, 1, 1, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ).expand(q_shape)
        mask = torch.zeros(
            prefix + (sequence, 1, 1, sequence),
            device="cuda",
            dtype=torch.bfloat16,
        )
        bias = torch.zeros(
            bias_prefix + (1, heads, sequence, sequence),
            device="cuda",
            dtype=torch.bfloat16,
        )
        return self.support(q=q, k=q, v=q, mask=mask, bias=bias)

    def test_accepts_pair_bias_broadcast_at_rank_16(self):
        prefix = (2,) + (1,) * 15
        supported, reason = self._ranked_prefix_case(prefix, (1,) * 16)
        self.assertTrue(supported, reason)

    def test_rejects_pair_bias_broadcast_above_rank_16(self):
        prefix = (2,) + (1,) * 16
        supported, reason = self._ranked_prefix_case(prefix, (1,) * 17)
        self.assertFalse(supported)
        self.assertIn("at most 16 target-prefix dimensions", reason)

    def test_accepts_rank_17_when_pair_bias_does_not_broadcast(self):
        prefix = (2,) + (1,) * 16
        supported, reason = self._ranked_prefix_case(prefix, prefix)
        self.assertTrue(supported, reason)

    def test_accepts_sequence_above_the_legacy_grid_z_limit(self):
        sequence = 65_536
        q = torch.empty(
            (1, 1, 1, 1, 16), device="cuda", dtype=torch.bfloat16
        ).expand((1, sequence, 1, sequence, 16))
        mask = torch.empty(
            (1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        ).expand((1, sequence, 1, 1, sequence))
        bias = torch.empty(
            (1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        ).expand((1, 1, 1, sequence, sequence))
        supported, reason = self.support(
            q=q, k=q, v=q, mask=mask, bias=bias
        )
        self.assertTrue(supported, reason)

    def test_rejects_sequence_requiring_too_many_query_blocks(self):
        sequence = 65_535 * 32 + 1
        q = torch.empty(
            (1, 1, 1, 1, 16), device="cuda", dtype=torch.bfloat16
        ).expand((1, sequence, 1, sequence, 16))
        mask = torch.empty(
            (1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        ).expand((1, sequence, 1, 1, sequence))
        bias = torch.empty(
            (1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        ).expand((1, 1, 1, sequence, sequence))
        supported, reason = self.support(
            q=q, k=q, v=q, mask=mask, bias=bias
        )
        self.assertFalse(supported)
        self.assertIn("too many CUDA query blocks", reason)

    def test_rejects_heads_beyond_cuda_grid_y(self):
        heads = 65_536
        q = torch.empty(
            (1, 1, 1, 1, 16), device="cuda", dtype=torch.bfloat16
        ).expand((1, 1, heads, 1, 16))
        mask = torch.zeros((1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
        bias = torch.empty(
            (1, 1, 1, 1, 1), device="cuda", dtype=torch.bfloat16
        ).expand((1, 1, heads, 1, 1))
        supported, reason = self.support(
            q=q, k=q, v=q, mask=mask, bias=bias
        )
        self.assertFalse(supported)
        self.assertIn("grid-y", reason)

    def test_rejects_non_bf16_k_or_v(self):
        for name in ("k", "v"):
            with self.subTest(name=name):
                tensor = getattr(self, name).float()
                supported, reason = self.support(**{name: tensor})
                self.assertFalse(supported)
                self.assertIn("q/k/v must be BF16", reason)

    def test_accepts_fp32_pair_bias(self):
        supported, reason = self.support(bias=self.bias.float())
        self.assertTrue(supported, reason)

    def test_rejects_fp16_pair_bias(self):
        supported, reason = self.support(bias=self.bias.half())
        self.assertFalse(supported)
        self.assertIn("pair_bias must be BF16 or FP32", reason)

    def test_rejects_cross_device_inputs(self):
        supported, reason = self.support(bias=self.bias.cpu())
        self.assertFalse(supported)
        self.assertIn("share one device", reason)

    def test_rejects_differentiable_mask_that_backward_would_drop(self):
        supported, reason = self.support(mask=self.mask.float().requires_grad_(True))
        self.assertFalse(supported)
        self.assertIn("must not require gradients", reason)


if __name__ == "__main__":
    unittest.main()
