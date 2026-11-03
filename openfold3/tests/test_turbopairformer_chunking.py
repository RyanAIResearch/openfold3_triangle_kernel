"""Integration checks for Turbo outer-row chunking and backend-aware tuning."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from openfold3.core.model.heads.prediction_heads import PairformerEmbedding
from openfold3.core.model.latent.base_stacks import MSAStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplatePairStack
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.utils.chunk_utils import ChunkSizeTuner


def initialized_attention(starting=True):
    module = TriangleAttention(64, 16, 4, starting=starting).eval()
    # The default output projection is zero: use nonzero weights so numerical
    # and gradient checks detect wrong row/bias indexing rather than zeros.
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return module


def record_attention_math(module, calls):
    original = module.mha.forward

    def forward(q_x, kv_x, biases, use_stable_triangle_attention=False, **kwargs):
        assert use_stable_triangle_attention
        calls.append(
            {
                "input_shape": tuple(q_x.shape),
                "mask_shape": tuple(biases[0].shape),
                "pair_shape": tuple(biases[1].shape),
                "pair_pointer": biases[1].data_ptr(),
            }
        )
        # CPU checks the integration's partitioning against the same complete
        # attention math. The Hopper test below forbids a real stock fallback.
        return original(q_x, kv_x, biases=biases, **kwargs)

    return forward


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("transpose_bias", [True, False])
@pytest.mark.parametrize("inplace_safe", [True, False])
def test_outer_chunks_preserve_samples_masks_and_full_pair_bias(
    starting, transpose_bias, inplace_safe
):
    module = initialized_attention(starting)
    torch.manual_seed(7)
    x = torch.randn(2, 5, 7, 7, 64)
    mask = torch.randint(0, 2, (2, 5, 7, 7)).float()
    # Leave a valid key in each orientation to avoid all-masked softmax rows.
    mask[..., 0, :] = 1
    mask[..., :, 0] = 1
    calls = []
    with (
        torch.no_grad(),
        mock.patch.object(
            module.mha, "forward", side_effect=record_attention_math(module, calls)
        ),
    ):
        dense = module(
            x.clone(),
            mask,
            transpose_bias=transpose_bias,
            use_stable_triangle_attention=True,
        )
        calls.clear()
        chunked = module(
            x.clone(),
            mask,
            transpose_bias=transpose_bias,
            chunk_size=3,
            inplace_safe=inplace_safe,
            use_stable_triangle_attention=True,
        )
    torch.testing.assert_close(chunked, dense, rtol=2e-5, atol=2e-6)
    assert chunked.abs().max() > 0
    assert len(calls) == 2 * 5 * 3
    for index, call in enumerate(calls):
        rows = 1 if index % 3 == 2 else 3
        assert call["input_shape"] == (1, rows, 7, 64)
        assert call["mask_shape"] == (1, rows, 1, 1, 7)
        assert call["pair_shape"] == (1, 1, 4, 7, 7)
        # All three chunks of an individual sample share the same full bias.
        assert call["pair_pointer"] == calls[index - index % 3]["pair_pointer"]


def test_outer_chunk_inference_replay_and_broadcast_mask():
    module = initialized_attention()
    torch.manual_seed(11)
    x = torch.randn(2, 3, 7, 7, 64)
    mask = torch.ones(1, 1, 7, 7)
    mask[..., -1] = 0
    calls = []
    with (
        torch.no_grad(),
        mock.patch.object(
            module.mha, "forward", side_effect=record_attention_math(module, calls)
        ),
    ):
        dense = module(x, mask, use_stable_triangle_attention=True)
        chunked = module(x, mask, chunk_size=3, use_stable_triangle_attention=True)
        replay = module(x, mask, chunk_size=3, use_stable_triangle_attention=True)
    torch.testing.assert_close(chunked, dense, rtol=2e-5, atol=2e-6)
    assert torch.equal(chunked, replay)


@pytest.mark.parametrize("stack_kind", ["msa", "pairformer", "template"])
@pytest.mark.parametrize("max_viable_chunk", [3, 2])
def test_tuner_uses_full_inputs_and_retunes_after_backend_change(
    stack_kind, max_viable_chunk
):
    calls = []
    first = torch.randn(1, 7, 64)
    pair = torch.randn(1, 7, 7, 64)
    mask = torch.ones(1, 7, 7)

    def block(*inputs, **kwargs):
        calls.append((inputs, kwargs))
        if kwargs["chunk_size"] > max_viable_chunk:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return inputs[0] if stack_kind == "template" else inputs

    stack = SimpleNamespace(
        blocks=[block],
        training=False,
        clear_cache_between_blocks=False,
        chunk_size_tuner=ChunkSizeTuner(),
    )

    def prepare(stable):
        common = dict(
            chunk_size=3,
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=False,
            use_triton_triangle_kernels=False,
            use_stable_triangle_attention=stable,
            use_stable_triangle_multiplicative=stable,
            use_lma=False,
            inplace_safe=True,
            _mask_trans=True,
        )
        if stack_kind == "msa":
            return MSAStack._prep_blocks(
                stack,
                m=first,
                z=pair,
                transition_ckpt_chunk_size=None,
                msa_mask=torch.ones(1, 7),
                pair_mask=mask,
                **common,
            )
        if stack_kind == "pairformer":
            return PairFormerStack._prep_blocks(
                stack,
                s=first,
                z=pair,
                single_mask=torch.ones(1, 7),
                pair_mask=mask,
                **common,
            )
        return TemplatePairStack._prep_blocks(stack, t=pair, mask=mask, **common)

    prepare(False)
    stock_probe_count = len(calls)
    assert stock_probe_count > 0
    prepared = prepare(True)
    stable_probe_count = len(calls)
    assert stable_probe_count > stock_probe_count
    for inputs, keywords in calls[stock_probe_count:]:
        expected_inputs = (pair,) if stack_kind == "template" else (first, pair)
        for actual, expected in zip(inputs, expected_inputs, strict=True):
            torch.testing.assert_close(actual, expected)
            assert actual.data_ptr() != expected.data_ptr()
        assert keywords["use_stable_triangle_attention"] is True
        assert keywords["use_stable_triangle_multiplicative"] is True
        assert keywords.get("pair_mask", keywords.get("mask")) is mask
    prepare(True)
    assert len(calls) == stable_probe_count
    inputs = (pair,) if stack_kind == "template" else (first, pair)
    prepared[0](*inputs)
    # Turbo must use the tuned flash-kernel chunk, not the stock quarter-size.
    assert calls[-1][1]["_attn_chunk_size"] == max_viable_chunk
    assert calls[-1][1]["chunk_size"] == max_viable_chunk


@pytest.mark.parametrize("stable", [False, True])
def test_confidence_keeps_chunk_size_for_multiple_turbo_samples(stable):
    recorded = []

    class RecordingStack(nn.Module):
        def forward(self, si, zij, *masks, **kwargs):
            recorded.append((si.shape, zij.shape, kwargs))
            return si, zij

    module = PairformerEmbedding.__new__(PairformerEmbedding)
    nn.Module.__init__(module)
    module.pairformer_stack = RecordingStack()
    with (
        mock.patch.object(
            module, "embed_zij", side_effect=lambda si_input, zij, x_pred: zij
        ),
        mock.patch("torch.amp.autocast", side_effect=lambda **kwargs: nullcontext()),
    ):
        module.pairformer_emb(
            si_input=torch.zeros(1, 1, 7, 64),
            si=torch.zeros(1, 1, 7, 64),
            zij=torch.zeros(1, 1, 7, 7, 64),
            x_pred=torch.zeros(1, 5, 7, 3),
            single_mask=torch.ones(1, 1, 7),
            pair_mask=torch.ones(1, 1, 7, 7),
            chunk_size=3,
            use_stable_triangle_attention=stable,
            use_cueq_triangle_kernels=True,
        )
    assert recorded[0][0] == (5, 7, 64)
    assert recorded[0][2]["chunk_size"] == (3 if stable else None)
    assert recorded[0][2]["use_stable_triangle_attention"] is stable


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hopper_chunks_execute_turbo_without_stock_fallback(dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Real Turbo route qualification requires Hopper sm90")
    from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_attention import (
        invocation_count,
    )
    from openfold3.core.model.primitives import attention as attention_module

    module = initialized_attention(starting=False).cuda().to(dtype=dtype)
    torch.manual_seed(31)
    x = torch.randn(1, 5, 33, 33, 64, device="cuda", dtype=dtype)
    mask = torch.ones(1, 5, 33, 33, device="cuda", dtype=dtype)
    mask[..., -2:] = 0
    mask[..., 0, :] = 1
    with (
        torch.inference_mode(),
        mock.patch.object(
            attention_module,
            "_attention",
            side_effect=AssertionError("unexpected stock attention fallback"),
        ),
    ):
        before = invocation_count()
        dense = module(
            x.clone(), mask, transpose_bias=True, use_stable_triangle_attention=True
        )
        assert invocation_count() > before
        before = invocation_count()
        chunked = module(
            x.clone(),
            mask,
            transpose_bias=True,
            chunk_size=16,
            use_stable_triangle_attention=True,
            inplace_safe=True,
        )
        assert invocation_count() - before == 5 * 3
    # Existing facade BF16 conversion is retained for FP32. Projection GEMM
    # tiling can differ with chunk shape, so compare within that dtype's error.
    torch.testing.assert_close(chunked, dense, rtol=2e-2, atol=2e-3)
