"""CPU routing tests for the optional FP32 confidence TriMul backend.

These check dispatch, sample batching, and chunk propagation without loading or
executing a CUDA kernel. Numerical qualification belongs to the GPU tests.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from openfold3.core.model.heads import head_modules
from openfold3.core.model.heads.prediction_heads import PairformerEmbedding
from openfold3.projects.of3_all_atom import model as model_module


class MemorySettings(dict):
    __getattr__ = dict.__getitem__


@pytest.mark.parametrize(
    "configured,training,grad_enabled,is_cuda,input_dtype,amp_dtype,expected",
    [
        (None, False, False, True, torch.float32, None, False),
        (False, False, False, True, torch.float32, None, False),
        (True, False, False, True, torch.float32, None, True),
        (True, False, False, True, torch.float32, torch.float32, True),
        (True, False, False, True, torch.float32, torch.bfloat16, False),
        (True, False, False, True, torch.float32, torch.float16, False),
        (True, True, False, True, torch.float32, None, False),
        (True, False, True, True, torch.float32, None, False),
        (True, True, True, True, torch.float32, torch.bfloat16, False),
        (True, False, False, True, torch.bfloat16, None, False),
        (True, False, False, False, torch.float32, None, False),
    ],
)
def test_model_confidence_route_uses_callers_precision_policy(
    configured, training, grad_enabled, is_cuda, input_dtype, amp_dtype, expected
):
    settings = MemorySettings(
        chunk_size=3,
        use_deepspeed_evo_attention=False,
        use_triton_triangle_kernels=False,
        use_cueq_triangle_kernels=True,
        use_lma=False,
    )
    if configured is not None:
        settings["use_stable_triangle_multiplicative"] = configured

    # Metadata stand-ins exercise CUDA dispatch on a CPU test host. Diffusion
    # and auxiliary heads are mocked; no tensor math uses these stand-ins.
    trunk = SimpleNamespace(
        dtype=input_dtype, is_cuda=is_cuda, device="cuda" if is_cuda else "cpu"
    )
    auxiliary_heads = mock.Mock(return_value={})
    model = SimpleNamespace(
        training=training,
        _get_mode_mem_settings=lambda: settings,
        _do_inference_offload=lambda **kwargs: False,
        shared=SimpleNamespace(
            diffusion=SimpleNamespace(
                no_mini_rollout_steps=1,
                no_full_rollout_steps=200,
                no_mini_rollout_samples=1,
                no_full_rollout_samples=5,
            ),
            use_confidence_emb_prob=1.0,
        ),
        config=SimpleNamespace(architecture=SimpleNamespace(noise_schedule={})),
        sample_diffusion=mock.Mock(return_value=mock.sentinel.positions),
        clear_autocast_cache=mock.Mock(),
        aux_heads=auxiliary_heads,
    )
    amp_state = {"dtype": amp_dtype}

    @contextmanager
    def local_autocast(*, device_type, dtype):
        previous = amp_state["dtype"]
        amp_state["dtype"] = dtype
        try:
            yield
        finally:
            amp_state["dtype"] = previous

    with (
        torch.set_grad_enabled(grad_enabled),
        mock.patch.object(
            torch,
            "is_autocast_enabled",
            side_effect=lambda device: amp_state["dtype"] is not None,
        ),
        mock.patch.object(
            torch, "get_autocast_dtype", side_effect=lambda device: amp_state["dtype"]
        ),
        mock.patch.object(torch.amp, "autocast", side_effect=local_autocast),
        mock.patch.object(
            model_module, "autocast_device_type", return_value=trunk.device
        ),
        mock.patch.object(
            model_module,
            "create_noise_schedule",
            return_value=mock.sentinel.noise_schedule,
        ),
    ):
        model_module.OpenFold3._rollout(
            model,
            {"token_mask": SimpleNamespace(shape=(1, 7))},
            trunk,
            trunk,
            trunk,
        )

    keywords = auxiliary_heads.call_args.kwargs
    assert keywords["use_stable_triangle_multiplicative"] is expected
    assert keywords["chunk_size"] == 3
    assert keywords["use_cueq_triangle_kernels"] is True
    assert amp_state["dtype"] == amp_dtype


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("stable_attention", [False, True])
def test_confidence_embedding_routes_all_five_samples_and_preserves_chunking(
    enabled, per_sample, stable_attention
):
    calls = []

    class RecordingStack(nn.Module):
        def forward(self, si, zij, single_mask, pair_mask, **kwargs):
            calls.append((tuple(si.shape), tuple(zij.shape), kwargs))
            return si, zij

    embedding = PairformerEmbedding.__new__(PairformerEmbedding)
    nn.Module.__init__(embedding)
    embedding.pairformer_stack = RecordingStack()
    si = torch.arange(7 * 4, dtype=torch.float32).reshape(1, 1, 7, 4)
    zij = torch.arange(7 * 7 * 4, dtype=torch.float32).reshape(1, 1, 7, 7, 4)
    extra = {} if enabled is None else {"use_stable_triangle_multiplicative": enabled}

    with (
        torch.no_grad(),
        mock.patch.object(
            embedding, "embed_zij", side_effect=lambda si_input, zij, x_pred: zij
        ),
        mock.patch.object(
            torch.amp, "autocast", side_effect=lambda **kwargs: nullcontext()
        ),
    ):
        result_si, result_zij = embedding(
            si_input=si,
            si=si,
            zij=zij,
            x_pred=torch.zeros(1, 5, 7, 3),
            single_mask=torch.ones(1, 1, 7),
            pair_mask=torch.ones(1, 1, 7, 7),
            apply_per_sample=per_sample,
            chunk_size=3,
            use_cueq_triangle_kernels=True,
            use_stable_triangle_attention=stable_attention,
            **extra,
        )

    assert len(calls) == (5 if per_sample else 1)
    flattened_samples = 1 if per_sample else 5
    expected_chunk = 3 if per_sample or stable_attention else None
    for si_shape, zij_shape, keywords in calls:
        assert si_shape == (flattened_samples, 7, 4)
        assert zij_shape == (flattened_samples, 7, 7, 4)
        assert keywords["use_stable_triangle_multiplicative"] is bool(enabled)
        assert keywords["use_stable_triangle_attention"] is stable_attention
        assert keywords["chunk_size"] == expected_chunk
    torch.testing.assert_close(result_si, si.expand(1, 5, 7, 4))
    torch.testing.assert_close(result_zij, zij.expand(1, 5, 7, 7, 4))


@pytest.mark.parametrize("enabled", [None, True])
@pytest.mark.parametrize(
    "tokens,samples,grad_enabled,per_sample",
    [
        (750, 5, False, False),
        (751, 5, False, True),
        (751, 1, False, False),
        (751, 5, True, False),
    ],
)
def test_auxiliary_heads_preserve_per_sample_cutoff(
    enabled, tokens, samples, grad_enabled, per_sample
):
    class ReachedEmbedding(Exception):
        pass

    embedding = mock.Mock(side_effect=ReachedEmbedding)
    heads = SimpleNamespace(
        per_sample_token_cutoff=750,
        distogram=lambda z: z,
        pairformer_embedding=embedding,
    )
    si = torch.zeros(1, 1, tokens, 1)
    zij = torch.zeros(1, 1, tokens, tokens, 1)
    positions = torch.zeros(1, samples, tokens, 3)
    token_mask = torch.ones(1, 1, tokens)
    extra = {} if enabled is None else {"use_stable_triangle_multiplicative": enabled}

    with (
        torch.set_grad_enabled(grad_enabled),
        mock.patch.object(
            head_modules,
            "get_token_representative_atoms",
            return_value=(positions, None),
        ),
        pytest.raises(ReachedEmbedding),
    ):
        head_modules.AuxiliaryHeadsAllAtom.forward(
            heads,
            batch={"token_mask": token_mask, "atom_mask": token_mask},
            si_input=si,
            output={
                "si_trunk": si,
                "zij_trunk": zij,
                "atom_positions_predicted": positions,
            },
            use_zij_trunk_embedding=True,
            chunk_size=3,
            **extra,
        )

    keywords = embedding.call_args.kwargs
    assert keywords["use_stable_triangle_multiplicative"] is bool(enabled)
    assert keywords["apply_per_sample"] is per_sample
    assert keywords["chunk_size"] == 3
