"""CPU ownership/precision regressions; native numerical tests remain separate."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from ml_collections import ConfigDict

from openfold3.core.model.heads import head_modules
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.layers.triangular_multiplicative_update import BaseTriangleMultiplicativeUpdate
from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.utils import checkpointing
from openfold3.core.utils.checkpointing import (
    frozen_confidence_inference_scope,
    get_checkpoint_fn,
    is_inference_without_checkpoint,
    is_in_frozen_confidence_scope,
)
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom import model as model_module
from openfold3.projects.of3_all_atom.model import OpenFold3
from tools.select_training_backend import configure_training_backend


def frozen_module():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25)).train().requires_grad_(False)


def test_scope_owns_only_its_subtree_and_restores_on_exception():
    module, outsider = frozen_module(), frozen_module()
    value = torch.ones(2, 4)
    assert not is_inference_without_checkpoint(module, value)
    with pytest.raises(LookupError):
        with frozen_confidence_inference_scope(module, value):
            assert not torch.is_grad_enabled()
            assert all(is_inference_without_checkpoint(child, value) for child in module.modules())
            assert not is_inference_without_checkpoint(outsider, value)
            assert all(child.training for child in module.modules())
            assert not is_inference_without_checkpoint(module, value.requires_grad_())
            raise LookupError("test unwind")
    assert torch.is_grad_enabled()
    assert not is_inference_without_checkpoint(module, value.detach())


@pytest.mark.parametrize("bad", ["input_grad", "input_bf16", "parameter_grad", "parameter_bf16", "nested", "autocast", "compile", "tracing"])
def test_scope_rejects_unsupported_ownership_and_precision(bad, monkeypatch):
    module, value = frozen_module(), torch.ones(2, 4)
    if bad == "input_grad": value.requires_grad_()
    if bad == "input_bf16": value = value.bfloat16()
    if bad == "parameter_grad": next(module.parameters()).requires_grad_()
    if bad == "parameter_bf16": module.bfloat16()
    if bad == "compile": monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    if bad == "tracing": monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    outer = (frozen_confidence_inference_scope(module, value) if bad == "nested"
             else torch.autocast("cpu", dtype=torch.bfloat16) if bad == "autocast"
             else mock.MagicMock())
    with outer, pytest.raises((ValueError, RuntimeError)):
        with frozen_confidence_inference_scope(module, value):
            pytest.fail("unsupported scope entered")
    assert not checkpointing._FROZEN_INFERENCE_MODULES.get()


def test_scope_rejects_marked_reentrant_checkpoint_without_changing_default_policy():
    module = frozen_module()
    def body(value):
        with frozen_confidence_inference_scope(module, value.detach()):
            return value.square()
    with pytest.raises(RuntimeError, match="marked checkpoint"):
        get_checkpoint_fn(True)(body, torch.ones(4, requires_grad=True))
    assert not checkpointing._CHECKPOINT_ACTIVE.get()
    assert not checkpointing._FROZEN_INFERENCE_MODULES.get()


def test_default_training_compile_does_not_read_scope_context():
    from torch._dynamo import is_dynamo_supported
    if not is_dynamo_supported():
        pytest.skip("torch.compile is unavailable on this Python/PyTorch platform")
    module = frozen_module()
    def body(value):
        assert not is_inference_without_checkpoint(module, value)
        return value.sin()
    value = torch.ones(4, requires_grad=True)
    try:
        compiled = torch.compile(body, backend="eager", fullgraph=True)
        assert torch.equal(compiled(value), body(value))
    finally:
        torch.compiler.reset()


@pytest.fixture
def real_heads(monkeypatch):
    config = ConfigDict(model_config.architecture.heads.to_dict())
    config.pae.enabled = True
    heads = head_modules.AuxiliaryHeadsAllAtom(config).train()
    for name in ("pairformer_embedding", "pde", "plddt", "experimentally_resolved", "pae"):
        getattr(heads, name).requires_grad_(False)
    # The geometry helpers are outside this routing regression. Keep all six
    # actual heads and the full four-block embedding, including real dropout.
    monkeypatch.setattr(head_modules, "get_token_representative_atoms", lambda batch, x, atom_mask: (x, atom_mask))
    monkeypatch.setattr(head_modules, "broadcast_token_feat_to_atoms", lambda token_mask, **kw: token_mask.repeat_interleave(heads.max_atoms_per_token, dim=-1))
    return heads


def inputs(samples=1):
    torch.manual_seed(17)
    count = 4
    return dict(
        batch={"token_mask": torch.ones(1, 1, count), "atom_mask": torch.ones(1, samples, count), "num_atoms_per_token": torch.ones(1, 1, count, dtype=torch.long)},
        si_input=torch.randn(1, 1, count, 449, requires_grad=True),
        output={"si_trunk": torch.randn(1, 1, count, 384, requires_grad=True),
                "zij_trunk": torch.randn(1, 1, count, count, 128, requires_grad=True),
                "atom_positions_predicted": torch.randn(1, samples, count, 3)},
        use_zij_trunk_embedding=True,
    )


def check_actual_heads(heads, values):
    stack = heads.pairformer_embedding.pairformer_stack
    original = stack.forward
    observations = []
    def observe(*args, **kwargs):
        selected = kwargs["use_stable_triangle_attention"]
        if selected:
            assert kwargs["use_stable_triangle_multiplicative"]
            assert not any(kwargs[key] for key in ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma"))
            assert not torch.is_grad_enabled()
            assert all(is_inference_without_checkpoint(m, *args) for m in stack.modules())
            observations.append(True)
            # CPU verifies ownership using actual stock math. It cannot claim
            # native CUDA correctness. Separate GPU qualification owns that.
            kwargs = dict(kwargs, use_stable_triangle_attention=False, use_stable_triangle_multiplicative=False)
        return original(*args, **kwargs)
    with mock.patch.object(stack, "forward", side_effect=observe):
        torch.manual_seed(314)
        ordinary = heads(**values)
        ordinary_rng = torch.get_rng_state()
        torch.manual_seed(314)
        selected = heads(**values, use_frozen_tpf_confidence=True,
                         use_deepspeed_evo_attention=True, use_cueq_triangle_kernels=True,
                         use_triton_triangle_kernels=True, use_lma=True)
    assert observations == [True]
    assert all(torch.equal(ordinary[name], selected[name]) for name in ordinary)
    assert torch.equal(torch.get_rng_state(), ordinary_rng)
    assert all(child.training for child in heads.modules())
    assert all(not value.requires_grad for name, value in selected.items() if name != "distogram_logits")
    selected["distogram_logits"].square().mean().backward()
    assert values["output"]["zij_trunk"].grad is not None
    assert all(p.grad is not None for p in heads.distogram.parameters())
    assert all(p.grad is None for name, p in heads.named_parameters() if not name.startswith("distogram."))
    assert torch.is_grad_enabled() and not checkpointing._FROZEN_INFERENCE_MODULES.get()


def test_actual_auxiliary_heads_preserve_dropout_outputs_and_distogram_gradients(real_heads):
    check_actual_heads(real_heads, inputs())


def test_real_four_blocks_select_only_eight_triangle_and_eight_trimul_sites(real_heads):
    observations = []
    originals = []
    for name, child in real_heads.pairformer_embedding.named_modules():
        if not isinstance(child, (TriangleAttention, BaseTriangleMultiplicativeUpdate, AttentionPairBias)):
            continue
        original = child.forward
        def observe(*args, name=name, child=child, original=original, **kwargs):
            active = is_in_frozen_confidence_scope(child)
            if isinstance(child, AttentionPairBias):
                assert kwargs['use_stable_triangle_attention'] is False
                assert kwargs['use_high_precision_attention'] is True
                observations.append(('apb', active))
            else:
                key = ('use_stable_triangle_attention' if isinstance(child, TriangleAttention)
                       else 'use_stable_triangle_multiplicative')
                assert kwargs[key] is active
                observations.append(('attention' if isinstance(child, TriangleAttention) else 'trimul', active))
                # CPU substitutes only the numerical leaf, after checking the
                # actual full-stack flags and scope at every real module.
                kwargs[key] = False
            return original(*args, **kwargs)
        child.forward = observe
        originals.append((child, original))
    try:
        values = inputs()
        real_heads(**values, use_frozen_tpf_confidence=False)
        real_heads(**values, use_frozen_tpf_confidence=True)
    finally:
        for child, original in originals:
            child.forward = original
    from collections import Counter
    assert Counter(observations) == {
        ('attention', False): 8, ('attention', True): 8,
        ('trimul', False): 8, ('trimul', True): 8,
        ('apb', False): 4, ('apb', True): 4,
    }


def test_ordinary_stable_apb_flag_is_not_changed(real_heads):
    block = real_heads.pairformer_embedding.pairformer_stack.blocks[0]
    with mock.patch.object(block.pair_stack, 'forward', side_effect=lambda z, **kw: z), mock.patch.object(block.attn_pair_bias, 'forward', side_effect=lambda a, **kw: a * 0) as apb:
        block(torch.ones(1, 4, 384), torch.ones(1, 4, 4, 128), torch.ones(1, 4), torch.ones(1, 4, 4), use_stable_triangle_attention=True)
    assert apb.call_args.kwargs['use_stable_triangle_attention'] is True
    assert apb.call_args.kwargs['use_high_precision_attention'] is False


@pytest.mark.parametrize("bad", ["unfrozen", "distogram_frozen", "dtype", "samples", "offload", "inplace", "pairformer_dtype"])
def test_actual_heads_fail_closed(real_heads, bad):
    values = inputs(samples=2 if bad == "samples" else 1)
    if bad == "unfrozen": next(real_heads.pde.parameters()).requires_grad_()
    if bad == "distogram_frozen": real_heads.distogram.requires_grad_(False)
    if bad == "dtype": values["batch"]["token_mask"] = values["batch"]["token_mask"].bfloat16()
    if bad == "offload": values["offload_inference"] = True
    if bad == "inplace": values["inplace_safe"] = True
    if bad == "pairformer_dtype": values["pairformer_dtype"] = torch.bfloat16
    with pytest.raises((ValueError, RuntimeError)), mock.patch.object(real_heads.pairformer_embedding, "forward") as embedding:
        real_heads(**values, use_frozen_tpf_confidence=True)
    embedding.assert_not_called()


def test_actual_p2_auxiliary_heads_when_checkpoint_is_available(real_heads):
    path = os.environ.get("OPENFOLD3_TEST_P2_CHECKPOINT")
    if not path:
        pytest.skip("set OPENFOLD3_TEST_P2_CHECKPOINT for the explicit CPU P2 regression")
    state = torch.load(Path(path), map_location="cpu", mmap=True, weights_only=False)
    selected = {name.removeprefix("aux_heads."): value for name, value in state.items() if name.startswith("aux_heads.")}
    real_heads.load_state_dict(selected, strict=True)
    check_actual_heads(real_heads, inputs())


def test_selector_preserves_existing_confidence_policy_unless_explicit():
    config = configure_training_backend({}, backend="protenix")
    expected = deepcopy(config)
    assert "use_frozen_tpf_confidence" not in config["model_update"]["custom"]["settings"]["memory"]["train"]
    configure_training_backend(config, backend="protenix", frozen_tpf_confidence=True)
    expected["model_update"]["custom"]["settings"]["memory"]["train"]["use_frozen_tpf_confidence"] = True
    assert config == expected
    configure_training_backend(config, backend="cueq")
    assert config["model_update"]["custom"]["settings"]["memory"]["train"]["use_frozen_tpf_confidence"] is True
    configure_training_backend(config, backend="cueq", frozen_tpf_confidence=False)
    assert config["model_update"]["custom"]["settings"]["memory"]["train"]["use_frozen_tpf_confidence"] is False


def test_model_defaults_do_not_enable_frozen_confidence():
    assert model_config.settings.memory.train.use_frozen_tpf_confidence is False
    assert model_config.settings.memory.eval.use_frozen_tpf_confidence is False


@pytest.mark.parametrize("enabled", [False, True])
def test_model_forwards_option_only_to_confidence(monkeypatch, enabled):
    memory = ConfigDict(model_config.settings.memory.train.to_dict())
    memory.use_frozen_tpf_confidence = enabled
    memory.use_cueq_triangle_kernels = True
    host = SimpleNamespace(
        training=True, _get_mode_mem_settings=lambda: memory,
        _do_inference_offload=lambda **kw: False, clear_autocast_cache=lambda: None,
        shared=SimpleNamespace(use_confidence_emb_prob=1.0,
            diffusion=SimpleNamespace(no_mini_rollout_steps=1, no_mini_rollout_samples=1)),
        config=SimpleNamespace(architecture=SimpleNamespace(noise_schedule={})),
        sample_diffusion=mock.Mock(return_value=torch.ones(1, 1, 4, 3)),
        aux_heads=mock.Mock(return_value={}),
    )
    monkeypatch.setattr(model_module, "create_noise_schedule", lambda **kw: None)
    x, z = torch.ones(1, 4, 384), torch.ones(1, 4, 4, 128)
    OpenFold3._rollout(host, {"token_mask": torch.ones(1, 4)}, x, x, z)
    assert host.aux_heads.call_args.kwargs["use_frozen_tpf_confidence"] is enabled
    assert host.aux_heads.call_args.kwargs["use_cueq_triangle_kernels"] is True
    assert "use_frozen_tpf_confidence" not in host.sample_diffusion.call_args.kwargs
