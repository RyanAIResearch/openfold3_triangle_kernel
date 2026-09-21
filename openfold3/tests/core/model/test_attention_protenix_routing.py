# SPDX-License-Identifier: Apache-2.0
"""CPU routing contracts for the optional Protenix + cuEq training backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from ml_collections import ConfigDict

from openfold3.core.model.latent.base_blocks import PairBlock
from openfold3.core.model.latent.evoformer import EvoformerBlock
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplatePairStack
from openfold3.core.model.layers import triangular_multiplicative_update as trimul
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.primitives import attention
from openfold3.core.model.primitives.attention import Attention
from openfold3.projects.of3_all_atom import model as model_module
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3
from tools.select_training_backend import BACKEND_FLAGS


@pytest.fixture
def native_attention(monkeypatch):
    def reference(q, k, v, mask, bias):
        return attention._attention(q, k, v, [mask, bias]).transpose(-2, -3)

    native = mock.Mock(side_effect=reference)
    monkeypatch.setitem(
        sys.modules,
        "openfold3.core.kernels.protenix",
        SimpleNamespace(attention_scaled=native),
    )
    return native


def _attention_inputs(dtype=torch.bfloat16, sequence=17):
    x = torch.randn(1, sequence, sequence, 32, dtype=dtype, requires_grad=True)
    mask = torch.zeros(1, sequence, 1, 1, sequence, dtype=dtype)
    bias = torch.randn(1, 1, 2, sequence, sequence, dtype=dtype, requires_grad=True)
    return x, mask, bias


def test_protenix_prescales_q_and_keeps_projection_and_bias_gradients(native_attention):
    module = Attention(32, 32, 32, 16, 2, gating=False).to(torch.bfloat16)
    torch.nn.init.normal_(module.linear_o.weight, std=0.1)
    x, mask, bias = _attention_inputs()
    with mock.patch.object(module, "_prep_qkv", wraps=module._prep_qkv) as prep:
        out = module(
            x,
            x,
            [mask, bias],
            use_protenix_triangle_attention=True,
            use_cueq_triangle_kernels=True,
        )
    prep.assert_called_once_with(x, x, apply_scale=True)
    native_attention.assert_called_once()
    assert out.shape == x.shape
    out.float().square().sum().backward()
    for value in (x.grad, bias.grad, module.linear_q.weight.grad):
        assert value is not None and torch.isfinite(value).all()
        assert torch.count_nonzero(value) > 0


def test_float32_site_stays_exactly_stock_with_no_provider_cast(native_attention):
    module = Attention(32, 32, 32, 16, 2, gating=False)
    x, mask, bias = _attention_inputs(torch.float32)
    stock = module(x, x, [mask, bias])
    selected = module(
        x,
        x,
        [mask, bias],
        use_protenix_triangle_attention=True,
        use_cueq_triangle_kernels=True,
    )
    assert torch.equal(stock, selected)
    native_attention.assert_not_called()


@pytest.mark.parametrize("sequence", (1, 16))
def test_protenix_short_sequences_use_stock_attention(native_attention, sequence):
    module = Attention(32, 32, 32, 16, 2, gating=False).to(torch.bfloat16)
    torch.nn.init.normal_(module.linear_o.weight, std=0.1)
    x, mask, bias = _attention_inputs(sequence=sequence)
    stock = module(x, x, [mask, bias])
    selected = module(
        x,
        x,
        [mask, bias],
        use_protenix_triangle_attention=True,
        use_cueq_triangle_kernels=True,
    )
    assert torch.equal(stock, selected)
    stock_grads = torch.autograd.grad(stock.float().square().sum(), (x, bias))
    selected_grads = torch.autograd.grad(selected.float().square().sum(), (x, bias))
    assert all(torch.equal(a, b) for a, b in zip(stock_grads, selected_grads))
    native_attention.assert_not_called()


def test_default_route_does_not_import_optional_provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "openfold3.core.kernels.protenix", None)
    module = Attention(32, 32, 32, 16, 2, gating=False).to(torch.bfloat16)
    x, mask, bias = _attention_inputs()
    assert module(x, x, [mask, bias]).shape == x.shape


def test_provider_failure_is_not_silently_replaced(native_attention):
    native_attention.side_effect = RuntimeError("provider launch failed")
    module = Attention(32, 32, 32, 16, 2, gating=False).to(torch.bfloat16)
    x, mask, bias = _attention_inputs()
    with pytest.raises(RuntimeError, match="provider launch failed"):
        module(x, x, [mask, bias], use_protenix_triangle_attention=True)


@pytest.mark.parametrize(
    "conflict",
    (
        "use_stable_triangle_attention",
        "use_deepspeed_evo_attention",
        "use_triton_triangle_kernels",
        "use_lma",
        "use_high_precision",
    ),
)
def test_attention_rejects_conflicting_routes_before_projection(conflict):
    module = Attention(32, 32, 32, 16, 2)
    x, mask, bias = _attention_inputs()
    with (
        mock.patch.object(module, "_prep_qkv") as prep,
        pytest.raises(ValueError, match="mutually exclusive"),
    ):
        module(
            x, x, [mask, bias], use_protenix_triangle_attention=True, **{conflict: True}
        )
    prep.assert_not_called()


@pytest.mark.parametrize("options", ({"chunk_size": 2}, {"inplace_safe": True}))
def test_triangle_rejects_unsupported_execution_modes(options):
    module = TriangleAttention(32, 16, 2)
    with pytest.raises(ValueError, match="chunk_size=None"):
        module(
            torch.zeros(1, 3, 3, 32), use_protenix_triangle_attention=True, **options
        )


@pytest.mark.parametrize("stack_name", ("pairformer", "msa", "template"))
def test_stack_routes_only_triangle_attention_and_scopes_cueq_trimul(
    stack_name,
    native_attention,
):
    architecture = model_config.architecture
    if stack_name == "pairformer":
        config = architecture.pairformer.to_dict()
        cls = PairFormerStack
    elif stack_name == "msa":
        config = architecture.msa.msa_module.to_dict()
        cls = MSAModuleStack
    else:
        config = architecture.template.template_pair_stack.to_dict()
        cls = TemplatePairStack
    config = ConfigDict(config)
    config.update(no_blocks=1, blocks_per_ckpt=None)
    module = cls(**config).to(torch.bfloat16)
    common = dict(use_protenix_triangle_attention=True, use_cueq_triangle_kernels=True)
    sequence = 17
    z = torch.randn(
        1,
        sequence,
        sequence,
        config.get("c_z", config.get("c_t")),
        dtype=torch.bfloat16,
    )
    mask = torch.ones(1, sequence, sequence, dtype=torch.bfloat16)
    with mock.patch.object(
        trimul, "_cueq_triangle_mult", side_effect=lambda z, **kw: z * 0.1
    ) as cueq:
        if stack_name == "pairformer":
            s = torch.randn(1, sequence, config["c_s"], dtype=torch.bfloat16)
            apb = module.blocks[0].attn_pair_bias
            with mock.patch.object(apb, "forward", wraps=apb.forward) as single:
                module(s, z, mask[:, 0], mask, **common)
            assert single.call_args.kwargs["use_high_precision_attention"]
            assert "use_protenix_triangle_attention" not in single.call_args.kwargs
        elif stack_name == "msa":
            m = torch.randn(1, 2, sequence, config["c_m"], dtype=torch.bfloat16)
            module(
                m, z, torch.ones(1, 2, sequence, dtype=torch.bfloat16), mask, **common
            )
        else:
            module(z.unsqueeze(1), mask.unsqueeze(1), **common)
    assert native_attention.call_count == 2
    assert cueq.call_count == (0 if stack_name == "template" else 2)


def _host_stub(protenix=True):
    flags = BACKEND_FLAGS["protenix" if protenix else "cueq"]
    memory = ConfigDict(
        dict(
            flags,
            chunk_size=None,
            msa_module={
                "swiglu_chunk_token_cutoff": None,
                "swiglu_seq_chunk_size": None,
            },
        )
    )
    x = torch.ones(1, 3, 4)
    z = torch.ones(1, 3, 3, 4)
    host = SimpleNamespace(
        training=True,
        settings=SimpleNamespace(train_confidence_only=False),
        _get_mode_mem_settings=lambda: memory,
        _do_inference_offload=lambda **kw: False,
        clear_autocast_cache=lambda: None,
        input_embedder=mock.Mock(return_value=(x, x, z)),
        template_embedder=mock.Mock(return_value=torch.zeros_like(z)),
        msa_module_embedder=mock.Mock(
            return_value=(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3))
        ),
        msa_module=mock.Mock(return_value=z),
        pairformer_stack=mock.Mock(return_value=(x, z)),
        linear_s=lambda a: a,
        layer_norm_s=lambda a: a,
        linear_z=lambda a: a,
        layer_norm_z=lambda a: a,
        shared=SimpleNamespace(
            use_confidence_emb_prob=1.0,
            diffusion=SimpleNamespace(
                no_mini_rollout_steps=1, no_mini_rollout_samples=1
            ),
        ),
        config=SimpleNamespace(architecture=SimpleNamespace(noise_schedule={})),
        sample_diffusion=mock.Mock(return_value=x),
        aux_heads=mock.Mock(return_value={}),
    )
    return host, memory, x, z


@pytest.mark.parametrize("protenix", (True, False))
def test_full_model_scope_preserves_confidence_and_diffusion(monkeypatch, protenix):
    host, _, x, z = _host_stub(protenix)
    batch = {"token_mask": torch.ones(1, 3)}
    OpenFold3.run_trunk(host, batch, 1)
    for call in (host.template_embedder, host.msa_module, host.pairformer_stack):
        assert call.call_args.kwargs["use_protenix_triangle_attention"] is protenix
    assert host.template_embedder.call_args.kwargs["use_cueq_triangle_kernels"] is (
        not protenix
    )
    assert host.msa_module.call_args.kwargs["use_cueq_triangle_kernels"]
    assert host.pairformer_stack.call_args.kwargs["use_cueq_triangle_kernels"]
    monkeypatch.setattr(model_module, "create_noise_schedule", lambda **kw: None)
    OpenFold3._rollout(host, batch, x, x, z)
    confidence = host.aux_heads.call_args.kwargs
    assert confidence["use_cueq_triangle_kernels"] is (not protenix)
    assert "use_protenix_triangle_attention" not in confidence
    diffusion = host.sample_diffusion.call_args.kwargs
    assert diffusion["use_high_precision_attention"]
    assert "use_protenix_triangle_attention" not in diffusion
    assert "use_cueq_triangle_kernels" not in diffusion


@pytest.mark.parametrize(
    "conflict",
    (
        "use_stable_triangle_attention",
        "use_stable_triangle_multiplicative",
        "use_deepspeed_evo_attention",
        "use_triton_triangle_kernels",
        "use_lma",
    ),
)
def test_full_model_conflicts_fail_before_any_compute(conflict):
    host, memory, _, _ = _host_stub()
    memory[conflict] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        OpenFold3.run_trunk(host, {"token_mask": torch.ones(1, 3)}, 1)
    host.input_embedder.assert_not_called()


def test_pair_block_rejects_conflict_before_cueq_mutation():
    block = PairBlock(32, 32, 16, 2, "relu", 2, 0.0, False, 1e9)
    with (
        mock.patch.object(block, "tri_mul_out_in") as multiply,
        pytest.raises(ValueError, match="other backends"),
    ):
        block(
            torch.ones(1, 3, 3, 32),
            torch.ones(1, 3, 3),
            use_protenix_triangle_attention=True,
            use_cueq_triangle_kernels=True,
            use_stable_triangle_multiplicative=True,
        )
    multiply.assert_not_called()


@pytest.mark.parametrize("protenix", (True, False))
@pytest.mark.parametrize("c_hidden", (16, 32))
def test_protenix_cueq_trimul_requires_matching_channels(protenix, c_hidden):
    block = PairBlock(32, c_hidden, 16, 2, "relu", 2, 0.0, False, 1e9)
    torch.nn.init.normal_(block.tri_mul_out.linear_z.weight, std=0.1)
    torch.nn.init.normal_(block.tri_mul_in.linear_z.weight, std=0.1)
    z, mask = torch.randn(1, 3, 3, 32), torch.ones(1, 3, 3)
    with (
        mock.patch.object(
            trimul, "_cueq_triangle_mult", side_effect=lambda z, **kw: z * 0.1
        ) as cueq,
        mock.patch.object(block, "tri_att_start_end", side_effect=lambda z, **kw: z),
    ):
        stock = block(z, mask)
        selected = block(
            z,
            mask,
            use_protenix_triangle_attention=protenix,
            use_cueq_triangle_kernels=True,
        )
    if protenix and c_hidden != 32:
        cueq.assert_not_called()
        assert torch.equal(selected, stock)
    else:
        assert cueq.call_count == 2


@pytest.mark.parametrize("protenix", (True, False))
def test_legacy_evoformer_scopes_cueq_to_trimul_for_protenix(protenix):
    block = EvoformerBlock(
        c_m=32,
        c_z=32,
        c_hidden_msa_att=16,
        c_hidden_opm=16,
        c_hidden_mul=32,
        c_hidden_pair_att=16,
        no_heads_msa=2,
        no_heads_pair=2,
        transition_type="relu",
        transition_n=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        no_column_attention=False,
        opm_first=True,
        fuse_projection_weights=False,
        inf=1e9,
        eps=1e-5,
    )
    m, z = torch.ones(1, 2, 3, 32), torch.ones(1, 3, 3, 32)
    with (
        mock.patch.object(block, "_compute_opm", return_value=(m, z)),
        mock.patch.object(block.msa_att_row, "forward", return_value=m) as row,
        mock.patch.object(block.msa_att_col, "forward", return_value=m) as column,
        mock.patch.object(block.msa_transition, "forward", return_value=m),
        mock.patch.object(block.pair_stack, "forward", return_value=z) as pair,
    ):
        block(
            m,
            z,
            torch.ones(1, 2, 3),
            torch.ones(1, 3, 3),
            use_cueq_triangle_kernels=True,
            use_protenix_triangle_attention=protenix,
        )
    for msa_attention in (row, column):
        assert msa_attention.call_args.kwargs["use_cueq_triangle_kernels"] is (
            not protenix
        )
        assert "use_protenix_triangle_attention" not in msa_attention.call_args.kwargs
    assert pair.call_args.kwargs["use_cueq_triangle_kernels"]
    assert pair.call_args.kwargs["use_protenix_triangle_attention"] is protenix
