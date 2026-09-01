# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the StableFold/AQLab diffusion boundary."""

from __future__ import annotations

import ast
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "projects" / "of3_all_atom" / "model.py"
)


def _openfold3_method(name: str) -> ast.FunctionDef:
    module = ast.parse(MODEL_PATH.read_text(), filename=str(MODEL_PATH))
    model = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenFold3"
    )
    return next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _self_call(method: ast.FunctionDef, attribute: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == attribute
    )


def _keywords(call: ast.Call) -> dict[str, ast.expr]:
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _is_mode_setting(value: ast.expr, name: str) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "mode_mem_settings"
        and value.attr == name
    )


def test_rollout_keeps_aqlab_diffusion_and_disables_stablefold():
    rollout = _openfold3_method("_rollout")
    sample_diffusion = _self_call(rollout, "sample_diffusion")
    keywords = _keywords(sample_diffusion)

    for name in (
        "use_deepspeed_evo_attention",
        "use_triton_triangle_kernels",
        "use_cueq_triangle_kernels",
        "use_lma",
    ):
        assert _is_mode_setting(keywords[name], name)
    assert isinstance(keywords["use_stable_triangle_attention"], ast.Constant)
    assert keywords["use_stable_triangle_attention"].value is False

    enclosing = next(
        node
        for node in ast.walk(rollout)
        if isinstance(node, ast.With) and sample_diffusion in ast.walk(node)
    )
    contexts = [item.context_expr for item in enclosing.items]
    assert any(
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and isinstance(context.func.value, ast.Name)
        and context.func.value.id == "torch"
        and context.func.attr == "no_grad"
        for context in contexts
    )
    autocast = next(
        context
        for context in contexts
        if isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and context.func.attr == "autocast"
    )
    autocast_keywords = _keywords(autocast)
    dtype = autocast_keywords["dtype"]
    assert isinstance(dtype, ast.Attribute)
    assert isinstance(dtype.value, ast.Name)
    assert (dtype.value.id, dtype.attr) == ("torch", "float32")


def test_train_diffusion_keeps_aqlab_stock_high_precision_attention():
    train_diffusion = _openfold3_method("_train_diffusion")
    diffusion_module = _self_call(train_diffusion, "diffusion_module")
    keywords = _keywords(diffusion_module)

    for name in (
        "use_deepspeed_evo_attention",
        "use_triton_triangle_kernels",
        "use_cueq_triangle_kernels",
        "use_stable_triangle_attention",
        "use_lma",
    ):
        assert name not in keywords
    assert isinstance(keywords["use_high_precision_attention"], ast.Constant)
    assert keywords["use_high_precision_attention"].value is True


def test_stablefold_remains_selected_only_at_non_diffusion_sites():
    trunk = _openfold3_method("run_trunk")
    pairformer = _self_call(trunk, "pairformer_stack")
    pairformer_stable = _keywords(pairformer)["use_stable_triangle_attention"]
    assert isinstance(pairformer_stable, ast.Name)
    assert pairformer_stable.id == "use_stable_triangle_attention"

    rollout = _openfold3_method("_rollout")
    confidence = _self_call(rollout, "aux_heads")
    confidence_stable = _keywords(confidence)["use_stable_triangle_attention"]
    assert not (
        isinstance(confidence_stable, ast.Constant) and confidence_stable.value is False
    )
