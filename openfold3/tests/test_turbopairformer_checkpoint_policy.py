"""Checkpointed training must keep the trainable Turbo dispatch in both passes."""

import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from openfold3.core.kernels.turbopairformer import (
    turbopairformer_triangle_attention as facade,
)
from openfold3.core.kernels.turbopairformer import (
    turbopairformer_trimul as stable_trimul,
)
from openfold3.core.kernels.turbopairformer import (
    turbopairformer_trimul_fp32 as native_trimul,
)
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.utils import checkpointing
from openfold3.core.utils.checkpointing import (
    get_checkpoint_fn,
    is_inference_without_checkpoint,
)


def initialized_module(kind, channels=32):
    if kind == "primitive":
        module = Attention(channels, channels, channels, 16, channels // 16)
    elif kind == "attention":
        module = TriangleAttention(
            channels, 32 if channels == 128 else 16, 4 if channels == 128 else 2
        )
    else:
        module = TriangleMultiplicationOutgoing(channels, channels)
    generator = torch.Generator().manual_seed(429)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = torch.randn(parameter.shape, generator=generator) * 0.1
            if "norm" in name and name.endswith("weight"):
                value += 1
            parameter.copy_(value)
    return module


def call_module(module, kind, x, mask):
    if kind == "primitive":
        bias = x.new_zeros((1, 1, module.no_heads, x.shape[-2], x.shape[-2]))
        mask_bias = ((mask - 1) * 1e9)[..., :, None, None, :]
        return module(
            x, x, biases=[mask_bias, bias], use_stable_triangle_attention=True
        )
    if kind == "attention":
        return module(x, mask, use_stable_triangle_attention=True)
    return module(x, mask, use_stable_triangle_multiplicative=True)


@pytest.fixture
def cpu_cuda_math(monkeypatch):
    """Substitute CUDA arithmetic, retaining real model dispatch and autograd."""
    routes = []
    monkeypatch.setenv("OPENFOLD3_SR_BACKWARD", "1")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    monkeypatch.setattr(
        facade, "is_turbopairformer_supported", lambda *args: (True, "test")
    )

    def attention(q, k, v, mask, bias):
        routes.append(("attention", q.dtype, torch.is_grad_enabled()))
        scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        result = torch.softmax(scores + mask.float() + bias.float(), -1) @ v.float()
        return result.to(q.dtype).transpose(-2, -3)

    def trimul(module, z, mask):
        routes.append(("trimul", z.dtype, torch.is_grad_enabled()))
        return module(z, mask)

    monkeypatch.setattr(facade, "turbopairformer_triangle_attention", attention)
    monkeypatch.setattr(native_trimul, "run_trimul_fp32", trimul)
    return routes


def forward_and_gradients(module, kind, source, mask, mode, nested_projection=False):
    module.zero_grad(set_to_none=True)
    x = source.detach().clone().requires_grad_(True)

    def function(value):
        if nested_projection:
            # In reentrant forward this intermediate loses requires_grad.
            value = value * 0.75 + 0.1
        return call_module(module, kind, value, mask)

    if mode == "direct":
        output = function(x)
    elif mode == "torch_reentrant":
        output = checkpoint(function, x, use_reentrant=True)
    else:
        reentrant = mode == "wrapped_reentrant"
        output = get_checkpoint_fn(reentrant)(function, x, use_reentrant=reentrant)
    loss = (output.square() + output.sin() * 0.03).mean()
    loss.backward()
    gradients = [x.grad.detach().clone()]
    gradients.extend(
        parameter.grad.detach().clone() for parameter in module.parameters()
    )
    return output.detach().clone(), gradients


@pytest.mark.parametrize("kind", ["primitive", "attention", "trimul"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("nested_projection", [False, True])
def test_checkpoint_matches_dense_training_on_cpu(
    cpu_cuda_math, kind, training, nested_projection
):
    module = initialized_module(kind).train(training)
    source = torch.randn(1, 7, 7, 32)
    mask = torch.ones(1, 7, 7)
    ordinary, reference_gradients = forward_and_gradients(
        module, kind, source, mask, "direct", nested_projection
    )
    modes = ["wrapped_reentrant", "wrapped_nonreentrant"]
    # Raw torch checkpoint is detectable at the immediate module boundary.
    # Larger eval functions must use our wrapper to mark intermediate tensors.
    if not nested_projection:
        modes.append("torch_reentrant")
    for mode in modes:
        cpu_cuda_math.clear()
        actual, gradients = forward_and_gradients(
            module, kind, source, mask, mode, nested_projection
        )
        assert torch.equal(actual, ordinary)
        assert all(torch.equal(a, b) for a, b in zip(gradients, reference_gradients))
        assert not any(route[0] == "trimul" for route in cpu_cuda_math)
        assert all(route[1] == torch.bfloat16 for route in cpu_cuda_math)


@pytest.mark.parametrize("kind", ["primitive", "attention", "trimul"])
def test_automatic_fp32_inference_requires_eval_and_non_grad_inputs(
    cpu_cuda_math, kind
):
    module = initialized_module(kind)
    x = torch.randn(1, 7, 7, 32)
    mask = torch.ones(1, 7, 7)
    with torch.no_grad():
        call_module(module, kind, x, mask)
        assert not any(route[0] == "trimul" for route in cpu_cuda_math)
        assert all(route[1] == torch.bfloat16 for route in cpu_cuda_math)
        cpu_cuda_math.clear()
        module.eval()
        call_module(module, kind, x, mask)
        assert len(cpu_cuda_math) == 1 and cpu_cuda_math[0][1] == torch.float32
        cpu_cuda_math.clear()
        call_module(module, kind, x.requires_grad_(), mask)
        assert not any(route[0] == "trimul" for route in cpu_cuda_math)
        assert all(route[1] == torch.bfloat16 for route in cpu_cuda_math)


def test_checkpoint_context_restores_after_nested_calls_and_exception():
    module = torch.nn.Identity().eval()
    observations = []

    def inner(value):
        intermediate = value * 2
        observations.append(is_inference_without_checkpoint(module, intermediate))
        return intermediate.square()

    def outer(value):
        # The explicit nested no_grad call also runs while the marker is set.
        with torch.no_grad():
            get_checkpoint_fn(True)(inner, value, use_reentrant=True)
            observations.append(is_inference_without_checkpoint(module, value.detach()))
        return value.sin()

    x = torch.randn(3, requires_grad=True)
    get_checkpoint_fn(True)(outer, x, use_reentrant=True).sum().backward()
    assert observations and not any(observations)
    with torch.no_grad():
        assert is_inference_without_checkpoint(module, x.detach())

    def fails(value):
        assert not is_inference_without_checkpoint(module, value.detach())
        raise RuntimeError("checkpoint sentinel")

    with pytest.raises(RuntimeError, match="checkpoint sentinel"):
        get_checkpoint_fn(True)(fails, x, use_reentrant=True)
    with torch.no_grad():
        assert is_inference_without_checkpoint(module, x.detach())


@pytest.mark.parametrize("override_reentrant", [False, True])
def test_checkpoint_factory_binds_default_and_protects_explicit_override(
    override_reentrant,
):
    module = torch.nn.Identity().eval()
    observations = []

    def body(value):
        intermediate = value.sin()
        observations.append(
            (
                torch.is_grad_enabled(),
                is_inference_without_checkpoint(module, intermediate),
            )
        )
        return intermediate.square()

    x = torch.randn(5, requires_grad=True)
    expected = x.sin().square()
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    kwargs = {"use_reentrant": True} if override_reentrant else {}
    actual = get_checkpoint_fn(False)(body, x, **kwargs)
    actual.sum().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(x.grad, expected_gradient)
    assert observations[0] == (not override_reentrant, False)
    assert all(not inference for _, inference in observations)


@pytest.mark.parametrize("reentrant", [False, True])
def test_native_checkpoint_keeps_compile_support(reentrant):
    from torch._dynamo import is_dynamo_supported

    if not is_dynamo_supported():
        pytest.skip("torch.compile does not support this Python/PyTorch platform")
    module = torch.nn.Identity().eval()
    checkpoint_fn = get_checkpoint_fn(reentrant)

    def body(value):
        intermediate = value.sin() * 0.75 + 0.1
        assert not is_inference_without_checkpoint(module, intermediate)
        return intermediate.square()

    def function(value):
        return checkpoint_fn(body, value)

    x = torch.randn(5, requires_grad=True)
    expected = body(x)
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    try:
        compiled = torch.compile(function, backend="eager", fullgraph=True)
        actual = compiled(x)
        actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
        assert torch.equal(actual, expected)
        assert torch.equal(actual_gradient, expected_gradient)
    finally:
        torch.compiler.reset()


def test_deepspeed_checkpoint_keeps_context_during_compile_detection(monkeypatch):
    module = torch.nn.Identity().eval()

    def deepspeed_checkpoint(function, value):
        with torch.no_grad():
            return function(value)

    monkeypatch.setattr(checkpointing, "is_deepspeed_configured", lambda: True)
    monkeypatch.setattr(
        checkpointing,
        "deepspeed",
        SimpleNamespace(checkpointing=SimpleNamespace(checkpoint=deepspeed_checkpoint)),
        raising=False,
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def body(value):
        intermediate = value * 2
        assert not is_inference_without_checkpoint(module, intermediate)
        return intermediate

    x = torch.randn(3, requires_grad=True)
    assert torch.equal(get_checkpoint_fn(True)(body, x), x * 2)
    with torch.no_grad():
        assert is_inference_without_checkpoint(module, x.detach())


@pytest.mark.parametrize(
    "training,grad_enabled", [(True, True), (True, False), (False, True)]
)
def test_stable_outer_chunks_reject_training(training, grad_enabled):
    module = initialized_module("attention").train(training)
    x = torch.randn(1, 7, 7, 32)
    with (
        torch.set_grad_enabled(grad_enabled),
        pytest.raises(ValueError, match="chunking is inference-only"),
    ):
        module(x, chunk_size=3, use_stable_triangle_attention=True)


def test_stable_outer_chunks_reject_eval_checkpoint_with_intermediate_projection():
    module = initialized_module("attention").eval()
    x = torch.randn(1, 7, 7, 32, requires_grad=True)

    def function(value):
        return module(value * 2, chunk_size=3, use_stable_triangle_attention=True)

    with pytest.raises(ValueError, match="chunking is inference-only"):
        get_checkpoint_fn(True)(function, x, use_reentrant=True)


@pytest.mark.parametrize("fused_result", [None, torch.zeros(1, 7, 7, 32)])
def test_policy_is_validated_before_fused_dispatch(monkeypatch, fused_result):
    module = initialized_module("attention").eval()
    monkeypatch.setenv("OPENFOLD3_SR_BACKWARD", "0")
    with (
        torch.no_grad(),
        mock.patch.object(
            module, "_fused_fp32_inference", return_value=fused_result
        ) as fused,
        pytest.raises(RuntimeError, match="single-reconstruction backward"),
    ):
        module(torch.randn(1, 7, 7, 32), use_stable_triangle_attention=True)
    fused.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper CUDA kernels",
)
@pytest.mark.parametrize("kind", ["attention", "trimul"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("tf32", [False, True])
def test_hopper_reentrant_checkpoint_preserves_real_outputs_and_gradients(
    kind, training, tf32
):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        module = initialized_module(kind, 128).cuda().train(training)
        source = torch.randn(1, 64, 64, 128, device="cuda")
        mask = torch.ones(1, 64, 64, device="cuda")
        mask[..., -7:] = 0
        for nested_projection in (False, True):
            ordinary, reference_gradients = forward_and_gradients(
                module, kind, source, mask, "direct", nested_projection
            )
            mode = "wrapped_reentrant" if nested_projection else "torch_reentrant"
            actual, gradients = forward_and_gradients(
                module, kind, source, mask, mode, nested_projection
            )
            assert torch.equal(actual, ordinary)
            assert all(
                torch.equal(a, b) for a, b in zip(gradients, reference_gradients)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper CUDA kernels",
)
@pytest.mark.parametrize("training", [True, False])
def test_hopper_bf16_trimul_checkpoint_uses_aot_and_preserves_gradients(training):
    # FP32 TriMul training deliberately uses the stock route. N > 100 and BF16
    # inputs exercise the separate AOT training path with FP32 parameters.
    module = initialized_module("trimul", 128).cuda().train(training)
    source = torch.randn(1, 113, 113, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 113, 113, device="cuda")
    mask[..., -7:] = 0
    with mock.patch.object(
        stable_trimul,
        "_run_supported_stable_trimul",
        wraps=stable_trimul._run_supported_stable_trimul,
    ) as aot:
        ordinary, reference_gradients = forward_and_gradients(
            module, "trimul", source, mask, "direct", nested_projection=True
        )
        assert aot.call_count == 1
        for mode in ("wrapped_reentrant", "wrapped_nonreentrant"):
            aot.reset_mock()
            actual, gradients = forward_and_gradients(
                module, "trimul", source, mask, mode, nested_projection=True
            )
            # Both the original forward and recomputation must really use AOT.
            assert aot.call_count == 2
            assert all(
                call.args[1].dtype == torch.bfloat16 for call in aot.call_args_list
            )
            assert torch.equal(actual, ordinary)
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert all(
                torch.equal(a, b) for a, b in zip(gradients, reference_gradients)
            )
