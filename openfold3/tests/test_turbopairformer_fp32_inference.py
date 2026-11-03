"""Native FP32 inference contracts, independent of the implementation math.

The numerical envelope is fixed before candidate qualification: 1.25 times the
stock/TF32-RNE numerical expectation, plus 2e-6 normalized RMS and 2e-5 peak.
The full immutable boundary matrix lives in the experiment validation artifact;
these smaller regression cases retain the same numerical policy.
"""

import math

import pytest
import torch
from torch.nn import functional as F

from openfold3.core.kernels.turbopairformer.turbopairformer_attention_fp32 import (
    tf32_enabled,
    turbopairformer_attention_fp32,
)
from openfold3.core.kernels.turbopairformer.turbopairformer_trimul_fp32 import (
    run_trimul_fp32,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate,
)

CUDA_SM90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Native FP32 qualification targets CUDA SM90",
)


@pytest.fixture(autouse=True)
def restore_precision_policy():
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _set_mode(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _round_tf32(value):
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    return torch.where(
        torch.isfinite(value), rounded.to(torch.int32).view(torch.float32), value
    )


def _error(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    return (
        float(delta.square().mean().sqrt())
        / max(float(expected.square().mean().sqrt()), 1e-12),
        float(delta.abs().max()) / max(float(expected.abs().max()), 1e-12),
    )


def _fixed_gate(actual, oracle, stock, surrogates=()):
    assert actual.dtype == torch.float32 and actual.shape == oracle.shape
    assert actual.is_contiguous() and torch.isfinite(actual).all()
    actual_errors = _error(actual, oracle)
    references = [_error(stock, oracle), *(_error(item, oracle) for item in surrogates)]
    for index, floor in enumerate((2e-6, 2e-5)):
        limit = 1.25 * max(item[index] for item in references) + floor
        assert actual_errors[index] <= limit, (actual_errors, limit)


def _attention(q, k, v, mask, bias, *, double=False, tf32=False, scale_after=False):
    if double:
        q, k, v, mask, bias = (value.double() for value in (q, k, v, mask, bias))
    scale = 1 / math.sqrt(q.shape[-1])
    if tf32:
        scores = _round_tf32(q if scale_after else q * scale).double() @ _round_tf32(
            k
        ).double().transpose(-1, -2)
        scores = (scores * scale if scale_after else scores).float()
    else:
        scores = (q * scale) @ k.transpose(-1, -2)
    scores = scores + mask
    scores = scores + bias
    all_infinite = torch.isneginf(scores).all(-1, keepdim=True)
    probabilities = torch.softmax(
        torch.where(all_infinite, torch.zeros_like(scores), scores), -1
    )
    probabilities = torch.where(
        all_infinite, torch.zeros_like(probabilities), probabilities
    )
    if double:
        all_finite_masked = torch.isfinite(mask).all(-1, keepdim=True) & (
            mask <= -1e8
        ).all(-1, keepdim=True)
        probabilities = torch.where(
            all_finite_masked,
            torch.full_like(probabilities, 1 / q.shape[-2]),
            probabilities,
        )
    output = (
        (_round_tf32(probabilities).double() @ _round_tf32(v).double()).float()
        if tf32
        else probabilities @ v
    )
    return output.transpose(-2, -3).contiguous()


def _trimul(module, z, mask, *, double=False, tf32=False):
    dtype = torch.float64 if double else torch.float32
    z = z.to(dtype)

    def linear(value, layer):
        weight = layer.weight.to(dtype)
        output = (
            (_round_tf32(value).double() @ _round_tf32(weight).double().T).float()
            if tf32
            else value @ weight.T
        )
        return output if layer.bias is None else output + layer.bias.to(dtype)

    def norm(value, layer):
        return F.layer_norm(
            value, layer.c_in, layer.weight.to(dtype), layer.bias.to(dtype), layer.eps
        )

    normalized = norm(z, module.layer_norm_in)
    mask = mask.to(dtype).unsqueeze(-1)
    a = (
        mask
        * torch.sigmoid(linear(normalized, module.linear_a_g))
        * linear(normalized, module.linear_a_p)
    )
    b = (
        mask
        * torch.sigmoid(linear(normalized, module.linear_b_g))
        * linear(normalized, module.linear_b_p)
    )
    a, b = a.movedim(-1, -3), b.movedim(-1, -3)
    lhs, rhs = (
        (a, b.transpose(-1, -2)) if module._outgoing else (a.transpose(-1, -2), b)
    )
    product = (
        (_round_tf32(lhs).double() @ _round_tf32(rhs).double()).float()
        if tf32
        else lhs @ rhs
    )
    output = linear(
        norm(product.movedim(-3, -1), module.layer_norm_out), module.linear_z
    )
    return (output * torch.sigmoid(linear(normalized, module.linear_g))).contiguous()


def _module(channels, outgoing):
    module = (
        TriangleMultiplicativeUpdate(channels, channels, _outgoing=outgoing)
        .cuda()
        .eval()
    )
    # Default final projections are zero; randomized nonzero weights are
    # essential for a meaningful output and gate/layout regression.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "layer_norm" in name and name.endswith("weight"):
                parameter.copy_(1 + torch.randn_like(parameter) * 0.1)
            elif parameter.ndim == 1:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
            else:
                parameter.copy_(
                    torch.randn_like(parameter) / math.sqrt(parameter.shape[-1])
                )
    return module


@CUDA_SM90
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "batch,n,d", [((1,), 33, 16), ((2, 3), 17, 32), ((1,), 129, 64)]
)
def test_attention_numeric_broadcast_strides_stream_and_policy(enabled, batch, n, d):
    torch.manual_seed(1701)
    _set_mode(enabled)
    q, k, v = [
        torch.randn((*batch, 2, 2, n, 2 * d), device="cuda")[..., ::2] for _ in range(3)
    ]
    mask = torch.zeros((*([1] * len(batch)), 1, 1, 1, n), device="cuda")
    mask[..., 1::3] = -1e9
    bias = torch.randn((*([1] * len(batch)), 1, 1, n, n), device="cuda") * 0.2
    inputs = (q, k, v, mask, bias)
    original = [value.clone() for value in inputs]
    with torch.inference_mode():
        oracle, stock = _attention(*inputs, double=True), _attention(*inputs)
        surrogates = (
            [
                _attention(*inputs, tf32=True, scale_after=after)
                for after in (False, True)
            ]
            if enabled
            else []
        )
        workspace = 4 * (n + d) * (128 * 3 if n == 129 else 5)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = turbopairformer_attention_fp32(*inputs, workspace_bytes=workspace)
        torch.cuda.current_stream().wait_stream(stream)
        _fixed_gate(actual, oracle, stock, surrogates)
    assert all(torch.equal(value, before) for value, before in zip(inputs, original))
    assert torch.backends.cuda.matmul.allow_tf32 == enabled
    assert torch.backends.cudnn.allow_tf32 == enabled


@CUDA_SM90
@pytest.mark.parametrize("mask_value", [-1e9, -float("inf")])
def test_attention_all_masked_and_nan_contract(mask_value):
    _set_mode(False)
    torch.manual_seed(1702)
    q, k, v = [torch.randn(1, 2, 2, 17, 16, device="cuda") for _ in range(3)]
    mask = torch.full((1, 2, 1, 1, 17), mask_value, device="cuda")
    bias = torch.randn(1, 1, 2, 17, 17, device="cuda") * 0.2
    with torch.inference_mode():
        actual = turbopairformer_attention_fp32(q, k, v, mask, bias)
        expected = _attention(q, k, v, mask, bias, double=True)
        _fixed_gate(actual, expected, _attention(q, k, v, mask, bias))
        if math.isinf(mask_value):
            assert torch.count_nonzero(actual) == 0
        q[0, 0, 0, 0, 0] = float("nan")
        assert torch.isnan(turbopairformer_attention_fp32(q, k, v, mask, bias)).any()


@CUDA_SM90
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("outgoing", [False, True])
@pytest.mark.parametrize("channels", [32, 64])
def test_trimul_numeric_chunk_tails_and_input_preservation(enabled, outgoing, channels):
    torch.manual_seed(1703)
    _set_mode(enabled)
    module = _module(channels, outgoing)
    z = torch.randn(2, 33, 33, channels, device="cuda").transpose(-3, -2)
    mask = (torch.rand(1, 33, 33, device="cuda") > 0.2).float()
    before = (
        z.clone(),
        mask.clone(),
        {name: value.clone() for name, value in module.state_dict().items()},
    )
    with torch.inference_mode():
        oracle = _trimul(module, z, mask, double=True)
        stock = module(z, mask, inplace_safe=False)
        surrogates = [_trimul(module, z, mask, tf32=True)] if enabled else []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = run_trimul_fp32(
                module,
                z,
                mask,
                output_row_chunk_size=5,
                projection_workspace_bytes=6 * channels * 4 * 7,
            )
        torch.cuda.current_stream().wait_stream(stream)
        _fixed_gate(actual, oracle, stock, surrogates)
    assert torch.equal(z, before[0]) and torch.equal(mask, before[1])
    assert all(
        torch.equal(value, before[2][name])
        for name, value in module.state_dict().items()
    )
    assert torch.backends.cuda.matmul.allow_tf32 == enabled
    assert torch.backends.cudnn.allow_tf32 == enabled


def test_native_fp32_cpu_requests_are_rejected_before_extension_load():
    q = torch.zeros(1, 2, 2, 3, 16)
    mask = torch.zeros(1, 2, 1, 1, 3)
    bias = torch.zeros(1, 1, 2, 3, 3)
    module = TriangleMultiplicativeUpdate(32, 32).eval()
    with torch.no_grad():
        with pytest.raises(ValueError):
            turbopairformer_attention_fp32(q, q, q, mask, bias)
        with pytest.raises(ValueError):
            run_trimul_fp32(module, torch.zeros(1, 3, 3, 32))


@CUDA_SM90
def test_native_training_calls_reject_and_tf32_override_is_respected(monkeypatch):
    q = torch.zeros(1, 2, 2, 3, 16, device="cuda", requires_grad=True)
    mask = torch.zeros(1, 2, 1, 1, 3, device="cuda")
    bias = torch.zeros(1, 1, 2, 3, 3, device="cuda")
    module = _module(32, True)
    with torch.enable_grad():
        with pytest.raises((ValueError, RuntimeError)):
            turbopairformer_attention_fp32(q, q, q, mask, bias)
        with pytest.raises(RuntimeError):
            run_trimul_fp32(module, torch.zeros(1, 3, 3, 32, device="cuda"))
    _set_mode(True)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    assert tf32_enabled() is False


@CUDA_SM90
@pytest.mark.parametrize("outgoing", [False, True])
def test_trimul_model_route_preserves_bf16_autocast_for_fp32_residual(
    monkeypatch, outgoing
):
    """Model dispatch respects mixed precision; direct FP32 calls remain explicit."""
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_trimul_fp32 as native_module,
    )
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_trimul_support as support_module,
    )

    torch.manual_seed(1704)
    _set_mode(False)
    module = _module(32, outgoing)
    z = torch.randn(1, 17, 17, 32, device="cuda", dtype=torch.float32)
    mask = (torch.rand(1, 17, 17, device="cuda") > 0.2).float()
    original_z = z.clone()
    original_mask = mask.clone()
    native_calls = []
    support_codes = []
    original_support = support_module.check_stable_trimul_support

    def record_native_route(*args, **kwargs):
        # Without the model-dispatch guard this rejects the accidental native
        # call during autocast, even if numerical tolerances would pass.
        assert not torch.is_autocast_enabled("cuda")
        native_calls.append(True)
        return run_trimul_fp32(*args, **kwargs)

    def record_old_support(*args, **kwargs):
        result = original_support(*args, **kwargs)
        support_codes.append(result.code)
        return result

    monkeypatch.setattr(native_module, "run_trimul_fp32", record_native_route)
    monkeypatch.setattr(
        support_module, "check_stable_trimul_support", record_old_support
    )

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stock = module(z, mask, inplace_safe=False)
            mixed = module(
                z,
                mask,
                inplace_safe=False,
                use_stable_triangle_multiplicative=True,
            )
            assert torch.is_autocast_enabled("cuda")
        assert stock.dtype == mixed.dtype == torch.bfloat16
        assert torch.equal(mixed, stock)
        assert support_codes == ["z_dtype"]
        assert not native_calls

        native = module(
            z,
            mask,
            inplace_safe=False,
            use_stable_triangle_multiplicative=True,
        )
        assert len(native_calls) == 1
        assert support_codes == ["z_dtype"]
        assert native.dtype == torch.float32 and torch.isfinite(native).all()

        # The directly imported function deliberately opts into FP32 compute
        # and disables the caller's autocast only within its own implementation.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            direct = run_trimul_fp32(module, z, mask)
            assert torch.is_autocast_enabled("cuda")
        assert direct.dtype == torch.float32
        assert torch.equal(direct, native)
        assert len(native_calls) == 1

    assert torch.equal(z, original_z) and torch.equal(mask, original_mask)


@CUDA_SM90
def test_attention_model_route_respects_autocast_with_forced_fp32_projections(
    monkeypatch,
):
    """FP32 projections inside autocast still use the established BF16 facade."""
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_triangle_attention as facade,
    )
    from openfold3.core.model.primitives.attention import Attention

    torch.manual_seed(1705)
    _set_mode(False)
    module = Attention(32, 32, 32, 16, 2, gating=False).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.1)
    for projection in (module.linear_q, module.linear_k, module.linear_v):
        projection.precision = torch.float32
    x = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float32)
    mask = torch.zeros(1, 2, 1, 1, 17, device="cuda", dtype=torch.float32)
    mask[..., 1::3] = -1e9
    zero_bias = torch.zeros(1, 1, 2, 17, 17, device="cuda", dtype=torch.float32)
    original_x, original_mask = x.clone(), mask.clone()
    routed_dtypes = []
    original_facade = facade.turbopairformer_triangle_attention

    def record_facade(q, k, v, res_mask, pair_bias):
        routed_dtypes.append((q.dtype, k.dtype, v.dtype, pair_bias.dtype))
        return original_facade(q, k, v, res_mask, pair_bias)

    monkeypatch.setattr(facade, "turbopairformer_triangle_attention", record_facade)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, k, v = module._prep_qkv(x, x, apply_scale=False)
            assert all(value.dtype == torch.float32 for value in (q, k, v))
            mixed = module(x, x, biases=[mask], use_stable_triangle_attention=True)
            expected_attention = original_facade(
                q.bfloat16(), k.bfloat16(), v.bfloat16(), mask, zero_bias.bfloat16()
            ).float()
            expected = module._wrap_up(expected_attention, x)
            assert mixed.dtype == torch.bfloat16 and torch.equal(mixed, expected)
            assert routed_dtypes == [(torch.bfloat16,) * 4]

            # The post-QKV rejection branch must preserve the same legacy
            # fallback policy rather than misclassify these FP32 projections.
            with monkeypatch.context() as rejected:
                rejected.setattr(
                    facade,
                    "is_turbopairformer_supported",
                    lambda *args: (False, "synthetic BF16 shape rejection"),
                )
                fallback = module(
                    x, x, biases=[mask], use_stable_triangle_attention=True
                )
                stock = module(x, x, biases=[mask])
                assert torch.equal(fallback, stock)
            assert routed_dtypes == [(torch.bfloat16,) * 4]
            assert torch.is_autocast_enabled("cuda")

        native = module(x, x, biases=[mask], use_stable_triangle_attention=True)
        assert routed_dtypes == [(torch.bfloat16,) * 4, (torch.float32,) * 4]
        q, k, v = module._prep_qkv(x, x, apply_scale=False)
        native_attention = turbopairformer_attention_fp32(q, k, v, mask, zero_bias)
        assert native.dtype == torch.float32
        assert torch.equal(native, module._wrap_up(native_attention, x))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            direct = turbopairformer_attention_fp32(q, k, v, mask, zero_bias)
            assert torch.is_autocast_enabled("cuda")
        assert direct.dtype == torch.float32 and torch.equal(direct, native_attention)

    assert torch.equal(x, original_x) and torch.equal(mask, original_mask)


@CUDA_SM90
@pytest.mark.parametrize("nested_bf16", [False, True])
@pytest.mark.parametrize("outgoing", [False, True])
def test_trimul_model_route_accepts_enabled_fp32_autocast(
    monkeypatch, outgoing, nested_bf16
):
    """The confidence head's FP32 autocast context still selects native FP32."""
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_trimul_fp32 as native_module,
    )

    torch.manual_seed(1706)
    _set_mode(False)
    module = _module(32, outgoing)
    z = torch.randn(1, 17, 17, 32, device="cuda", dtype=torch.float32)
    mask = (torch.rand(1, 17, 17, device="cuda") > 0.2).float()
    original_z, original_mask = z.clone(), mask.clone()
    native_contexts = []

    def record_native_route(*args, **kwargs):
        native_contexts.append(
            (
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )
        )
        return run_trimul_fp32(*args, **kwargs)

    monkeypatch.setattr(native_module, "run_trimul_fp32", record_native_route)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        expected = module(
            z,
            mask,
            inplace_safe=False,
            use_stable_triangle_multiplicative=True,
        )
        assert len(native_contexts) == 1 and not native_contexts[0][0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=nested_bf16):
            with torch.autocast("cuda", dtype=torch.float32):
                # PyTorch 2.10 keeps this context enabled; earlier versions
                # can disable it. Both cases still request FP32 compute.
                fp32_context_enabled = torch.is_autocast_enabled("cuda")
                assert torch.get_autocast_dtype("cuda") == torch.float32
                actual = module(
                    z,
                    mask,
                    inplace_safe=False,
                    use_stable_triangle_multiplicative=True,
                )
                assert torch.is_autocast_enabled("cuda") == fp32_context_enabled
                assert torch.get_autocast_dtype("cuda") == torch.float32
            assert torch.is_autocast_enabled("cuda") == nested_bf16
            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        assert not torch.is_autocast_enabled("cuda")

    assert native_contexts[1:] == [(fp32_context_enabled, torch.float32)]
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.equal(actual, expected)
    assert torch.equal(z, original_z) and torch.equal(mask, original_mask)


@CUDA_SM90
@pytest.mark.parametrize("nested_bf16", [False, True])
def test_attention_model_route_accepts_enabled_fp32_autocast(monkeypatch, nested_bf16):
    """Model FP32 autocast reaches the native API without a BF16 conversion."""
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_attention_fp32 as native_module,
    )
    from openfold3.core.model.primitives.attention import Attention

    torch.manual_seed(1707)
    _set_mode(False)
    module = Attention(32, 32, 32, 16, 2, gating=False).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.1)
    x = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float32)
    mask = torch.zeros(1, 2, 1, 1, 17, device="cuda", dtype=torch.float32)
    mask[..., 1::3] = -1e9
    original_x, original_mask = x.clone(), mask.clone()
    native_contexts = []

    def record_native_route(q, k, v, res_mask, pair_bias):
        assert all(
            value.dtype == torch.float32 for value in (q, k, v, res_mask, pair_bias)
        )
        native_contexts.append(
            (
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )
        )
        return turbopairformer_attention_fp32(q, k, v, res_mask, pair_bias)

    monkeypatch.setattr(
        native_module, "turbopairformer_attention_fp32", record_native_route
    )
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        expected = module(x, x, biases=[mask], use_stable_triangle_attention=True)
        assert len(native_contexts) == 1 and not native_contexts[0][0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=nested_bf16):
            with torch.autocast("cuda", dtype=torch.float32):
                fp32_context_enabled = torch.is_autocast_enabled("cuda")
                assert torch.get_autocast_dtype("cuda") == torch.float32
                actual = module(x, x, biases=[mask], use_stable_triangle_attention=True)
                assert torch.is_autocast_enabled("cuda") == fp32_context_enabled
                assert torch.get_autocast_dtype("cuda") == torch.float32
            assert torch.is_autocast_enabled("cuda") == nested_bf16
            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        assert not torch.is_autocast_enabled("cuda")

    assert native_contexts[1:] == [(fp32_context_enabled, torch.float32)]
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.equal(actual, expected)
    assert torch.equal(x, original_x) and torch.equal(mask, original_mask)
