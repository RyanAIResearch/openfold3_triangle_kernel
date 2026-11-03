"""The fused TF32 TriangleAttention inference route keeps the unfused route's bits.

TriangleAttention._fused_fp32_inference runs LayerNorm and the q/k/v/gate/
pair-bias projections in two TurboPairFormer kernels and applies the gate in the
attention kernel. Its output must equal, bit for bit, the unfused route
(PyTorch LayerNorm, cuBLAS TF32 linears, torch.sigmoid, the fused attention,
``o * g`` and linear_o), which these tests obtain by disabling the fused route.
"""

from unittest import mock

import pytest
import torch

from openfold3.core.model.layers.triangular_attention import TriangleAttention

CUDA_SM90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the fused route targets CUDA SM90",
)


@pytest.fixture(autouse=True)
def restore_precision_policy():
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _module(heads, hidden, seed):
    torch.manual_seed(seed)
    module = TriangleAttention(128, hidden, heads)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim == 2:
                parameter.normal_(0.0, 0.05)
        module.layer_norm.weight.copy_(1 + 0.3 * torch.randn(128))
        module.layer_norm.bias.copy_(0.2 * torch.randn(128))
    return module.cuda().eval()


def _inputs(prefix, n, ending, seed):
    torch.manual_seed(seed)
    z = torch.randn(*prefix, n, n, 128, device="cuda") * 2 + 0.3
    mask = torch.ones(*prefix, n, n, device="cuda")
    mask[..., -max(1, n // 7) :] = 0
    if ending:
        return z.transpose(-2, -3), mask.transpose(-1, -2)
    return z, mask


def _run(module, x, mask, *, fused, **kwargs):
    taken = []
    original = TriangleAttention._fused_fp32_inference

    def spy(self, *args, **spy_kwargs):
        result = original(self, *args, **spy_kwargs) if fused else None
        taken.append(result is not None)
        return result

    with mock.patch.object(TriangleAttention, "_fused_fp32_inference", spy):
        output = module(x, mask=mask, use_stable_triangle_attention=True, **kwargs)
    return output, any(taken)


@CUDA_SM90
@pytest.mark.parametrize(
    "heads,hidden,prefix,n",
    [(4, 32, (1,), 64), (4, 32, (1,), 97), (4, 32, (2,), 70), (8, 16, (1,), 46)],
)
@pytest.mark.parametrize("ending", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1024])
def test_fused_route_matches_unfused_bits(heads, hidden, prefix, n, ending, chunk_size):
    torch.backends.cuda.matmul.allow_tf32 = True
    module = _module(heads, hidden, seed=n)
    x, mask = _inputs(prefix, n, ending, seed=7 * n + ending)
    with torch.inference_mode():
        options = dict(transpose_bias=ending, chunk_size=chunk_size)
        fused, took_fused = _run(module, x, mask, fused=True, **options)
        unfused, _ = _run(module, x, mask, fused=False, **options)
    assert took_fused
    assert fused.shape == unfused.shape and fused.dtype == torch.float32
    assert torch.equal(fused.view(torch.int32), unfused.view(torch.int32))


# Every chunk, including the last, holds at least 128 pair rows (the fused
# route's bound): 97 % 33 = 31 rows of 97, 70 % 4 = 2 rows of 70, 1 row of 130.
@CUDA_SM90
@pytest.mark.parametrize(
    "prefix,n,chunk_size", [((1,), 97, 33), ((2,), 70, 4), ((1,), 130, 1)]
)
@pytest.mark.parametrize("ending", [False, True])
def test_chunked_fused_route_matches_unfused_bits(prefix, n, chunk_size, ending):
    torch.backends.cuda.matmul.allow_tf32 = True
    module = _module(4, 32, seed=n + chunk_size)
    x, mask = _inputs(prefix, n, ending, seed=11 * n + ending)
    with torch.inference_mode():
        options = dict(transpose_bias=ending, chunk_size=chunk_size, inplace_safe=True)
        fused, took_fused = _run(module, x, mask, fused=True, **options)
        unfused, _ = _run(module, x, mask, fused=False, **options)
    assert took_fused
    assert torch.equal(fused.view(torch.int32), unfused.view(torch.int32))


@CUDA_SM90
def test_fused_route_is_not_taken_outside_its_contract():
    module = _module(4, 32, seed=3)
    with torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = True
        # Below N^2 = 4096 cuBLAS selects small-matrix kernels with other bits.
        x, mask = _inputs((1,), 63, False, seed=1)
        assert not _run(module, x, mask, fused=True)[1]
        # A chunk below 128 pair rows (here the last, one-row chunk of N = 64)
        # keeps the chunked unfused route.
        x, mask = _inputs((1,), 64, False, seed=2)
        assert not _run(module, x, mask, fused=True, chunk_size=63)[1]
        # Strict FP32 keeps the unfused cuBLAS PEDANTIC route.
        torch.backends.cuda.matmul.allow_tf32 = False
        assert not _run(module, x, mask, fused=True)[1]
    # Training never takes the inference-only route.
    torch.backends.cuda.matmul.allow_tf32 = True
    assert module._fused_fp32_inference(x, mask, False, None) is None
