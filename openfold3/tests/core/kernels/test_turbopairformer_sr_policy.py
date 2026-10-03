# SPDX-License-Identifier: Apache-2.0

"""Source-development SR policy and Hopper numerical regression checks.

The historical wheel verifier remains frozen. Its nine input specifications,
FP64 oracle, BF16 stock reference, and error envelope are reused here, with
every case routed through the current production SR facade. GPU tests require
an explicit OPENFOLD3_TRIANGLE_ALLOW_JIT=1 source-development invocation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
FROZEN_PATH = ROOT / "tools/aot_numeric_smoke.py"
FROZEN_SHA256 = "c33ad3e3dbe3f4b881db132240ba673d52976b9719f1a4de185271e439f56067"
_SPEC = importlib.util.spec_from_file_location("_frozen_aot_numeric_smoke", FROZEN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(SMOKE)
SHAPES = [dict(shape) for _, _, shape in SMOKE.CASES]
CASE_IDS = [f"sr-{i + 1}-N{s['N']}-D{s['D']}" for i, s in enumerate(SHAPES)]


def _module(name):
    module = importlib.import_module("openfold3.core.kernels.cuda." + name)
    assert Path(module.__file__).resolve() == ROOT / (
        "openfold3/core/kernels/cuda/" + name + ".py"
    ), "This check must import the current source checkout"
    return module


def _set_flag(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def _forbidden(*args, **kwargs):
    raise AssertionError("The public facade executed a separate reconstruction path")


def _block_separate_backward(monkeypatch):
    dkdv = _module("turbopairformer_triangle_dkdv")
    dq = _module("turbopairformer_triangle_dq_dbias")
    monkeypatch.setattr(dkdv, "turbopairformer_triangle_dkdv", _forbidden)
    monkeypatch.setattr(dkdv, "load_extension", _forbidden)
    monkeypatch.setattr(dq, "turbopairformer_triangle_dq_dbias", _forbidden)
    return dq


def test_historical_numeric_verifier_is_unchanged():
    assert hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() == FROZEN_SHA256
    assert len(SHAPES) == 9


@pytest.mark.parametrize("sr,shadow", [(None, None), ("1", "1"), (None, "0")])
def test_cpu_facade_routes_one_sr_backward(monkeypatch, sr, shadow):
    """Mock native arithmetic only; exercise the real facade and autograd route."""
    facade = _module("turbopairformer_triangle_attention")
    dq = _block_separate_backward(monkeypatch)
    _set_flag(monkeypatch, "OPENFOLD3_SR_BACKWARD", sr)
    _set_flag(monkeypatch, "OPENFOLD3_SR_SHADOW", shadow)
    calls = {"sr": 0, "shadow": 0, "plain": 0}

    def forward(q, k, v, mask, bias, out, lse, residual=None):
        calls["plain" if residual is None else "shadow"] += 1
        out.zero_()
        lse.zero_()
        if residual is not None:
            residual.zero_()

    def sr_backward(q, k, v, dout, lse, delta, mask, bias):
        calls["sr"] += 1
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
            torch.zeros_like(bias, dtype=torch.float32),
        )

    monkeypatch.setattr(
        facade, "is_turbopairformer_supported", lambda *a: (True, "test metadata")
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        facade,
        "_forward_extension",
        lambda: SimpleNamespace(
            turbopairformer_triangle_forward_out=forward,
            turbopairformer_triangle_forward_shadow_out=forward,
        ),
    )
    monkeypatch.setattr(
        facade,
        "_delta_extension",
        lambda: SimpleNamespace(
            turbopairformer_triangle_delta_out=lambda *a: a[-1].zero_(),
            turbopairformer_triangle_delta_shadow_out=lambda *a: a[-1].zero_(),
        ),
    )
    monkeypatch.setattr(dq, "turbopairformer_triangle_sr_backward", sr_backward)
    q, k, v = (
        torch.zeros(1, 2, 2, 64, 16, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    bias = torch.zeros(1, 1, 2, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    out = facade.turbopairformer_triangle_attention(
        q, k, v, torch.zeros(1, 2, 1, 1, 64), bias
    )
    out.backward(torch.ones_like(out))
    assert calls == {"sr": 1, "shadow": int(shadow != "0"), "plain": int(shadow == "0")}
    assert all(t.grad is not None for t in (q, k, v, bias))


def test_cpu_explicit_sr_zero_fails_before_input_validation(monkeypatch):
    facade = _module("turbopairformer_triangle_attention")
    monkeypatch.setenv("OPENFOLD3_SR_BACKWARD", "0")
    monkeypatch.setattr(facade, "is_turbopairformer_supported", _forbidden)
    with pytest.raises((ValueError, RuntimeError), match="OPENFOLD3_SR_BACKWARD"):
        facade.turbopairformer_triangle_attention(None, None, None, None, None)


@pytest.fixture
def hopper_source_routes(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper compute capability 9.0 is required")
    if os.environ.get("OPENFOLD3_TRIANGLE_ALLOW_JIT") != "1":
        pytest.skip("Source GPU checks require OPENFOLD3_TRIANGLE_ALLOW_JIT=1")
    assert hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() == FROZEN_SHA256
    dq = _block_separate_backward(monkeypatch)
    extension = dq.load_extension()
    calls = {"python_sr": 0, "native_sr": 0}
    python_sr = dq.turbopairformer_triangle_sr_backward
    native_sr = extension.turbopairformer_triangle_sr_backward_out

    def counted_python(*args, **kwargs):
        calls["python_sr"] += 1
        return python_sr(*args, **kwargs)

    def counted_native(*args, **kwargs):
        calls["native_sr"] += 1
        return native_sr(*args, **kwargs)

    monkeypatch.setattr(dq, "turbopairformer_triangle_sr_backward", counted_python)
    monkeypatch.setattr(
        extension, "turbopairformer_triangle_sr_backward_out", counted_native
    )
    monkeypatch.setattr(extension, "turbopairformer_triangle_dq_dbias_out", _forbidden)
    yield calls
    runtime = _module("_aot_runtime")
    for key, stem in (
        ("turbopairformer_forward", "turbopairformer_triangle_forward"),
        ("turbopairformer_delta", "turbopairformer_triangle_delta"),
        ("turbopairformer_dq_dbias", "turbopairformer_triangle_dq_dbias"),
    ):
        source = ROOT / f"openfold3/core/kernels/cuda/{stem}.cu"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        # _load_development_jit generates this source-hashed name. Torch may
        # append a cache-version suffix; a wheel module cannot match it.
        assert runtime._MODULE_CACHE[key].__name__.startswith(
            f"openfold3_triangle_dev_{key}_{digest}"
        ), f"{key} was not built from the current source"


def _run_facade(inputs, calls):
    q, k, v, mask, bias, dout = inputs
    # Preserve producer views: clone() can materialize some strided layouts.
    q, k, v, bias = (t.detach().requires_grad_(True) for t in (q, k, v, bias))
    before = dict(calls)
    out = _module(
        "turbopairformer_triangle_attention"
    ).turbopairformer_triangle_attention(q, k, v, mask, bias)
    assert out.shape == dout.shape and out.is_contiguous()
    out.backward(dout)
    # These bounded test shapes fit one native SR call. Larger workspace plans
    # may legitimately split that call into multiple query/batch stripes.
    assert calls == {name: value + 1 for name, value in before.items()}
    return out.detach(), q.grad, k.grad, v.grad, bias.grad


def _check_numerics(inputs, full, monkeypatch, calls):
    reference, stock = SMOKE._oracle(*inputs), SMOKE._stock(*inputs)
    _set_flag(monkeypatch, "OPENFOLD3_SR_BACKWARD", None)
    _set_flag(monkeypatch, "OPENFOLD3_SR_SHADOW", None)
    default = _run_facade(inputs, calls)
    monkeypatch.setenv("OPENFOLD3_SR_BACKWARD", "1")
    monkeypatch.setenv("OPENFOLD3_SR_SHADOW", "1")
    explicit = _run_facade(inputs, calls)
    replay = _run_facade(inputs, calls)
    assert all(
        SMOKE._bits_equal(a, b) and SMOKE._bits_equal(b, c)
        for a, b, c in zip(default, explicit, replay, strict=True)
    )
    assert all(torch.isfinite(t).all().item() for t in default)
    if full:
        assert all(t[:, :full].abs().max().item() == 0 for t in default[:4])
        reference = (reference[0][:, full:], *reference[1:])
        stock = (stock[0][:, full:], *stock[1:])
        default = (default[0][:, full:], *default[1:])
    for label, value, baseline, oracle in zip(
        ("out", "dq", "dk", "dv", "dbias"), default, stock, reference, strict=True
    ):
        error, baseline_error = SMOKE._rel(value, oracle), SMOKE._rel(baseline, oracle)
        # Exact envelope from the hash-locked historical verifier.
        limit = 1.1 * baseline_error + 1e-6
        assert error <= limit, f"{label}: relative L2 {error} exceeds {limit}"


@pytest.mark.parametrize("shape", SHAPES, ids=CASE_IDS)
def test_hopper_all_nine_frozen_shapes_use_sr(
    shape, monkeypatch, hopper_source_routes, record_property
):
    torch.manual_seed(0)
    inputs = SMOKE._case(bias_dtype=torch.bfloat16, **shape)
    _check_numerics(inputs, shape["full_planes"], monkeypatch, hopper_source_routes)
    record_property("oracle_sha256", FROZEN_SHA256)
    record_property("sr_backward_calls", hopper_source_routes["python_sr"])


@pytest.mark.parametrize(
    "sequence,dim",
    [(64, 32), (67, 20)],
    ids=["producer-views-broadcast", "padded-producer-views-broadcast"],
)
def test_hopper_strided_inputs_and_broadcast_bias(
    sequence, dim, monkeypatch, hopper_source_routes
):
    torch.manual_seed(0)
    q, k, v, mask, bias, dout = SMOKE._case(
        B=2,
        O=3,
        H=2,
        N=sequence,
        D=dim,
        mask_frac=0.2,
        full_planes=1,
        bias_dtype=torch.bfloat16,
    )
    q, k, v = (t.transpose(-2, -3).contiguous().transpose(-2, -3) for t in (q, k, v))
    bias = bias.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert all(not t.is_contiguous() for t in (q, k, v, bias))
    # Both the additive mask and pair bias broadcast across the batch prefix.
    inputs = q, k, v, mask[:1], bias, dout
    _check_numerics(inputs, 1, monkeypatch, hopper_source_routes)
