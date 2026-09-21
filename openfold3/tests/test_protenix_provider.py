# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the optional provider's layout and runtime boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import openfold3.core.kernels.protenix as provider
from openfold3.core.kernels.protenix._cache import UPSTREAM_REVISION, get_config_dir
from openfold3.core.kernels.protenix._layout import _normalize_inputs


def _inputs(shape=(3, 2, 5, 16)):
    q, k, v = (torch.randn(shape, dtype=torch.bfloat16) for _ in range(3))
    mask = torch.zeros((3, 1, 1, 5), dtype=torch.bfloat16)
    bias = torch.randn((1, 2, 5, 5), dtype=torch.bfloat16)
    return q, k, v, mask, bias


def test_provider_import_does_not_import_torch_triton_or_global_protenix():
    # Isolate this provider from the unrelated optional DeepSpeed import in
    # openfold3.__init__, which may itself load Torch and Triton.
    kernel_root = Path(provider.__file__).parents[1]
    script = f"""
import sys
import types
from pathlib import Path
kernel_root = Path({str(kernel_root)!r})
for name, path in [
    ("openfold3", kernel_root.parents[1]),
    ("openfold3.core", kernel_root.parent),
    ("openfold3.core.kernels", kernel_root),
]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
import openfold3.core.kernels.protenix
assert "torch" not in sys.modules
assert "triton" not in sys.modules
assert not any(n == "protenix" or n.startswith("protenix.") for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("separate_bias", [False, True])
def test_4d_layout_keeps_chunk_row_bias(separate_bias):
    q, k, v, mask, bias = _inputs()
    if separate_bias:
        bias = torch.arange(3 * 2 * 5 * 5).reshape(3, 2, 5, 5)
    nq, _, _, nm, nb, original = _normalize_inputs(q, k, v, mask, bias)
    assert original == tuple(q.shape)
    assert tuple(nq.shape) == ((3, 1, 2, 5, 16) if separate_bias else (1, 3, 2, 5, 16))
    assert torch.equal(nq.reshape(q.shape), q)
    assert nm.shape == ((3, 1, 1, 1, 5) if separate_bias else (1, 3, 1, 1, 5))
    assert torch.equal(nb[:, 0], bias) if separate_bias else torch.equal(nb[0], bias)


def test_5d_batch_broadcast_reduces_pair_bias_gradients():
    q = torch.randn(2, 3, 2, 5, 16)
    mask = torch.zeros(1, 3, 1, 1, 5)
    bias = torch.randn(1, 1, 2, 5, 5, requires_grad=True)
    _, _, _, nm, nb, _ = _normalize_inputs(q, q, q, mask, bias)
    assert nm.shape == (2, 3, 1, 1, 5)
    nb.sum().backward()
    assert torch.equal(bias.grad, torch.full_like(bias, 2))


def test_6d_template_broadcast_preserves_distinct_batch_bias():
    q = torch.randn(2, 3, 4, 2, 5, 16)
    mask = torch.zeros(2, 1, 4, 1, 1, 5)
    bias = torch.arange(2 * 2 * 5 * 5, dtype=torch.float32).reshape(2, 1, 1, 2, 5, 5)
    bias.requires_grad_()
    nq, _, _, nm, nb, _ = _normalize_inputs(q, q, q, mask, bias)
    assert nq.shape == (6, 4, 2, 5, 16)
    assert nm.shape == (6, 4, 1, 1, 5)
    assert torch.equal(nb[:3], bias[0].expand(3, 1, 2, 5, 5))
    assert torch.equal(nb[3:], bias[1].expand(3, 1, 2, 5, 5))
    nb.sum().backward()
    assert torch.equal(bias.grad, torch.full_like(bias, 3))


@pytest.mark.parametrize(
    ("field", "transform", "message"),
    [
        (0, lambda t: t[..., :-1, :], "identical shapes"),
        (0, lambda t: t.float(), "bfloat16"),
        (4, lambda t: t.float(), "bfloat16"),
        (3, lambda t: t.bool(), "additive mask must"),
        (3, lambda t: t.requires_grad_(), "trainable additive mask"),
        (2, lambda t: t.to("meta"), "same CUDA device"),
    ],
)
def test_validation_rejects_invalid_inputs_before_loading_kernel(
    field, transform, message
):
    inputs = list(_inputs())
    inputs[field] = transform(inputs[field])
    with patch.object(provider, "_native_function") as native:
        with pytest.raises((ValueError, TypeError), match=message):
            provider.attention_scaled(*inputs)
        native.assert_not_called()


def test_cpu_execution_is_explicitly_rejected():
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        provider.attention_scaled(*_inputs())


@pytest.mark.parametrize("dimension", [8, 24])
def test_invalid_head_dimension_is_rejected(dimension):
    with pytest.raises(ValueError, match="power of two"):
        provider.attention_scaled(*_inputs((3, 2, 5, dimension)))


def test_adapter_preserves_scaled_q_and_makes_dout_contiguous():
    observed = {}

    class Echo(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, mask, bias, deterministic):
            observed["q"] = q.detach().clone()
            observed["deterministic"] = deterministic
            return q.contiguous().clone()

        @staticmethod
        def backward(ctx, dout):
            observed["contiguous"] = dout.is_contiguous()
            return dout, None, None, None, None, None

    inputs = list(_inputs())
    inputs[0].requires_grad_()
    with (
        patch.object(provider, "_validate_inputs"),
        patch.object(provider, "_native_function", return_value=Echo),
    ):
        output = provider.attention_scaled(*inputs)
    assert output.shape == (3, 5, 2, 16)
    assert torch.equal(observed["q"], inputs[0].unsqueeze(0).transpose(-2, -3))
    assert observed["deterministic"] is False
    dout = torch.randn(3, 5, 2, 32, dtype=torch.bfloat16)[..., ::2]
    assert not dout.is_contiguous()
    output.backward(dout)
    assert observed["contiguous"]
    assert torch.equal(inputs[0].grad, dout.transpose(-2, -3))


def test_runtime_failure_propagates():
    inputs = _inputs()
    native = SimpleNamespace(
        apply=lambda *_: (_ for _ in ()).throw(RuntimeError("kernel failed"))
    )
    with (
        patch.object(provider, "_validate_inputs"),
        patch.object(provider, "_native_function", return_value=native),
        pytest.raises(RuntimeError, match="kernel failed"),
    ):
        provider.attention_scaled(*inputs)


def test_missing_triton_has_actionable_error():
    provider._native_function.cache_clear()
    with (
        patch.object(
            provider, "import_module", side_effect=ModuleNotFoundError(name="triton")
        ),
        pytest.raises(ImportError, match="GPU dependencies"),
    ):
        provider._native_function()
    provider._native_function.cache_clear()


def test_cache_respects_override_xdg_device_and_rank(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA H100")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.delenv("OPENFOLD3_PROTENIX_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    directory = get_config_dir(torch.device("cuda:2"))
    assert directory.is_relative_to(tmp_path / "xdg" / "openfold3" / "protenix")
    assert UPSTREAM_REVISION in directory.parts
    assert "NVIDIA-H100-sm90-device2" in directory.parts
    assert directory.name == "rank-7"
    assert not directory.exists()
    monkeypatch.setenv("OPENFOLD3_PROTENIX_CACHE_DIR", str(tmp_path / "override"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    directory = get_config_dir(torch.device("cuda:0"))
    assert directory.is_relative_to(tmp_path / "override")
    assert directory.name == "rank-3"


def test_vendored_file_integrity_matches_provenance():
    root = Path(provider.__file__).parent
    manifest = json.loads((root / "PROVENANCE.json").read_text())
    assert manifest["commit"] == UPSTREAM_REVISION
    assert manifest["algorithm_changes"] is False
    for row in manifest["files"]:
        assert (
            hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
            == row["vendored_sha256"]
        )


def _cache_only_autotuner(tmp_path):
    """Exercise our disk-cache changes without Triton imports or compilation."""

    class BaseAutotuner:
        def __init__(self, fn, *args, **kwargs):
            self.base_fn = fn
            self.cache = {}

    root = Path(provider.__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_protenix_cache_unit_test", root / "_vendor" / "autotune.py"
    )
    module = importlib.util.module_from_spec(spec)
    fake_modules = {
        "triton": SimpleNamespace(runtime=SimpleNamespace(Autotuner=BaseAutotuner)),
        "openfold3.core.kernels.protenix._vendor.autotune_helpers": SimpleNamespace(
            dict_to_config=lambda value: value,
            config_to_dict=lambda value: value,
        ),
    }
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    module.get_config_dir = lambda device: tmp_path / f"device-{device.index}"

    def kernel():
        pass

    return module.Autotuner(kernel, [], [], [], None, None)


def _cache_args(device_index):
    return (SimpleNamespace(device=SimpleNamespace(type="cuda", index=device_index)),)


def test_autotuner_keeps_device_caches_separate_and_persists(tmp_path):
    tuner = _cache_only_autotuner(tmp_path)
    tuner._select_cache(_cache_args(0), {})
    tuner.cache["shape"] = {"choice": 0}
    tuner._write_cache_to_disk()
    tuner._select_cache(_cache_args(1), {})
    assert tuner.cache == {}
    tuner.cache["shape"] = {"choice": 1}
    tuner._select_cache(_cache_args(0), {})
    assert tuner.cache == {"shape": {"choice": 0}}
    reloaded = _cache_only_autotuner(tmp_path)
    reloaded._select_cache(_cache_args(0), {})
    assert reloaded.cache == tuner.cache


@pytest.mark.parametrize("payload", ["[]", "null", "4", "{broken"])
def test_autotuner_ignores_corrupted_cache(tmp_path, payload):
    cache_dir = tmp_path / "device-0"
    cache_dir.mkdir()
    (cache_dir / "kernel.json").write_text(payload)
    tuner = _cache_only_autotuner(tmp_path)
    tuner._select_cache(_cache_args(0), {})
    assert tuner.cache == {}


def test_autotuner_can_retry_after_unwritable_cache(tmp_path):
    tuner = _cache_only_autotuner(tmp_path)
    with (
        patch.object(Path, "mkdir", side_effect=PermissionError("read only")),
        pytest.raises(RuntimeError, match="OPENFOLD3_PROTENIX_CACHE_DIR"),
    ):
        tuner._select_cache(_cache_args(0), {})
    assert tuner.cache_file is None
    tuner._select_cache(_cache_args(0), {})
    assert tuner.cache_file.parent.is_dir()
