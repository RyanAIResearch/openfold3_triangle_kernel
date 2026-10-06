"""CPU checks of Hopper host admission; these do not execute GPU kernels."""

import sys
from types import SimpleNamespace

import pytest

from tools import production_training


@pytest.fixture
def runtime(monkeypatch):
    devices = [
        SimpleNamespace(
            name="NVIDIA H200",
            capability=(9, 0),
            total_memory=141 * 1024**3,
            multi_processor_count=132,
        )
    ]
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: len(devices),
        get_device_properties=lambda index: devices[index],
        get_device_capability=lambda index: devices[index].capability,
    )
    torch = SimpleNamespace(
        __version__="2.10.0+cu129", version=SimpleNamespace(cuda="12.9"), cuda=cuda
    )
    lightning = SimpleNamespace(__version__="2.6.1")
    python = SimpleNamespace(version_info=(3, 14, 0), version="3.14.0")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)
    monkeypatch.setattr(production_training, "sys", python)
    monkeypatch.setattr(production_training, "source_commit", lambda: "test-commit")
    return SimpleNamespace(
        devices=devices, torch=torch, lightning=lightning, python=python
    )


@pytest.mark.parametrize(
    ("name", "memory_gib"),
    (("NVIDIA H100 80GB HBM3", 80), ("NVIDIA H200", 141)),
)
def test_doctor_admits_h100_and_h200_and_reports_device(runtime, name, memory_gib):
    runtime.devices[0].name = name
    runtime.devices[0].total_memory = memory_gib * 1024**3

    result = production_training.doctor_hopper(SimpleNamespace())

    assert result["status"] == "PASS"
    assert result["devices"] == [
        {
            "index": 0,
            "name": name,
            "capability": [9, 0],
            "total_memory": memory_gib * 1024**3,
            "multiprocessor_count": 132,
        }
    ]


@pytest.mark.parametrize(
    ("name", "capability"),
    (
        ("NVIDIA H200", (8, 0)),
        ("NVIDIA H200", (10, 0)),
        ("NVIDIA H100", (8, 0)),
        ("NVIDIA H800", (9, 0)),
        ("NVIDIA B200", (9, 0)),
    ),
)
def test_doctor_rejects_other_devices_even_on_later_indices(runtime, name, capability):
    runtime.devices.append(
        SimpleNamespace(
            name=name,
            capability=capability,
            total_memory=141 * 1024**3,
            multi_processor_count=132,
        )
    )

    with pytest.raises(production_training.WorkflowError, match="device 1"):
        production_training.doctor_hopper(SimpleNamespace())


def test_doctor_still_rejects_partial_memory_devices(runtime):
    runtime.devices[0].total_memory = 74 * 1024**3

    with pytest.raises(production_training.WorkflowError, match="75 GiB"):
        production_training.doctor_hopper(SimpleNamespace())


@pytest.mark.parametrize("available", (False, True))
def test_doctor_requires_visible_cuda_devices(runtime, available):
    runtime.torch.cuda.is_available = lambda: available
    runtime.devices.clear()

    with pytest.raises(production_training.WorkflowError, match="CUDA"):
        production_training.doctor_hopper(SimpleNamespace())


@pytest.mark.parametrize("component", ("python", "torch", "cuda", "lightning"))
def test_doctor_keeps_pinned_runtime_requirements(runtime, component):
    if component == "python":
        runtime.python.version_info = (3, 13, 0)
    elif component == "torch":
        runtime.torch.__version__ = "2.9.0"
    elif component == "cuda":
        runtime.torch.version.cuda = "12.8"
    else:
        runtime.lightning.__version__ = "2.6.0"

    with pytest.raises(production_training.WorkflowError, match="expected"):
        production_training.doctor_hopper(SimpleNamespace())


def test_legacy_doctor_h100_wrapper_accepts_h200(runtime):
    assert production_training.doctor_h100(SimpleNamespace()) == (
        production_training.doctor_hopper(SimpleNamespace())
    )


@pytest.mark.parametrize("command", ("doctor-hopper", "doctor-h100"))
def test_both_doctor_commands_dispatch_to_hopper(monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", ["production_training", command])
    monkeypatch.setattr(
        production_training, "doctor_hopper", lambda args: {"command": args.command}
    )

    production_training.main()

    assert command in capsys.readouterr().out
