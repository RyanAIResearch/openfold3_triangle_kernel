# SPDX-License-Identifier: Apache-2.0

"""The installed API guide must remain usable without Torch or CUDA."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PACKAGE_ROOT / "turbopairformer"
TOPICS = ("attention", "trimul", "support", "distributed", "environment", "internals")


def _public_names() -> tuple[str, ...]:
    """Read exports without making test collection depend on GPU libraries."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("The package must declare its public API in __all__")


def _python(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # -S excludes installed packages; replacing PYTHONPATH also excludes host
    # checkouts and proves the standalone source package carries its own guide.
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-S", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _cli(tmp_path: Path, *arguments: str) -> str:
    result = _python(tmp_path, "-m", "turbopairformer", *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.strip()
    return result.stdout


def test_no_arguments_prints_help_without_site_packages(tmp_path):
    help_text = _cli(tmp_path, "--help")
    assert _cli(tmp_path) == help_text
    for command in ("turbopairformer", "api", "--help", "--version"):
        assert command in help_text


def test_version_matches_the_package_without_site_packages(tmp_path):
    expected = _python(
        tmp_path, "-c", "import turbopairformer; print(turbopairformer.__version__)"
    )
    assert expected.returncode == 0, expected.stdout + expected.stderr
    version_text = _cli(tmp_path, "--version")
    assert expected.stdout.strip() in version_text
    assert "usage:" not in version_text.lower()


@pytest.mark.parametrize("arguments", [("api",), ("api", "--all")])
def test_api_catalog_covers_every_top_level_public_name(tmp_path, arguments):
    text = _cli(tmp_path, *arguments)
    for name in _public_names():
        assert name in text, f"Undocumented public export: {name}"
    for topic in TOPICS:
        assert topic in text.lower()


@pytest.mark.parametrize("topic", TOPICS)
def test_each_guide_topic_is_available_without_site_packages(tmp_path, topic):
    text = _cli(tmp_path, "api", topic)
    assert topic in text.lower()
    assert len(text.splitlines()) > 3


@pytest.mark.parametrize("name", _public_names())
def test_public_api_names_resolve_with_or_without_package_prefix(tmp_path, name):
    short = _cli(tmp_path, "api", name)
    qualified = _cli(tmp_path, "api", f"turbopairformer.{name}")
    assert short == qualified
    assert name in short


@pytest.mark.parametrize(
    "name",
    (
        "_trimul.turbo_trimul",
        "trimul_support.check_trimul_tensor_support",
        "triangle_dq_dbias.workspace_plan",
        "distributed.TurboPairFormerKernelDDPCallback",
    ),
)
def test_internal_function_and_class_references_resolve_without_imports(tmp_path, name):
    text = _cli(tmp_path, "api", name)
    assert f"turbopairformer.{name}" in text
    assert "```python" in text
    assert text == _cli(tmp_path, "api", f"turbopairformer.{name}")


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("policy.SR_BACKWARD", "TURBOPAIRFORMER_SR_BACKWARD"),
        ("trimul_support.SUPPORTED_CHANNELS", "frozenset"),
        ("distributed.DistributedContractError", "class DistributedContractError"),
    ),
)
def test_module_constants_and_reexports_are_available_without_imports(
    tmp_path, name, expected
):
    text = _cli(tmp_path, "api", name)
    assert name in text
    assert expected in text
    assert text == _cli(tmp_path, "api", f"turbopairformer.{name}")
    assert name in _cli(tmp_path, "api", "--all")


@pytest.mark.parametrize(
    "name",
    (
        "trimul_support.TensorMetadata",
        "trimul_support.SupportResult",
        "distributed.TurboPairFormerKernelDDPCallback",
        "distributed.DistributedContractError",
    ),
)
def test_rendered_class_examples_are_valid_python(tmp_path, name):
    text = _cli(tmp_path, "api", name)
    blocks = [part.split("```", 1)[0] for part in text.split("```python\n")[1:]]
    assert blocks, f"No Python class reference for {name}"
    for block in blocks:
        tree = ast.parse(block)
        assert any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_class_reference_preserves_dataclass_and_property_semantics(tmp_path):
    text = _cli(tmp_path, "api", "trimul_support.TensorMetadata")
    block = text.split("```python\n", 1)[1].split("```", 1)[0]
    tree = ast.parse(block)
    record = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "dataclass"
        for decorator in record.decorator_list
    )
    device_key = next(
        node
        for node in record.body
        if isinstance(node, ast.FunctionDef) and node.name == "device_key"
    )
    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in device_key.decorator_list
    ), "device_key is an attribute; the guide must not advertise a callable method"


@pytest.mark.parametrize(
    "arguments", [("api", "attention", "--all"), ("api", "--all", "attention")]
)
def test_full_guide_and_individual_topic_are_mutually_exclusive(tmp_path, arguments):
    result = _python(tmp_path, "-m", "turbopairformer", *arguments)
    assert result.returncode == 2
    assert "--all" in result.stderr
    assert "not allowed" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_unknown_topic_reports_actionable_usage_error(tmp_path):
    result = _python(tmp_path, "-m", "turbopairformer", "api", "not_a_real_api")
    assert result.returncode == 2
    assert "not_a_real_api" in result.stderr
    assert "attention" in result.stderr
    assert "trimul" in result.stderr
    assert "Traceback" not in result.stderr


def test_console_entry_point_runs_the_same_help(tmp_path):
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    entry_point = metadata["project"]["scripts"]["turbopairformer"]
    assert entry_point == "turbopairformer.cli:main"
    module, function = entry_point.split(":")
    script = f"""
import importlib
import sys
sys.argv = ["turbopairformer", "--help"]
raise SystemExit(getattr(importlib.import_module({module!r}), {function!r})())
"""
    result = _python(tmp_path, "-c", script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout == _cli(tmp_path, "--help")


@pytest.mark.parametrize(
    "arguments",
    [("--help",), ("--version",), ("api",), ("api", "--all")],
)
def test_documentation_does_not_attempt_optional_runtime_imports(tmp_path, arguments):
    # Record attempts too: checking sys.modules alone misses optional imports
    # whose failures were caught and suppressed by application code.
    script = f"""
import importlib.abc
import sys

blocked = ("torch", "lightning", "pytorch_lightning", "openfold3", "turbopairformer._ext")
attempted = []

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            attempted.append(fullname)
            raise RuntimeError("Documentation attempted to import " + fullname)

sys.meta_path.insert(0, RejectRuntimeImports())
from turbopairformer.cli import main
try:
    status = main({list(arguments)!r})
except SystemExit as exc:
    status = exc.code
assert status in (None, 0), status
assert not attempted, attempted
assert not any(name == root or name.startswith(root + ".") for name in sys.modules for root in blocked)
"""
    result = _python(tmp_path, "-c", script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip()
    assert result.stderr == ""
