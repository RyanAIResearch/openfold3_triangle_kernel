# SPDX-License-Identifier: Apache-2.0

"""The offline CLI presents only the two operator usages, without GPU imports."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PACKAGE_ROOT / "turbopairformer"
OPERATOR_HEADINGS = {
    "attention": r"^Triangle Attention\s*$",
    "trimul": r"^Triangle Multiplicative Update(?: \(TriMul\))?\s*$",
}


def _python(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Exclude installed dependencies and any host checkout from these processes.
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


def _assert_operator_usage(text: str, expected: tuple[str, ...]) -> None:
    for topic, heading in OPERATOR_HEADINGS.items():
        assert len(re.findall(heading, text, re.MULTILINE)) == int(topic in expected)
        assert (f"turbo_{topic}(" in text) == (topic in expected)
    assert len(text.splitlines()) <= 80, "CLI help should remain a short usage reference"
    assert "`" not in text and not re.search(r"^#{1,6} ", text, re.MULTILINE)
    assert not re.search(r"\b(?:reason|ok)\b", text, re.IGNORECASE)
    for unrelated in (
        "api support",
        "api distributed",
        "api environment",
        "api internals",
        "check_trimul_support",
        "is_triangle_attention_supported",
        "workspace_plan",
        "_NOT_GIVEN",
        "stock_attention",
    ):
        assert unrelated not in text


def test_no_arguments_and_help_show_both_operator_usages(tmp_path):
    help_text = _cli(tmp_path, "--help")
    assert _cli(tmp_path) == help_text
    assert "usage:" in help_text.lower()
    assert "--version" in help_text
    _assert_operator_usage(help_text, ("attention", "trimul"))


def test_api_and_all_show_the_same_two_operator_sections(tmp_path):
    text = _cli(tmp_path, "api")
    assert text == _cli(tmp_path, "api", "--all")
    _assert_operator_usage(text, ("attention", "trimul"))


@pytest.mark.parametrize(
    ("topic", "aliases"),
    (
        ("attention", ("turbo_attention", "TurboTriangleAttention", "triangle_attention")),
        ("trimul", ("turbo_trimul", "TurboTriangleMultiplicative", "run_trimul")),
    ),
)
def test_operator_aliases_show_only_the_selected_usage(tmp_path, topic, aliases):
    expected = _cli(tmp_path, "api", topic)
    _assert_operator_usage(expected, (topic,))
    for alias in aliases:
        assert _cli(tmp_path, "api", alias) == expected
        assert _cli(tmp_path, "api", f"turbopairformer.{alias}") == expected


def test_version_matches_the_package_without_site_packages(tmp_path):
    expected = _python(
        tmp_path, "-c", "import turbopairformer; print(turbopairformer.__version__)"
    )
    assert expected.returncode == 0, expected.stdout + expected.stderr
    assert _cli(tmp_path, "--version").strip() == f"turbopairformer {expected.stdout.strip()}"


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
    "topic",
    (
        "support",
        "distributed",
        "environment",
        "internals",
        "check_trimul_support",
        "_trimul._run_supported_trimul",
        "turbopairformer.triangle_dq_dbias.workspace_plan",
        "not_a_real_operator",
    ),
)
def test_removed_topics_and_internal_symbols_are_rejected(tmp_path, topic):
    result = _python(tmp_path, "-m", "turbopairformer", "api", topic)
    assert result.returncode == 2
    assert "attention" in result.stderr.lower() and "trimul" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_full_usage_and_individual_operator_are_mutually_exclusive(tmp_path):
    for arguments in (("api", "attention", "--all"), ("api", "--all", "attention")):
        result = _python(tmp_path, "-m", "turbopairformer", *arguments)
        assert result.returncode == 2
        assert "--all" in result.stderr
        assert "Traceback" not in result.stderr
        assert result.stdout == ""


def test_documented_python_examples_are_syntactically_valid():
    guide = (PACKAGE / "API_GUIDE.md").read_text()
    snippets = re.findall(r"^```python\s*\n(.*?)^```\s*$", guide, re.MULTILINE | re.DOTALL)
    assert snippets, "The operator usage guide must include Python examples"
    operators = set()
    for snippet in snippets:
        tree = ast.parse(snippet)
        operators.update(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("turbo_")
        )
    assert operators == {"turbo_attention", "turbo_trimul"}


@pytest.mark.parametrize("arguments", [("--help",), ("--version",), ("api", "--all")])
def test_usage_does_not_attempt_optional_runtime_imports(tmp_path, arguments):
    # Record attempts even when application code suppresses an import failure.
    script = f"""
import importlib.abc
import sys
blocked = ("torch", "lightning", "pytorch_lightning", "openfold3", "turbopairformer._ext")
attempted = []

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            attempted.append(fullname)
            raise RuntimeError("Usage help attempted to import " + fullname)

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
