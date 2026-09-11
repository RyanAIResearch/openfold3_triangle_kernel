# SPDX-License-Identifier: Apache-2.0

"""Packaging contract of the standalone Trionce sm_90a wheel."""

from __future__ import annotations

import ast
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "trionce"
LOADERS = (
    "triangle_forward.py",
    "triangle_delta.py",
    "triangle_dkdv.py",
    "triangle_dq_dbias.py",
    "trimul_cuda.py",
)
EXPECTED_MODULES = {
    "_trionce_forward_sm90a",
    "_trionce_delta_sm90a",
    "_trionce_dkdv_sm90a",
    "_trionce_dq_dbias_sm90a",
    "_trionce_trimul_sm90a",
}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_package_has_no_host_dependency():
    for path in PACKAGE.glob("*.py"):
        text = path.read_text()
        assert "import openfold3" not in text, path.name
        assert "from openfold3" not in text, path.name


def test_production_loaders_do_not_contain_jit_compilation():
    for loader in LOADERS:
        text = (PACKAGE / loader).read_text()
        assert "torch.utils.cpp_extension" not in text, loader
        assert "load_extension_module" in text, loader
        assert "trionce._ext." in text, loader


def test_jit_import_is_isolated_to_explicit_development_function():
    path = PACKAGE / "_aot_runtime.py"
    tree = ast.parse(path.read_text())
    imports = [
        (node.module, function.name)
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "torch.utils.cpp_extension"
    ]
    assert imports == [("torch.utils.cpp_extension", "_load_development_jit")]
    text = path.read_text()
    assert '_DEVELOPMENT_JIT_ENV = "TRIONCE_ALLOW_JIT"' in text
    assert '_LEGACY_DEVELOPMENT_JIT_ENV = "OPENFOLD3_TRIANGLE_ALLOW_JIT"' in text
    assert "if not _development_jit_enabled():" in text


def test_facades_only_import_package_modules_lazily():
    facade = PACKAGE / "_triangle_attention.py"
    imports = _top_level_imports(facade)
    assert "trionce.policy" in imports
    assert not any(name.startswith("trionce.trionce_") for name in imports)


def test_setup_defines_five_unique_sm90a_extensions():
    text = (REPOSITORY / "setup.py").read_text()
    discovered = {name for name in EXPECTED_MODULES if text.count(name) == 1}
    assert discovered == EXPECTED_MODULES
    assert text.count("CUDAExtension(") == 1
    assert "EXPECTED_PYTHON = (3, 14)" in text
    assert 'EXPECTED_TORCH = "2.10.0"' in text
    assert 'EXPECTED_TORCH_CUDA = "12.9"' in text
    assert 'EXPECTED_ARCH_LIST = "9.0a"' in text
    assert '"--frandom-seed=0"' in text
    assert "sources=[f\"trionce/{value['source']}\"]," in text
    assert 'build_lib / "trionce/_build_manifest.json"' in text


def test_loader_contract_matches_setup_symbols():
    runtime = (PACKAGE / "_aot_runtime.py").read_text()
    setup = (REPOSITORY / "setup.py").read_text()
    for module in EXPECTED_MODULES:
        assert f"trionce._ext.{module}" in runtime
        assert f"trionce._ext.{module}" in setup


def test_pyproject_ships_sources_and_manifest_as_package_data():
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    assert project["project"]["name"] == "trionce"
    assert any(dep.startswith("torch") for dep in project["project"]["dependencies"])
    package_data = project["tool"]["setuptools"]["package-data"]["trionce"]
    for pattern in ("*.cu", "*.json", "*.md", "_ext/*.so"):
        assert pattern in package_data
    manifest = (REPOSITORY / "MANIFEST.in").read_text()
    assert "recursive-include trionce *.cu *.py *.json *.md" in manifest


def test_notice_is_self_contained():
    notice = (REPOSITORY / "NOTICE").read_text()
    assert "UPSTREAM.md" not in notice
    assert "c615a7f8d0e7bb04095a35ec6dc7b30192ac19d3" in notice
    assert "6db02ff4b31e87bfced98331b22613e96e8ced37" in notice


def test_checked_in_wheel_contains_the_corrected_claims_and_notice():
    wheel = (
        REPOSITORY
        / "release_artifacts"
        / "trionce-0.1.0-cp314-cp314-linux_x86_64.whl"
    )
    if not wheel.is_file():
        pytest.skip(
            "no trionce release wheel has been cut yet; release_artifacts still "
            "holds the StableFold wheel this package was renamed from"
        )
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read("trionce-0.1.0.dist-info/METADATA").decode()
        notice = archive.read("trionce-0.1.0.dist-info/licenses/NOTICE").decode()

    assert "6.94 s" not in metadata
    assert "at or below the stock" not in metadata
    assert "relative error plus `1e-6`" in metadata
    assert "does not cover TriMul or" in metadata
    assert "UPSTREAM.md" not in notice
    assert "c615a7f8d0e7bb04095a35ec6dc7b30192ac19d3" in notice
    assert "6db02ff4b31e87bfced98331b22613e96e8ced37" in notice


def test_source_tree_manifest_is_fail_closed():
    """A fresh checkout must not claim to carry an AOT build.

    ``pip install -e`` with ``TRIONCE_BUILD_AOT=1`` builds the extensions in
    place and writes the matching receipt next to them, so the working copy of
    this file legitimately says ``built: true`` on a developer machine.  The
    invariant worth protecting is that the *committed* manifest stays the
    placeholder, which is what a clone without a build gets.
    """

    relative = (PACKAGE / "_build_manifest.json").relative_to(
        REPOSITORY.parent
    )
    try:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=REPOSITORY.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, or the manifest is not committed yet")

    manifest = json.loads(committed)
    assert manifest["schema_version"] == 1
    assert manifest["built"] is False


def test_facade_metadata_contract_is_release_bounded():
    text = (PACKAGE / "_triangle_attention.py").read_text()
    assert "_PAD_HEAD_LANES = (16, 32, 64)" in text
    assert "_MAX_BROADCAST_PREFIX_RANK = 16" in text
    assert "if bias_requires_broadcast and len(prefix) >" in text
    assert "D<=64" in text
    assert "trionce_trimul" not in text


def test_build_entrypoint_is_abi_locked_and_smoke_tested():
    text = (REPOSITORY / "tools/build_sm90_wheel.sh").read_text()
    for contract in (
        "TORCH_CUDA_ARCH_LIST=9.0a",
        "sys.version_info[:2] == (3, 14)",
        'torch.__version__.split("+", 1)[0] == "2.10.0"',
        'torch.version.cuda == "12.9"',
        "TRIONCE_BUILD_AOT=1",
        "git status --porcelain --untracked-files=all",
        "CUDA_HOME=/nonexistent",
        "distributed_aot_identity",
        "-u TRIONCE_ALLOW_JIT -u OPENFOLD3_TRIANGLE_ALLOW_JIT",
    ):
        assert contract in text, contract


def test_manifest_receipt_covers_build_and_runtime_files():
    text = (REPOSITORY / "setup.py").read_text()
    for relative in (
        "MANIFEST.in",
        "README.md",
        "PRODUCTION_MANIFEST.md",
        "_ext/__init__.py",
        "policy.py",
        "distributed.py",
        "tools/build_sm90_wheel.sh",
    ):
        assert relative in text, relative


def test_binary_hash_is_checked_before_extension_import():
    runtime = (PACKAGE / "_aot_runtime.py").read_text()
    assert "_preflight_aot_artifact" in runtime
    assert "binary hash mismatch" in runtime
    assert runtime.index("artifact_path = _preflight_aot_artifact") < runtime.index(
        "module = importlib.import_module(aot_module)"
    )
    assert 'relative.startswith("trionce/")' in runtime
    assert "Path(__file__).resolve().parents[1]" in runtime


def test_trimul_custom_ops_use_the_package_namespace():
    text = (PACKAGE / "trimul.py").read_text()
    assert '"trionce::_trimul_fwd_v1"' in text
    assert '"trionce::_trimul_bwd_v1"' in text
    assert "openfold3_trionce" not in text
