# SPDX-License-Identifier: Apache-2.0

"""Packaging contract of the standalone TurboPairFormer sm_90a wheel."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "turbopairformer"
LOADERS = (
    "triangle_forward.py",
    "triangle_delta.py",
    "triangle_dkdv.py",
    "triangle_dq_dbias.py",
    "trimul_cuda.py",
)
EXPECTED_MODULES = {
    "_stablefold_forward_sm90a",
    "_stablefold_delta_sm90a",
    "_stablefold_dkdv_sm90a",
    "_stablefold_dq_dbias_sm90a",
    "_stablefold_trimul_sm90a",
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
        assert "turbopairformer._ext." in text, loader


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
    assert "distributed_execution_flag(ALLOW_JIT)" in text
    assert "if not _development_jit_enabled():" in text


def test_facades_only_import_package_modules_lazily():
    facade = PACKAGE / "_triangle_attention.py"
    imports = _top_level_imports(facade)
    assert "turbopairformer.policy" in imports
    assert not any(name.startswith("turbopairformer.stablefold_") for name in imports)


def test_setup_defines_five_unique_sm90a_extensions():
    text = (REPOSITORY / "setup.py").read_text()
    discovered = {name for name in EXPECTED_MODULES if text.count(name) == 1}
    assert discovered == EXPECTED_MODULES
    assert text.count("CUDAExtension(") == 1
    assert "EXPECTED_PYTHON = (3, 14)" in text
    assert 'EXPECTED_TORCH = "2.10.0"' in text
    assert 'EXPECTED_TORCH_CUDA = "12.8"' in text
    assert 'EXPECTED_ARCH_LIST = "9.0a"' in text
    assert '"--frandom-seed=0"' in text
    assert "sources=[f\"turbopairformer/{value['source']}\"]," in text
    assert 'build_lib / "turbopairformer/_build_manifest.json"' in text


def test_loader_contract_matches_setup_symbols():
    runtime = (PACKAGE / "_aot_runtime.py").read_text()
    setup = (REPOSITORY / "setup.py").read_text()
    for module in EXPECTED_MODULES:
        assert f"turbopairformer._ext.{module}" in runtime
        assert f"turbopairformer._ext.{module}" in setup


def test_pyproject_ships_sources_and_manifest_as_package_data():
    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    assert project["project"]["name"] == "turbopairformer"
    assert project["project"]["requires-python"] == "==3.14.*"
    assert project["project"]["dependencies"] == ["torch==2.10.0"]
    package_data = project["tool"]["setuptools"]["package-data"]["turbopairformer"]
    for pattern in ("*.cu", "_build_manifest.json", "*.md"):
        assert pattern in package_data
    manifest = (REPOSITORY / "MANIFEST.in").read_text()
    assert "recursive-include turbopairformer *.cu *.py *.json *.md" in manifest


def test_notice_is_self_contained():
    notice = (REPOSITORY / "NOTICE").read_text()
    assert "UPSTREAM.md" not in notice
    assert "c615a7f8d0e7bb04095a35ec6dc7b30192ac19d3" in notice
    assert "6db02ff4b31e87bfced98331b22613e96e8ced37" in notice


def test_candidate_wheel_has_a_completed_validated_receipt():
    candidate = os.environ.get("TURBOPAIRFORMER_TEST_WHEEL")
    if candidate is None:
        pytest.skip(
            "set TURBOPAIRFORMER_TEST_WHEEL to validate an actual candidate wheel"
        )
    module = _wheel_validator()
    result = module.validate_wheel(
        Path(candidate),
        allow_dirty=os.environ.get("TURBOPAIRFORMER_ALLOW_DIRTY_BUILD") == "1",
    )
    assert len(result["extensions"]) == 5


def test_source_tree_manifest_is_fail_closed():
    """The working placeholder is immutable, including after editable builds."""
    path = PACKAGE / "_build_manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["built"] is False
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    if root.returncode:
        return  # Standalone sdist: the actual source placeholder was checked above.
    relative = path.relative_to(Path(root.stdout.strip())).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root.stdout.strip(),
        capture_output=True,
        text=True,
    )
    if committed.returncode:
        staged = subprocess.run(
            ["git", "show", f":{relative}"],
            cwd=root.stdout.strip(),
            capture_output=True,
            text=True,
        )
        if staged.returncode == 0:
            assert json.loads(staged.stdout)["built"] is False
        return  # A new package can exist only in the index or working tree.
    assert json.loads(committed.stdout)["built"] is False


def _wheel_validator():
    spec = importlib.util.spec_from_file_location(
        "stablefold_wheel_validator", REPOSITORY / "tools/validate_wheel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_copy(tmp_path):
    destination = tmp_path / "package"
    shutil.copytree(
        REPOSITORY,
        destination,
        ignore=shutil.ignore_patterns(
            "build",
            "dist",
            "*.egg-info",
            "*.so",
            "*.local.json",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "release_artifacts",
        ),
    )
    return destination


def _setup(source, *args, source_only=False):
    env = os.environ.copy()
    for key in ("TURBOPAIRFORMER_BUILD_AOT", "TURBOPAIRFORMER_SOURCE_ONLY"):
        env.pop(key, None)
    if source_only:
        env["TURBOPAIRFORMER_SOURCE_ONLY"] = "1"
    return subprocess.run(
        [sys.executable, "setup.py", *args],
        cwd=source,
        env=env,
        capture_output=True,
        text=True,
    )


def test_metadata_can_be_generated_without_cuda(source_copy):
    result = _setup(source_copy, "egg_info")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("command", ["bdist_wheel", "editable_wheel"])
def test_default_source_build_refuses_an_unusable_install(source_copy, command):
    result = _setup(source_copy, command)
    assert result.returncode != 0
    assert "TURBOPAIRFORMER_SOURCE_ONLY=1" in result.stderr


def test_explicit_source_only_wheel_excludes_editable_receipts_and_binaries(
    source_copy,
):
    (source_copy / "turbopairformer/_build_manifest.local.json").write_text(
        '{"built":true}'
    )
    (source_copy / "turbopairformer/_ext/stale.so").write_bytes(b"stale binary")
    result = _setup(source_copy, "bdist_wheel", source_only=True)
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = (source_copy / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(
            name.endswith((".local.json", ".so")) for name in archive.namelist()
        )
        assert (
            json.loads(archive.read("turbopairformer/_build_manifest.json"))["built"]
            is False
        )
    with pytest.raises(ValueError, match="binary"):
        _wheel_validator().validate_wheel(wheel)


def test_facade_metadata_contract_is_release_bounded():
    text = (PACKAGE / "_triangle_attention.py").read_text()
    assert "_PAD_HEAD_LANES = (16, 32, 64)" in text
    assert "_MAX_BROADCAST_PREFIX_RANK = 16" in text
    assert "if bias_requires_broadcast and len(prefix) >" in text
    assert "D<=64" in text
    assert "stablefold_trimul" not in text


def test_build_entrypoint_is_abi_locked_and_smoke_tested():
    text = (REPOSITORY / "tools/build_sm90_wheel.sh").read_text()
    for contract in (
        "TORCH_CUDA_ARCH_LIST=9.0a",
        "sys.version_info[:2] == (3, 14)",
        'torch.__version__.split("+", 1)[0] == "2.10.0"',
        'torch.version.cuda == "12.8"',
        "TURBOPAIRFORMER_BUILD_AOT=1",
        "git status --porcelain --untracked-files=all",
        "CUDA_HOME=/nonexistent",
        "load_extension()",
        "tools/validate_wheel.py",
        "pip install --no-deps --no-compile --target",
        "-u TURBOPAIRFORMER_ALLOW_JIT -u OPENFOLD3_TRIANGLE_ALLOW_JIT",
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
    assert 'relative.startswith("turbopairformer/")' in runtime
    assert "Path(__file__).resolve().parents[1]" in runtime


def test_trimul_custom_ops_use_the_package_namespace():
    text = (PACKAGE / "trimul.py").read_text()
    assert '"turbopairformer::_trimul_fwd_v1"' in text
    assert '"turbopairformer::_trimul_bwd_v1"' in text
    assert "openfold3_stablefold" not in text


def test_sdist_excludes_generated_artifacts(source_copy):
    import tarfile

    (source_copy / "turbopairformer/_build_manifest.local.json").write_text(
        '{"built":true}'
    )
    (source_copy / "turbopairformer/_ext/stale.so").write_bytes(b"stale binary")
    result = _setup(source_copy, "sdist")
    assert result.returncode == 0, result.stdout + result.stderr
    (archive_path,) = (source_copy / "dist").glob("*.tar.gz")
    with tarfile.open(archive_path) as archive:
        names = archive.getnames()
        assert not any(name.endswith((".local.json", ".so")) for name in names)
        assert any(name.endswith("/tools/validate_wheel.py") for name in names)


def _elf_fixture(runtime_path=None, *, tag=15):
    """Minimal structurally valid ELF64 dynamic object for parser edge cases."""
    import struct

    base = 0x10000
    dynamic_offset = 64 + 2 * 56
    dynamic_size = (3 if runtime_path is None else 4) * 16
    string_offset = dynamic_offset + dynamic_size
    strings = (
        b"\x00" if runtime_path is None else b"\x00" + runtime_path.encode() + b"\x00"
    )
    total_size = string_offset + len(strings)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\x00" * 9,
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, base, base, total_size, total_size, 4096)
    dynamic_header = struct.pack(
        "<IIQQQQQQ",
        2,
        6,
        dynamic_offset,
        base + dynamic_offset,
        base + dynamic_offset,
        dynamic_size,
        dynamic_size,
        8,
    )
    entries = [(5, base + string_offset), (10, len(strings))]
    if runtime_path is not None:
        entries.append((tag, 1))
    entries.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *entry) for entry in entries)
    return header + load + dynamic_header + dynamic + strings


@pytest.fixture
def synthetic_aot_wheel(tmp_path):
    """An archive fixture tests integrity gates; these are not runnable kernels."""
    import hashlib

    validator = _wheel_validator()
    source = b"# Synthetic package source fixture.\n"
    files = {"turbopairformer/__init__.py": source}
    records = {}
    for key, module in validator.MODULES.items():
        filename = module.replace(".", "/") + ".cpython-314-x86_64-linux-gnu.so"
        binary = _elf_fixture() + key.encode()
        files[filename] = binary
        records[key] = {
            "module": module,
            "filename": filename,
            "size_bytes": len(binary),
            "sha256": hashlib.sha256(binary).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "built": True,
        "git": {"commit": "a" * 40, "dirty": False},
        "abi": {
            "python_version": "3.14.0",
            "torch_version": "2.10.0+cu128",
            "torch_cuda_version": "12.8",
        },
        "target": {
            "cuda_arch": "sm_90a",
            "torch_cuda_arch_list": "9.0a",
            "ptx_embedded": False,
        },
        "extensions": records,
        "sources": {
            "turbopairformer/__init__.py": hashlib.sha256(source).hexdigest(),
        },
    }
    metadata = (
        "Metadata-Version: 2.4\nName: turbopairformer\nVersion: 0.1.0\n"
        "Requires-Python: ==3.14.*\nRequires-Dist: torch==2.10.0\n"
    )
    files["turbopairformer-0.1.0.dist-info/METADATA"] = metadata.encode()

    def make(
        *,
        dirty=False,
        corruption=False,
        local_receipt=False,
        platform="linux_x86_64",
        missing_source_record=False,
        extra_source=False,
        runtime_path=None,
    ):
        manifest["git"]["dirty"] = dirty
        if runtime_path is not None:
            for record in records.values():
                binary = _elf_fixture(runtime_path)
                files[record["filename"]] = binary
                record["size_bytes"] = len(binary)
                record["sha256"] = hashlib.sha256(binary).hexdigest()
        if missing_source_record:
            manifest["sources"].pop("turbopairformer/__init__.py")
        if extra_source:
            files["turbopairformer/unrecorded.py"] = b"# Unrecorded source file.\n"
        files["turbopairformer/_build_manifest.json"] = json.dumps(manifest).encode()
        files["turbopairformer-0.1.0.dist-info/WHEEL"] = (
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp314-cp314-{platform}\n"
        ).encode()
        result = tmp_path / f"turbopairformer-0.1.0-cp314-cp314-{platform}.whl"
        with zipfile.ZipFile(result, "w") as archive:
            for name, content in files.items():
                if corruption and name.endswith(".so"):
                    content = content[:-1] + b"!"
                archive.writestr(name, content)
            if local_receipt:
                archive.writestr("turbopairformer/_build_manifest.local.json", "{}")
        return result

    return make


def test_wheel_validator_accepts_a_complete_local_receipt(synthetic_aot_wheel):
    result = _wheel_validator().validate_wheel(synthetic_aot_wheel())
    assert result["git"] == {"commit": "a" * 40, "dirty": False}


def test_wheel_validator_rejects_corrupt_binaries(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="hash mismatch"):
        _wheel_validator().validate_wheel(synthetic_aot_wheel(corruption=True))


def test_wheel_validator_rejects_dirty_release_and_allows_explicit_development(
    synthetic_aot_wheel,
):
    wheel = synthetic_aot_wheel(dirty=True)
    with pytest.raises(ValueError, match="dirty"):
        _wheel_validator().validate_wheel(wheel)
    assert _wheel_validator().validate_wheel(wheel, allow_dirty=True)["git"]["dirty"]


def test_wheel_validator_rejects_local_receipt_leak(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="editable build receipt"):
        _wheel_validator().validate_wheel(synthetic_aot_wheel(local_receipt=True))


def test_wheel_validator_refuses_linux_tag_for_pypi(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="audited manylinux"):
        _wheel_validator().validate_wheel(synthetic_aot_wheel(), require_pypi=True)


def test_wheel_validator_rejects_retagging_without_an_audit(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="auditwheel repair receipt"):
        _wheel_validator().validate_wheel(
            synthetic_aot_wheel(platform="manylinux_2_28_x86_64"), require_pypi=True
        )


@pytest.fixture
def wheel_repair(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY / "tools"))
    spec = importlib.util.spec_from_file_location(
        "turbopairformer_wheel_repair", REPOSITORY / "tools/repair_wheel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "test-audit")
    monkeypatch.setattr(module, "_providers", lambda _: {})

    def audit(source, destination, target, exclusions):
        assert not exclusions
        members = module._members(source)
        manifest = json.loads(members[module.MANIFEST])
        if "repair" not in manifest:
            # Exercise a real binary transformation without claiming that this
            # CPU fixture is an actual successful auditwheel policy audit.
            for record in manifest["extensions"].values():
                members[record["filename"]] += b"repaired fixture"
        metadata = "turbopairformer-0.1.0.dist-info/"
        members[metadata + "WHEEL"] = (
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp314-cp314-{target}\n"
        ).encode()
        members[metadata + "RECORD"] = b""
        output = destination / f"turbopairformer-0.1.0-cp314-cp314-{target}.whl"
        module._write_wheel(output, members)
        return output, "synthetic audit test only"

    monkeypatch.setattr(module, "_audit", audit)
    return module


def test_repair_preserves_original_and_final_binary_hashes(
    synthetic_aot_wheel, wheel_repair, tmp_path
):
    original = synthetic_aot_wheel()
    result = wheel_repair.repair_wheel(
        original, tmp_path / "releases", "manylinux_2_28_x86_64"
    )
    output = tmp_path / "releases" / result["wheel"]
    manifest = json.loads(wheel_repair._members(output)[wheel_repair.MANIFEST])
    for key, final in manifest["extensions"].items():
        before = manifest["repair"]["input_extensions"][key]
        assert before["sha256"] != final["sha256"]
        assert before["size_bytes"] < final["size_bytes"]
    assert manifest["repair"]["input_sha256"] != result["sha256"]
    assert wheel_repair.check_wheel(output)["sha256"] == result["sha256"]


def test_release_record_detects_nonbinary_metadata_tampering(
    synthetic_aot_wheel, wheel_repair, tmp_path
):
    result = wheel_repair.repair_wheel(
        synthetic_aot_wheel(), tmp_path / "releases", "manylinux_2_28_x86_64"
    )
    output = tmp_path / "releases" / result["wheel"]
    members = wheel_repair._members(output)
    members["turbopairformer-0.1.0.dist-info/METADATA"] += b"Summary: tampered\n"
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError, match="RECORD mismatch"):
        _wheel_validator().validate_wheel(output, require_pypi=True)


def test_release_audit_rejects_unexpected_bundled_library(
    synthetic_aot_wheel, wheel_repair, monkeypatch, tmp_path
):
    audit = wheel_repair._audit

    def bundle(*args):
        output, log = audit(*args)
        members = wheel_repair._members(output)
        members["turbopairformer.libs/libunreviewed.so"] = _elf_fixture()
        wheel_repair._write_wheel(output, members)
        return output, log

    monkeypatch.setattr(wheel_repair, "_audit", bundle)
    with pytest.raises(ValueError, match="unexpected bundled library"):
        wheel_repair.repair_wheel(
            synthetic_aot_wheel(), tmp_path / "releases", "manylinux_2_28_x86_64"
        )
    assert not list((tmp_path / "releases").glob("*.whl"))


def test_wheel_validator_binds_artifact_to_selected_source(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="another commit"):
        _wheel_validator().validate_wheel(
            synthetic_aot_wheel(), expected_commit="b" * 40
        )


@pytest.mark.parametrize("change", ["missing_source_record", "extra_source"])
def test_wheel_validator_requires_all_packaged_source_hashes(
    synthetic_aot_wheel, change
):
    with pytest.raises(ValueError, match="source receipt coverage mismatch"):
        _wheel_validator().validate_wheel(synthetic_aot_wheel(**{change: True}))


@pytest.mark.parametrize("tag", [15, 29])
def test_elf_parser_finds_rpath_and_runpath(tag):
    paths = "$ORIGIN/../lib:/tmp/old-conda/lib"
    assert _wheel_validator().elf_runtime_paths(
        _elf_fixture(paths, tag=tag)
    ) == paths.split(":")


@pytest.mark.parametrize("content", [b"", b"\x7fELF", _elf_fixture()[:-1]])
def test_elf_parser_rejects_malformed_binaries(content):
    with pytest.raises(ValueError):
        _wheel_validator().elf_runtime_paths(content)


def test_elf_parser_rejects_invalid_dynamic_string_address():
    import struct

    binary = bytearray(_elf_fixture("$ORIGIN"))
    struct.pack_into("<Q", binary, 64 + 2 * 56 + 8, 0xFFFFFFFFFFFFFFFF)
    with pytest.raises(ValueError, match="string table"):
        _wheel_validator().elf_runtime_paths(bytes(binary))


def test_wheel_validator_rejects_absolute_build_host_rpath(synthetic_aot_wheel):
    with pytest.raises(ValueError, match="absolute ELF runtime search path"):
        _wheel_validator().validate_wheel(
            synthetic_aot_wheel(runtime_path="$ORIGIN/../lib:/tmp/old-conda/lib")
        )


def test_wheel_validator_accepts_origin_relative_rpath(synthetic_aot_wheel):
    _wheel_validator().validate_wheel(
        synthetic_aot_wheel(runtime_path="$ORIGIN/../turbopairformer.libs")
    )


@pytest.mark.parametrize("dtags", ["--enable-new-dtags", "--disable-new-dtags"])
def test_elf_parser_and_patchelf_on_a_linked_shared_object(tmp_path, dtags):
    compiler = shutil.which("cc")
    patchelf = shutil.which("patchelf")
    if sys.platform != "linux" or not compiler or not patchelf:
        pytest.skip("this ELF integration test needs Linux, cc, and patchelf")
    source = tmp_path / "rpath_test.c"
    source.write_text("int stablefold_rpath_fixture(void) { return 0; }\n")
    binary = tmp_path / "rpath_test.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(binary),
            f"-Wl,{dtags},-rpath,/tmp/turbopairformer-build-host/lib",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    validator = _wheel_validator()
    assert validator.elf_runtime_paths(binary.read_bytes()) == [
        "/tmp/turbopairformer-build-host/lib"
    ]
    subprocess.run([patchelf, "--remove-rpath", str(binary)], check=True)
    assert validator.elf_runtime_paths(binary.read_bytes()) == []
