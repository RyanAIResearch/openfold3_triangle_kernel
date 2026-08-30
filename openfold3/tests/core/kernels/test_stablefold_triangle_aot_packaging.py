from __future__ import annotations

import ast
import json
import tomllib
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
CUDA_PACKAGE = REPOSITORY / "openfold3/core/kernels/cuda"
LOADERS = (
    "stablefold_triangle_forward.py",
    "stablefold_triangle_delta.py",
    "stablefold_triangle_dkdv.py",
    "stablefold_triangle_dq_dbias.py",
    "stablefold_trimul_cuda.py",
)
EXPECTED_MODULES = {
    "_stablefold_forward_sm90a",
    "_stablefold_delta_sm90a",
    "_stablefold_dkdv_sm90a",
    "_stablefold_dq_dbias_sm90a",
    "_stablefold_trimul_sm90a",
}


class StableFoldTriangleAOTPackagingTest(unittest.TestCase):
    def test_production_loaders_do_not_contain_jit_compilation(self):
        for loader in LOADERS:
            text = (CUDA_PACKAGE / loader).read_text()
            self.assertNotIn("torch.utils.cpp_extension", text, loader)
            self.assertIn("load_extension_module", text, loader)
            self.assertIn(".cuda._ext.", text, loader)

    def test_jit_import_is_isolated_to_explicit_development_function(self):
        path = CUDA_PACKAGE / "_aot_runtime.py"
        tree = ast.parse(path.read_text())
        imports: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch.utils.cpp_extension"
            ):
                parent_function = next(
                    (
                        function.name
                        for function in ast.walk(tree)
                        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node in ast.walk(function)
                    ),
                    "<module>",
                )
                imports.append((node.module, parent_function))
        self.assertEqual(
            imports,
            [("torch.utils.cpp_extension", "_load_development_jit")],
        )
        text = path.read_text()
        self.assertIn(
            '_DEVELOPMENT_JIT_ENV = "OPENFOLD3_TRIANGLE_ALLOW_JIT"', text
        )
        self.assertIn(
            'os.environ.get(_DEVELOPMENT_JIT_ENV) == "1"', text
        )
        self.assertIn("if not _development_jit_enabled():", text)

    def test_setup_defines_five_unique_sm90a_extensions(self):
        text = (REPOSITORY / "setup.py").read_text()
        discovered = {
            name for name in EXPECTED_MODULES if text.count(name) == 1
        }
        self.assertEqual(discovered, EXPECTED_MODULES)
        self.assertEqual(text.count("CUDAExtension("), 1)
        self.assertIn('EXPECTED_PYTHON = (3, 14)', text)
        self.assertIn('EXPECTED_TORCH = "2.10.0"', text)
        self.assertIn('EXPECTED_TORCH_CUDA = "12.9"', text)
        self.assertIn('EXPECTED_ARCH_LIST = "9.0a"', text)
        self.assertIn('"stablefold_pack_qkv"', text)
        self.assertIn('"stablefold_triangle_forward_shadow_out"', text)
        self.assertIn('"stablefold_triangle_delta_shadow_out"', text)
        self.assertIn('"stablefold_triangle_dkdv_delta_true_out"', text)
        self.assertIn('"stablefold_triangle_sr_backward_out"', text)
        for symbol in (
            "trimul_in_fwd",
            "trimul_out_fwd",
            "trimul_gate_split_bwd",
            "sig_mul_bwd_out",
            "ln_bwd_dx",
            "ln_bwd_dgb",
            "transpose_ln_bwd",
        ):
            self.assertIn(f'"{symbol}"', text)
        self.assertNotIn("--use_fast_math", text)

    def test_trimul_aot_excludes_legacy_row_gather_symbols(self):
        production_members = (
            "openfold3/core/kernels/cuda/stablefold_trimul.py",
            "openfold3/core/kernels/cuda/stablefold_trimul_cuda.py",
            "openfold3/core/kernels/cuda/stablefold_trimul_cuda.cu",
            "setup.py",
            "tools/attest_sm90_wheel.py",
        )
        for member in production_members:
            text = (REPOSITORY / member).read_text()
            for symbol in ("pack_rows3", "gather_rows1"):
                self.assertNotIn(symbol, text, member)

    def test_pyproject_pins_torch_family_and_ships_manifest(self):
        project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["requires-python"], "==3.14.*")
        self.assertIn("torch==2.10.0", project["project"]["dependencies"])
        self.assertIn(
            "torch==2.10.0",
            project["project"]["optional-dependencies"]["cuequivariance"],
        )
        python_classifiers = {
            value
            for value in project["project"]["classifiers"]
            if value.startswith("Programming Language :: Python ::")
        }
        self.assertEqual(
            python_classifiers,
            {"Programming Language :: Python :: 3.14"},
        )
        package_data = project["tool"]["setuptools"]["package-data"][
            "openfold3"
        ]
        self.assertIn("core/kernels/cuda/_build_manifest.json", package_data)
        self.assertIn("core/kernels/cuda/stablefold_trimul_cuda.cu", package_data)
        self.assertNotIn("cibuildwheel", project.get("tool", {}))

    def test_source_tree_manifest_is_fail_closed(self):
        manifest = json.loads(
            (CUDA_PACKAGE / "_build_manifest.json").read_text()
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIs(manifest["built"], False)

    def test_facade_metadata_contract_is_release_bounded(self):
        text = (CUDA_PACKAGE / "stablefold_triangle_attention.py").read_text()
        self.assertIn("_PAD_HEAD_LANES = (16, 32, 64)", text)
        self.assertIn("_MAX_BROADCAST_PREFIX_RANK = 16", text)
        self.assertIn("if bias_requires_broadcast and len(prefix) >", text)
        self.assertIn("D<=64", text)
        self.assertNotIn("stablefold_trimul", text)

    def test_build_entrypoint_is_abi_locked_and_nonisolated(self):
        text = (REPOSITORY / "tools/build_sm90_wheel.sh").read_text()
        for contract in (
            "TORCH_CUDA_ARCH_LIST=9.0a",
            "sys.version_info[:2] == (3, 14)",
            'torch.__version__.split("+", 1)[0] == "2.10.0"',
            'torch.version.cuda == "12.9"',
            "--no-build-isolation",
            "tools/attest_sm90_wheel.py",
            "--require-binary-tools",
            "tools/aot_clean_install_smoke.py",
            "DS_ACCELERATOR=cpu",
            "git clone --quiet --no-local --no-checkout",
            'git -C "$FRESH_SOURCE" checkout --quiet --detach',
            'git status --porcelain --untracked-files=all',
            'mktemp -d "${TMPDIR:-/tmp}/openfold3-aot-sm90a-build.XXXXXX"',
            'chmod -R u+w -- "$BUILD_ROOT"',
            'rm -r -- "$BUILD_ROOT"',
            'SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"',
            'CXX="${CXX:-$(command -v c++)}"',
        ):
            self.assertIn(contract, text)
        self.assertNotIn("OPENFOLD3_AOT_BUILD_ROOT", text)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn("ALLOW_DIRTY", text)

    def test_clean_install_smoke_forbids_compiler_and_jit(self):
        text = (REPOSITORY / "tools/aot_clean_install_smoke.py").read_text()
        self.assertIn('shutil.which("nvcc")', text)
        self.assertIn("OPENFOLD3_TRIANGLE_ALLOW_JIT", text)
        self.assertIn("TORCH_EXTENSIONS_DIR", text)
        self.assertIn('os.environ.get("TORCH_EXTENSIONS_DIR")', text)
        self.assertNotIn('os.environ["TORCH_EXTENSIONS_DIR"]', text)
        self.assertIn("torch.autograd.grad", text)
        self.assertIn('"dBias": gradients[3]', text)
        self.assertIn("prefix, sequence, heads, head_dim = 1, 64, 4, 32", text)
        self.assertIn("non-contiguous fused pack differs", text)
        self.assertIn('for arm in ("two_pass", "sr_r")', text)
        self.assertIn("_stablefold_trimul_sm90a", text)
        self.assertIn('"outgoing": run_trimul(True)', text)
        self.assertIn('"incoming": run_trimul(False)', text)
        self.assertIn('os.environ["OPENFOLD3_SR_BACKWARD"] = "1"', text)
        self.assertIn('os.environ["OPENFOLD3_SR_SHADOW"] = "1"', text)
        self.assertIn("mask = torch.zeros(", text)

    def test_clean_install_smoke_covers_versioned_trimul_graph_contract(self):
        text = (REPOSITORY / "tools/aot_clean_install_smoke.py").read_text()
        for contract in (
            "openfold3_stablefold::_trimul_fwd_v1",
            "openfold3_stablefold::_trimul_bwd_v1",
            "torch.ops.openfold3_stablefold._trimul_fwd_v1.default._schema",
            "torch.ops.openfold3_stablefold._trimul_bwd_v1.default._schema",
            "stable._trimul_fwd_v1(",
            "stable._trimul_bwd_v1(",
            "self.c_z = channels",
            "self.c_hidden = channels",
            "bias=False",
            "eps=1e-5",
            "stable.run_stable_trimul(module, z, mask)",
            "stable._run_supported_stable_trimul(self, z, mask)",
            "torch.compile(",
            'backend="aot_eager"',
            "fullgraph=True",
            'torch.export.Dim("sequence", min=1, max=256)',
            "torch.export.export(",
            "torch.export.save(",
            "torch.export.load(",
            "openfold3_stablefold._trimul_fwd_v1.default",
            "requires_grad_(False)",
            "dynamic export/save/load differs from eager inference",
            "aot_eager fullgraph F+B differs from eager",
            'assert_no_jit_cache("custom-op import")',
            'assert_no_jit_cache("custom-op fake")',
            'assert_no_jit_cache("dynamic export registration")',
            'assert_no_jit_cache("TriMul eager and aot_eager")',
            '"trimul_custom_ops"',
        ):
            self.assertIn(contract, text)
        self.assertNotIn('backend="inductor"', text)
        self.assertLess(
            text.index('assert_no_jit_cache("dynamic export registration")'),
            text.index("expected = module(dynamic_z, dynamic_mask)"),
        )

    def test_attestation_is_cp314_sass_only_and_checks_unique_binaries(self):
        text = (REPOSITORY / "tools/attest_sm90_wheel.py").read_text()
        for contract in (
            "-cp314-cp314-linux_x86_64",
            "Tag: cp314-cp314-linux_x86_64",
            "Root-Is-Purelib: false",
            'r"[0-9a-f]{40}"',
            "unexpected extension module name",
            "unexpected extension symbol contract",
            "unrecorded or missing private shared objects",
            'for tool in ("nm", "cuobjdump")',
            'command("cuobjdump", "--dump-ptx"',
        ):
            self.assertIn(contract, text)

    def test_manifest_receipt_covers_build_and_runtime_contract(self):
        text = (REPOSITORY / "setup.py").read_text()
        for relative in (
            "MANIFEST.in",
            "README.md",
            "PRODUCTION_MANIFEST.md",
            "_ext/__init__.py",
            "tools/build_sm90_wheel.sh",
            "tools/attest_sm90_wheel.py",
            "tools/aot_clean_install_smoke.py",
        ):
            self.assertIn(relative, text)
        self.assertIn('"--frandom-seed=0"', text)
        self.assertIn(
            'sources=[f"openfold3/core/kernels/cuda/{value[\'source\']}"],',
            text,
        )
        source_manifest = (REPOSITORY / "MANIFEST.in").read_text()
        self.assertIn("recursive-include tools *.py *.sh", source_manifest)
        self.assertIn(
            "recursive-include openfold3/core/kernels/cuda *.cu *.py *.json *.md",
            source_manifest,
        )

    def test_binary_hash_is_checked_before_extension_import(self):
        runtime = (CUDA_PACKAGE / "_aot_runtime.py").read_text()
        self.assertIn("_preflight_aot_artifact", runtime)
        self.assertIn("binary hash mismatch", runtime)
        self.assertLess(
            runtime.index("artifact_path = _preflight_aot_artifact"),
            runtime.index("module = importlib.import_module(aot_module)"),
        )


if __name__ == "__main__":
    unittest.main()
