from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
CUDA_PACKAGE = REPOSITORY / "openfold3/core/kernels/cuda"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


class StableFoldTrianglePackagingTest(unittest.TestCase):
    def test_runtime_facade_does_not_depend_on_benchmark_modules(self):
        facade = CUDA_PACKAGE / "stablefold_triangle_attention.py"
        text = facade.read_text()
        self.assertNotIn("scripts.benchmarks", text)

    def test_runtime_loader_closure_is_package_owned(self):
        runtime_modules = {
            "stablefold_triangle_attention.py",
            "stablefold_triangle_forward.py",
            "stablefold_triangle_delta.py",
            "stablefold_triangle_dkdv.py",
            "stablefold_triangle_dq_dbias.py",
        }
        for module in runtime_modules:
            imports = _top_level_imports(CUDA_PACKAGE / module)
            self.assertFalse(
                any(name.startswith("scripts.benchmarks") for name in imports),
                module,
            )

    def test_runtime_cuda_sources_are_shipped_as_package_data(self):
        project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
        package_data = set(
            project["tool"]["setuptools"]["package-data"]["openfold3"]
        )
        required = {
            "core/kernels/cuda/stablefold_triangle_forward.cu",
            "core/kernels/cuda/stablefold_triangle_delta.cu",
            "core/kernels/cuda/stablefold_triangle_dkdv.cu",
            "core/kernels/cuda/stablefold_triangle_dq_dbias.cu",
        }
        self.assertTrue(required.issubset(package_data))
        for relative in required:
            self.assertTrue((REPOSITORY / "openfold3" / relative).is_file())


if __name__ == "__main__":
    unittest.main()
