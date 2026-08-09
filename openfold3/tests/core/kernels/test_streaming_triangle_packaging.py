from __future__ import annotations

import ast
import unittest
from pathlib import Path

import tomllib

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


class StreamingTrianglePackagingTest(unittest.TestCase):
    def test_runtime_facade_does_not_depend_on_benchmark_modules(self):
        facade = CUDA_PACKAGE / "streaming_triangle_attention.py"
        text = facade.read_text()
        self.assertNotIn("scripts.benchmarks", text)
        self.assertNotIn("streaming_backward_m2_gate", text)
        self.assertNotIn("fused_triangle_dq_dbias_v7", text)
        self.assertNotIn("stable_triangle_dkdv_hmma_resident", text)

    def test_runtime_loader_closure_is_package_owned(self):
        runtime_modules = {
            "streaming_triangle_attention.py",
            "streaming_triangle_forward_v1.py",
            "streaming_triangle_delta_v1.py",
            "stable_triangle_dkdv_v2.py",
            "fused_triangle_dq_dbias_v9.py",
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
            "core/kernels/cuda/streaming_triangle_forward_v1.cu",
            "core/kernels/cuda/streaming_triangle_delta_v1.cu",
            "core/kernels/cuda/stable_triangle_dkdv_v2.cu",
            "core/kernels/cuda/fused_triangle_dq_dbias_v9.cu",
        }
        self.assertTrue(required.issubset(package_data))
        for relative in required:
            self.assertTrue((REPOSITORY / "openfold3" / relative).is_file())


if __name__ == "__main__":
    unittest.main()
