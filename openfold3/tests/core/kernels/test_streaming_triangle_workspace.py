from __future__ import annotations

import struct
import unittest
from pathlib import Path

from openfold3.core.kernels.cuda.fused_triangle_dq_dbias_v9 import (
    outer_leaf_for,
    workspace_elements,
    workspace_plan,
)


class StreamingTriangleWorkspaceTest(unittest.TestCase):
    def test_outer_leaf_is_sequence_independent(self):
        for sequence in (1, 17, 65, 383, 384, 640, 768, 769, 2048):
            with self.subTest(sequence=sequence):
                self.assertEqual(outer_leaf_for(sequence), 4)

    def test_workspace_respects_target_budget_above_one_query_tile(self):
        budget = 256 * 1024 * 1024
        for sequence in (65, 384, 640, 768, 1024, 2048, 4096, 8192):
            with self.subTest(sequence=sequence):
                plan = workspace_plan(
                    sequence,
                    4,
                    workspace_budget_bytes=budget,
                )
                minimum_tile_bytes = 4 * 32 * sequence * 2
                self.assertEqual(plan.query_span % 32, 0)
                self.assertGreaterEqual(plan.resident_leaves, 1)
                self.assertLessEqual(
                    plan.elements * 2,
                    max(budget, minimum_tile_bytes),
                )
                self.assertEqual(
                    plan.elements,
                    plan.resident_leaves
                    * plan.batch_chunk
                    * 4
                    * plan.query_span
                    * sequence,
                )

    def test_workspace_budget_accounts_for_flattened_batch(self):
        sequence = 640
        batch = 6
        plan = workspace_plan(
            sequence,
            4,
            batch=batch,
            workspace_budget_bytes=256 * 1024 * 1024,
        )
        self.assertLessEqual(plan.batch_chunk, batch)
        self.assertEqual(
            plan.batch_chunks,
            (batch + plan.batch_chunk - 1) // plan.batch_chunk,
        )
        self.assertLessEqual(plan.elements * 2, 256 * 1024 * 1024)

    def test_query_tile_replaces_the_old_full_plane_lower_bound(self):
        sequence = 8192
        plan = workspace_plan(
            sequence,
            4,
            workspace_budget_bytes=256 * 1024 * 1024,
        )
        old_plane_elements = 4 * sequence * sequence
        self.assertLess(plan.elements, old_plane_elements)
        self.assertEqual(plan.query_span, 32)
        self.assertLessEqual(plan.elements * 2, 256 * 1024 * 1024)

    def test_long_n_plans_reduce_root_waves_without_extra_launch_pairs(self):
        for sequence in (1024, 1536, 2048):
            with self.subTest(sequence=sequence):
                slab = workspace_plan(sequence, 4)
                full = workspace_plan(sequence, 4, debug_full_plane=True)
                self.assertLessEqual(slab.launch_pairs, full.launch_pairs)
                self.assertLessEqual(slab.leaf_waves, full.leaf_waves)
                self.assertLess(slab.query_span, sequence)

    def test_debug_full_plane_preserves_the_old_workspace_shape(self):
        sequence = 641
        batch = 3
        plan = workspace_plan(
            sequence,
            4,
            batch=batch,
            debug_full_plane=True,
        )
        plane_elements = batch * 4 * sequence * sequence
        self.assertTrue(plan.debug_full_plane)
        self.assertEqual(plan.query_span, sequence)
        self.assertEqual(plan.batch_chunk, batch)
        self.assertEqual(plan.elements % plane_elements, 0)

    def test_forced_query_spans_retain_the_same_global_leaf_order(self):
        sequence = 257
        expected_leaves = list(range((sequence + 3) // 4))
        for query_span in (32, 64, 128, 288):
            with self.subTest(query_span=query_span):
                plan = workspace_plan(
                    sequence,
                    4,
                    workspace_budget_bytes=2 * 1024 * 1024,
                    query_span=query_span,
                )
                observed: list[int] = []
                for leaf_base in range(0, plan.leaves, plan.resident_leaves):
                    wave_leaves = min(
                        plan.resident_leaves,
                        plan.leaves - leaf_base,
                    )
                    observed.extend(
                        range(leaf_base, leaf_base + wave_leaves)
                    )
                self.assertEqual(observed, expected_leaves)

    def test_wave_boundaries_preserve_the_fp32_left_fold_bits(self):
        def f32(value: float) -> float:
            return struct.unpack("=f", struct.pack("=f", value))[0]

        leaves = [
            f32(((-1.0) ** index) * (index + 1) / 257.0)
            for index in range(193)
        ]

        def fold(wave_size: int) -> bytes:
            root: float | None = None
            for base in range(0, len(leaves), wave_size):
                for value in leaves[base : base + wave_size]:
                    root = value if root is None else f32(root + value)
                # A wave boundary is an FP32 store/reload, not an arithmetic
                # node and therefore cannot change the root bits.
                assert root is not None
                root = f32(root)
            assert root is not None
            return struct.pack("=f", root)

        expected = fold(len(leaves))
        for wave_size in (1, 2, 3, 7, 32, 64, 192, 193):
            with self.subTest(wave_size=wave_size):
                self.assertEqual(fold(wave_size), expected)

    def test_cuda_source_owns_a_fixed_non_atomic_left_fold(self):
        source = (
            Path(__file__).parents[3]
            / "core/kernels/cuda/fused_triangle_dq_dbias_v9.cu"
        ).read_text()
        self.assertIn("root = __fadd_rn", source)
        self.assertIn("leaf_base += resident_leaves", source)
        self.assertIn("TaskInGridZ", source)
        self.assertIn("task_in_grid_z = leaf_tasks <= 65535", source)
        self.assertNotIn("atomicAdd", source)
        self.assertNotIn("semaphore", source.lower())

    def test_invalid_dimensions_fail_closed(self):
        for kwargs in (
            {"sequence": 0, "heads": 4},
            {"sequence": 64, "heads": 0},
            {"sequence": 64, "heads": 4, "batch": 0},
            {"sequence": 64, "heads": 65_536},
            {"sequence": 65_535 * 32 + 1, "heads": 4},
            {"sequence": 65_536, "heads": 4, "batch": 65_536},
            {
                "sequence": 64,
                "heads": 4,
                "workspace_budget_bytes": 0,
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                workspace_elements(**kwargs)

    def test_invalid_forced_plans_fail_closed(self):
        for kwargs in (
            {"sequence": 64, "heads": 4, "query_span": 31},
            {"sequence": 64, "heads": 4, "query_span": 0},
            {"sequence": 64, "heads": 4, "batch": 2, "batch_chunk": 3},
            {
                "sequence": 64,
                "heads": 4,
                "query_span": 32,
                "debug_full_plane": True,
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                workspace_plan(**kwargs)


if __name__ == "__main__":
    unittest.main()
