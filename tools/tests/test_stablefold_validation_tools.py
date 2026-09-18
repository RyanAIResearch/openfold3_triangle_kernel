# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for the evidence tools used to validate a packaged release."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


def _load_tool(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_tool("compare_checkpoints")
accuracy = _load_tool("benchmark_stablefold_accuracy")


class TensorByteComparisonTest(unittest.TestCase):
    def test_signed_zero_is_not_bitwise_equal(self):
        left, right = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(torch.equal(left, right))
        self.assertFalse(compare._bitwise_equal(left, right))
        self.assertFalse(accuracy._bitwise_equal(left, right))
        self.assertIn("differing bytes = 1/4", compare._describe("zero", left, right))

    def test_nan_payloads_are_compared_as_bytes(self):
        left = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
        right = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
        self.assertFalse(torch.equal(left, left))
        for comparator in (compare._bitwise_equal, accuracy._bitwise_equal):
            self.assertTrue(comparator(left, left.clone()))
            self.assertFalse(comparator(left, right))

    def test_scalar_empty_bfloat16_and_noncontiguous_values(self):
        for value in (
            torch.tensor(3.0),
            torch.empty((0, 3)),
            torch.tensor([1.0, -0.0], dtype=torch.bfloat16),
            torch.arange(6.0).reshape(2, 3).t(),
        ):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                self.assertTrue(compare._bitwise_equal(value, value.contiguous()))
                self.assertTrue(accuracy._bitwise_equal(value, value.contiguous()))
        self.assertFalse(compare._bitwise_equal(torch.ones(1), torch.ones(1, 1)))
        self.assertFalse(compare._bitwise_equal(torch.ones(1), torch.ones(1).double()))

    def test_checkpoint_cli_rejects_signed_zero_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.ckpt", Path(directory) / "right.ckpt"
            torch.save({"state_dict": {"weight": torch.tensor([0.0])}}, left)
            torch.save({"state_dict": {"weight": torch.tensor([-0.0])}}, right)
            output = io.StringIO()
            with (
                mock.patch("sys.argv", ["compare_checkpoints", str(left), str(right)]),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(compare.main(), 1)
            self.assertIn("differing bytes = 1/4", output.getvalue())
            self.assertNotIn("bitwise identical", output.getvalue())

    def test_checkpoint_cli_does_not_pass_without_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "empty.ckpt"
            torch.save({"epoch": 1}, checkpoint)
            with (
                mock.patch(
                    "sys.argv",
                    ["compare_checkpoints", str(checkpoint), str(checkpoint)],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(compare.main(), 1)


class AccuracyBenchmarkTest(unittest.TestCase):
    def test_case_seed_does_not_use_python_hash(self):
        expected = int.from_bytes(
            hashlib.sha256(b"trunk_n384_d32").digest()[:4], "big"
        ) % (2**31)
        with mock.patch("builtins.hash", side_effect=AssertionError("randomized hash")):
            self.assertEqual(accuracy._case_seed("trunk_n384_d32"), expected)
        self.assertNotEqual(expected, accuracy._case_seed("trunk_n640_d32"))

    def test_nonfinite_error_cannot_pass_the_accuracy_gate(self):
        values = tuple(torch.ones(1) for _ in accuracy.TENSOR_NAMES)
        reference = tuple(torch.full((1,), float("nan")) for _ in values)
        with (
            mock.patch(
                "sys.argv",
                [
                    "benchmark_stablefold_accuracy",
                    "--providers",
                    "stablefold",
                    "--cases",
                    "trunk_n384_d32",
                ],
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_name", return_value="CPU mock"),
            mock.patch.object(accuracy, "_case_inputs", return_value=()),
            mock.patch.object(accuracy, "_oracle", return_value=reference),
            mock.patch.object(
                accuracy,
                "_providers",
                return_value={
                    "stock": lambda *args: values,
                    "stablefold": lambda *args: values,
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(accuracy.main(), 1)

    def test_provider_selection_without_stock_still_validates_against_stock(self):
        values = tuple(torch.ones(1) for _ in accuracy.TENSOR_NAMES)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "benchmark_stablefold_accuracy",
                        "--providers",
                        "stablefold",
                        "--cases",
                        "trunk_n384_d32",
                        "--json",
                        str(report_path),
                    ],
                ),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(
                    torch.cuda, "get_device_name", return_value="CPU mock"
                ),
                mock.patch.object(
                    accuracy, "_case_inputs", return_value=values
                ) as inputs,
                mock.patch.object(accuracy, "_oracle", return_value=values),
                mock.patch.object(
                    accuracy,
                    "_providers",
                    return_value={
                        "stock": lambda *args: values,
                        "stablefold": lambda *args: values,
                    },
                ) as providers,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(accuracy.main(), 0)
            providers.assert_called_once_with(["stock", "stablefold"])
            self.assertEqual(
                inputs.call_args.kwargs["seed"], accuracy._case_seed("trunk_n384_d32")
            )
            report = json.loads(report_path.read_text())
            self.assertIn("stock", report["trunk_n384_d32"])
            self.assertTrue(report["trunk_n384_d32"]["stablefold"]["deterministic"])


if __name__ == "__main__":
    unittest.main()
