# SPDX-License-Identifier: Apache-2.0

"""Guard the two properties the StableFold kernels exist to provide.

Bitwise equivalence between the providers is checked by
``test_turbopairformer_provider_equivalence.py``. That is necessary but not sufficient:
a packaging change could leave the numbers identical and still lose the speedup
that motivates the kernels, or a kernel change could keep the speed and quietly
give up accuracy. This module holds both to a floor.

Thresholds are deliberately loose. A shared GPU makes wall-clock noisy, so the
speed bounds are set to catch a structural regression -- a lost fast path, a
fallback to stock, a dispatch that stopped being cached -- not a few percent of
jitter. Reference numbers measured on one H100 80 GB are in the comments.

Run with::

    pytest openfold3/tests/test_turbopairformer_performance_regression.py \\
        --import-mode=importlib -v
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import statistics
import unittest

import torch

WARMUP = 5
ITERATIONS = 15

# Layouts drawn from the repository's own SR benchmarks, kept small enough that
# the whole module runs in well under a minute.
SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "trunk_n384_d32": (1, 384, 4, 384, 32),
    "trunk_n640_d32": (1, 640, 4, 640, 32),
}

# Observed forward+backward speedups over stock attention were 3.43x and 4.16x
# at these two layouts. A floor of 2.0 leaves generous headroom while still
# failing loudly if the kernels stop being used at all.
MIN_SPEEDUP_OVER_STOCK = 2.0

# Observed packaging cost was 0.99x-1.04x across six layouts. 1.25 catches a
# structural regression without tripping on GPU jitter.
MAX_PACKAGING_COST = 1.25


def _has_real_turbopairformer() -> bool:
    try:
        spec = importlib.util.find_spec("turbopairformer")
    except ImportError, ValueError:
        return False
    return spec is not None and spec.origin is not None


def _turbopairformer_distribution_installed() -> bool:
    try:
        importlib.metadata.distribution("turbopairformer")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


HAS_TURBOPAIRFORMER = _has_real_turbopairformer()
IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)

if _turbopairformer_distribution_installed() and not HAS_TURBOPAIRFORMER:
    raise RuntimeError(
        "the 'turbopairformer' distribution is installed but "
        "'import turbopairformer' resolves "
        "to a namespace package -- run with --import-mode=importlib, or from "
        "outside the checkout"
    )

requires_both = unittest.skipUnless(
    HAS_TURBOPAIRFORMER and IS_HOPPER,
    "needs the turbopairformer package and a Hopper sm_90a GPU",
)


def _inputs(shape, *, grad: bool):
    batch, outer, heads, seq, head_dim = shape
    generator = torch.Generator(device="cuda").manual_seed(7)

    def randn(*size):
        return torch.randn(
            *size, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    q = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    k = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    v = randn(batch, outer, heads, seq, head_dim).requires_grad_(grad)
    mask = torch.zeros(batch, outer, 1, 1, seq, device="cuda", dtype=torch.bfloat16)
    bias = randn(batch, 1, heads, seq, seq).requires_grad_(grad)
    return q, k, v, mask, bias


def _stock_attention(q, k, v, mask, bias):
    from openfold3.core.model.primitives.attention import _attention

    return _attention(q * (1.0 / math.sqrt(q.shape[-1])), k, v, [mask, bias]).transpose(
        -2, -3
    )


def _implementations():
    from turbopairformer import triangle_attention

    from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_attention import (
        turbopairformer_triangle_attention,
    )

    return {
        "turbopairformer": triangle_attention,
        "intree": turbopairformer_triangle_attention,
        "stock": _stock_attention,
    }


def _median_ms(attention, shape) -> float:
    """Forward+backward milliseconds, CUDA-event timed, median of the run."""

    inputs = _inputs(shape, grad=True)
    q, k, v, _, bias = inputs
    grad_out = torch.randn_like(attention(*inputs))

    def step():
        for tensor in (q, k, v, bias):
            tensor.grad = None
        attention(*inputs).backward(grad_out)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    samples = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


@requires_both
class SpeedRegressionTest(unittest.TestCase):
    """The kernels must stay faster than stock, and packaging must stay free."""

    @classmethod
    def setUpClass(cls):
        cls.implementations = _implementations()
        cls.timings = {
            name: {
                impl: _median_ms(attention, shape)
                for impl, attention in cls.implementations.items()
            }
            for name, shape in SHAPES.items()
        }

    def test_kernels_stay_faster_than_stock(self):
        for name in SHAPES:
            timing = self.timings[name]
            for provider in ("turbopairformer", "intree"):
                with self.subTest(layout=name, provider=provider):
                    speedup = timing["stock"] / timing[provider]
                    self.assertGreater(
                        speedup,
                        MIN_SPEEDUP_OVER_STOCK,
                        f"{provider} at {name} is only {speedup:.2f}x stock "
                        f"({timing[provider]:.3f} ms against "
                        f"{timing['stock']:.3f} ms); the fast path is likely "
                        "not being taken",
                    )

    def test_packaging_costs_nothing(self):
        for name in SHAPES:
            with self.subTest(layout=name):
                timing = self.timings[name]
                cost = timing["turbopairformer"] / timing["intree"]
                self.assertLess(
                    cost,
                    MAX_PACKAGING_COST,
                    f"the pip package is {cost:.2f}x the in-tree build at "
                    f"{name} ({timing['turbopairformer']:.3f} ms against "
                    f"{timing['intree']:.3f} ms)",
                )


@requires_both
class AccuracyRegressionTest(unittest.TestCase):
    """Packaging must not move the error, and the kernels must beat stock."""

    # Finite stand-in for -inf so the FP64 oracle stays well defined.
    NEG = -1e4
    TENSORS = ("out", "dq", "dk", "dv", "dbias")

    def _case(self, shape):
        batch, outer, heads, seq, head_dim = shape
        torch.manual_seed(23)
        q, k, v = (
            torch.randn(batch, outer, heads, seq, head_dim, device="cuda").to(
                torch.bfloat16
            )
            for _ in range(3)
        )
        mask = torch.zeros(batch, outer, 1, 1, seq, device="cuda")
        bias = (torch.randn(1, 1, heads, seq, seq, device="cuda") * 2).to(
            torch.bfloat16
        )
        d_out = torch.randn(batch, outer, seq, heads, head_dim, device="cuda").to(
            torch.bfloat16
        )
        return q, k, v, mask, bias, d_out

    def _oracle(self, q, k, v, mask, bias, d_out):
        q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
        b64 = bias.detach().double().requires_grad_(True)
        scores = torch.matmul(q64, k64.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores + mask.double() + b64
        out = torch.matmul(torch.softmax(scores, dim=-1), v64).transpose(-2, -3)
        out.backward(d_out.double())
        return out.detach(), q64.grad, k64.grad, v64.grad, b64.grad

    def _stock(self, q, k, v, mask, bias, d_out):
        q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        b_ = bias.detach().clone().requires_grad_(True)
        scores = torch.matmul(q_, k_.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
        scores = scores + mask + b_.float()
        out = torch.matmul(torch.softmax(scores, dim=-1).to(torch.bfloat16), v_)
        out = out.transpose(-2, -3)
        out.backward(d_out)
        return out.detach(), q_.grad, k_.grad, v_.grad, b_.grad

    def _kernel(self, attention, q, k, v, mask, bias, d_out):
        q_, k_, v_ = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        b_ = bias.detach().clone().requires_grad_(True)
        attention(q_, k_, v_, mask, b_).backward(d_out)
        return None, q_.grad, k_.grad, v_.grad, b_.grad

    @staticmethod
    def _relative(value, reference) -> float:
        return float(
            (value.double() - reference.double()).norm()
            / reference.double().norm().clamp_min(1e-30)
        )

    def test_packaging_does_not_move_the_error(self):
        """The two providers must report the same error to the last digit."""

        implementations = _implementations()
        shape = SHAPES["trunk_n384_d32"]
        q, k, v, mask, bias, d_out = self._case(shape)
        reference = self._oracle(q, k, v, mask, bias, d_out)

        errors = {}
        for provider in ("turbopairformer", "intree"):
            results = self._kernel(
                implementations[provider], q, k, v, mask, bias, d_out
            )
            errors[provider] = [
                self._relative(value, ref)
                for value, ref in zip(results[1:], reference[1:], strict=True)
            ]
        self.assertEqual(
            errors["turbopairformer"],
            errors["intree"],
            "packaging moved the numerical error; the two builds should be "
            "bitwise identical",
        )

    def test_kernels_stay_within_the_stock_envelope(self):
        """The package's own bound: at or below 1.1 x stock error + 1e-6."""

        implementations = _implementations()
        shape = SHAPES["trunk_n384_d32"]
        q, k, v, mask, bias, d_out = self._case(shape)
        reference = self._oracle(q, k, v, mask, bias, d_out)
        stock = self._stock(q, k, v, mask, bias, d_out)
        stablefold = self._kernel(
            implementations["turbopairformer"], q, k, v, mask, bias, d_out
        )

        for name, kernel_value, stock_value, ref in zip(
            self.TENSORS[1:], stablefold[1:], stock[1:], reference[1:], strict=True
        ):
            with self.subTest(tensor=name):
                kernel_error = self._relative(kernel_value, ref)
                stock_error = self._relative(stock_value, ref)
                self.assertLessEqual(
                    kernel_error,
                    1.1 * stock_error + 1e-6,
                    f"{name}: {kernel_error:.3e} exceeds the stock envelope "
                    f"built from {stock_error:.3e}",
                )

    def test_the_kernels_replay_bitwise(self):
        """Determinism is half of what these kernels are for."""

        implementations = _implementations()
        shape = SHAPES["trunk_n384_d32"]
        case = self._case(shape)
        for provider in ("turbopairformer", "intree"):
            with self.subTest(provider=provider):
                first = self._kernel(implementations[provider], *case)
                second = self._kernel(implementations[provider], *case)
                for left, right in zip(first[1:], second[1:], strict=True):
                    self.assertEqual(left.shape, right.shape)
                    self.assertEqual(left.dtype, right.dtype)
                    self.assertTrue(
                        torch.equal(
                            left.contiguous().reshape(-1).view(torch.uint8),
                            right.contiguous().reshape(-1).view(torch.uint8),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
