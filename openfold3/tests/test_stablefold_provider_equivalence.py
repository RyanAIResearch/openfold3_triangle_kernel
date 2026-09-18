# SPDX-License-Identifier: Apache-2.0

"""The ``stablefold`` pip package must be indistinguishable from the in-tree copy.

``openfold3/core/kernels/cuda/provider.py`` lets the stable triangle routes be
served either by the installed ``stablefold`` package or by the kernel copy
vendored under ``openfold3/core/kernels/cuda``.  Both providers build the same
``.cu`` sources, so switching between them must not move a single bit: these
tests compare the raw bytes of forward outputs and every input gradient.

The two implementations register their ``torch.library`` ops in different
namespaces (``stablefold::`` and ``openfold3_stablefold::``), so a single process
can hold both and compare them directly.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Numeric equality alone does not distinguish signed zeros or NaN payloads."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _has_real_stablefold() -> bool:
    """A bare ``stablefold`` directory on sys.path imports as an empty namespace.

    Interpreters started in the repository root see the package's own project
    folder that way, so ``find_spec`` alone would report an install that has no
    modules in it.  A real distribution has a spec origin.
    """

    try:
        spec = importlib.util.find_spec("stablefold")
    except ImportError, ValueError:
        return False
    return spec is not None and spec.origin is not None


def _stablefold_distribution_installed() -> bool:
    try:
        importlib.metadata.distribution("stablefold")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


HAS_STABLEFOLD = _has_real_stablefold()

# Skipping is the right answer when the package is genuinely absent.  It is the
# wrong answer when it is installed but shadowed: every equivalence test would
# skip, the run would report success, and nothing would have been compared.
if _stablefold_distribution_installed() and not HAS_STABLEFOLD:
    raise RuntimeError(
        "the 'stablefold' distribution is installed but 'import stablefold' resolves "
        "to a namespace package -- the repository root is on sys.path and its "
        "stablefold/ project folder shadows the install. Re-run with "
        "--import-mode=importlib, or from outside the checkout."
    )
IS_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)

requires_both = unittest.skipUnless(
    HAS_STABLEFOLD and IS_HOPPER,
    "needs the stablefold package and a Hopper sm_90a GPU",
)

# [batch, outer, heads, sequence, head_dim] with the pair-stack shapes the
# training run actually issues.
ATTENTION_SHAPES = (
    (1, 128, 4, 128, 32),
    (1, 256, 4, 256, 32),
    (1, 384, 4, 384, 32),
    (1, 384, 8, 384, 16),
)


def _attention_inputs(shape, seed):
    """Return grad-tracking ``(q, k, v, mask, bias)`` for one shape."""

    batch, outer, heads, seq, head_dim = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*size):
        return torch.randn(
            *size, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    q = randn(batch, outer, heads, seq, head_dim).requires_grad_(True)
    k = randn(batch, outer, heads, seq, head_dim).requires_grad_(True)
    v = randn(batch, outer, heads, seq, head_dim).requires_grad_(True)
    # Additive mask: 0 keeps a column, -inf drops it.  Masking the tail of the
    # sequence mirrors a padded crop.
    mask = torch.zeros(batch, outer, 1, 1, seq, device="cuda", dtype=torch.bfloat16)
    mask[..., seq - 3 :] = float("-inf")
    bias = randn(batch, 1, heads, seq, seq).requires_grad_(True)
    return q, k, v, mask, bias


def _run_attention(attention, inputs, weight):
    """Run one forward/backward and return ``(output, grads)`` detached."""

    q, k, v, mask, bias = inputs
    for tensor in (q, k, v, bias):
        tensor.grad = None
    out = attention(q, k, v, mask, bias)
    (out.float() * weight).sum().backward()
    grads = tuple(tensor.grad.detach().clone() for tensor in (q, k, v, bias))
    return out.detach().clone(), grads


@requires_both
class TriangleAttentionProviderEquivalenceTest(unittest.TestCase):
    """Forward and backward must agree bitwise across the two providers."""

    @classmethod
    def setUpClass(cls):
        from stablefold import (
            is_triangle_attention_supported,
            triangle_attention,
        )

        from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
            is_stablefold_supported,
            stablefold_triangle_attention,
        )

        # Held in a dict, not as class attributes: a plain function stored on a
        # class is a descriptor, so ``self.intree_attention`` would hand the
        # kernel a bound method and pass the test case itself as ``q``.
        cls.facades = {
            "intree": (is_stablefold_supported, stablefold_triangle_attention),
            "stablefold": (is_triangle_attention_supported, triangle_attention),
        }

    def test_support_verdicts_agree(self):
        for shape in ATTENTION_SHAPES:
            with self.subTest(shape=shape):
                inputs = _attention_inputs(shape, seed=17)
                intree = self.facades["intree"][0](*inputs)
                stablefold = self.facades["stablefold"][0](*inputs)
                self.assertEqual(intree[0], stablefold[0])
                self.assertTrue(intree[0], intree[1])

    def test_forward_and_gradients_are_bitwise_identical(self):
        for shape in ATTENTION_SHAPES:
            with self.subTest(shape=shape):
                inputs = _attention_inputs(shape, seed=101)
                weight = torch.randn(
                    inputs[0].shape[0],
                    inputs[0].shape[1],
                    inputs[0].shape[3],
                    inputs[0].shape[2],
                    inputs[0].shape[4],
                    generator=torch.Generator(device="cuda").manual_seed(5),
                    device="cuda",
                    dtype=torch.float32,
                )
                intree_out, intree_grads = _run_attention(
                    self.facades["intree"][1], inputs, weight
                )
                stablefold_out, stablefold_grads = _run_attention(
                    self.facades["stablefold"][1], inputs, weight
                )

                self.assertEqual(intree_out.dtype, stablefold_out.dtype)
                self.assertEqual(intree_out.shape, stablefold_out.shape)
                self.assertTrue(
                    _bitwise_equal(intree_out, stablefold_out),
                    f"forward differs at {shape}: max |delta| = "
                    f"{(intree_out.float() - stablefold_out.float()).abs().max()}",
                )
                for name, intree_grad, stablefold_grad in zip(
                    ("dq", "dk", "dv", "dbias"), intree_grads, stablefold_grads
                ):
                    self.assertTrue(
                        _bitwise_equal(intree_grad, stablefold_grad),
                        f"{name} differs at {shape}: max |delta| = "
                        f"{(intree_grad.float() - stablefold_grad.float()).abs().max()}",
                    )

    def test_each_provider_is_reproducible_against_itself(self):
        """A differing rerun would make the cross-provider check meaningless."""

        shape = ATTENTION_SHAPES[1]
        weight = torch.ones(
            shape[0],
            shape[1],
            shape[3],
            shape[2],
            shape[4],
            device="cuda",
            dtype=torch.float32,
        )
        for name, (_, attention) in self.facades.items():
            with self.subTest(provider=name):
                first = _run_attention(attention, _attention_inputs(shape, 7), weight)
                second = _run_attention(attention, _attention_inputs(shape, 7), weight)
                self.assertTrue(_bitwise_equal(first[0], second[0]))
                for left, right in zip(first[1], second[1]):
                    self.assertTrue(_bitwise_equal(left, right))


@requires_both
class TriMulProviderEquivalenceTest(unittest.TestCase):
    """The fused TriMul must also agree bitwise across the two providers."""

    C_Z = 128
    C_HIDDEN = 128
    SEQ = 256

    def _module(self):
        from openfold3.core.model.layers.triangular_multiplicative_update import (
            TriangleMultiplicationOutgoing,
        )

        torch.manual_seed(3)
        module = TriangleMultiplicationOutgoing(self.C_Z, self.C_HIDDEN)
        # The fused path takes a BF16 pair tensor against FP32 weights, which
        # is how the training stack invokes it; casting the module to BF16
        # would have the support check reject the call outright.
        return module.to(device="cuda", dtype=torch.float32).eval()

    def _inputs(self):
        generator = torch.Generator(device="cuda").manual_seed(23)
        z = torch.randn(
            1,
            self.SEQ,
            self.SEQ,
            self.C_Z,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_(True)
        mask = torch.ones(1, self.SEQ, self.SEQ, device="cuda", dtype=torch.bfloat16)
        return z, mask

    def _run(self, run_supported, module, z, mask, weight):
        z.grad = None
        for parameter in module.parameters():
            parameter.grad = None
        out = run_supported(module, z, mask)
        (out.float() * weight).sum().backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        return out.detach().clone(), z.grad.detach().clone(), grads

    def test_forward_and_gradients_are_bitwise_identical(self):
        from stablefold.trimul import _run_supported_trimul
        from stablefold.trimul_support import check_trimul_support

        from openfold3.core.kernels.cuda.stablefold_trimul import (
            _run_supported_stable_trimul,
        )
        from openfold3.core.kernels.cuda.stablefold_trimul_support import (
            check_stable_trimul_support,
        )

        module = self._module()
        z, mask = self._inputs()
        weight = torch.randn(
            z.shape,
            generator=torch.Generator(device="cuda").manual_seed(11),
            device="cuda",
            dtype=torch.float32,
        )

        intree_support = check_stable_trimul_support(module, z, mask)
        stablefold_support = check_trimul_support(module, z, mask)
        self.assertTrue(bool(intree_support), intree_support.reason)
        self.assertEqual(bool(intree_support), bool(stablefold_support))

        intree = self._run(_run_supported_stable_trimul, module, z, mask, weight)
        stablefold = self._run(_run_supported_trimul, module, z, mask, weight)

        self.assertTrue(
            _bitwise_equal(intree[0], stablefold[0]),
            "TriMul forward differs: max |delta| = "
            f"{(intree[0].float() - stablefold[0].float()).abs().max()}",
        )
        self.assertTrue(
            _bitwise_equal(intree[1], stablefold[1]),
            "TriMul dz differs: max |delta| = "
            f"{(intree[1].float() - stablefold[1].float()).abs().max()}",
        )
        self.assertEqual(sorted(intree[2]), sorted(stablefold[2]))
        for name in intree[2]:
            self.assertTrue(
                _bitwise_equal(intree[2][name], stablefold[2][name]),
                f"TriMul grad {name} differs: max |delta| = "
                f"{(intree[2][name].float() - stablefold[2][name].float()).abs().max()}",
            )


@unittest.skipUnless(HAS_STABLEFOLD and IS_HOPPER, "needs stablefold on a Hopper GPU")
class StableFoldAotProvenanceTest(unittest.TestCase):
    """The package must serve its own compiled artifacts, not a JIT rebuild.

    Development JIT may be enabled explicitly for the in-tree copy, but the
    package must still serve its wheel-owned extensions. Comparing two JIT
    builds would prove nothing about the installed wheel.
    """

    FACADES = (
        "stablefold.triangle_forward",
        "stablefold.triangle_delta",
        "stablefold.triangle_dkdv",
        "stablefold.triangle_dq_dbias",
        "stablefold.trimul_cuda",
    )

    def test_extensions_load_from_the_installed_package(self):
        import importlib
        import os

        import stablefold

        package_dir = os.path.dirname(os.path.realpath(stablefold.__file__))
        for name in self.FACADES:
            with self.subTest(facade=name):
                extension = importlib.import_module(name).load_extension()
                path = os.path.realpath(extension.__file__)
                self.assertTrue(
                    path.startswith(package_dir + os.sep),
                    f"{name} loaded from {path}, outside the installed package "
                    f"at {package_dir}; the AOT artifacts are not serving",
                )


class TensorBitwiseComparisonTest(unittest.TestCase):
    """Keep the GPU equivalence predicate stricter than numeric equality."""

    def test_signed_zero_difference_is_detected(self):
        self.assertFalse(_bitwise_equal(torch.tensor(0.0), torch.tensor(-0.0)))

    def test_nan_payloads_are_compared_exactly(self):
        first = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
        second = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
        self.assertTrue(_bitwise_equal(first, first.clone()))
        self.assertFalse(_bitwise_equal(first, second))


class ProviderResolutionTest(unittest.TestCase):
    """Provider selection is explicit, validated, and fails closed."""

    def setUp(self):
        from openfold3.core.kernels.cuda import provider

        self.provider = provider
        provider._reset_for_tests()
        self.addCleanup(provider._reset_for_tests)

    def _with_env(self, value):
        import os
        from unittest import mock

        environment = dict(os.environ)
        if value is None:
            environment.pop(self.provider.PROVIDER_ENV, None)
        else:
            environment[self.provider.PROVIDER_ENV] = value
        patcher = mock.patch.dict("os.environ", environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider._reset_for_tests()

    def test_explicit_intree_is_honoured(self):
        self._with_env("intree")
        self.assertEqual(self.provider.active_provider(), self.provider.INTREE)

    def test_unknown_provider_raises(self):
        self._with_env("cueq")
        with self.assertRaises(self.provider.ProviderError):
            self.provider.active_provider()

    def test_value_is_normalized(self):
        self._with_env("  InTree  ")
        self.assertEqual(self.provider.active_provider(), self.provider.INTREE)

    def test_default_stays_intree_even_when_package_is_available(self):
        self._with_env(None)
        with mock.patch.object(
            self.provider, "_stablefold_available", return_value=True
        ):
            self.assertEqual(self.provider.active_provider(), self.provider.INTREE)

    def test_explicit_missing_package_fails_closed(self):
        self._with_env("stablefold")
        with (
            mock.patch.object(
                self.provider, "_stablefold_available", return_value=False
            ),
            self.assertRaisesRegex(self.provider.ProviderError, "not importable"),
        ):
            self.provider.active_provider()

    def test_explicit_package_is_honoured(self):
        self._with_env("stablefold")
        with mock.patch.object(
            self.provider, "_stablefold_available", return_value=True
        ):
            self.assertEqual(self.provider.active_provider(), self.provider.STABLEFOLD)

    def test_failed_import_does_not_change_provider(self):
        self._with_env("intree")
        with (
            mock.patch.object(
                self.provider.importlib,
                "import_module",
                side_effect=ImportError("broken"),
            ),
            self.assertRaisesRegex(self.provider.ProviderError, "cannot import"),
        ):
            self.provider.triangle_attention_facade()
        self.assertEqual(self.provider.active_provider(), self.provider.INTREE)

    def test_facade_observes_instrumentation_without_reimporting(self):
        self._with_env("intree")
        module = SimpleNamespace(
            is_stablefold_supported=mock.sentinel.support,
            stablefold_triangle_attention=mock.sentinel.original,
        )
        with mock.patch.object(
            self.provider, "_import", return_value=module
        ) as importer:
            self.assertIs(
                self.provider.triangle_attention_facade()[1], mock.sentinel.original
            )
            module.stablefold_triangle_attention = mock.sentinel.instrumented
            self.assertIs(
                self.provider.triangle_attention_facade()[1], mock.sentinel.instrumented
            )
            importer.assert_called_once()

    @unittest.skipUnless(HAS_STABLEFOLD, "needs the stablefold package")
    def test_identity_reports_the_installed_package(self):
        self._with_env("stablefold")
        identity = self.provider.provider_identity()
        self.assertEqual(identity["provider"], "stablefold")
        self.assertIn("stablefold_version", identity)

    def test_resolution_is_memoized(self):
        self._with_env("intree")
        first = self.provider.active_provider()
        with mock.patch.dict("os.environ", {self.provider.PROVIDER_ENV: "stablefold"}):
            self.assertEqual(first, self.provider.active_provider())


if __name__ == "__main__":
    unittest.main()
