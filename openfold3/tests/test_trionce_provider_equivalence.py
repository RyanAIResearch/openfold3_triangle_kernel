# SPDX-License-Identifier: Apache-2.0

"""The ``trionce`` pip package must be indistinguishable from the in-tree copy.

``openfold3/core/kernels/cuda/provider.py`` lets the stable triangle routes be
served either by the installed ``trionce`` package or by the kernel copy
vendored under ``openfold3/core/kernels/cuda``.  Both providers build the same
``.cu`` sources, so switching between them must not move a single bit: these
tests compare forward outputs and every input gradient with ``torch.equal``.

The two implementations register their ``torch.library`` ops in different
namespaces (``trionce::`` and ``openfold3_stablefold::``), so a single process
can hold both and compare them directly.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import unittest

import torch


def _has_real_trionce() -> bool:
    """A bare ``trionce`` directory on sys.path imports as an empty namespace.

    Interpreters started in the repository root see the package's own project
    folder that way, so ``find_spec`` alone would report an install that has no
    modules in it.  A real distribution has a spec origin.
    """

    try:
        spec = importlib.util.find_spec("trionce")
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def _trionce_distribution_installed() -> bool:
    try:
        importlib.metadata.distribution("trionce")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


HAS_TRIONCE = _has_real_trionce()

# Skipping is the right answer when the package is genuinely absent.  It is the
# wrong answer when it is installed but shadowed: every equivalence test would
# skip, the run would report success, and nothing would have been compared.
if _trionce_distribution_installed() and not HAS_TRIONCE:
    raise RuntimeError(
        "the 'trionce' distribution is installed but 'import trionce' resolves "
        "to a namespace package -- the repository root is on sys.path and its "
        "trionce/ project folder shadows the install. Re-run with "
        "--import-mode=importlib, or from outside the checkout."
    )
IS_HOPPER = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
)

requires_both = unittest.skipUnless(
    HAS_TRIONCE and IS_HOPPER,
    "needs the trionce package and a Hopper sm_90a GPU",
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
    mask = torch.zeros(
        batch, outer, 1, 1, seq, device="cuda", dtype=torch.bfloat16
    )
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
        from openfold3.core.kernels.cuda.stablefold_triangle_attention import (
            is_stablefold_supported,
            stablefold_triangle_attention,
        )
        from trionce.triangle_attention import (
            is_triangle_attention_supported,
            triangle_attention,
        )

        # Held in a dict, not as class attributes: a plain function stored on a
        # class is a descriptor, so ``self.intree_attention`` would hand the
        # kernel a bound method and pass the test case itself as ``q``.
        cls.facades = {
            "intree": (is_stablefold_supported, stablefold_triangle_attention),
            "trionce": (is_triangle_attention_supported, triangle_attention),
        }

    def test_support_verdicts_agree(self):
        for shape in ATTENTION_SHAPES:
            with self.subTest(shape=shape):
                inputs = _attention_inputs(shape, seed=17)
                intree = self.facades["intree"][0](*inputs)
                trionce = self.facades["trionce"][0](*inputs)
                self.assertEqual(intree[0], trionce[0])
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
                trionce_out, trionce_grads = _run_attention(
                    self.facades["trionce"][1], inputs, weight
                )

                self.assertEqual(intree_out.dtype, trionce_out.dtype)
                self.assertEqual(intree_out.shape, trionce_out.shape)
                self.assertTrue(
                    torch.equal(intree_out, trionce_out),
                    f"forward differs at {shape}: max |delta| = "
                    f"{(intree_out.float() - trionce_out.float()).abs().max()}",
                )
                for name, intree_grad, trionce_grad in zip(
                    ("dq", "dk", "dv", "dbias"), intree_grads, trionce_grads
                ):
                    self.assertTrue(
                        torch.equal(intree_grad, trionce_grad),
                        f"{name} differs at {shape}: max |delta| = "
                        f"{(intree_grad.float() - trionce_grad.float()).abs().max()}",
                    )

    def test_each_provider_is_reproducible_against_itself(self):
        """A differing rerun would make the cross-provider check meaningless."""

        shape = ATTENTION_SHAPES[1]
        weight = torch.ones(
            shape[0], shape[1], shape[3], shape[2], shape[4],
            device="cuda", dtype=torch.float32,
        )
        for name, (_, attention) in self.facades.items():
            with self.subTest(provider=name):
                first = _run_attention(attention, _attention_inputs(shape, 7), weight)
                second = _run_attention(attention, _attention_inputs(shape, 7), weight)
                self.assertTrue(torch.equal(first[0], second[0]))
                for left, right in zip(first[1], second[1]):
                    self.assertTrue(torch.equal(left, right))


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
            1, self.SEQ, self.SEQ, self.C_Z,
            generator=generator, device="cuda", dtype=torch.bfloat16,
        ).requires_grad_(True)
        mask = torch.ones(
            1, self.SEQ, self.SEQ, device="cuda", dtype=torch.bfloat16
        )
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
        from openfold3.core.kernels.cuda.stablefold_trimul import (
            _run_supported_stable_trimul,
        )
        from openfold3.core.kernels.cuda.stablefold_trimul_support import (
            check_stable_trimul_support,
        )
        from trionce.trimul import _run_supported_trimul
        from trionce.trimul_support import check_trimul_support

        module = self._module()
        z, mask = self._inputs()
        weight = torch.randn(
            z.shape, generator=torch.Generator(device="cuda").manual_seed(11),
            device="cuda", dtype=torch.float32,
        )

        intree_support = check_stable_trimul_support(module, z, mask)
        trionce_support = check_trimul_support(module, z, mask)
        self.assertTrue(bool(intree_support), intree_support.reason)
        self.assertEqual(bool(intree_support), bool(trionce_support))

        intree = self._run(_run_supported_stable_trimul, module, z, mask, weight)
        trionce = self._run(_run_supported_trimul, module, z, mask, weight)

        self.assertTrue(
            torch.equal(intree[0], trionce[0]),
            "TriMul forward differs: max |delta| = "
            f"{(intree[0].float() - trionce[0].float()).abs().max()}",
        )
        self.assertTrue(
            torch.equal(intree[1], trionce[1]),
            "TriMul dz differs: max |delta| = "
            f"{(intree[1].float() - trionce[1].float()).abs().max()}",
        )
        self.assertEqual(sorted(intree[2]), sorted(trionce[2]))
        for name in intree[2]:
            self.assertTrue(
                torch.equal(intree[2][name], trionce[2][name]),
                f"TriMul grad {name} differs: max |delta| = "
                f"{(intree[2][name].float() - trionce[2][name].float()).abs().max()}",
            )


@unittest.skipUnless(HAS_TRIONCE and IS_HOPPER, "needs trionce on a Hopper GPU")
class TrionceAotProvenanceTest(unittest.TestCase):
    """The package must serve its own compiled artifacts, not a JIT rebuild.

    The equivalence tests run with the development JIT lane enabled so the
    in-tree copy can compile.  Were the package's AOT extensions to fail to
    load, trionce would quietly JIT the same sources and the comparison would
    pass while proving nothing about the installed package.
    """

    FACADES = (
        "trionce.triangle_forward",
        "trionce.triangle_delta",
        "trionce.triangle_dkdv",
        "trionce.triangle_dq_dbias",
        "trionce.trimul_cuda",
    )

    def test_extensions_load_from_the_installed_package(self):
        import importlib
        import os

        import trionce

        package_dir = os.path.dirname(os.path.realpath(trionce.__file__))
        for name in self.FACADES:
            with self.subTest(facade=name):
                extension = importlib.import_module(name).load_extension()
                path = os.path.realpath(extension.__file__)
                self.assertTrue(
                    path.startswith(package_dir + os.sep),
                    f"{name} loaded from {path}, outside the installed package "
                    f"at {package_dir}; the AOT artifacts are not serving",
                )


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

    def test_default_prefers_the_package_when_installed(self):
        self._with_env(None)
        expected = self.provider.TRIONCE if HAS_TRIONCE else self.provider.INTREE
        self.assertEqual(self.provider.active_provider(), expected)

    @unittest.skipUnless(HAS_TRIONCE, "needs the trionce package")
    def test_identity_reports_the_installed_package(self):
        self._with_env("trionce")
        identity = self.provider.provider_identity()
        self.assertEqual(identity["provider"], "trionce")
        self.assertIn("trionce_version", identity)

    def test_resolution_is_memoized(self):
        self._with_env("intree")
        first = self.provider.active_provider()
        self.assertEqual(first, self.provider.active_provider())


if __name__ == "__main__":
    unittest.main()
