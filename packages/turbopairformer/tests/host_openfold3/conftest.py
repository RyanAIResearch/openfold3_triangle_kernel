"""Tests in this directory exercise TurboPairFormer through an installed OpenFold3.

They are skipped when OpenFold3 is not importable, so the standalone package
test suite stays self-contained.
"""

import pytest

pytest.importorskip("openfold3", reason="host tests need an OpenFold3 install")


@pytest.fixture(autouse=True)
def _isolate_stablefold_triangle_provider(monkeypatch):
    """Package integration tests must exercise the installed package provider."""
    from openfold3.core.kernels.cuda import provider

    monkeypatch.setenv(provider.PROVIDER_ENV, provider.TURBOPAIRFORMER)
    provider._reset_for_tests()
    try:
        yield
    finally:
        provider._reset_for_tests()
