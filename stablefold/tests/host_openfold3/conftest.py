"""Tests in this directory exercise StableFold through an installed OpenFold3.

They are skipped when OpenFold3 is not importable, so the standalone package
test suite stays self-contained.
"""

import pytest

pytest.importorskip("openfold3", reason="host tests need an OpenFold3 install")
