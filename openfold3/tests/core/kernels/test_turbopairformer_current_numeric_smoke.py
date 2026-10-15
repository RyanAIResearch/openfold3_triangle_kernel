"""CPU-only checks for the installed-wheel numerical qualification boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[4]
TOOL = REPOSITORY / "tools/aot_current_numeric_smoke.py"
SPEC = importlib.util.spec_from_file_location("current_numeric_smoke", TOOL)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

COMMIT = "a" * 40
MANIFEST = "openfold3/core/kernels/turbopairformer/_build_manifest.json"
PYTHON_MEMBER = "openfold3/core/kernels/turbopairformer/loader.py"
BINARY_MEMBER = "openfold3/core/kernels/turbopairformer/_ext/kernel.so"


class CurrentNumericWheelIdentityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.install_root = self.directory / "installed"
        self.wheel = self.directory / "candidate.whl"
        self.manifest = {"git": {"commit": COMMIT, "dirty": False}}
        self.files = {
            "openfold3/__init__.py": b"# package\n",
            PYTHON_MEMBER: b"VALUE = 1\n",
            BINARY_MEMBER: b"synthetic binary bytes\n",
            MANIFEST: json.dumps(self.manifest).encode(),
        }
        self.identity = {
            "git": {"commit": COMMIT, "dirty": False},
            "manifest_sha256": hashlib.sha256(self.files[MANIFEST]).hexdigest(),
        }
        for member, content in self.files.items():
            destination = self.install_root / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        self.write_wheel()

    def write_wheel(self, extra_members=()):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for member, content in self.files.items():
                archive.writestr(member, content)
            for member, content in extra_members:
                archive.writestr(member, content)

    def verify(self):
        return VALIDATOR.verify_wheel_installation(
            self.wheel, COMMIT, self.identity, self.install_root
        )

    def test_matches_every_installed_package_file(self):
        # Distribution metadata is outside the package identity checked here.
        self.write_wheel([("openfold3-0.0.0.dist-info/METADATA", b"Name: openfold3")])
        expected = {
            member: hashlib.sha256(content).hexdigest()
            for member, content in self.files.items()
        }
        self.assertEqual(self.verify(), expected)

    def test_unchanged_manifest_does_not_hide_changed_python(self):
        (self.install_root / PYTHON_MEMBER).write_bytes(b"VALUE = 2\n")
        with self.assertRaisesRegex(RuntimeError, "installed file differs"):
            self.verify()

    def test_missing_installed_binary_fails(self):
        (self.install_root / BINARY_MEMBER).unlink()
        with self.assertRaisesRegex(RuntimeError, "installed file differs"):
            self.verify()

    def test_wrong_manifest_identity_fails(self):
        self.identity["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "selected wheel"):
            self.verify()

    def test_wrong_commit_or_dirty_build_fails(self):
        for location in ("wheel", "loaded binaries"):
            for invalid_git in (
                {"commit": "b" * 40, "dirty": False},
                {"commit": COMMIT, "dirty": True},
            ):
                with self.subTest(location=location, git=invalid_git):
                    self.identity["git"] = {"commit": COMMIT, "dirty": False}
                    manifest = {"git": {"commit": COMMIT, "dirty": False}}
                    if location == "wheel":
                        manifest["git"] = invalid_git
                    else:
                        self.identity["git"] = invalid_git
                    self.files[MANIFEST] = json.dumps(manifest).encode()
                    (self.install_root / MANIFEST).write_bytes(self.files[MANIFEST])
                    self.identity["manifest_sha256"] = hashlib.sha256(
                        self.files[MANIFEST]
                    ).hexdigest()
                    self.write_wheel()
                    with self.assertRaisesRegex(RuntimeError, "clean commit"):
                        self.verify()

    def test_commit_must_be_full_lowercase_sha(self):
        for invalid_commit in ("a9dea1e5", "A" * 40, "g" * 40, ""):
            with (
                self.subTest(commit=invalid_commit),
                self.assertRaisesRegex(RuntimeError, "full lowercase Git SHA"),
            ):
                VALIDATOR.verify_wheel_installation(
                    self.wheel, invalid_commit, self.identity, self.install_root
                )

    def test_duplicate_archive_member_fails(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.write_wheel([(PYTHON_MEMBER, self.files[PYTHON_MEMBER])])
        with self.assertRaisesRegex(RuntimeError, "duplicate members"):
            self.verify()

    def test_archive_member_cannot_escape_package(self):
        self.write_wheel([("openfold3/../outside.py", b"VALUE = 3\n")])
        with self.assertRaisesRegex(RuntimeError, "escapes the package"):
            self.verify()

    def test_installed_symlink_cannot_escape_package(self):
        outside = self.directory / "outside.py"
        outside.write_bytes(self.files[PYTHON_MEMBER])
        installed = self.install_root / PYTHON_MEMBER
        installed.unlink()
        installed.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes the package"):
            self.verify()

    def test_imported_module_must_match_verified_install(self):
        hashes = self.verify()
        imported = {
            "openfold3": SimpleNamespace(
                __file__=str(self.install_root / "openfold3/__init__.py")
            ),
            "openfold3.loader": SimpleNamespace(
                __file__=str(self.install_root / PYTHON_MEMBER)
            ),
            "openfold3.absent": None,
            "unrelated": SimpleNamespace(__file__="/outside/unrelated.py"),
        }
        with patch.object(VALIDATOR.sys, "modules", imported):
            VALIDATOR.verify_imported_modules(self.install_root, hashes)

    def test_imported_module_without_file_origin_fails(self):
        hashes = self.verify()
        imported = {"openfold3.loader": SimpleNamespace()}
        with (
            patch.object(VALIDATOR.sys, "modules", imported),
            self.assertRaisesRegex(RuntimeError, "no verifiable file"),
        ):
            VALIDATOR.verify_imported_modules(self.install_root, hashes)

    def test_imported_module_outside_install_fails(self):
        hashes = self.verify()
        outside = self.directory / "other_install/loader.py"
        outside.parent.mkdir()
        outside.write_bytes(self.files[PYTHON_MEMBER])
        imported = {"openfold3.loader": SimpleNamespace(__file__=str(outside))}
        with (
            patch.object(VALIDATOR.sys, "modules", imported),
            self.assertRaisesRegex(RuntimeError, "outside the selected wheel"),
        ):
            VALIDATOR.verify_imported_modules(self.install_root, hashes)

    def test_imported_module_changed_after_verification_fails(self):
        hashes = self.verify()
        installed = self.install_root / PYTHON_MEMBER
        installed.write_bytes(b"VALUE = 4\n")
        imported = {"openfold3.loader": SimpleNamespace(__file__=str(installed))}
        with (
            patch.object(VALIDATOR.sys, "modules", imported),
            self.assertRaisesRegex(RuntimeError, "differs from selected wheel"),
        ):
            VALIDATOR.verify_imported_modules(self.install_root, hashes)

    def test_imported_module_absent_from_wheel_fails(self):
        hashes = self.verify()
        installed = self.install_root / "openfold3/unlisted.py"
        installed.write_bytes(b"VALUE = 5\n")
        imported = {"openfold3.unlisted": SimpleNamespace(__file__=str(installed))}
        with (
            patch.object(VALIDATOR.sys, "modules", imported),
            self.assertRaisesRegex(RuntimeError, "differs from selected wheel"),
        ):
            VALIDATOR.verify_imported_modules(self.install_root, hashes)


class CurrentNumericErrorGateTest(unittest.TestCase):
    def test_exact_historical_boundary_passes(self):
        for stock_error in (0.0, 0.001, 1.0):
            with self.subTest(stock_error=stock_error):
                limit = 1.1 * stock_error + 1e-6
                self.assertEqual(
                    VALIDATOR.check_error_envelope(limit, stock_error), limit
                )
                with self.assertRaisesRegex(RuntimeError, "fixed FP64 gate"):
                    VALIDATOR.check_error_envelope(
                        math.nextafter(limit, math.inf), stock_error
                    )

    def test_nonfinite_or_negative_error_never_passes(self):
        for invalid in (math.nan, math.inf, -math.inf, -0.001):
            for error, stock_error in ((invalid, 0.1), (0.1, invalid)):
                with (
                    self.subTest(error=error, stock_error=stock_error),
                    self.assertRaisesRegex(RuntimeError, "fixed FP64 gate"),
                ):
                    VALIDATOR.check_error_envelope(error, stock_error)

    def test_overflowed_limit_fails(self):
        with self.assertRaisesRegex(RuntimeError, "fixed FP64 gate"):
            VALIDATOR.check_error_envelope(0.0, sys.float_info.max)

    def test_tampered_oracle_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            oracle = Path(directory) / "tampered_oracle.py"
            oracle.write_text("raise AssertionError('must not execute')\n")
            with self.assertRaisesRegex(RuntimeError, "oracle SHA-256 mismatch"):
                VALIDATOR.load_frozen_oracle(oracle)


class CurrentNumericCommandLineTest(unittest.TestCase):
    def run_cli(self, arguments, directory, *, jit="0"):
        environment = dict(os.environ, OPENFOLD3_TRIANGLE_ALLOW_JIT=jit)
        # -I -S removes site-packages and PYTHONPATH, including any Torch install.
        return subprocess.run(
            [sys.executable, "-I", "-S", str(TOOL), *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_help_does_not_require_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(["--help"], directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-commit", result.stdout)
        self.assertIn("--output-json", result.stdout)

    def test_jit_opt_in_is_rejected_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh-receipt.json"
            result = self.run_cli(
                [
                    "--wheel",
                    "absent.whl",
                    "--expected-commit",
                    COMMIT,
                    "--output-json",
                    str(output),
                ],
                directory,
                jit="1",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("numeric qualification forbids JIT", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse(output.exists())

    def test_reused_receipt_is_refused_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing-receipt.json"
            original = b'{"status": "PASS", "previous_run": true}\n'
            output.write_bytes(original)
            result = self.run_cli(
                [
                    "--wheel",
                    "absent.whl",
                    "--expected-commit",
                    COMMIT,
                    "--output-json",
                    str(output),
                ],
                directory,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output receipt already exists", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
