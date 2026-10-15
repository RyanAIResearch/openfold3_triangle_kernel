# SPDX-License-Identifier: Apache-2.0

"""Qualify an installed current-namespace wheel on one H100, with JIT disabled.

Reuse the frozen validator's inputs, FP64 oracle, stock reference, and error
envelope. All nine Attention cases use the current SR + shadow facade. Compare
default flags, explicit flags, and replay bitwise. TriMul checks only the fixed
FP32 weight-gradient fold; full-operator accuracy and training are outside scope.

Run this file directly with the wheel installed non-editably. Keep the original
aot_numeric_smoke.py beside it: its SHA-256 is checked before importing helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path

ORACLE_PATH = Path(__file__).with_name("aot_numeric_smoke.py")
ORACLE_SHA256 = "c33ad3e3dbe3f4b881db132240ba673d52976b9719f1a4de185271e439f56067"
MANIFEST_MEMBER = "openfold3/core/kernels/turbopairformer/_build_manifest.json"
MODULE_KEYS = (
    "turbopairformer_forward",
    "turbopairformer_delta",
    "turbopairformer_dkdv",
    "turbopairformer_dq_dbias",
    "turbopairformer_trimul",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_oracle(path: Path = ORACLE_PATH):
    """Import only after checking the original, immutable helper implementation."""
    if sha256_file(path) != ORACLE_SHA256:
        raise RuntimeError("frozen numerical oracle SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("_current_wheel_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen numerical oracle")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return oracle


def verify_wheel_installation(
    wheel: Path, expected_commit: str, identity: dict, install_root: Path
) -> dict[str, str]:
    """Bind the supplied wheel to the actual installed Python and binary files."""
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise RuntimeError("expected commit must be a full lowercase Git SHA")
    install_root = install_root.resolve()
    package_hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("wheel contains duplicate members")
        manifest_bytes = archive.read(MANIFEST_MEMBER)
        manifest = json.loads(manifest_bytes)
        expected_git = {"commit": expected_commit, "dirty": False}
        if manifest.get("git") != expected_git or identity.get("git") != expected_git:
            raise RuntimeError("wheel and loaded binaries must match the clean commit")
        if (
            identity.get("manifest_sha256")
            != hashlib.sha256(manifest_bytes).hexdigest()
        ):
            raise RuntimeError("loaded binaries do not belong to the selected wheel")
        for member in names:
            if not member.startswith("openfold3/") or member.endswith("/"):
                continue
            installed = (install_root / member).resolve()
            if not installed.is_relative_to(install_root / "openfold3"):
                raise RuntimeError(f"wheel member escapes the package: {member}")
            digest = hashlib.sha256(archive.read(member)).hexdigest()
            if not installed.is_file() or sha256_file(installed) != digest:
                raise RuntimeError(
                    f"installed file differs from selected wheel: {member}"
                )
            package_hashes[member] = digest
    return package_hashes


def verify_imported_modules(install_root: Path, package_hashes: dict[str, str]) -> None:
    """Reject mixed imports even when another valid distribution is installed."""
    install_root = install_root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "openfold3" and not name.startswith("openfold3."):
            continue
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RuntimeError(f"imported module has no verifiable file: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(install_root):
            raise RuntimeError(f"module imported outside the selected wheel: {name}")
        member = path.relative_to(install_root).as_posix()
        if member not in package_hashes or sha256_file(path) != package_hashes[member]:
            raise RuntimeError(f"imported module differs from selected wheel: {name}")


def run_attention(inputs):
    from openfold3.core.kernels.turbopairformer import (
        turbopairformer_triangle_attention as attention,
    )

    q, k, v, mask, bias, dout = inputs
    q, k, v, bias = (t.detach().clone().requires_grad_(True) for t in (q, k, v, bias))
    out = attention.turbopairformer_triangle_attention(q, k, v, mask, bias)
    out.backward(dout)
    return out.detach(), q.grad, k.grad, v.grad, bias.grad


def check_error_envelope(error: float, stock_error: float) -> float:
    """Keep the historical gate exactly; NaN and infinity never pass."""
    limit = 1.1 * stock_error + 1e-6
    if (
        not all(math.isfinite(value) and value >= 0 for value in (error, stock_error))
        or not math.isfinite(limit)
        or error > limit
    ):
        raise RuntimeError(f"relative L2 {error} exceeds the fixed FP64 gate {limit}")
    return limit


def attention_checks(oracle) -> list[dict]:
    torch = oracle.torch
    receipts = []
    for index, (_, _, shape) in enumerate(oracle.CASES, 1):
        # Preserve all nine specifications, including the two formerly classic
        # cases. Their historical flags are deliberately not applied here.
        name = f"sr-{index}-N{shape['N']}-D{shape['D']}"
        torch.manual_seed(0)
        inputs = oracle._case(bias_dtype=torch.bfloat16, **shape)
        reference, stock = oracle._oracle(*inputs), oracle._stock(*inputs)
        os.environ.pop("OPENFOLD3_SR_BACKWARD", None)
        os.environ.pop("OPENFOLD3_SR_SHADOW", None)
        actual = run_attention(inputs)
        os.environ.update(oracle.PROD_FLAGS)
        explicit = run_attention(inputs)
        replay = run_attention(inputs)
        if not all(
            oracle._bits_equal(a, b) and oracle._bits_equal(b, c)
            for a, b, c in zip(actual, explicit, replay, strict=True)
        ):
            raise RuntimeError(f"{name}: default, explicit, or replay results differ")
        if not all(torch.isfinite(value).all().item() for value in actual):
            raise RuntimeError(f"{name}: non-finite output or gradients")
        full = shape["full_planes"]
        if full:
            if any(value[:, :full].abs().max().item() != 0 for value in actual[:4]):
                raise RuntimeError(f"{name}: fully masked planes must be zero")
            # The oracle uses a finite mask sentinel. Exclude its fully masked
            # output rows, while still checking every gradient against it.
            reference = (reference[0][:, full:], *reference[1:])
            stock = (stock[0][:, full:], *stock[1:])
            actual = (actual[0][:, full:], *actual[1:])
        errors = {}
        for label, value, baseline, expected in zip(
            ("out", "dq", "dk", "dv", "dbias"), actual, stock, reference, strict=True
        ):
            error, baseline_error = (
                oracle._rel(value, expected),
                oracle._rel(baseline, expected),
            )
            try:
                limit = check_error_envelope(error, baseline_error)
            except RuntimeError as error_message:
                raise RuntimeError(
                    f"{name} {label}: {error_message}"
                ) from error_message
            errors[label] = {
                "relative_l2": error,
                "stock_relative_l2": baseline_error,
                "limit": limit,
            }
        receipts.append(
            {
                "name": name,
                "shape": shape,
                "flags": dict(oracle.PROD_FLAGS),
                "status": "PASS",
                "bitwise_replay": True,
                "finite": True,
                "default_matches_explicit": True,
                "fully_masked_zero": True if full else None,
                "errors": errors,
            }
        )
    return receipts


def trimul_fold_checks(oracle) -> list[dict]:
    from openfold3.core.kernels.turbopairformer import turbopairformer_trimul_cuda

    torch = oracle.torch
    extension = turbopairformer_trimul_cuda.load_extension()
    receipts = []
    for channels in (32, 64, 96, 128):
        torch.manual_seed(9187 + channels)
        partials = torch.randn(
            17, 5 * channels, channels, device="cuda", dtype=torch.float32
        )
        expected = torch.zeros_like(partials[0])
        for chunk in partials.unbind(0):
            expected.add_(chunk)
        for _ in range(2):
            actual = torch.empty_like(expected)
            returned = extension.trimul_wgrad_fixed_fold(partials, actual, True)
            if returned.data_ptr() != actual.data_ptr() or not oracle._bits_equal(
                actual, expected
            ):
                raise RuntimeError(
                    f"C={channels}: TriMul FP32 fold differs from reference"
                )
        receipts.append(
            {
                "channels": channels,
                "status": "PASS",
                "bitwise_equal": True,
                "bitwise_replay": True,
            }
        )
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    validator_hash = sha256_file(Path(__file__))
    # Reject a reused receipt path so a failed rerun cannot leave an old PASS.
    if args.output_json.exists():
        raise RuntimeError("output receipt already exists; choose a fresh path")
    if os.environ.get("OPENFOLD3_TRIANGLE_ALLOW_JIT") not in (None, "0"):
        raise RuntimeError("numeric qualification forbids JIT")
    # Torch is loaded after argument parsing so --help works without a GPU stack.
    oracle = load_frozen_oracle()
    torch = oracle.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("numeric qualification requires exactly one visible GPU")
    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    if capability != (9, 0) or "H100" not in device_name:
        raise RuntimeError("this qualification requires an H100 with CC 9.0")
    if torch.__version__.split("+", 1)[0] != "2.10.0" or torch.version.cuda != "12.9":
        raise RuntimeError("numeric qualification requires Torch 2.10 / CUDA 12.9")

    import openfold3
    from openfold3.core.kernels.turbopairformer._aot_runtime import (
        distributed_aot_identity,
    )
    from openfold3.core.utils.distributed_contract import (
        installed_distribution_identity,
    )

    install_root = Path(openfold3.__file__).resolve().parent.parent
    if install_root == Path(__file__).resolve().parents[1]:
        raise RuntimeError("numeric qualification must use an installed wheel")
    distribution = installed_distribution_identity()
    identity = distributed_aot_identity(MODULE_KEYS)
    wheel_hash = sha256_file(args.wheel)
    package_hashes = verify_wheel_installation(
        args.wheel, args.expected_commit, identity, install_root
    )
    verify_imported_modules(install_root, package_hashes)

    attention = attention_checks(oracle)
    folds = trimul_fold_checks(oracle)
    torch.cuda.synchronize()
    if (
        distribution != installed_distribution_identity()
        or identity != distributed_aot_identity(MODULE_KEYS)
    ):
        raise RuntimeError("installed wheel identity changed during numerical checks")
    verify_imported_modules(install_root, package_hashes)
    if wheel_hash != sha256_file(args.wheel):
        raise RuntimeError("selected wheel changed during numerical checks")
    if (
        validator_hash != sha256_file(Path(__file__))
        or sha256_file(ORACLE_PATH) != ORACLE_SHA256
    ):
        raise RuntimeError("numerical validation scripts changed during checks")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": (
            "Current SR Attention FP64 envelope and TriMul FP32 weight-gradient fold"
        ),
        "validator": {
            "path": "tools/aot_current_numeric_smoke.py",
            "sha256": validator_hash,
        },
        "oracle": {"path": "tools/aot_numeric_smoke.py", "sha256": ORACLE_SHA256},
        "git_commit": args.expected_commit,
        "wheel_sha256": wheel_hash,
        "device": device_name,
        "compute_capability": list(capability),
        "gpu_count": 1,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "aot_identity": identity,
        "installed_distribution": distribution,
        "wheel_package_files_verified": len(package_hashes),
        "attention": attention,
        "trimul_fp32_fold": folds,
        "limitations": [
            "Same-process replay of the tested cases only; "
            "not a cross-device determinism proof",
            "No multi-GPU, full-model training, or performance qualification",
            "TriMul reference covers FP32 weight-gradient folding, "
            "not full-operator accuracy",
        ],
    }
    with args.output_json.open("x") as handle:
        handle.write(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "attention_cases": len(attention),
                "trimul_fold_cases": len(folds),
            }
        )
    )


if __name__ == "__main__":
    main()
