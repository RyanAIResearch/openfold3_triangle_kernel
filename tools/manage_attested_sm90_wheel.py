#!/usr/bin/env python3
"""Install, validate, or maintain the attested StableFold H100 wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "dist"
PREBUILT_COMMIT = "ce15d03b0a3b8793b6cf7e40a03c9475e432d6ca"
PREBUILT_ARTIFACT_DIR = ROOT / "release_artifacts" / "stablefold-h100-ce15d03b"
PREBUILT_FILE_SHA256 = {
    "RECEIPT.json": (
        "e85451dbffe40dd9f23b81f064982cc3d6bab8d8068f64b8804a13ae0b7629e1"
    ),
    "SHA256SUMS": ("71618162eb488451a01c1aaa8dd58bc4b63b411f354be4d0ff278697fe0ddcad"),
    "aot_clean_install_smoke.json": (
        "2ead03061b1b0a99df50d5ee4e5a7333395b841406b826d7681ae7f546e3bd20"
    ),
    (
        "openfold3-0.4.3.dev5512+gce15d03b0-cp314-cp314-linux_x86_64.whl"
    ): "e3e6b80e08aa16e0580b2809a54523458ac6503ce185ea276e05179ae8a27d19",
    (
        "openfold3-0.4.3.dev5512+gce15d03b0-cp314-cp314-"
        "linux_x86_64.whl.attestation.json"
    ): "e5c7d468d1887f1bca56e7e59af204ef5e0c0ef2f96cb27e017830dd6570be0f",
    (
        "openfold3-0.4.3.dev5512+gce15d03b0-cp314-cp314-linux_x86_64.whl.sha256"
    ): "ba20026faa198d54b9c1a9f1958661685bad6a18e3afd592dc7f1a836eb5da95",
}
MANIFEST_MEMBER = "openfold3/core/kernels/cuda/_build_manifest.json"
MODULE_KEYS = (
    "stablefold_forward",
    "stablefold_delta",
    "stablefold_dkdv",
    "stablefold_dq_dbias",
    "stablefold_trimul",
)


class ArtifactError(RuntimeError):
    """Raised when a StableFold wheel or installation fails its contract."""


@dataclass(frozen=True)
class ArtifactBundle:
    """The exact four runtime files emitted by the canonical AOT builder."""

    wheel: Path
    checksum: Path
    attestation: Path
    smoke: Path
    wheel_sha256: str
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    installed_distribution: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(ROOT), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_commit() -> str:
    """Return the exact clean source commit that artifacts must attest."""

    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ArtifactError("StableFold AOT artifacts require a clean Git checkout")
    return _git("rev-parse", "HEAD")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _wheel_distribution_identity(archive: zipfile.ZipFile) -> dict[str, Any]:
    """Return the installed-package identity implied by a wheel RECORD."""

    record_members = [
        name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
    ]
    if len(record_members) != 1:
        raise ArtifactError("wheel does not contain one unique RECORD")
    record_bytes = archive.read(record_members[0])
    verified_members: list[tuple[str, bytes]] = []
    for relative, encoded_hash, raw_size in csv.reader(
        record_bytes.decode().splitlines()
    ):
        if not relative.startswith("openfold3/") or not encoded_hash:
            continue
        try:
            mode, expected_encoded = encoded_hash.split("=", 1)
            data = archive.read(relative)
            digest = hashlib.new(mode, data).digest()
        except (KeyError, ValueError) as error:
            raise ArtifactError(f"invalid wheel RECORD member: {relative}") from error
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if encoded != expected_encoded:
            raise ArtifactError(f"wheel RECORD hash mismatch: {relative}")
        if raw_size and len(data) != int(raw_size):
            raise ArtifactError(f"wheel RECORD size mismatch: {relative}")
        verified_members.append((relative, digest))
    verified_files = len(verified_members)
    if verified_files < 1:
        raise ArtifactError("wheel RECORD has no hashed OpenFold3 package members")
    tree_hasher = hashlib.sha256()
    for relative, digest in sorted(verified_members):
        relative_bytes = relative.encode()
        tree_hasher.update(len(relative_bytes).to_bytes(8, "big"))
        tree_hasher.update(relative_bytes)
        tree_hasher.update(digest)
    return {
        "verified_package_files": verified_files,
        "installed_package_tree_sha256": tree_hasher.hexdigest(),
        "runtime_from_verified_distribution": True,
    }


def discover_bundle(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    expected_commit: str | None = None,
    verify_checkout_sources: bool = True,
) -> ArtifactBundle:
    """Discover and fail-closed validate one canonical artifact bundle."""

    artifact_dir = artifact_dir.resolve()
    wheels = sorted(artifact_dir.glob("openfold3-*.whl"))
    if len(wheels) != 1:
        raise ArtifactError(
            f"expected exactly one wheel in {artifact_dir}, found {len(wheels)}"
        )
    wheel = wheels[0]
    checksum = wheel.with_suffix(wheel.suffix + ".sha256")
    attestation = wheel.with_suffix(wheel.suffix + ".attestation.json")
    smoke = artifact_dir / "aot_clean_install_smoke.json"
    missing = [path for path in (checksum, attestation, smoke) if not path.is_file()]
    if missing:
        raise ArtifactError(f"incomplete AOT artifact bundle: {missing}")

    wheel_sha256 = sha256_file(wheel)
    expected_sidecar = f"{wheel_sha256}  {wheel.name}\n"
    if checksum.read_text() != expected_sidecar:
        raise ArtifactError("wheel SHA-256 sidecar does not match the wheel")

    receipt = _read_json(attestation)
    smoke_receipt = _read_json(smoke)
    if smoke_receipt.get("verdict") != "PASS":
        raise ArtifactError("canonical clean-install AOT smoke did not pass")
    if receipt.get("wheel") != wheel.name:
        raise ArtifactError("attestation names a different wheel")
    if receipt.get("wheel_sha256") != wheel_sha256:
        raise ArtifactError("attestation wheel digest does not match")

    try:
        with zipfile.ZipFile(wheel) as archive:
            manifest_bytes = archive.read(MANIFEST_MEMBER)
            manifest = json.loads(manifest_bytes)
            installed_distribution = _wheel_distribution_identity(archive)
            members = set(archive.namelist())
            extensions = manifest["extensions"]
            if set(extensions) != set(MODULE_KEYS):
                raise ArtifactError("wheel does not contain exactly five AOT modules")
            for key, extension in extensions.items():
                filename = extension["filename"]
                if filename not in members:
                    raise ArtifactError(f"wheel is missing AOT member {filename}")
                digest = hashlib.sha256(archive.read(filename)).hexdigest()
                if digest != extension["sha256"]:
                    raise ArtifactError(f"AOT member hash mismatch: {key}")
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise ArtifactError("wheel contains an invalid AOT manifest") from error

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if receipt.get("embedded_manifest_sha256") != manifest_sha256:
        raise ArtifactError("attestation manifest digest does not match")
    if manifest.get("schema_version") != 1 or manifest.get("built") is not True:
        raise ArtifactError("wheel does not contain a built AOT manifest")
    target = manifest.get("target")
    if target != {
        "cuda_arch": "sm_90a",
        "ptx_embedded": False,
        "torch_cuda_arch_list": "9.0a",
    }:
        raise ArtifactError("wheel is not the sm_90a SASS-only artifact")
    abi = manifest.get("abi", {})
    if not str(abi.get("python_soabi", "")).startswith("cpython-314"):
        raise ArtifactError("wheel does not target CPython 3.14")
    if str(abi.get("torch_version", "")).split("+", 1)[0] != "2.10.0":
        raise ArtifactError("wheel does not target Torch 2.10.0")
    if abi.get("torch_cuda_version") != "12.9":
        raise ArtifactError("wheel does not target Torch CUDA 12.9")
    git = manifest.get("git", {})
    if git.get("dirty") is not False:
        raise ArtifactError("wheel was not built from a clean checkout")
    if expected_commit is not None and git.get("commit") != expected_commit:
        raise ArtifactError(
            f"wheel commit {git.get('commit')} does not match {expected_commit}"
        )
    if receipt.get("git") != git or receipt.get("abi") != abi:
        raise ArtifactError("attestation disagrees with the embedded manifest")
    if receipt.get("target") != target:
        raise ArtifactError("attestation target disagrees with the manifest")
    if receipt.get("extension_sha256") != {
        key: value["sha256"] for key, value in extensions.items()
    }:
        raise ArtifactError("attestation extension digests do not match")

    if verify_checkout_sources:
        for relative, expected_sha256 in manifest.get("sources", {}).items():
            source = ROOT / relative
            if not source.is_file() or sha256_file(source) != expected_sha256:
                raise ArtifactError(f"packaged source receipt mismatch: {relative}")

    bundle = ArtifactBundle(
        wheel=wheel,
        checksum=checksum,
        attestation=attestation,
        smoke=smoke,
        wheel_sha256=wheel_sha256,
        manifest=manifest,
        receipt=receipt,
        installed_distribution=installed_distribution,
    )
    smoke_identity = smoke_receipt.get("aot_identity")
    if not isinstance(smoke_identity, dict) or _canonical_aot_identity(
        smoke_identity
    ) != _canonical_aot_identity(_expected_aot_identity(bundle)):
        raise ArtifactError("clean-install smoke names a different AOT artifact")
    smoke_distribution = smoke_receipt.get("installed_distribution", {})
    if any(
        smoke_distribution.get(key) != value
        for key, value in installed_distribution.items()
    ):
        raise ArtifactError("clean-install smoke names a different wheel RECORD")
    return bundle


def prebuilt_bundle() -> ArtifactBundle:
    """Return the repository-pinned, qualified H100 runtime bundle."""

    if not PREBUILT_ARTIFACT_DIR.is_dir():
        raise ArtifactError(
            f"prebuilt StableFold bundle is missing: {PREBUILT_ARTIFACT_DIR}"
        )
    actual_names = {
        path.name for path in PREBUILT_ARTIFACT_DIR.iterdir() if path.is_file()
    }
    if actual_names != set(PREBUILT_FILE_SHA256):
        raise ArtifactError(
            "prebuilt StableFold bundle has missing or unexpected files: "
            f"{sorted(actual_names)}"
        )
    for name, expected_sha256 in PREBUILT_FILE_SHA256.items():
        path = PREBUILT_ARTIFACT_DIR / name
        if sha256_file(path) != expected_sha256:
            raise ArtifactError(f"prebuilt StableFold artifact digest mismatch: {name}")

    receipt = _read_json(PREBUILT_ARTIFACT_DIR / "RECEIPT.json")
    if receipt.get("schema_version") != 1:
        raise ArtifactError("unsupported prebuilt StableFold receipt schema")
    if receipt.get("status") != "PASS_SINGLE_H100_PRODUCTION_CANDIDATE":
        raise ArtifactError("prebuilt StableFold artifact is not qualified")
    if receipt.get("git_commit") != PREBUILT_COMMIT:
        raise ArtifactError("prebuilt StableFold receipt names another commit")
    wheel_receipt = receipt.get("wheel", {})
    if wheel_receipt.get("name") not in PREBUILT_FILE_SHA256 or wheel_receipt.get(
        "sha256"
    ) != PREBUILT_FILE_SHA256.get(wheel_receipt.get("name")):
        raise ArtifactError("prebuilt StableFold receipt names another wheel")
    return discover_bundle(
        PREBUILT_ARTIFACT_DIR,
        expected_commit=PREBUILT_COMMIT,
        verify_checkout_sources=False,
    )


def _expected_aot_identity(bundle: ArtifactBundle) -> dict[str, Any]:
    extensions = bundle.manifest["extensions"]
    return {
        "schema_version": 1,
        "manifest_sha256": bundle.receipt["embedded_manifest_sha256"],
        "git": {"commit": bundle.manifest["git"]["commit"], "dirty": False},
        "target": bundle.manifest["target"],
        "abi": bundle.manifest["abi"],
        "packaged_source_sha256": {
            path: digest
            for path, digest in bundle.manifest["sources"].items()
            if path.startswith("openfold3/")
        },
        "extensions": {
            key: {
                "filename": Path(value["filename"]).name,
                "module": value["module"],
                "sha256": value["sha256"],
                "symbols": value["symbols"],
            }
            for key, value in extensions.items()
        },
    }


def _canonical_aot_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Normalize order-insensitive AOT metadata before identity comparison."""

    canonical = json.loads(json.dumps(identity))
    for extension in canonical.get("extensions", {}).values():
        symbols = extension.get("symbols")
        if isinstance(symbols, list):
            extension["symbols"] = sorted(symbols)
    return canonical


def _verification_program(bundle: ArtifactBundle, expected_commit: str) -> str:
    expected_aot = json.dumps(
        _canonical_aot_identity(_expected_aot_identity(bundle)), sort_keys=True
    )
    expected_distribution = json.dumps(bundle.installed_distribution, sort_keys=True)
    return f"""
import json
import os
import sys
from pathlib import Path

if os.environ.get('OPENFOLD3_TRIANGLE_ALLOW_JIT') not in (None, '0'):
    raise SystemExit('production verification forbids StableFold JIT')
import openfold3
module = Path(openfold3.__file__).resolve()
prefix = Path(sys.prefix).resolve()
if not module.is_relative_to(prefix):
    raise SystemExit(f'OpenFold3 imported outside the Pixi prefix: {{module}}')
from openfold3.core.utils.distributed_contract import installed_distribution_identity
distribution = installed_distribution_identity()
expected_distribution = json.loads({expected_distribution!r})
for key, expected in expected_distribution.items():
    if distribution.get(key) != expected:
        raise SystemExit(
            f'installed distribution mismatch for {{key}}: '
            f'{{distribution.get(key)}} != {{expected}}'
        )
from openfold3.core.kernels.cuda._aot_runtime import distributed_aot_identity
identity = distributed_aot_identity({MODULE_KEYS!r})
expected_aot = json.loads({expected_aot!r})
for extension in identity.get('extensions', {{}}).values():
    symbols = extension.get('symbols')
    if isinstance(symbols, list):
        extension['symbols'] = sorted(symbols)
if identity != expected_aot:
    raise SystemExit('installed AOT identity differs from the selected artifact bundle')
if identity['git'] != {{'commit': {expected_commit!r}, 'dirty': False}}:
    raise SystemExit('installed AOT commit differs from the clean checkout')
print(json.dumps({{
    'module': str(module),
    'distribution': distribution,
    'identity': identity,
}}, sort_keys=True))
"""


def verify_installed(
    bundle: ArtifactBundle,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify the installed wheel from a fresh process outside the source tree."""

    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENFOLD3_TRIANGLE_ALLOW_JIT": "0",
        }
    )
    with tempfile.TemporaryDirectory(prefix="stablefold-wheel-verify-") as directory:
        result = subprocess.run(
            (sys.executable, "-c", _verification_program(bundle, expected_commit)),
            cwd=directory,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ArtifactError(
            "installed-wheel verification returned invalid JSON"
        ) from error


def install_bundle(bundle: ArtifactBundle, expected_commit: str) -> dict[str, Any]:
    """Install one exact wheel and verify the resulting runtime."""

    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(bundle.wheel),
        ),
        check=True,
    )
    return verify_installed(bundle, expected_commit)


def ensure_bundle(bundle: ArtifactBundle, expected_commit: str) -> dict[str, Any]:
    """Return a valid runtime, reinstalling the exact wheel when necessary."""

    try:
        return verify_installed(bundle, expected_commit)
    except ArtifactError, subprocess.CalledProcessError:
        return install_bundle(bundle, expected_commit)


def build_bundle() -> None:
    """Run the one canonical five-extension builder."""

    environment = os.environ.copy()
    conda_prefix = environment.get("CONDA_PREFIX")
    if not conda_prefix:
        raise ArtifactError("build-install must run inside the H100 Pixi environment")
    environment.setdefault("CUDA_HOME", conda_prefix)
    environment.setdefault("CUDACXX", str(Path(conda_prefix) / "bin" / "nvcc"))
    subprocess.run(
        ("bash", str(ROOT / "tools" / "build_sm90_wheel.sh")),
        cwd=ROOT,
        env=environment,
        check=True,
    )


def _bundle_files(bundle: ArtifactBundle) -> tuple[Path, ...]:
    return (bundle.wheel, bundle.checksum, bundle.attestation, bundle.smoke)


def _copy_bundle(
    bundle: ArtifactBundle,
    destination: Path,
    expected_commit: str,
) -> ArtifactBundle:
    """Publish a fully validated bundle without leaving mixed artifacts."""

    destination = destination.resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ArtifactError(
                f"artifact destination is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            existing = discover_bundle(destination, expected_commit=expected_commit)
            source_hashes = {
                path.name: sha256_file(path) for path in _bundle_files(bundle)
            }
            existing_hashes = {
                path.name: sha256_file(path) for path in _bundle_files(existing)
            }
            if source_hashes != existing_hashes:
                raise ArtifactError(
                    f"refusing to replace a different artifact bundle: {destination}"
                )
            return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        for source in _bundle_files(bundle):
            shutil.copy2(source, staging / source.name)
        discover_bundle(staging, expected_commit=expected_commit)
        if destination.exists() and (
            not destination.is_dir() or any(destination.iterdir())
        ):
            raise ArtifactError(
                f"artifact destination changed while publishing: {destination}"
            )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return discover_bundle(destination, expected_commit=expected_commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "install-prebuilt",
            "verify-prebuilt",
            "build-install",
            "install",
            "ensure",
            "verify",
            "check",
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    checkout_commit = source_commit()
    is_prebuilt = args.action in ("install-prebuilt", "verify-prebuilt")
    expected_commit = PREBUILT_COMMIT if is_prebuilt else checkout_commit
    if args.action == "build-install":
        build_bundle()
    if is_prebuilt:
        if args.artifact_dir is not None or args.copy_to is not None:
            raise ArtifactError(
                "prebuilt actions use only the repository-pinned artifact bundle"
            )
        bundle = prebuilt_bundle()
    else:
        artifact_dir = args.artifact_dir or DEFAULT_ARTIFACT_DIR
        bundle = discover_bundle(artifact_dir, expected_commit=expected_commit)
    if args.copy_to is not None:
        bundle = _copy_bundle(bundle, args.copy_to, expected_commit)

    result: dict[str, Any] = {
        "action": args.action,
        "checkout_commit": checkout_commit,
        "artifact_commit": expected_commit,
        "wheel": str(bundle.wheel),
        "wheel_sha256": bundle.wheel_sha256,
    }
    if args.action in ("install-prebuilt", "build-install", "install"):
        result["runtime"] = install_bundle(bundle, expected_commit)
    elif args.action == "ensure":
        result["runtime"] = ensure_bundle(bundle, expected_commit)
    elif args.action in ("verify-prebuilt", "verify"):
        result["runtime"] = verify_installed(bundle, expected_commit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
