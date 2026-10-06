#!/usr/bin/env python3
"""Launch a command with the pinned source package, without changing the Pixi lock."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = Path(__file__).with_name("turbopairformer_source.json")


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def read_pin(path: Path = PIN) -> dict:
    pin = json.loads(path.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]):
        raise RuntimeError(
            "The TurboPairFormer source revision must be a full Git SHA."
        )
    if pin["subdirectory"] != "packages/turbopairformer":
        raise RuntimeError("Unexpected TurboPairFormer source subdirectory.")
    return pin


def source_environment(site: Path, inherited: dict[str, str]) -> dict[str, str]:
    env = inherited.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(p) for p in (site, env.get("PYTHONPATH", "")) if p
    )
    # Keep a Python command's script directory from shadowing the verified overlay.
    env["PYTHONSAFEPATH"] = "1"
    env["TURBOPAIRFORMER_ALLOW_JIT"] = "1"
    env["TURBOPAIRFORMER_BUILD_AOT"] = "0"
    env["TURBOPAIRFORMER_SOURCE_ONLY"] = "1"
    # PyTorch's plain 9.0 target cannot compile this package's WGMMA instructions.
    env["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    # Preserve explicit SR/shadow settings: the package validates their policy.
    return env


def verify_site(checkout: Path, site: Path, pin: dict) -> dict[str, str]:
    if run("git", "rev-parse", "HEAD", cwd=checkout) != pin["revision"]:
        raise RuntimeError("Cached source checkout does not match the pinned revision.")
    prefix = pin["subdirectory"] + "/"
    tree = run("git", "ls-tree", "-r", "-z", "HEAD", "--", prefix, cwd=checkout)
    hashes = {}
    for entry in tree.split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        relative = name.removeprefix(prefix)
        if not relative.startswith("turbopairformer/"):
            continue
        target = site / relative
        if not target.is_file():
            raise RuntimeError(f"Pinned package file is missing: {target}")
        data = target.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != metadata.split()[2]:
            raise RuntimeError(f"Pinned package file was changed: {target}")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    if not hashes:
        raise RuntimeError("Pinned source has no package files.")
    package = site / "turbopairformer"
    actual = {
        str(path.relative_to(site))
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != set(hashes):
        raise RuntimeError(
            "Source overlay contains unexpected package files or AOT code."
        )
    manifest = json.loads((package / "_build_manifest.json").read_text())
    if manifest.get("built") is not False:
        raise RuntimeError("Current-source launcher requires the source-only manifest.")
    return hashes


def prepare_overlay(pin: dict, cache: Path) -> tuple[Path, dict[str, str]]:
    cache.mkdir(parents=True, exist_ok=True)
    # Multiple training processes may enter the same installer on a shared host.
    with (cache / ".install.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        checkout = cache / "checkout"
        site = cache / "site"
        if not checkout.exists():
            with tempfile.TemporaryDirectory(dir=cache, prefix="checkout-") as tmp:
                temp = Path(tmp)
                run("git", "init", "--quiet", str(temp))
                run("git", "remote", "add", "origin", pin["repository"], cwd=temp)
                run("git", "sparse-checkout", "set", pin["subdirectory"], cwd=temp)
                run(
                    "git",
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    pin["revision"],
                    cwd=temp,
                )
                run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=temp)
                temp.rename(checkout)
        if run("git", "rev-parse", "HEAD", cwd=checkout) != pin["revision"]:
            raise RuntimeError("Cached source checkout has the wrong revision.")
        run(
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            pin["subdirectory"],
            cwd=checkout,
        )
        if not site.exists():
            with tempfile.TemporaryDirectory(dir=cache, prefix="install-") as tmp:
                target = Path(tmp) / "site"
                env = source_environment(target, os.environ)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-build-isolation",
                        "--no-compile",
                        "--target",
                        str(target),
                        str(checkout / pin["subdirectory"]),
                    ],
                    env=env,
                    check=True,
                )
                verify_site(checkout, target, pin)
                target.rename(site)
        hashes = verify_site(checkout, site, pin)
        return site, hashes


def check_runtime() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("Use the Linux x86-64 openfold3 Pixi environment.")
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("The pinned source package requires Python 3.14.")
    import torch

    if torch.__version__.split("+")[0] != "2.10.0" or torch.version.cuda != "12.8":
        raise RuntimeError("Use the locked PyPI torch 2.10.0 / CUDA 12.8 environment.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify source without a GPU call"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.check and not args.command:
        parser.error("provide a command, or --check")
    check_runtime()
    pin = read_pin()
    cache = ROOT / "build" / "turbopairformer-source" / pin["revision"]
    site, hashes = prepare_overlay(pin, cache)
    env = source_environment(site, os.environ)
    # A fresh interpreter confirms import precedence and policy before the command.
    check = (
        "import pathlib, sys, turbopairformer; "
        "from turbopairformer.policy import validate_attention_policy; "
        "assert pathlib.Path(turbopairformer.__file__).resolve() == "
        "pathlib.Path(sys.argv[1]).resolve(); validate_attention_policy()"
    )
    subprocess.run(
        [sys.executable, "-c", check, str(site / "turbopairformer" / "__init__.py")],
        env=env,
        check=True,
    )
    receipt = {**pin, "package_path": str(site / "turbopairformer"), "sha256": hashes}
    (cache / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"TurboPairFormer source {pin['revision']} verified at {site}", flush=True)
    if not args.check:
        command = args.command[1:] if args.command[0] == "--" else args.command
        if not command:
            parser.error("provide a command after --")
        executable = shutil.which(command[0], path=env.get("PATH"))
        if executable is None:
            raise RuntimeError(f"Command not found: {command[0]}")
        os.execve(executable, command, env)


if __name__ == "__main__":
    main()
