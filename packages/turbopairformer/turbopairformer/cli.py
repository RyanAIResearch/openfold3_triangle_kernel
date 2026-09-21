# SPDX-License-Identifier: Apache-2.0

"""Concise operator help; uses only the Python standard library."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from turbopairformer import __version__

_PACKAGE = Path(__file__).resolve().parent
_OPERATOR_NAMES = {
    "attention": "attention",
    "turbo_attention": "attention",
    "TurboTriangleAttention": "attention",
    "triangle_attention": "attention",
    "trimul": "trimul",
    "turbo_trimul": "trimul",
    "TurboTriangleMultiplicative": "trimul",
    "run_trimul": "trimul",
}


def _operator_help() -> dict[str, str]:
    """Read the two shipped descriptions without importing the operators."""
    descriptions = {}
    guide = (_PACKAGE / "API_GUIDE.md").read_text(encoding="utf-8")
    for section in guide.split("\n## ")[1:]:
        heading, body = section.split("\n", 1)
        name, title = heading.split(" - ", 1)
        if name not in {"attention", "trimul"}:
            continue
        lines = [title]
        in_code = False
        for line in body.splitlines():
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                lines.append("  " + line if line else "")
            else:
                lines.append(line.replace("**", "").replace("`", ""))
        descriptions[name] = "\n".join(lines).strip()
    return descriptions


def main(argv: list[str] | None = None) -> int:
    """Print operator usage without initializing Torch or CUDA."""
    descriptions = _operator_help()
    combined = "\n\n".join(descriptions[name] for name in ("attention", "trimul"))
    parser = argparse.ArgumentParser(
        prog="turbopairformer",
        description=combined,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")
    api = subparsers.add_parser(
        "api",
        help="show Triangle Attention and TriMul usage",
        description="Usage for turbo_attention and turbo_trimul.",
        allow_abbrev=False,
    )
    selection = api.add_mutually_exclusive_group()
    selection.add_argument("operator", nargs="?", help="attention or trimul")
    selection.add_argument("--all", action="store_true", help="show both operators")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if not args.operator:
        print(combined)
        return 0
    name = args.operator.removeprefix("turbopairformer.")
    if name not in _OPERATOR_NAMES:
        api.error("choose attention or trimul")
    print(descriptions[_OPERATOR_NAMES[name]])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
