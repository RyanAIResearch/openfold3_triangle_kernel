# SPDX-License-Identifier: Apache-2.0

"""Offline API help using only the standard library; never imports a kernel."""

from __future__ import annotations

import argparse
import ast
import copy
import sys
from pathlib import Path

from turbopairformer import __version__

_PACKAGE = Path(__file__).resolve().parent
_TOPICS = {
    "attention": "Triangle Attention: inputs, masks, output layout and autograd",
    "trimul": "TriMul: functional weights, directions and module integration",
    "support": "Support checks, metadata records and host fallback routing",
    "distributed": "Optional Lightning DDP kernel preflight and identity helpers",
    "environment": "Execution flags, defaults, aliases and package version",
    "internals": "Source-derived signatures for internal Python APIs and loaders",
}
_SYMBOL_TOPICS = {
    "turbo_attention": "attention",
    "TurboTriangleAttention": "attention",
    "triangle_attention": "attention",
    "turbo_trimul": "trimul",
    "TurboTriangleMultiplicative": "trimul",
    "run_trimul": "trimul",
    "TRIMUL_STOCK_FALLBACK_THRESHOLD": "support",
    "SupportResult": "support",
    "TrimulSupportMetadata": "support",
    "check_trimul_support": "support",
    "evaluate_trimul_support": "support",
    "is_triangle_attention_supported": "support",
    "triangle_attention_invocation_count": "support",
    "DistributedContractError": "distributed",
    "distributed_execution_flag": "environment",
    "distributed_strict_enabled": "environment",
    "__version__": "environment",
}


def _catalog() -> str:
    topics = "\n".join(f"  {name:14} {summary}" for name, summary in _TOPICS.items())
    symbols = "\n".join(
        f"  {name:38} -> {topic}" for name, topic in _SYMBOL_TOPICS.items()
    )
    return (
        f"API topics:\n{topics}\n\n"
        f"Top-level Python API (from turbopairformer import ...):\n{symbols}\n\n"
        "Use turbopairformer api <topic-or-symbol> for contracts and examples.\n"
        "Use turbopairformer api internals for module APIs, or look up a symbol\n"
        "such as trimul_support.check_trimul_tensor_support or\n"
        "triangle_dq_dbias.workspace_plan directly. Fully qualified\n"
        "turbopairformer.<module>.<symbol> names also work.\n"
        "Use turbopairformer api --all to print the entire offline reference."
    )


def _guide_topics() -> dict[str, str]:
    """Split the shipped Markdown guide at its six level-two topic headings."""
    sections = {}
    for section in (
        (_PACKAGE / "API_GUIDE.md").read_text(encoding="utf-8").split("\n## ")[1:]
    ):
        name = section.split("\n", 1)[0].split(" - ", 1)[0]
        sections[name] = "## " + section.rstrip()
    return sections


def _source_entries() -> dict[str, ast.AST]:
    """Read signatures without executing modules, decorators or default values."""
    entries = {}
    reexports = {}
    for path in sorted(_PACKAGE.glob("*.py")):
        if path.stem in {"__init__", "__main__", "cli"}:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                entries[f"{path.stem}.{node.name}"] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                entries[f"{path.stem}.{node.target.id}"] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        entries[f"{path.stem}.{target.id}"] = node
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("turbopairformer.")
            ):
                module = node.module.removeprefix("turbopairformer.")
                for alias in node.names:
                    reexports[f"{path.stem}.{alias.asname or alias.name}"] = (
                        f"{module}.{alias.name}"
                    )
    for name, original in reexports.items():
        if original in entries:
            entries[name] = entries[original]
    return entries


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = copy.deepcopy(node.args)
    # Raw sentinel names are useful to maintainers; their meaning is explained
    # in the TriMul topic. Never evaluate defaults to pretty-print a signature.
    result = f"{node.name}({ast.unparse(arguments)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def _source_reference(name: str, node: ast.AST) -> str:
    parts = [f"### turbopairformer.{name}"]
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        lines = ["@" + ast.unparse(item) for item in node.decorator_list]
        lines.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
        header_lines = len(lines)
        for member in node.body:
            if isinstance(member, ast.AnnAssign):
                lines.append("    " + ast.unparse(member))
            elif isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                not member.name.startswith("_")
                or member.name in {"__init__", "__call__", "__bool__"}
            ):
                lines.extend(
                    "    @" + ast.unparse(item) for item in member.decorator_list
                )
                lines.append("    def " + _signature(member) + ": ...")
        if len(lines) == header_lines:
            lines.append("    ...")
        parts.append("```python\n" + "\n".join(lines) + "\n```")
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts.append("```python\n" + _signature(node) + "\n```")
    else:
        parts.append("```python\n" + ast.unparse(node) + "\n```")
        parts.append("Source declaration; displayed without evaluating its value.")
        return "\n\n".join(parts)
    parts.append(
        ast.get_docstring(node)
        or "Internal helper; see the module source for its implementation contract."
    )
    return "\n\n".join(parts)


def _internal_reference(entries: dict[str, ast.AST]) -> str:
    return "\n\n".join(
        _source_reference(name, node)
        for name, node in entries.items()
        if not name.rsplit(".", 1)[1].startswith("_")
    )


def main(argv: list[str] | None = None) -> int:
    """Print help; return zero on success, argparse exits with 2 on usage errors."""
    parser = argparse.ArgumentParser(
        prog="turbopairformer",
        description=(
            "TurboPairFormer offline API guide for Triangle Attention and\n"
            "Triangle Multiplicative Update (PyTorch autograd).\n"
            "Help requires no Torch import, CUDA device, compiler or network."
        ),
        epilog=(
            "Start here:\n"
            "  turbopairformer api turbo_attention   Shapes, masks and backward\n"
            "  turbopairformer api turbo_trimul      Weights and both directions\n"
            "  turbopairformer api support           Support checks and routing\n"
            "  turbopairformer api distributed       Optional Lightning DDP preflight\n"
            "  turbopairformer api environment       Execution flags and defaults\n"
            "  turbopairformer api internals         Low-level Python API reference\n"
            "  turbopairformer api --all             Complete offline manual\n\n"
            "Python: from turbopairformer import turbo_attention, turbo_trimul\n"
            "Equivalent command: python -m turbopairformer --help\n"
            "GPU calls require the supported Hopper wheel runtime; see api environment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")
    api = subparsers.add_parser(
        "api",
        help="browse every package API, contract and usage example",
        description=_catalog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    selection = api.add_mutually_exclusive_group()
    selection.add_argument(
        "topic", nargs="?", help="topic, exported name, or module.symbol"
    )
    selection.add_argument(
        "--all", action="store_true", help="print every topic and module API"
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if not args.topic and not args.all:
        print(_catalog())
        return 0
    topics = _guide_topics()
    entries = _source_entries()
    if args.all:
        print("# TurboPairFormer API reference\n\n" + _catalog())
        print("\n\n" + "\n\n".join(topics[name] for name in _TOPICS))
        print("\n\n" + _internal_reference(entries))
        return 0
    name = args.topic.removeprefix("turbopairformer.")
    topic = _SYMBOL_TOPICS.get(name, name)
    if topic in topics:
        print(topics[topic])
        if topic == "internals":
            print("\n\n" + _internal_reference(entries))
        return 0
    if name not in entries:
        matches = [key for key in entries if key.rsplit(".", 1)[1] == name]
        if len(matches) == 1:
            name = matches[0]
    if name in entries:
        print(_source_reference(name, entries[name]))
        print("\nSource reference from this package; see api internals for ABI limits.")
        return 0
    api.error(
        f"unknown API topic or symbol {args.topic!r}; choose "
        + ", ".join(_TOPICS)
        + ", a name from 'turbopairformer api', or a module.symbol from 'api internals'"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
