"""Command-line interface for JTF.

Usage::

    jtf encode input.json [-o output.jtf] [--tiktoken]
    jtf decode input.jtf  [-o output.json]
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import encode, decode
from .cost import heuristic_cost, tiktoken_cost


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, text: str) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")


def cmd_encode(args) -> int:
    data = json.loads(_read(args.input))
    cost_fn = tiktoken_cost if args.tiktoken else heuristic_cost
    _write(args.output, encode(data, cost_fn=cost_fn))
    return 0


def cmd_decode(args) -> int:
    data = decode(_read(args.input))
    _write(args.output, json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="jtf",
        description="JTF — lossless JSON <-> compact token-efficient format",
    )
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("encode", help="JSON -> JTF")
    e.add_argument("input", help="input JSON file, or '-' for stdin")
    e.add_argument("-o", "--output", help="output file (default: stdout)")
    e.add_argument(
        "--tiktoken",
        action="store_true",
        help="use tiktoken (cl100k_base) for the value-dictionary break-even "
        "analysis instead of the deterministic heuristic",
    )
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("decode", help="JTF -> JSON")
    d.add_argument("input", help="input JTF file, or '-' for stdin")
    d.add_argument("-o", "--output", help="output file (default: stdout)")
    d.set_defaults(func=cmd_decode)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
