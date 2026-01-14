from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import dumps as dumps_config
from .config import format as format_config
from .config import load


def _render(args: argparse.Namespace) -> int:
    cfg = load(
        args.paths,
        overrides=args.overrides or None,
        resolve_lazy=not args.no_resolve_lazy,
    )
    if args.no_pretty:
        output = format_config(cfg)
    else:
        output = format_config(cfg, format="pprint")
    sys.stdout.write(f"{output}\n")
    return 0


def _dump(args: argparse.Namespace) -> int:
    cfg = load(args.paths, overrides=args.overrides or None)
    sys.stdout.write(
        dumps_config(cfg, format=args.format, sort_keys=args.sort_keys)
    )
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="Config files to load.")
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="+",
        action="extend",
        default=[],
        metavar="OVERRIDE",
        help="Override values such as key=value.",
    )
def _add_render_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "render",
        aliases=["print"],
        help="Load configs and print the result.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--no-resolve-lazy",
        action="store_true",
        help="Print Lazies without resolving them.",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Print raw repr output instead of pprint formatting.",
    )
    parser.set_defaults(func=_render)


def _add_dump_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "dump",
        aliases=["freeze"],
        help="Load configs and print a snapshot.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--format",
        choices=("pprint", "ruff"),
        default=False,
        help="Optional formatter to apply.",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort mapping keys before formatting.",
    )
    parser.set_defaults(func=_dump)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfgx", description="Config loader utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_render_parser(subparsers)
    _add_dump_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
