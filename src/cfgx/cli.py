from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import _format_snapshot, format as format_config, load


def _render(args: argparse.Namespace) -> int:
    cfg = load(args.paths, overrides=args.overrides or None)
    sys.stdout.write(f"{format_config(cfg, formatter=args.formatter)}\n")
    return 0


def _dump(args: argparse.Namespace) -> int:
    cfg = load(args.paths, overrides=args.overrides or None)
    sys.stdout.write(_format_snapshot(cfg, formatter=args.formatter))
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
    parser.add_argument(
        "--formatter",
        choices=("auto", "ruff", "pprint"),
        default="auto",
        help="Output formatter.",
    )


def _add_render_parser(subparsers) -> None:
    parser = subparsers.add_parser("render", help="Load configs and print the result.")
    _add_common_args(parser)
    parser.set_defaults(func=_render)


def _add_dump_parser(subparsers) -> None:
    parser = subparsers.add_parser("dump", help="Load configs and print a snapshot.")
    _add_common_args(parser)
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
