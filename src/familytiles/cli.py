"""Small explicit command interface for the gated experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .records import resolve_under_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="familytiles", description="Gated exact-inference research")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Render supported status from saved records")
    report.add_argument("--results", required=True)
    report.add_argument("--out", required=True)
    for name in ("doctor", "survey", "convert", "verify", "bench"):
        commands.add_parser(name, help="Available after its prerequisite implementation gate")
    args = parser.parse_args(argv)
    if args.command != "report":
        print(f"{args.command} is not implemented at the current gate", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[2]
    try:
        from .report import render_report

        render_report(resolve_under_root(root, args.results),
                      resolve_under_root(root, args.out))
    except (OSError, ValueError) as exc:
        print(f"report: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
