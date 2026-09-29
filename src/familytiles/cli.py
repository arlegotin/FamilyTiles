"""Small explicit command interface for the gated experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .records import publish_gate, resolve_under_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="familytiles",
        description="Research archive; no validated compressed-model inference command",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Render supported status from saved records")
    report.add_argument("--results", required=True)
    report.add_argument("--out", required=True)
    doctor_parser = commands.add_parser("doctor", help="Run the G0 machine and Metal smoke")
    doctor_parser.add_argument("--out", required=True)
    survey_parser = commands.add_parser("survey", help="Run the bounded real-family G1 screen")
    survey_parser.add_argument("--families", required=True)
    survey_parser.add_argument("--sample-mib-per-model", required=True, type=int)
    survey_parser.add_argument("--seed", required=True, type=int)
    survey_parser.add_argument("--out", required=True)
    convert_parser = commands.add_parser("convert", help="Build the selected full-family artifact")
    convert_parser.add_argument("--family", required=True)
    convert_parser.add_argument("--policy", required=True)
    convert_parser.add_argument("--out", required=True)
    verify_parser = commands.add_parser("verify", help="Verify a complete family artifact")
    verify_parser.add_argument("--artifact", required=True)
    verify_parser.add_argument("--device", choices=("cpu", "gpu"))
    verify_parser.add_argument("--scope", choices=("model",))
    bench_parser = commands.add_parser("bench", help="Run a gated benchmark suite")
    bench_parser.add_argument("--artifact", required=True)
    bench_parser.add_argument("--suite", choices=("kernel",), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "doctor":
        from .measure import doctor, evaluate_g0
        import json

        try:
            output = resolve_under_root(root, args.out)
            environment = doctor(output)
            audit_path = root / "results/prior-art.json"
            audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
            gate = evaluate_g0(environment, audit)
            publish_gate(root / "results/gates", gate)
            return {"pass": 0, "fail": 3, "blocked": 4}[gate.decision]
        except (OSError, ValueError) as exc:
            print(f"doctor: {exc}", file=sys.stderr)
            return 2
    if args.command == "survey":
        from .survey import survey
        try:
            record = survey(resolve_under_root(root, args.families),
                            args.sample_mib_per_model, args.seed,
                            resolve_under_root(root, args.out))
            return {"pass": 0, "fail": 3, "blocked": 4}[record["decision"]]
        except (OSError, ValueError) as exc:
            print(f"survey: {exc}", file=sys.stderr)
            return 2
    if args.command == "convert":
        from .convert import convert_family
        try:
            convert_family(resolve_under_root(root, args.family),
                           resolve_under_root(root, args.policy),
                           resolve_under_root(root, args.out))
            return 0
        except (OSError, ValueError) as exc:
            print(f"convert: {exc}", file=sys.stderr)
            return 2
    if args.command == "verify" and args.device == "cpu" and args.scope is None:
        from .convert import load_artifact, publish_g2, verify_cpu
        try:
            artifact = load_artifact(resolve_under_root(root, args.artifact))
            gate = publish_g2(artifact, verify_cpu(artifact))
            return 0 if gate.decision == "pass" else 3
        except (OSError, ValueError) as exc:
            print(f"verify: {exc}", file=sys.stderr)
            return 2
    if args.command == "verify" and args.device == "gpu" and args.scope is None:
        from .convert import load_artifact
        from .metal import verify_gpu
        from .records import read_gate
        try:
            artifact = load_artifact(resolve_under_root(root, args.artifact))
            gate = read_gate(root / "results/gates/G2.json")
            if gate.decision != "pass" or gate.context.artifact_hash != artifact.manifest_hash:
                raise ValueError("G2 CPU gate does not match this artifact")
            result = verify_gpu(artifact)
            return 0 if result["representation_exact"] else 3
        except (OSError, ValueError) as exc:
            print(f"verify: {exc}", file=sys.stderr)
            return 2
    if args.command == "bench" and args.suite == "kernel":
        from .measure import run_kernel_benchmark
        try:
            gate = run_kernel_benchmark(resolve_under_root(root, args.artifact))
            return {"pass": 0, "fail": 3, "blocked": 4}[gate.decision]
        except (OSError, ValueError) as exc:
            print(f"bench: {exc}", file=sys.stderr)
            return 2
    if args.command != "report":
        print(f"{args.command} is not implemented at the current gate", file=sys.stderr)
        return 2
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
