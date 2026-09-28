"""Render supported research status from saved records only."""

from __future__ import annotations

from pathlib import Path

from .records import GATES, GateRecord, read_gate


def render_report(results: Path, out: Path) -> None:
    gate_dir = results / "gates"
    records: dict[str, GateRecord] = {}
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.json")):
            record = read_gate(path)
            if path.stem.upper() != record.gate:
                raise ValueError("gate filename and record disagree")
            records[record.gate] = record
    identities = {
        (record.context.family_id, tuple(sorted(record.context.revisions.items())))
        for record in records.values() if record.context.family_id is not None
    }
    if len(identities) > 1:
        raise ValueError("gate records mix model identities")
    rows = []
    for gate in GATES:
        record = records.get(gate)
        decision = record.decision if record else "not_run"
        reason = record.reason.replace("|", "\\|").replace("\n", " ") if record else "No saved evidence"
        rows.append(f"| {gate} | {decision} | {reason} |")
    last = next((records[gate] for gate in reversed(GATES) if gate in records), None)
    if last is None:
        claim = "No research gate has been executed."
    elif last.decision == "fail":
        claim = f"The research stopped at {last.gate}: {last.reason}"
    elif last.decision == "blocked":
        claim = f"The research is blocked at {last.gate}: {last.reason}"
    elif last.gate == "G5" and last.decision == "pass":
        claim = "G5 passed according to the saved gate decision; inspect linked measurements before citing a result."
    else:
        claim = f"The latest saved gate is {last.gate} ({last.decision}); the research result is partial."
    lines = ["# FamilyTiles results", "", claim, "", "| Gate | Decision | Reason |",
             "| --- | --- | --- |", *rows, "",
             "Measurements and reproduction commands are added as the corresponding gates run.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
