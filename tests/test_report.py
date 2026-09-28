from familytiles.records import GateRecord, RunContext, write_json_atomic
from familytiles.report import render_report


def test_report_can_finish_after_g1_failure(tmp_path):
    results = tmp_path / "results"
    ctx = RunContext("qwen-small", {"anchor": "a", "target": "b"},
                     "policy", None, "code", "machine", None)
    write_json_atomic(
        results / "gates" / "g1.json",
        GateRecord(1, "G1", ctx, {}, {"min_saving": 0.15},
                   {"saving": 0.04}, "fail", "real samples too large").to_dict(),
    )
    out = tmp_path / "RESULTS.md"
    render_report(results, out)
    text = out.read_text()
    assert "G1" in text and "fail" in text
    assert "G5" in text and "not_run" in text
    assert "real samples too large" in text
    assert "positive" not in text.lower()
    assert "tokens/s" not in text


def test_report_rejects_conflicting_gate_identity(tmp_path):
    results = tmp_path / "results"
    ctx = RunContext("f", {"anchor": "a", "target": "b"},
                     "p", None, "c", "e", None)
    changed = RunContext("f", {"anchor": "a", "target": "changed"},
                         "p", None, "c", "e", None)
    for gate, context in (("G0", ctx), ("G1", changed)):
        write_json_atomic(results / "gates" / f"{gate}.json",
                          GateRecord(1, gate, context, {}, {}, {},
                                     "pass", "recorded").to_dict())
    import pytest

    with pytest.raises(ValueError):
        render_report(results, tmp_path / "RESULTS.md")
