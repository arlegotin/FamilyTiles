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


def test_g3_early_stop_report_uses_saved_evidence(tmp_path):
    results = tmp_path / "results"
    ctx = RunContext("qwen-pair", {"anchor": "a" * 40, "target": "b" * 40},
                     "policy", "artifact", "code", "machine", "inputs")
    for gate, decision in (("G0", "pass"), ("G1", "pass"),
                           ("G2", "pass"), ("G3", "fail")):
        observed = ({"ledger": {"family_bytes": 80, "w0_bytes": 100,
                                 "b2_bytes": 100, "b3_bytes": 70,
                                 "b4_compressed_anchor_bytes": 60,
                                 "family_saving_vs_b1": 0.20,
                                 "family_beyond_b2_raw": 0.20},
                     "audit": {"verified_target_words": 1000,
                               "mismatched_words": 0}} if gate == "G2" else
                    {"pair_ratio_vs_best_native": 2.5,
                     "median_sweep_ns_by_mode": {"family_pair": 250,
                                                  "native_batched": 100,
                                                  "family_two_singles": 300},
                     "c1_zero_mismatches": True,
                     "c2_bitwise_equal": True} if gate == "G3" else {})
        write_json_atomic(results / "gates" / f"{gate}.json",
                          GateRecord(1, gate, ctx, {}, {}, observed, decision,
                                     "primitive too slow" if gate == "G3" else "passed").to_dict())
    write_json_atomic(results / "conversion/ledger.json",
                      {"family_bytes": 80, "w0_bytes": 100, "b2_bytes": 100,
                       "b3_bytes": 70, "b4_compressed_anchor_bytes": 60,
                       "family_saving_vs_b1": 0.20, "family_beyond_b2_raw": 0.20})
    write_json_atomic(results / "kernel/summary.json",
                      {"gate": GateRecord(1, "G3", ctx, {}, {},
                                          {"pair_ratio_vs_best_native": 2.5,
                                           "median_sweep_ns_by_mode": {"family_pair": 250,
                                                                        "native_batched": 100,
                                                                        "family_two_singles": 300},
                                           "c1_zero_mismatches": True,
                                           "c2_bitwise_equal": True},
                                          "fail", "primitive too slow").to_dict()})
    write_json_atomic(results / "correctness/single-model-comparison.json",
                      {"artifact_hash": "artifact", "bit_mismatches": 0,
                       "logit_words": 12, "two_model_loaded_physical_footprint_saving": 0.19,
                       "two_model_weight_allocation_saving": 0.20,
                       "physical_footprints_are_phase_samples_not_instantaneous_peaks": True})
    out = tmp_path / "RESULTS.md"
    render_report(results, out)
    body = out.read_text()
    assert "20.00%" in body and "2.50×" in body
    assert "19.00%" in body and "phase samples" in body
    assert "G4 | not_run" in body and "G5 | not_run" in body
    assert "full inference result remains unproven" in body
    assert "results/conversion/ledger.json" in body
    assert "results/gates/G3.json" in body
