import os

import pytest

from familytiles import measure


def test_timing_evaluates_new_work_before_stop():
    events = []
    sequence = iter([object(), object()])

    def step():
        events.append("step")
        return next(sequence)

    def evaluate(value):
        events.append("evaluate")

    def synchronize():
        events.append("synchronize")

    assert measure.time_evaluated(step, synchronize, evaluate=evaluate) > 0
    assert measure.time_evaluated(step, synchronize, evaluate=evaluate) > 0
    assert events == ["step", "evaluate", "synchronize"] * 2


def test_timeout_is_censored_and_kills_child():
    spec = measure.CellSpec(suite="_test", mode="sleep", requested_model_tokens=30,
                            timeout_seconds=0.25, extras={"delay_seconds": 0.05})
    row = measure.run_cell(spec)
    assert row["status"] == "timeout"
    assert row["completion_ns"] is None
    assert row["completed_model_tokens"] < spec.requested_model_tokens
    assert row["child_reaped"] is True


def test_budget_rechecked_before_launch(monkeypatch):
    class TinyMemory:
        total = 2**30
        available = 2**20

    monkeypatch.setattr(measure.psutil, "virtual_memory", lambda: TinyMemory())
    spec = measure.CellSpec(suite="_test", mode="sleep", requested_model_tokens=1,
                            estimated_peak_bytes=100 * 2**20)
    row = measure.run_cell(spec)
    assert row["status"] == "budget_blocked"
    assert row["pid"] is None


def test_memory_categories_are_not_added(monkeypatch):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return type("Info", (), {"rss": 1234})()

    monkeypatch.setattr(measure.psutil, "Process", FakeProcess)
    monkeypatch.setattr(measure, "read_footprint", lambda pid: 2345)
    sample = measure.sample_memory(os.getpid(), "decode")
    assert sample["rss_bytes"] == 1234
    assert sample["physical_footprint_bytes"] == 2345
    assert "total_bytes" not in sample


def test_missing_footprint_cannot_pass_physical_claim():
    assert not measure.physical_claim_verified({"physical_footprint_bytes": None})
    assert measure.physical_claim_verified({"physical_footprint_bytes": 100})


def test_summarize_trials_uses_trial_summaries_not_pooled_tokens():
    rows = [
        {"trial": 0, "status": "complete", "completion_ns": 100, "completed_model_tokens": 10,
         "latency_ns": [1] * 100},
        {"trial": 1, "status": "complete", "completion_ns": 200, "completed_model_tokens": 10,
         "latency_ns": [1000]},
        {"trial": 2, "status": "complete", "completion_ns": 300, "completed_model_tokens": 10,
         "latency_ns": [1000]},
    ]
    result = measure.summarize_trials(rows)
    assert result["median_completion_ns"] == 200
    assert result["median_trial_p95_latency_ns"] == 1000


def _kernel_rows(family_ns=135, native_two_ns=100, native_batch_ns=90,
                 source_hash="kernel-v1"):
    rows = []
    for trial in range(3):
        for mode, duration in (("family_pair", family_ns),
                               ("native_two", native_two_ns),
                               ("native_batched", native_batch_ns),
                               ("raw_pair", native_two_ns + 5),
                               ("family_two_singles", family_ns + 20)):
            rows.append({"trial": trial, "mode": mode, "status": "complete",
                         "completion_ns": duration, "completed_model_tokens": 42,
                         "artifact_hash": "artifact-1", "input_hash": "inputs-1",
                         "source_hash": source_hash, "inventory_hash": "inventory-1",
                         "matrix_times_ns": [1, duration - 1],
                         "peak_active_bytes": 1000, "peak_extra_bytes": 1000,
                         "decoded_matrix_bytes": 0})
    return rows


def _kernel_correctness(source_hash="kernel-v1", layout_changes=0):
    return {"artifact_hash": "artifact-1", "input_hash": "inputs-1",
            "source_hash": source_hash, "inventory_hash": "inventory-1",
            "c1_zero_mismatches": True, "c2_bitwise_equal": True,
            "layout_changes": layout_changes}


def test_kernel_sweep_uses_summed_real_work_and_fastest_native_pair():
    at_boundary = measure.evaluate_g3(_kernel_rows(), _kernel_correctness())
    assert at_boundary.decision == "pass"
    assert at_boundary.observed["pair_ratio_vs_best_native"] == 1.5
    slow_large_matrix = _kernel_rows(family_ns=145)
    for row in slow_large_matrix:
        if row["mode"] == "family_pair":
            row["matrix_times_ns"] = [0.1, 144.9]
    assert measure.evaluate_g3(slow_large_matrix, _kernel_correctness()).decision == "fail"
    assert measure.evaluate_g3(_kernel_rows(family_ns=135.009),
                               _kernel_correctness()).decision == "fail"


def test_kernel_evidence_rejects_changed_source_and_c2_mismatch():
    assert measure.evaluate_g3(_kernel_rows(source_hash="old"),
                               _kernel_correctness(source_hash="new")).decision == "blocked"
    incorrect = _kernel_correctness()
    incorrect["c2_bitwise_equal"] = False
    assert measure.evaluate_g3(_kernel_rows(family_ns=50), incorrect).decision == "fail"


def test_kernel_layout_budget_is_two_after_first_correct():
    assert measure.evaluate_g3(_kernel_rows(),
                               _kernel_correctness(layout_changes=2)).decision == "pass"
    assert measure.evaluate_g3(_kernel_rows(),
                               _kernel_correctness(layout_changes=3)).decision == "fail"


def test_kernel_cells_cover_early_middle_late_and_all_projection_roles(tmp_path):
    from familytiles.convert import FamilyArtifact

    roles = ("q", "k", "v", "o", "gate", "up", "down")
    tensors = {}
    for layer in range(5):
        for role in roles:
            prefix = "self_attn" if role in ("q", "k", "v", "o") else "mlp"
            tensors[f"model.layers.{layer}.{prefix}.{role}_proj.weight"] = {
                "shape": [32 + layer, 64], "target": {"kind": "packed"}}
    fixture = tmp_path / "inputs.json"
    fixture.write_text('{"seed": 20260928, "pool_size": 8, "sweeps": 2}')
    artifact = FamilyArtifact(tmp_path, {"family_id": "test", "tensors": tensors,
                                         "revisions": {"anchor": "a", "target": "b"}},
                              "artifact-hash")
    cells = measure.build_kernel_cells(artifact, fixture)
    assert cells
    names = cells[0].extras["tensor_names"]
    assert len(names) == 21
    assert {int(name.split(".")[2]) for name in names} == {0, 2, 4}
    assert {name.rsplit(".", 2)[-2].removesuffix("_proj") for name in names} == set(roles)
    assert {cell.trial for cell in cells if not cell.trace} == {0, 1, 2}
    assert {cell.mode for cell in cells} >= {"family_pair", "native_two",
                                             "native_batched", "raw_pair",
                                             "family_two_singles", "family_single"}
