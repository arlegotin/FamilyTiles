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
