"""Admission decision must preserve numerical and memory evidence limits."""

import pytest

from tools.summarize_admission import decide_candidate


def test_latency_failure_cannot_be_relabelled_fast():
    result = decide_candidate(native_p95_ns=30, candidate_p95_ns=120,
                              native_footprint=1000, candidate_footprint=850,
                              logit_mismatches=0)
    assert result["fast_admission"] is False
    assert result["sampled_footprint_saving"] == pytest.approx(0.15)


def test_unavailable_process_footprint_cannot_support_peak_claim():
    result = decide_candidate(native_p95_ns=30, candidate_p95_ns=35,
                              native_footprint=None, candidate_footprint=None,
                              logit_mismatches=0)
    assert result["fast_admission"] is True
    assert result["sampled_footprint_saving"] is None
    assert result["peak_memory_verified"] is False
