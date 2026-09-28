"""Fixed cached-step fixture and numerical comparison rules."""

import numpy as np
import pytest

from tools.trace_decode import compare_words, token_schedule


def test_teacher_forced_schedule_keeps_saved_first_step_and_request_independence():
    saved = {"anchor_prompt": [1, 2, 3], "target_prompt": [4, 5],
             "anchor_next": 3, "target_next": 5}
    schedule = token_schedule(saved, 8)
    assert schedule[0] == (3, 5)
    assert len(schedule) == 8
    assert schedule[1] != schedule[0]
    assert all(0 <= a and 0 <= b for a, b in schedule)


def test_compare_words_uses_fp64_global_normalized_rms():
    ref = np.array([0x3f80, 0x4000], dtype="<u2")  # 1, 2 in BF16
    same = ref.copy()
    changed = np.array([0x4000, 0x4000], dtype="<u2")  # 2, 2
    assert compare_words(ref, same)["normalized_rms"] == 0.0
    metrics = compare_words(ref, changed)
    assert metrics["bit_mismatches"] == 1
    assert metrics["normalized_rms"] == pytest.approx((1 / 5) ** 0.5)
    assert metrics["max_absolute_error"] == 1.0
