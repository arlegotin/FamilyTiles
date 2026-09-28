"""Fixed-work paired admission inputs and latency summaries."""

from tools.admit_policy import fixed_prompt, percentile_ns


def test_fixed_prompt_preserves_original_prefix_and_length():
    assert fixed_prompt([1, 2, 3], 8) == [1, 2, 3, 1, 2, 3, 1, 2]
    assert percentile_ns([10, 20, 30, 40], 95) == 38.5
