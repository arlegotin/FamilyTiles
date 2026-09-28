"""Resident exact Zstd provider integration for the bounded continuation."""

import numpy as np

from tools.profile_execution import (build_resident_b3, build_resident_b4,
                                     reconstruct_resident_pair)


def test_resident_exact_providers_round_trip_distinct_bf16_words():
    rng = np.random.default_rng(20260928)
    anchor = rng.integers(0, 65536, size=(3, 129), dtype=np.uint16)
    target = anchor.copy()
    target[0, 0] ^= np.uint16(0x8000)
    target[2, 128] = np.uint16(0x7fc1)
    for builder in (build_resident_b3, build_resident_b4):
        backing = builder(anchor, target)
        actual_anchor, actual_target = reconstruct_resident_pair(backing)
        np.testing.assert_array_equal(actual_anchor, anchor)
        np.testing.assert_array_equal(actual_target, target)
        assert backing.allocated_bytes > 0
        assert backing.allocated_bytes == sum(array.nbytes for array in backing.owned_arrays)
