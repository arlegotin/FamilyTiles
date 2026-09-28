import numpy as np
import pytest

from familytiles.codec import (
    CodecPolicy,
    decode_residual,
    encode_residual,
    inverse_key,
    ordered_key,
)


def test_all_words_key_bijection():
    words = np.arange(65536, dtype=np.uint16)
    np.testing.assert_array_equal(inverse_key(ordered_key(words)), words)
    np.testing.assert_array_equal(ordered_key(inverse_key(words)), words)


def test_extreme_delta_needs_17_bits():
    low_key = np.array([0xFFFF], dtype="<u2")
    high_key = np.array([0x7FFF], dtype="<u2")
    forward = encode_residual(low_key, high_key, "ordered_delta")
    backward = encode_residual(high_key, low_key, "ordered_delta")
    assert forward.dtype == np.dtype("uint32")
    assert int(forward[0]) == 131070
    assert int(backward[0]) == 131069
    np.testing.assert_array_equal(decode_residual(low_key, forward, "ordered_delta"), high_key)
    np.testing.assert_array_equal(decode_residual(high_key, backward, "ordered_delta"), low_key)


@pytest.mark.parametrize("anchor_word", [0x0000, 0x8000, 0x0001, 0x7F80,
                                         0xFF80, 0x7FFF, 0xFFFF])
@pytest.mark.parametrize("transform", ["xor", "ordered_delta"])
def test_every_word_representative_anchors(anchor_word, transform):
    target = np.arange(65536, dtype=np.uint16)
    anchor = np.full(target.shape, anchor_word, dtype=np.uint16)
    coded = encode_residual(anchor, target, transform)
    np.testing.assert_array_equal(decode_residual(anchor, coded, transform), target)


@pytest.mark.parametrize("transform", ["xor", "ordered_delta"])
def test_million_seeded_pairs_roundtrip(transform):
    rng = np.random.default_rng(20260928)
    anchor = rng.integers(0, 65536, size=1_000_000, dtype=np.uint16)
    target = rng.integers(0, 65536, size=1_000_000, dtype=np.uint16)
    coded = encode_residual(anchor, target, transform)
    np.testing.assert_array_equal(decode_residual(anchor, coded, transform), target)


def test_special_bf16_words_survive_both_transforms():
    anchor = np.array([0, 0x8000, 0x0001, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2], dtype="<u2")
    target = anchor[::-1].copy()
    for transform in ("xor", "ordered_delta"):
        np.testing.assert_array_equal(
            decode_residual(anchor, encode_residual(anchor, target, transform), transform), target
        )


def test_residual_rejects_invalid_reconstruction_and_shape():
    anchor = np.array([0], dtype="<u2")
    with pytest.raises(ValueError):
        decode_residual(anchor, np.array([131070], dtype=np.uint32), "ordered_delta")
    with pytest.raises(ValueError):
        decode_residual(anchor, np.array([131071], dtype=np.uint32), "ordered_delta")
    with pytest.raises(ValueError):
        decode_residual(anchor, np.array([65536], dtype=np.uint32), "xor")
    with pytest.raises(ValueError):
        encode_residual(anchor, np.array([0, 1], dtype=np.uint16), "xor")


def test_policy_rejects_unsupported_version_block_and_transform():
    with pytest.raises(ValueError):
        CodecPolicy(2, 128, ("xor",))
    with pytest.raises(ValueError):
        CodecPolicy(1, 32, ("xor",))
    with pytest.raises(ValueError):
        CodecPolicy(1, 128, ("xor", "xor"))
    with pytest.raises(ValueError):
        CodecPolicy(1, 128, ("float_delta",))
