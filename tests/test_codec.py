import numpy as np
import pytest

from familytiles.codec import (
    CodecPolicy,
    EncodedTensor,
    allocated_bytes,
    decode_tensor,
    decode_residual,
    encode_tensor,
    encode_residual,
    inverse_key,
    ordered_key,
    validate_encoded,
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


@pytest.mark.parametrize("block", [64, 128, 256])
@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 63, 127, 128, 129, 255, 256, 257])
def test_roundtrip_tails_and_rows(block, length):
    rng = np.random.default_rng(20260928 + length)
    anchor = rng.integers(0, 65536, (2, length), dtype=np.uint16)
    target = anchor.copy()
    if length:
        target[:, ::11] ^= np.uint16(3)
    for modes in (("copy", "raw", "packed"), ("copy", "raw")):
        encoded = encode_tensor(anchor, target, CodecPolicy(1, block, ("xor",)), modes=modes)
        validate_encoded(anchor, encoded)
        np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
        assert allocated_bytes(encoded) <= target.nbytes


@pytest.mark.parametrize("block", [64, 128, 256])
@pytest.mark.parametrize("width,normal", [(0, 0), (2, 3), (4, 15), (8, 255)])
@pytest.mark.parametrize("exception_count", [0, 1, "all"])
def test_packed_width_exception_counts_and_rank(block, width, normal, exception_count):
    anchor = np.zeros((1, block), dtype=np.uint16)
    target = np.full_like(anchor, normal)
    count = block if exception_count == "all" else exception_count
    positions = list(range(block)) if count == block else ([block - 1] if count else [])
    for index in positions:
        target[0, index] = 0x7FC1
    encoded = encode_tensor(anchor, target, CodecPolicy(1, block, ("xor",)))
    np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
    if count != block and width in (2, 4, 8):
        assert encoded.kind == "packed"
        assert int(encoded.descriptors[0, 1] >> 5) & 0x1FF == count
        assert int(encoded.descriptors[0, 1] >> 2) & 3 == {2: 1, 4: 2, 8: 3}[width]
    if count == block:
        assert encoded.kind == "native"


def test_each_exception_position_crosses_mask_words():
    block = 128
    anchor = np.zeros((1, block), dtype=np.uint16)
    for position in range(block):
        target = np.full_like(anchor, 15)
        target[0, position] = 0x7FC1
        encoded = encode_tensor(anchor, target, CodecPolicy(1, block, ("xor",)))
        np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
        assert int(encoded.descriptors[0, 1] >> 5) & 0x1FF == 1
    target = np.full_like(anchor, 15)
    target[0, [31, 32]] = 0x7FC1
    encoded = encode_tensor(anchor, target, CodecPolicy(1, block, ("xor",)))
    assert allocated_bytes(encoded) == 92


def test_exception_count_256_uses_nine_bits():
    anchor = np.zeros((1, 256), dtype=np.uint16)
    target = np.full_like(anchor, 0x7FC1)
    encoded = encode_tensor(anchor, target, CodecPolicy(1, 256, ("xor",)), modes=("packed",))
    assert (int(encoded.descriptors[0, 1]) >> 5) & 0x1FF == 256
    np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)


def test_random_words_choose_native_and_ordered_delta_roundtrips():
    rng = np.random.default_rng(72)
    anchor = rng.integers(0, 65536, (2, 1024), dtype=np.uint16)
    target = rng.integers(0, 65536, (2, 1024), dtype=np.uint16)
    encoded = encode_tensor(anchor, target, CodecPolicy(1, 128, ("xor", "ordered_delta")))
    assert encoded.kind == "native"
    assert allocated_bytes(encoded) == target.nbytes
    np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
    anchor.fill(0x7F80)
    target.fill(0x7F81)
    encoded = encode_tensor(anchor, target, CodecPolicy(1, 128, ("ordered_delta",)))
    assert encoded.kind == "packed"
    np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)


def test_descriptor_fields_and_corruption_rejected():
    anchor = np.zeros((1, 128), dtype=np.uint16)
    target = np.full_like(anchor, 15)
    target[0, [31, 32]] = 0x7FC1
    good = encode_tensor(anchor, target, CodecPolicy(1, 128, ("xor",)))
    assert good.kind == "packed"
    assert good.descriptors.dtype == np.dtype("<u4")
    assert good.descriptors.shape == (1, 2)
    assert int(good.descriptors[0, 0]) == 0
    assert int(good.descriptors[0, 1]) == 2 | (2 << 2) | (2 << 5)

    def changed(*, descriptor=None, payload=None, policy=None):
        return EncodedTensor("packed", good.shape, policy or good.policy,
                             descriptor if descriptor is not None else good.descriptors.copy(),
                             payload if payload is not None else good.payload.copy(), None)

    for new_field in (3, 1 << 14, 1 << 4, 3 << 5, 129 << 5):
        descriptors = good.descriptors.copy()
        descriptors[0, 1] |= np.uint32(new_field)
        with pytest.raises(ValueError):
            validate_encoded(anchor, changed(descriptor=descriptors))
    descriptors = good.descriptors.copy()
    descriptors[0, 0] = np.uint32(0xFFFFFFFF)
    with pytest.raises(ValueError):
        validate_encoded(anchor, changed(descriptor=descriptors))
    for offset, replacement in [(15, 0xFF), (64, 1), (67, 0), (68, 0)]:
        payload = good.payload.copy()
        payload[offset] = replacement
        with pytest.raises(ValueError):
            validate_encoded(anchor, changed(payload=payload))
    with pytest.raises(ValueError):
        validate_encoded(anchor, changed(payload=good.payload[:-1]))

    # A literal replacement remains a valid encoding; container hashes detect corruption.
    literal = good.payload.copy()
    literal[80] ^= 1
    validate_encoded(anchor, changed(payload=literal))

    tail_anchor = np.zeros((1, 33), dtype=np.uint16)
    tail_target = tail_anchor.copy()
    tail_target[0, -1] = 0x7FC1
    tail = encode_tensor(tail_anchor, tail_target, CodecPolicy(1, 64, ("xor",)), modes=("packed",))
    field = int(tail.descriptors[0, 1])
    width = (0, 2, 4, 8)[(field >> 2) & 3]
    low_size = 4 * ((33 * width + 31) // 32)
    tail_bad = tail.payload.copy()
    tail_bad[low_size + 4] |= 2
    with pytest.raises(ValueError):
        validate_encoded(tail_anchor, EncodedTensor("packed", tail.shape, tail.policy,
                                                   tail.descriptors.copy(), tail_bad, None))


def test_copy_raw_and_native_validation():
    anchor = np.zeros((1, 129), dtype=np.uint16)
    target = anchor.copy()
    target[0, -1] = 0xFFFF
    encoded = encode_tensor(anchor, target, CodecPolicy(1, 128, ("xor",)), modes=("copy", "raw"))
    np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
    wrong_anchor = anchor.copy()
    wrong_anchor[0, 0] = 1
    # COPY uses the supplied anchor, so content identity is checked by the container hash later.
    np.testing.assert_array_equal(decode_tensor(wrong_anchor, encoded)[0, :128], wrong_anchor[0, :128])


def test_padding_overlap_and_truncation_rejected():
    policy = CodecPolicy(1, 64, ("xor",))
    anchor = np.zeros((1, 65), dtype=np.uint16)
    target = np.full_like(anchor, 15)
    target[0, -1] = 0x7FC1
    encoded = encode_tensor(anchor, target, policy, modes=("packed",))
    assert encoded.descriptors.shape == (2, 2)
    overlap = encoded.descriptors.copy()
    overlap[1, 0] = overlap[0, 0]
    with pytest.raises(ValueError):
        validate_encoded(anchor, EncodedTensor("packed", anchor.shape, policy,
                                              overlap, encoded.payload, None))
    with pytest.raises(ValueError):
        validate_encoded(anchor, EncodedTensor("packed", anchor.shape, policy,
                                              encoded.descriptors, encoded.payload[:-4], None))
    # The one-word tail has a padded two-byte exception literal.
    padded = encoded.payload.copy()
    padded[-1] = 1
    with pytest.raises(ValueError):
        validate_encoded(anchor, EncodedTensor("packed", anchor.shape, policy,
                                              encoded.descriptors, padded, None))

    raw = encode_tensor(anchor[:, -1:].copy(), target[:, -1:].copy(), policy,
                        modes=("raw",))
    assert raw.kind == "native"  # Whole-tensor fallback is cheaper.
    forced = encode_tensor(anchor[:, -1:].copy(), target[:, -1:].copy(), policy,
                           modes=("packed",))
    bad_mode = forced.descriptors.copy()
    bad_mode[0, 1] = 1
    with pytest.raises(ValueError):
        validate_encoded(anchor[:, -1:].copy(), EncodedTensor("packed", (1, 1), policy,
                                                             bad_mode, forced.payload, None))
