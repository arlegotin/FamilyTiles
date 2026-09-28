import numpy as np
import pytest

from familytiles.baselines import (ExactChunks, decode_exact, decode_family_exact,
                                   encode_exact, encode_family_exact,
                                   exact_allocated_bytes)


def test_field_split_all_bf16_words():
    words = np.arange(65536, dtype="<u2")
    encoded = encode_exact(words, "field", chunk_bytes=2048)
    np.testing.assert_array_equal(np.concatenate(list(decode_exact(encoded))), words)
    assert exact_allocated_bytes(encoded) >= len(encoded.payload)


@pytest.mark.parametrize("transform", ["native", "byte_plane", "field"])
@pytest.mark.parametrize("length", [0, 1, 31, 1024, 1025, 8193])
def test_exact_chunks_roundtrip_and_no_expansion(transform, length):
    rng = np.random.default_rng(length + 49)
    words = rng.integers(0, 65536, length, dtype=np.uint16)
    encoded = encode_exact(words, transform, chunk_bytes=2048)
    chunks = list(decode_exact(encoded))
    restored = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    np.testing.assert_array_equal(restored, words)
    assert exact_allocated_bytes(encoded) <= words.nbytes


@pytest.mark.parametrize("transform", ["xor", "ordered_delta"])
@pytest.mark.parametrize("anchor_form", ["raw", "compressed"])
def test_family_chunks_roundtrip_including_17bit(transform, anchor_form):
    rng = np.random.default_rng(20260928)
    anchor = rng.integers(0, 65536, 65537, dtype=np.uint16)
    target = anchor.copy()
    target[::17] ^= np.uint16(0x7FFF)
    anchor[0] = 0xFFFF
    target[0] = 0x7FFF
    backing = encode_family_exact(anchor, target, transform, anchor_form)
    pairs = list(decode_family_exact(backing))
    np.testing.assert_array_equal(np.concatenate([p[0] for p in pairs]), anchor)
    np.testing.assert_array_equal(np.concatenate([p[1] for p in pairs]), target)
    assert exact_allocated_bytes(backing) >= len(backing.anchor.payload) + len(backing.target.payload)


def test_corrupt_chunk_index_and_payload_rejected():
    words = np.zeros(8192, dtype=np.uint16)
    encoded = encode_exact(words, "byte_plane", chunk_bytes=2048)
    assert not encoded.native_fallback
    short = ExactChunks(encoded.shape, encoded.word_dtype, encoded.transform,
                        encoded.chunk_index.copy(), encoded.payload[:-1].copy(), False)
    with pytest.raises(ValueError):
        list(decode_exact(short))
    overlap = encoded.chunk_index.copy()
    overlap[1, 0] = 0
    bad = ExactChunks(encoded.shape, encoded.word_dtype, encoded.transform,
                      overlap, encoded.payload, False)
    with pytest.raises(ValueError):
        list(decode_exact(bad))
