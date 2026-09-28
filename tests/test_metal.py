import numpy as np
import pytest

from familytiles.codec import CodecPolicy, encode_tensor
from familytiles.metal import decode_words, device_operand_from_words, probe_anchor_view


pytestmark = pytest.mark.metal


@pytest.mark.parametrize("block", [64, 128, 256])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 127, 128, 129, 257])
def test_gpu_words_match_cpu_for_tails_and_modes(block, length):
    import mlx.core as mx

    rng = np.random.default_rng(20260928 + length)
    anchor = rng.integers(0, 65536, (3, length), dtype=np.uint16)
    target = anchor.copy()
    target[0, ::17] ^= np.uint16(3)
    target[1, ::11] = np.uint16(0x7FC1)
    target[2] = rng.integers(0, 65536, length, dtype=np.uint16)
    for modes in (("copy", "raw", "packed"), ("copy", "raw"), ("packed",)):
        encoded = encode_tensor(anchor, target, CodecPolicy(1, block, ("xor", "ordered_delta")), modes=modes)
        operand = device_operand_from_words(anchor, encoded)
        output = decode_words(operand)
        mx.eval(output)
        np.testing.assert_array_equal(np.asarray(output), target)
        middle = decode_words(operand, row_start=1, row_count=1)
        mx.eval(middle)
        np.testing.assert_array_equal(np.asarray(middle), target[1:2])


def test_gpu_words_preserve_special_payloads_and_256_exceptions():
    import mlx.core as mx

    anchor = np.full((1, 256), 0x8000, dtype=np.uint16)
    target = np.tile(np.array([0, 0x8000, 1, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2],
                              dtype=np.uint16), 37)[:256].reshape(1, 256)
    for transform in ("xor", "ordered_delta"):
        encoded = encode_tensor(anchor, target, CodecPolicy(1, 256, (transform,)),
                                modes=("packed",))
        output = decode_words(device_operand_from_words(anchor, encoded))
        mx.eval(output)
        np.testing.assert_array_equal(np.asarray(output), target)


def test_validation_rejects_corruption_before_gpu_allocation():
    anchor = np.zeros((1, 128), dtype=np.uint16)
    target = np.full_like(anchor, 15)
    target[0, 17] = 0x7FC1
    encoded = encode_tensor(anchor, target, CodecPolicy(1, 128, ("xor",)))
    encoded.descriptors[0, 1] |= np.uint32(1 << 14)
    with pytest.raises(ValueError):
        device_operand_from_words(anchor, encoded)


def test_decode_rejects_invalid_row_range_and_zero_gemv_shape():
    anchor = np.zeros((2, 64), dtype=np.uint16)
    encoded = encode_tensor(anchor, anchor, CodecPolicy(1, 64, ("xor",)))
    operand = device_operand_from_words(anchor, encoded)
    with pytest.raises(ValueError):
        decode_words(operand, row_start=-1)
    with pytest.raises(ValueError):
        decode_words(operand, row_start=2, row_count=1)
    with pytest.raises(ValueError):
        decode_words(operand, row_count=0)


def test_anchor_uint16_bf16_view_has_no_full_copy():
    evidence = probe_anchor_view()
    assert evidence["anchor_bytes"] == 16 * 2**20
    assert evidence["no_full_copy"]
