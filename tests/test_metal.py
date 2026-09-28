import numpy as np
import pytest

from familytiles.codec import CodecPolicy, encode_tensor
from familytiles.metal import decode_words, device_operand_from_words, probe_anchor_view
from familytiles import metal


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


@pytest.mark.parametrize("length", [1, 31, 33, 129, 257])
@pytest.mark.parametrize("modes", [("packed",), ("copy", "raw"), ("raw",)])
def test_single_gemv_matches_same_schedule_raw_control(length, modes):
    import mlx.core as mx

    rng = np.random.default_rng(20260928 + length)
    anchor_values = mx.array(rng.normal(0, 0.1, (3, length)).astype(np.float32), dtype=mx.bfloat16)
    target_values = mx.array(rng.normal(0, 0.1, (3, length)).astype(np.float32), dtype=mx.bfloat16)
    mx.eval(anchor_values, target_values)
    anchor_words = np.asarray(anchor_values.view(mx.uint16))
    target_words = np.asarray(target_values.view(mx.uint16))
    encoded = encode_tensor(anchor_words, target_words,
                            CodecPolicy(1, 128, ("xor", "ordered_delta")), modes=modes)
    operand = device_operand_from_words(anchor_words, encoded)
    x = mx.array(rng.normal(0, 0.5, length).astype(np.float32), dtype=mx.bfloat16)
    bias = mx.array(rng.normal(0, 0.1, 3).astype(np.float32), dtype=mx.bfloat16)
    raw_words = mx.array(target_words, dtype=mx.uint16)
    actual = metal.family_gemv(operand, x, bias)
    expected = metal.raw_gemv(raw_words, x, bias)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))


def test_single_gemv_supports_strided_activation_and_bias():
    import mlx.core as mx

    anchor = np.arange(3 * 33, dtype=np.uint16).reshape(3, 33)
    encoded = encode_tensor(anchor, anchor, CodecPolicy(1, 64, ("xor",)))
    operand = device_operand_from_words(anchor, encoded)
    x_full = mx.arange(66, dtype=mx.float32).astype(mx.bfloat16)
    bias_full = mx.arange(6, dtype=mx.float32).astype(mx.bfloat16)
    actual = metal.family_gemv(operand, x_full[::2], bias_full[::2])
    expected = metal.raw_gemv(mx.array(anchor, dtype=mx.uint16),
                              mx.contiguous(x_full[::2]), mx.contiguous(bias_full[::2]))
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))


def test_single_raw_control_supports_strided_weight_view():
    import mlx.core as mx

    rng = np.random.default_rng(55)
    values = mx.array(rng.normal(0, 0.1, (33, 3)).astype(np.float32),
                      dtype=mx.bfloat16)
    words = values.view(mx.uint16).T
    x = mx.ones((33,), dtype=mx.bfloat16)
    actual = metal.raw_gemv(words, x)
    expected = metal.raw_gemv(mx.contiguous(words), x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))


def test_single_gemv_rejects_unsupported_shapes_and_dtypes():
    import mlx.core as mx

    anchor = np.zeros((2, 32), dtype=np.uint16)
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, anchor, CodecPolicy(1, 64, ("xor",))))
    with pytest.raises(ValueError):
        metal.family_gemv(operand, mx.zeros((1, 32), dtype=mx.bfloat16))
    with pytest.raises(ValueError):
        metal.family_gemv(operand, mx.zeros((32,), dtype=mx.float32))
    with pytest.raises(ValueError):
        metal.family_gemv(operand, mx.zeros((32,), dtype=mx.bfloat16),
                          mx.zeros((3,), dtype=mx.bfloat16))
    with pytest.raises(ValueError):
        metal.raw_gemv(mx.zeros((0, 32), dtype=mx.uint16),
                       mx.zeros((32,), dtype=mx.bfloat16))


def test_single_gemv_allocates_only_output_sized_storage():
    import mlx.core as mx

    anchor = np.zeros((256, 2048), dtype=np.uint16)
    target = anchor.copy()
    target[:, ::257] = 0x3F80
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 256, ("xor",)), modes=("packed",)))
    x = mx.ones((2048,), dtype=mx.bfloat16)
    mx.eval(x)
    before = mx.get_active_memory()
    output = metal.family_gemv(operand, x)
    mx.eval(output)
    assert mx.get_active_memory() - before < 2**18
    assert output.shape == (256,)


@pytest.mark.parametrize("length", [1, 33, 128, 129])
@pytest.mark.parametrize("modes", [("packed",), ("copy", "raw"), ("raw",)])
def test_pair_independent_inputs_and_biases(length, modes):
    import mlx.core as mx

    rng = np.random.default_rng(4455 + length)
    anchor_float = mx.array(rng.normal(0, 0.1, (5, length)).astype(np.float32),
                            dtype=mx.bfloat16)
    target_float = mx.array(rng.normal(0, 0.1, (5, length)).astype(np.float32),
                            dtype=mx.bfloat16)
    mx.eval(anchor_float, target_float)
    anchor = np.asarray(anchor_float.view(mx.uint16))
    target = np.asarray(target_float.view(mx.uint16))
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 128, ("xor", "ordered_delta")), modes=modes))
    x_anchor = mx.array(rng.normal(0, 0.3, length).astype(np.float32),
                        dtype=mx.bfloat16)
    x_target = mx.array(rng.normal(1, 0.7, length).astype(np.float32),
                        dtype=mx.bfloat16)
    bias_anchor = mx.array(rng.normal(0, 0.2, 5).astype(np.float32),
                           dtype=mx.bfloat16)
    bias_target = mx.array(rng.normal(1, 0.2, 5).astype(np.float32),
                           dtype=mx.bfloat16)
    actual = metal.family_gemv_pair(operand, x_anchor, x_target,
                                    (bias_anchor, bias_target))
    expected = metal.raw_gemv_pair(mx.array(anchor, dtype=mx.uint16),
                                   mx.array(target, dtype=mx.uint16),
                                   x_anchor, x_target, (bias_anchor, bias_target))
    mx.eval(*actual, *expected)
    assert actual[0] is not actual[1]
    for packed, raw in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(packed.view(mx.uint16)),
                                      np.asarray(raw.view(mx.uint16)))


def test_pair_copy_native_and_identical_input_diagnostics():
    import mlx.core as mx

    anchor = np.arange(3 * 35, dtype=np.uint16).reshape(3, 35)
    x = mx.ones((35,), dtype=mx.bfloat16)
    for target, modes in ((anchor.copy(), ("copy",)),
                          (np.bitwise_xor(anchor, np.uint16(0x7777)), ("raw",))):
        operand = device_operand_from_words(anchor, encode_tensor(
            anchor, target, CodecPolicy(1, 64, ("xor",)), modes=modes))
        actual = metal.family_gemv_pair(operand, x, x)
        expected = metal.raw_gemv_pair(mx.array(anchor, dtype=mx.uint16),
                                       mx.array(target, dtype=mx.uint16), x, x)
        mx.eval(*actual, *expected)
        for packed, raw in zip(actual, expected):
            np.testing.assert_array_equal(np.asarray(packed.view(mx.uint16)),
                                          np.asarray(raw.view(mx.uint16)))


def test_pair_rejects_bad_inputs_and_allocates_no_decoded_matrix():
    import mlx.core as mx

    anchor = np.zeros((256, 2048), dtype=np.uint16)
    target = anchor.copy()
    target[:, ::257] = 0x3F80
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 256, ("xor",)), modes=("packed",)))
    x0 = mx.ones((2048,), dtype=mx.bfloat16)
    x1 = mx.full((2048,), 2, dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        metal.family_gemv_pair(operand, x0, mx.ones((2047,), dtype=mx.bfloat16))
    mx.eval(x0, x1)
    before = mx.get_active_memory()
    pair = metal.family_gemv_pair(operand, x0, x1)
    mx.eval(*pair)
    assert mx.get_active_memory() - before < 2**18
    assert pair[0].shape == pair[1].shape == (256,)
