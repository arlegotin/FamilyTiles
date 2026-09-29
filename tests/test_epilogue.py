"""BF16 bias ordering in the actual compiled raw and compressed consumers."""

import json
from pathlib import Path

import numpy as np
import pytest

from familytiles.codec import CodecPolicy, encode_tensor
from familytiles.metal import device_operand_from_words
from familytiles import metal


pytestmark = pytest.mark.metal
ROOT = Path(__file__).resolve().parents[1]


def _bf16_words(value):
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).copy()


def test_compiled_single_and_pair_round_dot_before_bias():
    import mlx.core as mx

    # The FP32 dot is 1 + 2^-8, exactly halfway between 1 and the next BF16.
    anchor = mx.array([[1.0, 2**-8]], dtype=mx.bfloat16)
    target = mx.array([[1.0, -2**-8]], dtype=mx.bfloat16)
    mx.eval(anchor, target)
    anchor_words, target_words = _bf16_words(anchor), _bf16_words(target)
    operand = device_operand_from_words(anchor_words, encode_tensor(
        anchor_words, target_words, CodecPolicy(1, 64, ("ordered_delta",)),
        modes=("packed",)))
    x_anchor = mx.array([1.0, 1.0], dtype=mx.bfloat16)
    x_target = mx.array([1.0, -1.0], dtype=mx.bfloat16)
    positive = mx.array([2**-8], dtype=mx.bfloat16)
    negative = mx.array([-2**-8], dtype=mx.bfloat16)
    zero = mx.array([0.0], dtype=mx.bfloat16)

    assert int(_bf16_words(metal.raw_gemv(
        anchor.view(mx.uint16), x_anchor, positive))[0]) == 0x3F80
    assert int(_bf16_words(metal.family_gemv(
        operand, x_target, negative))[0]) == 0x3F7F

    raw = metal.raw_gemv_pair(anchor.view(mx.uint16), target.view(mx.uint16),
                              x_anchor, x_target, (positive, negative))
    fused = metal.family_gemv_pair(operand, x_anchor, x_target,
                                   (positive, negative))
    for pair in (raw, fused):
        assert [int(_bf16_words(value)[0]) for value in pair] == [0x3F80, 0x3F7F]

    # Zero bias, bias-free execution, and cancellation keep their own paths.
    assert np.array_equal(_bf16_words(metal.raw_gemv(anchor.view(mx.uint16), x_anchor, zero)),
                          _bf16_words(metal.raw_gemv(anchor.view(mx.uint16), x_anchor)))
    cancellation = mx.array([-1.0], dtype=mx.bfloat16)
    result = metal.raw_gemv(mx.array([[0x3F80]], dtype=mx.uint16),
                            mx.array([1.0], dtype=mx.bfloat16), cancellation)
    assert int(_bf16_words(result)[0]) == 0


@pytest.mark.skipif(not (ROOT / "artifacts/family/manifest.json").is_file(),
                    reason="requires the pinned local Qwen artifact")
def test_saved_real_first_layer_qkv_uses_corrected_production_epilogue():
    import mlx.core as mx
    from familytiles.convert import load_artifact
    from familytiles.model import load_runtime, prefill

    artifact = load_artifact(ROOT / "artifacts/family")
    fixture = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    runtime = load_runtime(artifact, "raw_pair")
    for role in ("anchor", "target"):
        prefill(runtime, role, fixture[f"{role}_prompt"])
        model = runtime.models[role]
        layer = model.model.layers[0]
        token = mx.array([[fixture[f"{role}_next"]]], dtype=mx.int32)
        x = layer.input_layernorm(model.model.embed_tokens(token))[0, 0, :]
        for projection in ("q_proj", "k_proj", "v_proj"):
            module = getattr(layer.self_attn, projection)
            words = (module.operand.anchor_words if role == "anchor"
                     else module.operand.native_words)
            expected = mx.addmm(module.bias, x.reshape(1, 1, -1),
                                words.view(mx.bfloat16).T)[0, 0, :]
            actual = metal.raw_gemv(words, x, module.bias)
            assert np.array_equal(_bf16_words(actual), _bf16_words(expected)), (
                role, projection)
