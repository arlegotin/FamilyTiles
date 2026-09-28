"""Strict Qwen2 inventory and bounded materialization contracts."""

import numpy as np
import pytest
from dataclasses import replace
from dataclasses import asdict
import json
import struct

pytestmark = pytest.mark.metal

from familytiles.codec import CodecPolicy, encode_tensor
from familytiles.metal import device_operand_from_words
from familytiles.model import (FamilyLinear, expected_active_names,
                               expected_active_shapes, load_runtime,
                               loaded_weight_ledger, validate_active_inventory)


def _tiny_config(tied=True):
    return {"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "vocab_size": 32,
            "tie_word_embeddings": tied}


def test_active_inventory_is_strict():
    config = _tiny_config()
    names = expected_active_names(config)
    assert len(names) == 14
    assert "lm_head.weight" not in names
    shapes = expected_active_shapes(config)
    manifest = {"tensors": {name: {"shape": list(shapes[name]),
                                    "target": {"kind": "native"}}
                            for name in names},
                "tie_word_embeddings": {"anchor": True, "target": True}}
    validate_active_inventory(manifest, {"anchor": config, "target": config})
    manifest["tensors"]["model.layers.0.self_attn.k_proj.weight"]["shape"] = [8, 8]
    with pytest.raises(ValueError, match="shape"):
        validate_active_inventory(manifest, {"anchor": config, "target": config})
    manifest["tensors"]["model.layers.0.self_attn.k_proj.weight"]["shape"] = list(
        shapes["model.layers.0.self_attn.k_proj.weight"])
    del manifest["tensors"]["model.layers.0.mlp.down_proj.weight"]
    with pytest.raises(ValueError, match="inventory"):
        validate_active_inventory(manifest, {"anchor": config, "target": config})
    manifest["tensors"]["model.layers.0.mlp.down_proj.weight"] = {"shape": [8]}
    manifest["tensors"]["lm_head.weight"] = {"shape": [8]}
    with pytest.raises(ValueError, match="inventory|tied"):
        validate_active_inventory(manifest, {"anchor": config, "target": config})
    untied = _tiny_config(False)
    with pytest.raises(ValueError, match="tied"):
        validate_active_inventory(manifest, {"anchor": untied, "target": config})


def test_native_and_alias_storage_are_unique():
    import mlx.core as mx

    words = np.full((4, 8), 0x3F80, dtype=np.uint16)
    encoded_operand = device_operand_from_words(words, encode_tensor(
        words, words, CodecPolicy(1, 64, ("xor",))))
    # A whole-tensor alias is a container-level decision, separate from a
    # packed tensor whose individual tiles happen to be COPY.
    operand = replace(encoded_operand, kind="alias", descriptors=None, payload=None)
    anchor = FamilyLinear(operand, "anchor", None, "family_single")
    target = FamilyLinear(operand, "target", None, "family_single")
    assert anchor.operand.anchor_words is target.operand.anchor_words
    assert anchor.reconstruct_words() is operand.anchor_words
    assert target.operand.native_words is None
    np.testing.assert_array_equal(np.asarray(target.reconstruct_words()), words)
    mx.eval(anchor.reconstruct_words(), target.reconstruct_words())


def test_prefill_materializes_bounded_matrices(monkeypatch):
    import mlx.core as mx
    import familytiles.model as model_module

    anchor = np.full((4, 8), 0x3F80, dtype=np.uint16)
    target = anchor.copy()
    target[0, 2] = np.uint16(0x4000)
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 64, ("xor",)), modes=("packed",)))
    linear = FamilyLinear(operand, "target", None, "family_single")
    x = mx.array(np.ones((1, 3, 8), dtype=np.float32), dtype=mx.bfloat16)
    actual_decode = model_module.decode_words
    calls = []

    def tracked_decode(value):
        calls.append("decode")
        return actual_decode(value)

    monkeypatch.setattr(model_module, "decode_words", tracked_decode)
    result = linear.prefill(x)
    calls.append("returned")
    mx.eval(result)
    assert calls == ["decode", "returned"]
    assert result.shape == (1, 3, 4)
    # The wrapper must finish the matmul before it returns and drops its
    # decoded matrix. An unevaluated result would retain that full operand.
    assert linear.last_prefill_decoded_matrix_bytes == target.nbytes
    assert linear.live_decoded_matrices == 0
    expected = mx.matmul(x, mx.array(target, dtype=mx.uint16).view(mx.bfloat16).T)
    mx.eval(expected)
    np.testing.assert_array_equal(np.asarray(result.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))


def test_family_linear_single_step_matches_raw_schedule():
    import mlx.core as mx
    from familytiles.metal import raw_gemv

    anchor = np.full((4, 8), 0x3F80, dtype=np.uint16)
    target = anchor.copy()
    target[0, 2] = np.uint16(0x4000)
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 64, ("xor",)), modes=("packed",)))
    x = mx.array(np.ones(8, dtype=np.float32), dtype=mx.bfloat16)
    linear = FamilyLinear(operand, "target", None, "family_single")
    result = linear.decode_step(x)
    raw = raw_gemv(mx.array(target, dtype=mx.uint16), x)
    mx.eval(result, raw)
    np.testing.assert_array_equal(np.asarray(result.view(mx.uint16)),
                                  np.asarray(raw.view(mx.uint16)))


def test_prefill_bias_matches_stock_addmm():
    import mlx.core as mx
    import mlx.nn as nn

    rng = np.random.default_rng(419)
    anchor_bf = mx.array(rng.normal(0, 0.1, (32, 65)).astype(np.float32),
                         dtype=mx.bfloat16)
    target_bf = mx.array(rng.normal(0, 0.1, (32, 65)).astype(np.float32),
                         dtype=mx.bfloat16)
    mx.eval(anchor_bf, target_bf)
    anchor = np.asarray(anchor_bf.view(mx.uint16)).copy()
    target = np.asarray(target_bf.view(mx.uint16)).copy()
    operand = device_operand_from_words(anchor, encode_tensor(
        anchor, target, CodecPolicy(1, 128, ("ordered_delta",)), modes=("packed",)))
    bias = mx.array(rng.normal(0, 0.1, 32).astype(np.float32), dtype=mx.bfloat16)
    x = mx.array(rng.normal(0, 1, (1, 3, 65)).astype(np.float32), dtype=mx.bfloat16)
    stock = nn.Linear(65, 32, bias=True)
    stock.weight = target_bf
    stock.bias = bias
    actual = FamilyLinear(operand, "target", bias, "A1").prefill(x)
    expected = stock(x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))


def test_no_random_full_placeholder_is_evaluated(tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from familytiles.convert import convert_family
    from familytiles.data import TensorInfo
    import familytiles.model as model_module

    config = {**_tiny_config(), "rms_norm_eps": 1e-6}
    shapes = expected_active_shapes(config)
    anchor = {name: np.full(shape, 0 if name.endswith(".bias") else 0x3F80,
                            dtype=np.uint16) for name, shape in shapes.items()}
    target = {name: words.copy() for name, words in anchor.items()}
    target["model.layers.0.self_attn.q_proj.weight"][0, 0] ^= np.uint16(1)

    def write_source(path, arrays):
        header = {}
        offset = 0
        for name, values in arrays.items():
            header[name] = {"dtype": "BF16", "shape": list(values.shape),
                            "data_offsets": [offset, offset + values.nbytes]}
            offset += values.nbytes
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded +
                         b"".join(value.tobytes() for value in arrays.values()))
        return {name: TensorInfo(name, values.shape, "BF16", path.name,
                                 8 + len(encoded), header[name]["data_offsets"][0],
                                 values.nbytes)
                for name, values in arrays.items()}

    source = tmp_path / "sources"
    source.mkdir()
    a_path, t_path = source / "a.safetensors", source / "t.safetensors"
    a_info, t_info = write_source(a_path, anchor), write_source(t_path, target)
    survey = tmp_path / "survey"
    (survey / "tiny").mkdir(parents=True)
    revisions = {"anchor": "a" * 40, "target": "b" * 40}
    pinned = {
        role: {"repo": f"Qwen/{role}", "revision": revisions[role],
               "files": {path.name: path.stat().st_size},
               "tensors": {name: asdict(info) for name, info in infos.items()},
               "metadata": {"config": config}}
        for role, path, infos in (("anchor", a_path, a_info), ("target", t_path, t_info))
    }
    (survey / "tiny/pinned.json").write_text(json.dumps(pinned))
    selected = survey / "selected-family.json"
    selected.write_text(json.dumps({"family_id": "tiny", "anchor": "Qwen/anchor",
                                    "target": "Qwen/target", "revisions": revisions,
                                    "pinned_manifest": "tiny/pinned.json"}))
    policy = survey / "frozen-policy.json"
    policy.write_text(json.dumps({"family_id": "tiny", "version": 1,
                                  "block_values": 64, "transforms": ["xor"]}))
    monkeypatch.setattr("familytiles.convert._require_g1", lambda *args: None)
    monkeypatch.setattr("familytiles.convert._source_paths",
                        lambda *args: {"anchor": {a_path.name: a_path},
                                       "target": {t_path.name: t_path}})
    artifact = convert_family(selected, policy, tmp_path / "artifact")
    monkeypatch.setattr(model_module, "_load_configs",
                        lambda _: {"anchor": config, "target": config})

    placeholder_values = []
    original_model = model_module.Model
    original_eval = mx.eval

    def tracked_model(args):
        instance = original_model(args)
        placeholder_values.extend(value for _, value in tree_flatten(instance.parameters()))
        return instance

    def guarded_eval(*values):
        assert all(value is not placeholder for value in values
                   for placeholder in placeholder_values)
        return original_eval(*values)

    monkeypatch.setattr(model_module, "Model", tracked_model)
    monkeypatch.setattr(mx, "eval", guarded_eval)
    runtime = load_runtime(artifact, "A1")
    ledger = loaded_weight_ledger(runtime)
    assert ledger["random_placeholder_evaluated_bytes"] == 0
    assert ledger["unique_weight_allocation_bytes"] > 0
    assert set(runtime.operands) == {name for name in shapes if name.endswith("_proj.weight")}
