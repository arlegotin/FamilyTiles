import json
import hashlib
import struct
from dataclasses import asdict

import numpy as np
import pytest

from familytiles.data import TensorInfo
from familytiles.convert import (canonical_ledger, convert_family, iter_tensor_words,
                                 load_artifact, verify_cpu)


def _write_safetensors(path, arrays):
    header = {}
    offset = 0
    for name, values in arrays.items():
        header[name] = {"dtype": "BF16", "shape": list(values.shape),
                        "data_offsets": [offset, offset + values.nbytes]}
        offset += values.nbytes
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded +
                     b"".join(x.tobytes() for x in arrays.values()))
    return {name: TensorInfo(name, value.shape, "BF16", path.name, 8 + len(encoded),
                             header[name]["data_offsets"][0], value.nbytes)
            for name, value in arrays.items()}


@pytest.fixture
def tiny_family(tmp_path, monkeypatch):
    embed = np.arange(16, dtype=np.uint16).reshape(4, 4)
    q = np.full((4, 64), 0x3F80, dtype=np.uint16)
    tq = q.copy()
    tq[:, ::17] ^= np.uint16(1)
    anchor_arrays = {
        "model.embed_tokens.weight": embed,
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.input_layernorm.weight": np.ones(4, dtype=np.uint16),
        "model.norm.weight": np.ones(4, dtype=np.uint16),
        "unused.auxiliary": np.array([4], dtype=np.uint16),
    }
    target_arrays = {**anchor_arrays,
                     "model.layers.0.self_attn.q_proj.weight": tq,
                     "model.layers.0.input_layernorm.weight": np.full(4, 2, dtype=np.uint16)}
    source = tmp_path / "source"
    source.mkdir()
    a_path, t_path = source / "a.safetensors", source / "t.safetensors"
    a_tensors = _write_safetensors(a_path, anchor_arrays)
    t_tensors = _write_safetensors(t_path, target_arrays)
    family_dir = tmp_path / "survey"
    (family_dir / "tiny").mkdir(parents=True)
    sha_a, sha_t = "a" * 40, "b" * 40
    pinned = {"anchor": {"repo": "Qwen/a", "revision": sha_a,
                         "files": {a_path.name: a_path.stat().st_size},
                         "tensors": {k: asdict(v) for k, v in a_tensors.items()},
                         "metadata": {"config": {"tie_word_embeddings": True}}},
              "target": {"repo": "Qwen/b", "revision": sha_t,
                         "files": {t_path.name: t_path.stat().st_size},
                         "tensors": {k: asdict(v) for k, v in t_tensors.items()},
                         "metadata": {"config": {"tie_word_embeddings": True}}}}
    (family_dir / "tiny/pinned.json").write_text(json.dumps(pinned))
    selected = family_dir / "selected-family.json"
    selected.write_text(json.dumps({"family_id": "tiny", "anchor": "Qwen/a", "target": "Qwen/b",
                                    "revisions": {"anchor": sha_a, "target": sha_t},
                                    "pinned_manifest": "tiny/pinned.json"}))
    policy = family_dir / "frozen-policy.json"
    policy.write_text(json.dumps({"family_id": "tiny", "version": 1,
                                  "block_values": 64, "transforms": ["xor"]}))
    monkeypatch.setattr("familytiles.convert._require_g1", lambda *a: None)
    monkeypatch.setattr("familytiles.convert._source_paths",
                        lambda *a: {"anchor": {a_path.name: a_path},
                                    "target": {t_path.name: t_path}})
    return selected, policy, tmp_path / "artifact", anchor_arrays, target_arrays, pinned


def test_stream_conversion_accounts_active_tensors_and_exact_words(tiny_family):
    selected, policy, out, anchor, target, _ = tiny_family
    artifact = convert_family(selected, policy, out)
    audit = verify_cpu(artifact)
    assert audit["mismatched_words"] == 0
    assert audit["verified_active_tensors"] == 4
    ledger = canonical_ledger(artifact)
    assert ledger["w0_bytes"] == 2 * sum(v.nbytes for k, v in anchor.items() if not k.startswith("unused"))
    assert ledger["w1_bytes"] < ledger["w0_bytes"]
    assert ledger["family_bytes"] > ledger["w1_bytes"]  # Tiny fixtures expose container overhead.
    assert ledger["manifest_bytes"] > 0
    assert artifact.manifest["excluded_tensors"] == ["unused.auxiliary"]
    restored = np.concatenate(list(iter_tensor_words(artifact, "target",
                                                    "model.layers.0.self_attn.q_proj.weight", 2)))
    np.testing.assert_array_equal(restored, target["model.layers.0.self_attn.q_proj.weight"])


def test_hash_lookup_still_compares_bytes(tiny_family, monkeypatch):
    selected, policy, out, _, _, _ = tiny_family
    monkeypatch.setattr("familytiles.convert._hash_words", lambda x: "collision")
    artifact = convert_family(selected, policy, out)
    record = artifact.manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]
    assert record["target"]["kind"] != "alias"


def test_tied_head_conflict_is_error(tiny_family):
    selected, policy, out, anchor, target, pinned = tiny_family
    for model_id, arrays in (("anchor", anchor), ("target", target)):
        arrays["lm_head.weight"] = np.zeros((4, 4), dtype=np.uint16)
        path = selected.parent.parent / "source" / ("a.safetensors" if model_id == "anchor" else "t.safetensors")
        infos = _write_safetensors(path, arrays)
        pinned[model_id]["files"] = {path.name: path.stat().st_size}
        pinned[model_id]["tensors"] = {k: asdict(v) for k, v in infos.items()}
    (selected.parent / "tiny/pinned.json").write_text(json.dumps(pinned))
    with pytest.raises(ValueError, match="tied"):
        convert_family(selected, policy, out)


def test_interrupted_and_corrupt_manifest_rejected(tiny_family):
    selected, policy, out, _, _, _ = tiny_family
    with pytest.raises(ValueError):
        load_artifact(out)
    artifact = convert_family(selected, policy, out)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    alias = manifest["tensors"]["model.embed_tokens.weight"]["target"]
    assert alias["kind"] == "alias"
    alias["name"] = "model.norm.weight"
    manifest_path.write_text(json.dumps(manifest))
    (out / "manifest.sha256").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_artifact(out)
    manifest["tensors"]["model.embed_tokens.weight"]["target"]["name"] = "model.embed_tokens.weight"
    manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]["target"]["payload_path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))
    (out / "manifest.sha256").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_artifact(out)


def test_truncated_payload_and_mixed_revision_rejected(tiny_family):
    selected, policy, out, _, _, _ = tiny_family
    artifact = convert_family(selected, policy, out)
    packed = artifact.manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]["target"]
    payload = out / packed["payload_path"]
    payload.write_bytes(payload.read_bytes()[:-4])
    with pytest.raises(ValueError):
        load_artifact(out)
    selected_record = json.loads(selected.read_text())
    selected_record["revisions"]["target"] = "c" * 40
    selected.write_text(json.dumps(selected_record))
    with pytest.raises(ValueError, match="identity"):
        convert_family(selected, policy, out.parent / "other-artifact")


def test_wrong_anchor_source_rejected_before_decoding(tiny_family):
    selected, policy, out, _, _, _ = tiny_family
    artifact = convert_family(selected, policy, out)
    anchor_source = out / artifact.manifest["tensors"]["model.embed_tokens.weight"]["anchor"]["path"]
    with anchor_source.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="anchor"):
        load_artifact(out)
