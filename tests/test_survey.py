import json
from types import SimpleNamespace

import pytest

from familytiles.codec import CodecPolicy
from familytiles.data import PinnedModel, TensorInfo
from familytiles.survey import (choose_policy, make_sample_plan, project_family,
                                survey)


def _models():
    tensors = {}
    for layer in range(6):
        for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.self_attn.{role}.weight"
            tensors[name] = TensorInfo(name, (256, 512), "BF16", "w.safetensors", 100, 0, 256 * 512 * 2)
    return tuple(PinnedModel(f"Qwen/test{i}", "a" * 40, {"w.safetensors": 10**9}, tensors, {}) for i in range(2))


def test_sample_split_precedes_policy_selection():
    plan = make_sample_plan(_models(), seed=20260928, limit_bytes=32 * 2**20)
    dev = {w.id for w in plan.windows if w.split == "development"}
    hold = {w.id for w in plan.windows if w.split == "holdout"}
    assert dev.isdisjoint(hold)
    assert len(dev) == 49 and len(hold) == 21
    assert plan.bytes_per_model <= 32 * 2**20
    assert all(w.word_count % 512 == 0 for w in plan.windows)


def test_weighted_worse_stratum_projection_and_threshold_boundaries():
    inventory = {"raw_family_bytes": 2000, "b1_bytes": 1800,
                 "target_role_bytes": {"large": 600, "small": 100},
                 "unresolved_alias_bytes": 200}
    dev = {"roles": {"large": {"raw": 100, "b2": 95, "family": 60},
                     "small": {"raw": 100, "b2": 100, "family": 50}}}
    hold = {"roles": {"large": {"raw": 100, "b2": 100, "family": 70},
                      "small": {"raw": 100, "b2": 100, "family": 55}}}
    result = project_family(dev, hold, inventory, CodecPolicy(1, 128, ("xor",)))
    assert result["b2_bytes"] == 1800
    assert result["family_bytes"] == 1575
    assert result["g1_pass"] is False  # 12.5% < 15% versus B1
    assert result["beyond_b2_fraction_raw"] == pytest.approx(225 / 2000)
    hold["roles"]["large"]["family"] = 65
    result = project_family(dev, hold, inventory, CodecPolicy(1, 128, ("xor",)))
    assert result["family_bytes"] == 1545
    assert result["g1_pass"] is False
    hold["roles"]["large"]["family"] = 60
    hold["roles"]["small"]["family"] = 70
    result = project_family(dev, hold, inventory, CodecPolicy(1, 128, ("xor",)))
    assert result["family_bytes"] == 1530  # worse dev ratios still apply
    assert result["g1_pass"] is True  # exactly 15% below 1800


def test_dual_transform_requires_two_percent_raw_family_gain():
    candidates = [
        {"block": 128, "transforms": ["xor"], "family_bytes": 800, "raw_family_bytes": 2000},
        {"block": 128, "transforms": ["xor", "ordered_delta"], "family_bytes": 765, "raw_family_bytes": 2000},
    ]
    assert choose_policy({"candidates": candidates}).transforms == ("xor",)
    candidates[1]["family_bytes"] = 760
    assert choose_policy({"candidates": candidates}).transforms == ("xor", "ordered_delta")


def test_no_alias_from_samples_and_controls_cannot_select():
    inventory = {"raw_family_bytes": 2000, "b1_bytes": 1800,
                 "target_role_bytes": {"large": 700}, "unresolved_alias_bytes": 200}
    dev = {"roles": {"large": {"raw": 100, "b2": 100, "family": 50}}}
    hold = {"roles": {"large": {"raw": 100, "b2": 100, "family": 50}}}
    result = project_family(dev, hold, inventory, CodecPolicy(1, 128, ("xor",)))
    assert result["b1_bytes"] == 1800
    assert result["unresolved_alias_bytes"] == 200
    assert result["g1_pass"]
    with pytest.raises(ValueError):
        choose_policy({"candidates": [{"block": 128, "transforms": ["xor"],
                                        "family_bytes": 1, "raw_family_bytes": 2000,
                                        "source": "synthetic_control"}]})


def test_failed_acquisition_removes_stale_selection_pointer(tmp_path, monkeypatch):
    from familytiles import survey as module

    families = tmp_path / "families.json"
    families.write_text(json.dumps({"families": [
        {"id": "a", "anchor": "Qwen/a", "target": "Qwen/b", "role": "primary", "enabled": True},
        {"id": "b", "anchor": "Qwen/c", "target": "Qwen/d", "role": "fallback-scale-check", "enabled": True}
    ]}))
    out = tmp_path / "survey"
    out.mkdir()
    (out / "selected-family.json").write_text("stale")
    monkeypatch.setattr(module, "read_gate", lambda path: SimpleNamespace(
        decision="pass", context=SimpleNamespace(environment_hash="hash")))
    monkeypatch.setattr(module, "pin_model", lambda *a: (_ for _ in ()).throw(ValueError("offline")))
    monkeypatch.setattr(module, "publish_gate", lambda *a: None)
    result = survey(families, 32, 20260928, out)
    assert result["decision"] == "blocked"
    assert not (out / "selected-family.json").exists()
    assert not (out / "frozen-policy.json").exists()
