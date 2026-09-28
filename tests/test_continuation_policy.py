"""Whole-matrix promotion spends real net bytes and binds a frozen artifact."""

from familytiles.policy import choose_native_promotions, validate_execution_policy


def test_promotions_replace_packets_and_preserve_a_weight_budget():
    tensors = {f"model.layers.0.mlp.{role}.weight": {
        "shape": [10, 10], "target": {"kind": "packed", "allocated_bytes": packet}}
        for role, packet in (("gate_proj", 80), ("up_proj", 100))}
    gains = {"gate_proj": 120, "up_proj": 10}
    chosen = choose_native_promotions(tensors, gains, max_extra_bytes=120)
    assert chosen == ["model.layers.0.mlp.gate_proj.weight"]
    policy = {"artifact_hash": "abc", "native_names": chosen,
              "staged_names": ["model.layers.0.mlp.up_proj.weight"],
              "extra_resident_bytes": 120}
    validate_execution_policy(policy, "abc", tensors)


def test_policy_rejects_missing_dependency_and_uncredited_old_packet():
    tensors = {"model.layers.0.mlp.gate_proj.weight": {
        "shape": [10, 10], "target": {"kind": "packed", "allocated_bytes": 80}}}
    bad = {"artifact_hash": "abc", "native_names": ["model.layers.0.mlp.gate_proj.weight"],
           "staged_names": [], "extra_resident_bytes": 0}
    try:
        validate_execution_policy(bad, "abc", tensors)
    except ValueError as error:
        assert "extra" in str(error)
    else:
        raise AssertionError("missing native-promotion storage charge was accepted")
