from dataclasses import replace

import pytest

from familytiles.records import (
    GateRecord,
    RunContext,
    require_gate,
    resolve_under_root,
    write_json_atomic,
)


def context() -> RunContext:
    return RunContext(
        family_id="qwen-small",
        revisions={"anchor": "a1", "target": "b2"},
        policy_hash="policy-a",
        artifact_hash=None,
        implementation_hash="code-a",
        environment_hash="machine-a",
        input_hash=None,
    )


def test_gate_requires_matching_evidence(tmp_path):
    path = tmp_path / "g1.json"
    gate = GateRecord(1, "G1", context(), {}, {"saving": 0.15},
                      {"saving": 0.16}, "pass", "observed")
    write_json_atomic(path, gate.to_dict())
    assert require_gate(path, "G1", context()).decision == "pass"
    for changed in (
        replace(context(), policy_hash="policy-b"),
        replace(context(), artifact_hash="other"),
        replace(context(), revisions={"anchor": "a1", "target": "c3"}),
    ):
        with pytest.raises(ValueError):
            require_gate(path, "G1", changed)
    with pytest.raises(ValueError):
        require_gate(path, "G2", context())
    with pytest.raises(FileNotFoundError):
        require_gate(tmp_path / "missing.json", "G1", context())


def test_gate_rejects_nonpass_and_invalid_schema(tmp_path):
    path = tmp_path / "gate.json"
    gate = GateRecord(1, "G1", context(), {}, {}, {}, "fail", "too large")
    write_json_atomic(path, gate.to_dict())
    with pytest.raises(ValueError):
        require_gate(path, "G1", context())
    write_json_atomic(path, {**gate.to_dict(), "schema_version": 2})
    with pytest.raises(ValueError):
        require_gate(path, "G1", context())


def test_paths_reject_parent_absolute_and_symlink_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    assert resolve_under_root(root, "results/g1.json") == root / "results/g1.json"
    for value in ("../outside/file", str(outside / "file"), "link/file"):
        with pytest.raises(ValueError):
            resolve_under_root(root, value)
