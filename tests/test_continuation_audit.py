"""Continuation timing provenance and complete-sweep controls."""

from pathlib import Path

import numpy as np
import pytest

from tools.audit_g3 import audit_historical, native_pair_outputs, run_session, run_slot


ROOT = Path(__file__).resolve().parents[1]


def test_same_source_g3_records_keep_both_outcomes_and_missing_fields():
    audit = audit_historical(ROOT / "results")
    runs = audit["same_source_runs"]
    assert len(runs) == 2
    assert {run["decision"] for run in runs} == {"pass", "fail"}
    assert len({run["source_hash"] for run in runs}) == 1
    assert len({run["benchmark_code_hash"] for run in runs}) == 1
    assert len({run["input_hash"] for run in runs}) == 1
    assert len({run["inventory_hash"] for run in runs}) == 1
    assert all(run["native_batched_two_products"] == "unrecorded" for run in runs)
    assert audit["classification"] == "unresolved"


@pytest.mark.metal
def test_native_pair_computes_two_different_weight_activation_products():
    import mlx.core as mx

    anchor = mx.array([[1., 2., 3.], [4., 5., 6.]], dtype=mx.bfloat16)
    target = mx.array([[7., 8., 9.], [10., 11., 12.]], dtype=mx.bfloat16)
    xa = mx.array([1., 0., 1.], dtype=mx.bfloat16)
    xt = mx.array([0., 1., 2.], dtype=mx.bfloat16)
    stacked = mx.stack((anchor, target), axis=0)
    result = native_pair_outputs(stacked, mx.stack((xa, xt), axis=0))
    mx.eval(result)
    expected = mx.stack((anchor @ xa, target @ xt), axis=0)
    wrong = mx.stack((anchor @ xa, anchor @ xt), axis=0)
    mx.eval(expected, wrong)
    np.testing.assert_array_equal(np.asarray(result.view(mx.uint16)),
                                  np.asarray(expected.view(mx.uint16)))
    assert not np.array_equal(np.asarray(result.view(mx.uint16)),
                              np.asarray(wrong.view(mx.uint16)))


@pytest.mark.metal
def test_slot_executes_fresh_work_and_evaluates_both_outputs():
    import mlx.core as mx

    anchor = mx.array([[1., 2., 3., 4.]], dtype=mx.bfloat16)
    target = mx.array([[4., 3., 2., 1.]], dtype=mx.bfloat16)
    stack = mx.stack((anchor, target), axis=0)
    ring = [mx.array([[1., 1., 0., 0.], [0., 1., 1., 0.]], dtype=mx.bfloat16),
            mx.array([[0., 0., 1., 1.], [1., 0., 0., 1.]], dtype=mx.bfloat16)]
    record = run_slot("native_batched", ["one", "two"],
                      {"one": stack, "two": stack},
                      {"one": ring, "two": ring}, sweeps=3)
    assert record["operation_count"] == 6
    assert record["model_products"] == 12
    assert record["evaluated_outputs"] == 12
    assert record["last_output_shapes"] == [[2, 1], [2, 1]]
    assert record["duration_ns"] > 0
    assert record["last_output_hashes"][0] == record["last_output_hashes"][1]


@pytest.mark.metal
@pytest.mark.model
def test_real_interleaved_session_keeps_equal_work_and_swap_accounting():
    if not (ROOT / "artifacts/family/manifest.json").is_file():
        pytest.skip("requires the pinned converted family artifact")
    names = ("model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.mlp.down_proj.weight")
    record = run_session(ROOT / "artifacts/family", 0, names=names,
                         blocks=1, sweeps=1)
    assert record["artifact_hash"] == "ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585"
    assert record["policy_hash"] == "53628dd18b85be5a969284737dfb1e0f1243365fe431c3bf8e18d75b702281bd"
    assert [slot["mode"] for slot in record["slots"]] == [
        "native_batched", "family_pair", "family_pair", "native_batched"]
    assert all(slot["operation_count"] == 2 and slot["evaluated_outputs"] == 4
               for slot in record["slots"])
    assert all(slot["duration_ns"] > 0 for slot in record["slots"])
    assert record["pressure_before"]["swap_used_bytes"] is not None
    assert record["pressure_after"]["swap_used_bytes"] is not None
