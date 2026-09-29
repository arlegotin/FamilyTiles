"""Locate the first residual raw-pair error on identical stock layer inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from familytiles.records import write_json_atomic
from tools.trace_decode import compare_words


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/epilogue-check"


def _words(value) -> np.ndarray:
    import mlx.core as mx
    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).astype("<u2", copy=True)


def capture() -> None:
    import mlx.core as mx
    import familytiles.model as model_module
    from familytiles.convert import load_artifact

    artifact = load_artifact(ROOT / "artifacts/family")
    fixture = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    runtime = model_module.load_runtime(artifact, "native_pair")
    states = tuple(model_module.prefill(runtime, role, fixture[f"{role}_prompt"])[1]
                   for role in ("anchor", "target"))
    arrays, names = {}, []
    original = model_module._paired_linear

    class EndOfFirstLayer(Exception):
        pass

    def recorded(runtime_, name, x_anchor, x_target):
        output = original(runtime_, name, x_anchor, x_target)
        if name.startswith("model.layers.0."):
            index = len(names)
            names.append(name)
            arrays[f"{index}_anchor_input"] = _words(x_anchor)
            arrays[f"{index}_target_input"] = _words(x_target)
            arrays[f"{index}_anchor_output"] = _words(output[0])
            arrays[f"{index}_target_output"] = _words(output[1])
            if name.endswith("down_proj.weight"):
                raise EndOfFirstLayer
        return output

    model_module._paired_linear = recorded
    try:
        try:
            model_module.step_pair(runtime, states,
                                   (fixture["anchor_next"], fixture["target_next"]))
        except EndOfFirstLayer:
            pass
    finally:
        model_module._paired_linear = original
    if len(names) != 7:
        raise AssertionError(f"expected seven stock first-layer projections, got {len(names)}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "layer0-reference.npz", **arrays)
    write_json_atomic(OUT / "layer0-reference.json", {
        "artifact_hash": artifact.manifest_hash,
        "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
        "names_in_execution_order": names,
        "arrays_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest()
                          for key, value in arrays.items()},
        "scope": "one saved cached step; first-layer stock projection inputs and outputs"})
    print(json.dumps({"captured_projections": len(names)}))


def compare() -> None:
    import mlx.core as mx
    from familytiles.convert import load_artifact
    from familytiles.metal import raw_gemv_pair, kernel_config
    from familytiles.model import _attribute, load_runtime

    metadata = json.loads((OUT / "layer0-reference.json").read_text())
    arrays = np.load(OUT / "layer0-reference.npz", allow_pickle=False)
    for key, expected in metadata["arrays_sha256"].items():
        if hashlib.sha256(arrays[key].tobytes()).hexdigest() != expected:
            raise ValueError("stock trace array changed")
    artifact = load_artifact(ROOT / "artifacts/family")
    if artifact.manifest_hash != metadata["artifact_hash"]:
        raise ValueError("trace uses a different artifact")
    runtime = load_runtime(artifact, "raw_pair")
    observed_dispatch = []
    original_metal_kernel = mx.fast.metal_kernel

    def tracked_kernel(*args, **kwargs):
        observed_dispatch.append({"name": kwargs.get("name"),
            "source_sha256": hashlib.sha256(kwargs["source"].encode()).hexdigest(),
            "header_sha256": hashlib.sha256(kwargs["header"].encode()).hexdigest(),
            "compile_options": kwargs.get("compile_options"),
            "ensure_row_contiguous": kwargs.get("ensure_row_contiguous")})
        return original_metal_kernel(*args, **kwargs)

    mx.fast.metal_kernel = tracked_kernel
    rows = []
    try:
        for index, name in enumerate(metadata["names_in_execution_order"]):
            operand = runtime.operands[name]
            inputs = tuple(mx.array(arrays[f"{index}_{role}_input"].reshape(-1),
                                    dtype=mx.uint16).view(mx.bfloat16)
                           for role in ("anchor", "target"))
            modules = tuple(getattr(*_attribute(runtime.models[role],
                                                  name.removesuffix(".weight")))
                            for role in ("anchor", "target"))
            actual = raw_gemv_pair(operand.anchor_words, operand.native_words,
                                   *inputs, (modules[0].bias, modules[1].bias))
            row = {"name": name, "metrics": {}}
            for role, value in zip(("anchor", "target"), actual):
                expected = arrays[f"{index}_{role}_output"].reshape(-1)
                words = _words(value).reshape(-1)
                row["metrics"][role] = compare_words(expected, words)
                row["metrics"][role]["first_mismatched_rows"] = (
                    np.flatnonzero(expected != words)[:8].astype(int).tolist())
            rows.append(row)
    finally:
        mx.fast.metal_kernel = original_metal_kernel
    first = next((row for row in rows if any(
        row["metrics"][role]["bit_mismatches"] for role in ("anchor", "target"))), None)
    result = {"artifact_hash": artifact.manifest_hash,
              "fixture_sha256": metadata["fixture_sha256"],
              "kernel_config": kernel_config(),
              "metal_py_sha256": hashlib.sha256((ROOT / "src/familytiles/metal.py").read_bytes()).hexdigest(),
              "first_remaining_divergence": first,
              "rows": rows, "actual_dispatch": observed_dispatch,
              "reference": "results/epilogue-check/layer0-reference.json"}
    write_json_atomic(OUT / "remaining-trace.json", result)
    print(json.dumps({"first_remaining_divergence": first,
                      "dispatches": len(observed_dispatch)}))


def summarize() -> None:
    """Bind the corrected short raw run to the saved stock control and close its gate."""
    raw_record = json.loads((OUT / "numerical-raw_pair-8.json").read_text())
    ref_record = json.loads((ROOT / "results/continuation/numerical-b1-8.json").read_text())
    trace = json.loads((OUT / "remaining-trace.json").read_text())
    decision_path = OUT / "decision.json"
    decision = json.loads(decision_path.read_text())
    if raw_record["mode"] != "raw_pair" or ref_record["mode"] != "B1":
        raise ValueError("wrong raw/reference modes")
    for key in ("artifact_hash", "revisions", "fixture_sha256", "tokens"):
        if raw_record[key] != ref_record[key]:
            raise ValueError(f"raw/reference identity differs: {key}")
    if raw_record["artifact_hash"] != decision["artifact_hash"]:
        raise ValueError("run differs from frozen artifact")
    if trace["artifact_hash"] != decision["artifact_hash"]:
        raise ValueError("projection trace differs from frozen artifact")
    if trace["metal_py_sha256"] != hashlib.sha256(
            (ROOT / "src/familytiles/metal.py").read_bytes()).hexdigest():
        raise ValueError("projection trace differs from current Metal source")
    metrics = {}
    with np.load(OUT / raw_record["arrays_file"], allow_pickle=False) as raw, np.load(
            ROOT / "results/continuation" / ref_record["arrays_file"],
            allow_pickle=False) as reference:
        for role in ("anchor", "target"):
            candidate = raw[role]
            stock = reference[role]
            for record, words in ((raw_record, candidate), (ref_record, stock)):
                if hashlib.sha256(words.tobytes()).hexdigest() != record["logit_hashes"][role]:
                    raise ValueError(f"{record['mode']} {role} logit hash changed")
            if candidate.shape != stock.shape or candidate.shape[0] != 8:
                raise ValueError("raw/reference logit shapes differ")
            metrics[role] = {
                "aggregate": compare_words(stock, candidate),
                "per_step": [compare_words(stock[index], candidate[index])
                             for index in range(8)],
                "logit_shape": list(stock.shape),
            }
    threshold = 0.001
    passed = all(metrics[role]["aggregate"]["normalized_rms"] <= threshold
                 for role in ("anchor", "target"))
    if passed:
        raise ValueError("raw gate passed; do not close as a failure")
    record = {
        "outcome": "raw_short_failed",
        "artifact_hash": raw_record["artifact_hash"],
        "revisions": raw_record["revisions"],
        "fixture_sha256": raw_record["fixture_sha256"],
        "source_commit": "7467e00",
        "metal_py_sha256": trace["metal_py_sha256"],
        "kernel_config": raw_record["kernel_config"],
        "actual_dispatch": trace["actual_dispatch"],
        "stock_reference": "results/continuation/numerical-b1-8.json",
        "corrected_raw": "results/epilogue-check/numerical-raw_pair-8.json",
        "first_remaining_projection": trace["first_remaining_divergence"],
        "short_gate": {"steps": 8, "threshold": threshold,
                       "normalization": "sqrt(sum((test-ref)^2)/max(sum(ref^2),1e-24)) over BF16 logits expanded to FP64",
                       "passed": False, "metrics": metrics},
        "full_c3": "not_run",
        "corrected_fused_c2": "not_run",
        "corrected_fused_model": "not_run",
        "additional_32_step_fixtures": "frozen_before_raw_gate_but_not_run",
        "runtime": "not_run",
        "process_peak_memory": "not_run",
    }
    write_json_atomic(OUT / "regression.json", record)
    decision["outcome"] = "raw_short_failed"
    decision["reason"] = "corrected raw paired execution exceeds the unchanged 0.001 short cached-decode criterion"
    decision["regression"] = "results/epilogue-check/regression.json"
    decision["later_gates"] = "not_run_by_stopping_rule"
    write_json_atomic(decision_path, decision)
    print(json.dumps({"outcome": record["outcome"], "nrmse": {
        role: metrics[role]["aggregate"]["normalized_rms"]
        for role in ("anchor", "target")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capture", "compare", "summarize"), required=True)
    args = parser.parse_args()
    {"capture": capture, "compare": compare, "summarize": summarize}[args.mode]()
