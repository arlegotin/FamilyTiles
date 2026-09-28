"""Audit saved G3 measurements and run a bounded, interleaved reproduction."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _digest_words(value: Any) -> str:
    import mlx.core as mx

    mx.eval(value)
    words = np.asarray(value.view(mx.uint16))
    return hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()


def audit_historical(results: Path) -> dict[str, Any]:
    """Preserve both outcomes when code/work identities match but timings do not."""
    gate_files = sorted((results / "gates").glob("*/G3.json"))
    gate_records = [(path, json.loads(path.read_text())) for path in gate_files]
    runs: list[dict[str, Any]] = []
    for cells_path in sorted((results / "kernel/runs").glob("*/cells.json")):
        rows_path = cells_path.parent / "rows.jsonl"
        if not rows_path.is_file():
            continue
        cells = json.loads(cells_path.read_text())["cells"]
        extras = cells[0]["extras"]
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        if not rows or any(row.get("status") != "complete" for row in rows):
            continue
        modes = {mode: sorted((row for row in rows if row["mode"] == mode and not row["trace"]),
                              key=lambda row: row["trial"])
                 for mode in ("family_pair", "family_two_singles", "native_batched", "native_two")}
        if any(len(group) != 3 for group in modes.values()):
            continue
        medians = {mode: int(median(row["completion_ns"] for row in group))
                   for mode, group in modes.items()}
        matching = [(path, gate) for path, gate in gate_records
                    if gate["context"]["implementation_hash"] == extras["source_hash"]
                    and all(gate["observed"]["median_sweep_ns_by_mode"].get(mode) == value
                            for mode, value in medians.items())]
        gate_path, gate = matching[0] if len(matching) == 1 else (None, None)
        rows_by_mode = {mode: [row["completion_ns"] for row in group]
                        for mode, group in modes.items()}
        first = rows[0]
        runs.append({"run_id": cells_path.parent.name,
                     "source_hash": extras["source_hash"],
                     "benchmark_code_hash": first.get("implementation_hash"),
                     "config_hash": first.get("config_hash"),
                     "input_hash": extras["input_hash"],
                     "inventory_hash": extras["inventory_hash"],
                     "artifact_hash": extras["artifact_hash"],
                     "tensor_names": extras["tensor_names"],
                     "pool_size": extras["pool_size"],
                     "sweeps": extras["sweeps"],
                     "cache_limit_bytes": cells[0]["cache_limit_bytes"],
                     "trial_times_ns": rows_by_mode,
                     "median_times_ns": medians,
                     "activation_hashes": sorted(set(row.get("activation_hash") for row in rows
                                                       if row.get("activation_hash"))),
                     "swap_used_before_bytes": [row["pressure_before"].get("swap_used_bytes")
                                                for row in modes["family_pair"]],
                     "swap_used_after_bytes": [row["pressure_after"].get("swap_used_bytes")
                                               for row in modes["family_pair"]],
                     "decision": gate["decision"] if gate else "unmatched",
                     "gate_record": str(gate_path.relative_to(results.parent)) if gate_path else None,
                     "native_batched_two_products": "unrecorded",
                     "missing_historical_fields": ["git_commit", "dirty_diff", "generated_headers",
                                                   "compiled_dispatch_path", "warmup_duration_ns"]})
    by_source: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_source.setdefault(run["source_hash"], []).append(run)
    conflicting = [group for group in by_source.values()
                   if {run["decision"] for run in group} >= {"pass", "fail"}]
    if len(conflicting) != 1:
        raise ValueError("expected one saved same-source pass/fail conflict")
    group = sorted(conflicting[0], key=lambda run: run["median_times_ns"]["family_pair"])
    return {"same_source_runs": group, "classification": "unresolved",
            "explanation": ("Saved source, benchmark-code, config, inventory, fixture and "
                            "activation hashes match. Absolute custom-kernel times differ "
                            "while native batched times are closer. Historical swap usage "
                            "also differs, without recorded growth during either run. "
                            "This does not establish the cause; compiled headers, git state "
                            "and warmup duration were not saved."),
            "other_runs": [run for run in runs if run not in group]}


def native_pair_outputs(weights: Any, activations: Any) -> Any:
    """Two different pre-stacked BF16 matrices and activation vectors."""
    import mlx.core as mx

    if (weights.ndim != 3 or activations.ndim != 2 or weights.shape[0] != 2
            or activations.shape != (2, weights.shape[2]) or
            weights.dtype != mx.bfloat16 or activations.dtype != mx.bfloat16):
        raise ValueError("native pair expects two compatible BF16 products")
    return mx.matmul(weights, activations[..., None])[..., 0]


def run_slot(mode: str, names: list[str], matrices: dict[str, Any],
             input_pool: dict[str, list[Any]], sweeps: int) -> dict[str, Any]:
    """Evaluate fresh work for every declared matrix visit; hash outside timing."""
    import mlx.core as mx
    from familytiles.metal import family_gemv_pair, family_gemv, raw_gemv

    if mode not in ("native_batched", "family_pair", "family_two_singles") or sweeps <= 0:
        raise ValueError("unsupported interleaved sweep mode")
    count = evaluated = 0
    last: list[Any] = []
    start = time.perf_counter_ns()
    for sweep in range(sweeps):
        for name in names:
            pool = input_pool[name]
            if not pool:
                raise ValueError("empty activation ring")
            x = pool[sweep % len(pool)]
            if mode == "native_batched":
                result = native_pair_outputs(matrices[name], x)
                outputs = (result[0], result[1])
            elif mode == "family_pair":
                outputs = family_gemv_pair(matrices[name], x[0], x[1])
            else:
                operand = matrices[name]
                outputs = (raw_gemv(operand.anchor_words, x[0]),
                           family_gemv(operand, x[1]))
            mx.eval(*outputs)
            mx.synchronize()
            count += 1
            evaluated += len(outputs)
            if sweep == sweeps - 1:
                last.append(outputs)
    duration = time.perf_counter_ns() - start
    return {"mode": mode, "duration_ns": duration, "operation_count": count,
            "model_products": 2 * count, "evaluated_outputs": evaluated,
            "last_output_shapes": [list(outputs[0].shape) if mode != "native_batched"
                                   else [2, *outputs[0].shape]
                                   for outputs in last],
            "last_output_hashes": [hashlib.sha256(
                (_digest_words(outputs[0]) + _digest_words(outputs[1])).encode()).hexdigest()
                                   for outputs in last]}


def _pressure() -> dict[str, int | float | None]:
    import psutil

    memory = psutil.virtual_memory()
    try:
        swap = int(psutil.swap_memory().used)
    except OSError:
        swap = None
    return {"available_bytes": int(memory.available),
            "memory_percent": float(memory.percent), "swap_used_bytes": swap}


def _captured_ring(names: tuple[str, ...], shapes: dict[str, tuple[int, int]]) -> tuple[
        dict[str, list[Any]], dict[str, list[Any]], dict[str, Any]]:
    import mlx.core as mx

    captures = {}
    revisions = {}
    for role in ("anchor", "target"):
        record_path = ROOT / "results/inputs" / f"{role}-activations.json"
        record = json.loads(record_path.read_text())
        archive_path = record_path.parent / record["array_file"]
        arrays = np.load(archive_path, allow_pickle=False)
        captures[role] = {}
        for key, info in record["arrays"].items():
            words = arrays[key]
            digest = hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()
            if digest != info["sha256"] or list(words.shape) != info["shape"]:
                raise ValueError(f"captured activation changed: {role}:{key}")
            captures[role][key] = mx.array(words, dtype=mx.uint16).view(mx.bfloat16)
        revisions[role] = record["revision"]
    mx.eval(*(value for role in captures.values() for value in role.values()))
    family: dict[str, list[Any]] = {}
    native: dict[str, list[Any]] = {}
    mapping = {}
    for name in names:
        columns = shapes[name][1]
        role = "down_proj" if columns == 8960 else "q_proj" if columns == 1536 else None
        if role is None:
            raise ValueError(f"no real activation vectors with width {columns}")
        keys = [f"layer{layer}.{role}" for layer in (0, 14, 27)]
        family[name] = [(captures["anchor"][key], captures["target"][key]) for key in keys]
        native[name] = [mx.stack(pair, axis=0) for pair in family[name]]
        mapping[name] = keys
    mx.eval(*(value for pool in native.values() for value in pool))
    return family, native, {"role_revisions": revisions, "source_keys_by_tensor": mapping,
                            "note": "Three genuine saved Qwen activations per width; other same-width roles reuse them for timing."}


def run_session(artifact_path: Path, session_id: int, *, names: tuple[str, ...] | None = None,
                blocks: int = 5, sweeps: int = 8) -> dict[str, Any]:
    """Load both representations once and interleave full matrix sweeps."""
    import mlx.core as mx
    import psutil
    from familytiles.convert import load_artifact, _hash_words
    from familytiles.measure import build_kernel_cells, compute_budget, sample_mlx_memory
    from familytiles.metal import kernel_config, load_operand

    if blocks <= 0 or sweeps <= 0 or not 0 <= session_id < 3:
        raise ValueError("invalid controlled session")
    artifact = load_artifact(artifact_path)
    frozen = json.loads((ROOT / "results/survey/frozen-policy.json").read_text())
    if (frozen.get("family_id") != artifact.manifest["family_id"] or
            any(frozen.get(key) != value for key, value in artifact.manifest["policy"].items())):
        raise ValueError("continuation policy differs from the frozen original")
    if names is None:
        names = tuple(build_kernel_cells(artifact, ROOT / "experiments/kernel-inputs.json")[0]
                      .extras["tensor_names"])
    shapes = {name: tuple(artifact.manifest["tensors"][name]["shape"]) for name in names}
    available = psutil.virtual_memory()
    budget = compute_budget(available.total, available.available)
    raw_bytes = sum(2 * rows * columns for rows, columns in shapes.values())
    if budget.process_limit_bytes < 4 * raw_bytes + 512 * 2**20:
        raise RuntimeError("combined diagnostic operands exceed safe process budget")
    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)
    selected = json.loads((ROOT / "results/survey/selected-family.json").read_text())
    pinned = json.loads((ROOT / "results/survey" / selected["pinned_manifest"]).read_text())
    target_info = pinned["target"]
    if target_info["revision"] != artifact.manifest["revisions"]["target"]:
        raise ValueError("original target shard has the wrong revision")
    family = {}
    native = {}
    for name in names:
        operand = load_operand(artifact, name)
        family[name] = operand
        info = target_info["tensors"][name]
        if tuple(info["shape"]) != shapes[name] or info["dtype"] != "BF16":
            raise ValueError("native target matrix metadata disagrees")
        shard = (ROOT / "artifacts/downloaded" /
                 target_info["repo"].replace("/", "--") /
                 target_info["revision"] / info["shard"])
        words = np.memmap(shard, mode="r", dtype="<u2",
                          offset=info["data_start"] + info["byte_offset"],
                          shape=shapes[name])
        if _hash_words(words) != artifact.manifest["tensors"][name]["target"]["sha256"]:
            raise ValueError("original target matrix hash mismatch")
        target = mx.array(words, dtype=mx.uint16).view(mx.bfloat16)
        native[name] = mx.stack((operand.anchor_words.view(mx.bfloat16), target), axis=0)
        mx.eval(native[name])
        del target, words
    family_inputs, native_inputs, input_identity = _captured_ring(names, shapes)
    if input_identity["role_revisions"] != artifact.manifest["revisions"]:
        raise ValueError("captured activations have different model revisions")
    # Compilation and every shape are warmed before measured blocks.
    run_slot("native_batched", list(names), native, native_inputs, 1)
    run_slot("family_pair", list(names), family, family_inputs, 1)
    mx.synchronize()
    mx.reset_peak_memory()
    pressure_before = _pressure()
    memory_before = sample_mlx_memory()
    slots = []
    ratios = []
    for block in range(blocks):
        measured = []
        for mode in ("native_batched", "family_pair", "family_pair", "native_batched"):
            matrices, inputs = ((native, native_inputs) if mode == "native_batched"
                                else (family, family_inputs))
            slot = run_slot(mode, list(names), matrices, inputs, sweeps)
            slot["block"] = block
            slots.append(slot)
            measured.append(slot)
        native_time = median((measured[0]["duration_ns"], measured[3]["duration_ns"]))
        family_time = median((measured[1]["duration_ns"], measured[2]["duration_ns"]))
        ratios.append(family_time / native_time)
    single = None
    if session_id == 0 and blocks == 5:
        run_slot("family_two_singles", list(names), family, family_inputs, 1)
        single = run_slot("family_two_singles", list(names), family, family_inputs, sweeps)
    pressure_after = _pressure()
    memory_after = sample_mlx_memory()
    config = kernel_config()
    return {"session": session_id, "artifact_hash": artifact.manifest_hash,
            "policy_hash": frozen["policy_sha256"],
            "revisions": artifact.manifest["revisions"],
            "kernel_config": config,
            "benchmark_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "package_versions": {name: metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
            "tensor_names": list(names), "shapes": {name: list(shape) for name, shape in shapes.items()},
            "input_identity": input_identity,
            "work": {"sweeps_per_slot": sweeps, "blocks": blocks,
                     "visits_per_slot": len(names) * sweeps,
                     "biases": "absent in all modes, matching original G3 matrix-only sweep",
                     "native_stack": "prebuilt once from distinct original matrices",
                     "contiguity": "custom ensure_row_contiguous=False; native stack contiguous"},
            "slots": slots, "block_ratios": ratios, "median_ratio": float(median(ratios)),
            "two_singles_control": single,
            "pressure_before": pressure_before, "pressure_after": pressure_after,
            "mlx_before": memory_before, "mlx_after": memory_after,
            "combined_diagnostic_storage": True,
            "safe_budget_bytes": budget.process_limit_bytes}


def run_interleaved(artifact: Path, sessions: int = 3,
                    out: Path = ROOT / "results/continuation") -> dict[str, Any]:
    """Reproduce three sessions in fresh processes without overwriting G3."""
    from familytiles.records import write_json_atomic

    if sessions != 3:
        raise ValueError("controlled reproduction requires three sessions")
    out.mkdir(parents=True, exist_ok=True)
    historical = audit_historical(ROOT / "results")
    records = []
    for index in range(sessions):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker", "--session", str(index),
             "--artifact", str(artifact), "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"controlled session {index} failed: {completed.stderr[-1000:]}")
        path = out / "blocks" / f"session-{index}.json"
        records.append(json.loads(path.read_text()))
    ratios = [record["median_ratio"] for record in records]
    middle = median(ratios)
    spread = max(abs(ratio - middle) / middle for ratio in ratios)
    stable = spread <= 0.10
    audit = {"artifact_hash": records[0]["artifact_hash"],
             "historical": historical,
             "current_sessions": [f"results/continuation/blocks/session-{i}.json"
                                  for i in range(sessions)],
             "session_median_ratios": ratios, "ratio_relative_spread": spread,
             "stable": stable,
             "performance_classification": (
                 "stable_current_failure" if stable and middle > 1.5 else
                 "stable_current_pass" if stable else "inconclusive"),
             "historical_discrepancy": historical["classification"],
             "note": "Original G3 gate and all raw records remain unchanged."}
    write_json_atomic(out / "audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/family")
    parser.add_argument("--out", type=Path, default=ROOT / "results/continuation")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        from familytiles.records import write_json_atomic

        session = run_session(args.artifact, args.session)
        write_json_atomic(args.out / "blocks" / f"session-{args.session}.json", session)
        print(json.dumps({"session": args.session, "median_ratio": session["median_ratio"]}))
    else:
        result = run_interleaved(args.artifact, out=args.out)
        print(json.dumps({"ratios": result["session_median_ratios"],
                          "performance_classification": result["performance_classification"]}))
