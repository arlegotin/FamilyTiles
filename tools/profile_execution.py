"""Bounded per-matrix cost profile of exact resident execution choices."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from familytiles.baselines import (ExactChunks, ExactFamilyBacking, decode_exact,
                                   decode_family_exact, encode_exact,
                                   encode_family_exact, exact_allocated_bytes)
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation"


@dataclass(frozen=True)
class ResidentPair:
    kind: str
    anchor: ExactChunks | ExactFamilyBacking
    target: ExactChunks | None
    transform: str

    @property
    def owned_arrays(self) -> tuple[np.ndarray, ...]:
        chunks = ((self.anchor.anchor, self.anchor.target)
                  if isinstance(self.anchor, ExactFamilyBacking)
                  else (self.anchor, self.target))
        return tuple(array for chunk in chunks if chunk is not None
                     for array in (chunk.chunk_index, chunk.payload))

    @property
    def allocated_bytes(self) -> int:
        return sum(int(array.nbytes) for array in self.owned_arrays)


def build_resident_b3(anchor: np.ndarray, target: np.ndarray) -> ResidentPair:
    def best(words: np.ndarray) -> ExactChunks:
        return min((encode_exact(words, transform)
                    for transform in ("native", "byte_plane", "field")),
                   key=exact_allocated_bytes)
    return ResidentPair("B3", best(anchor), best(target), "independent")


def build_resident_b4(anchor: np.ndarray, target: np.ndarray) -> ResidentPair:
    backing = min((encode_family_exact(anchor, target, transform, "compressed")
                   for transform in ("xor", "ordered_delta")),
                  key=exact_allocated_bytes)
    return ResidentPair("B4", backing, None, backing.family_transform)


def reconstruct_resident_pair(backing: ResidentPair) -> tuple[np.ndarray, np.ndarray]:
    if backing.kind == "B3" and isinstance(backing.anchor, ExactChunks) and backing.target:
        a = np.concatenate(tuple(decode_exact(backing.anchor))).reshape(backing.anchor.shape)
        t = np.concatenate(tuple(decode_exact(backing.target))).reshape(backing.target.shape)
    elif backing.kind == "B4" and isinstance(backing.anchor, ExactFamilyBacking):
        pairs = tuple(decode_family_exact(backing.anchor))
        a = np.concatenate([part[0] for part in pairs]).reshape(backing.anchor.anchor.shape)
        t = np.concatenate([part[1] for part in pairs]).reshape(backing.anchor.target.shape)
    else:
        raise ValueError("invalid resident pair")
    if a.dtype != np.dtype("<u2") or t.dtype != np.dtype("<u2"):
        raise ValueError("baseline did not reconstruct BF16 words")
    return a, t


def _original_pair(artifact, pinned: dict, name: str) -> tuple[np.memmap, np.memmap]:
    shape = tuple(artifact.manifest["tensors"][name]["shape"])
    values = []
    for role in ("anchor", "target"):
        source = pinned[role]
        info = source["tensors"][name]
        if info["dtype"] != "BF16" or tuple(info["shape"]) != shape:
            raise ValueError("pinned source changed shape or dtype")
        path = (ROOT / "artifacts/downloaded" / source["repo"].replace("/", "--") /
                source["revision"] / info["shard"])
        words = np.memmap(path, mode="r", dtype="<u2",
                          offset=info["data_start"] + info["byte_offset"], shape=shape)
        if hashlib.sha256(memoryview(words).cast("B")).hexdigest() != artifact.manifest["tensors"][name][role]["sha256"]:
            raise ValueError(f"pinned source hash changed: {role}:{name}")
        values.append(words)
    return values[0], values[1]


def _run_pair(mode: str, data: Any, x: tuple[Any, Any]) -> tuple[Any, Any]:
    import mlx.core as mx
    from familytiles.metal import decode_words, family_gemv_pair, raw_gemv_pair

    if mode == "native":
        y = mx.matmul(data, x[..., None])[..., 0]
        return y[0], y[1]
    if mode == "raw_custom":
        return raw_gemv_pair(data[0], data[1], *x)
    if mode == "fused":
        return family_gemv_pair(data, *x)
    if mode == "staged_native":
        anchor = data.anchor_words.view(mx.bfloat16)
        target = decode_words(data).view(mx.bfloat16)
        first = x[0] @ anchor.T
        second = x[1] @ target.T
        mx.eval(first, second)
        return first, second
    if mode in ("B3", "B4"):
        anchor_words, target_words = reconstruct_resident_pair(data)
        a = mx.array(anchor_words, dtype=mx.uint16).view(mx.bfloat16)
        t = mx.array(target_words, dtype=mx.uint16).view(mx.bfloat16)
        first, second = x[0] @ a.T, x[1] @ t.T
        mx.eval(first, second)
        return first, second
    raise ValueError(f"unknown profile mode: {mode}")


def _time_mode(mode: str, data: Any, pool: list[tuple[Any, Any]], repeats: int = 3) -> int:
    import mlx.core as mx

    _run_pair(mode, data, pool[0])
    mx.synchronize()
    start = time.perf_counter_ns()
    for index in range(repeats):
        output = _run_pair(mode, data, pool[index % len(pool)])
        mx.eval(*output)
        mx.synchronize()
    return (time.perf_counter_ns() - start) // repeats


def profile(*, max_names: int = 21) -> dict:
    import mlx.core as mx
    import psutil

    from familytiles.convert import load_artifact
    from familytiles.measure import build_kernel_cells, compute_budget, sample_mlx_memory
    from familytiles.metal import load_operand
    from tools.audit_g3 import _captured_ring

    if max_names < 1 or max_names > 21:
        raise ValueError("profile is bounded to the original 21-matrix sweep")
    artifact = load_artifact(ROOT / "artifacts/family")
    selected = json.loads((ROOT / "results/survey/selected-family.json").read_text())
    pinned = json.loads((ROOT / "results/survey" / selected["pinned_manifest"]).read_text())
    names = tuple(build_kernel_cells(artifact, ROOT / "experiments/kernel-inputs.json")[0]
                  .extras["tensor_names"])[:max_names]
    shapes = {name: tuple(artifact.manifest["tensors"][name]["shape"]) for name in names}
    family_inputs, _, input_identity = _captured_ring(names, shapes)
    budget = compute_budget(psutil.virtual_memory().total, psutil.virtual_memory().available)
    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)
    records = []
    for name in names:
        a_words, t_words = _original_pair(artifact, pinned, name)
        operand = load_operand(artifact, name)
        target = mx.array(t_words, dtype=mx.uint16).view(mx.bfloat16)
        native = mx.stack((operand.anchor_words.view(mx.bfloat16), target), axis=0)
        raw = operand.anchor_words, target.view(mx.uint16)
        mx.eval(native, *raw)
        b3, b4 = build_resident_b3(a_words, t_words), build_resident_b4(a_words, t_words)
        for backing in (b3, b4):
            got_a, got_t = reconstruct_resident_pair(backing)
            if not np.array_equal(got_a, a_words) or not np.array_equal(got_t, t_words):
                raise AssertionError(f"{backing.kind} failed exact source verification: {name}")
        inputs = family_inputs[name]
        native_inputs = [mx.stack(pair, axis=0) for pair in inputs]
        mx.eval(*native_inputs)
        timings = {mode: _time_mode(mode, data, inputs) for mode, data in (
            ("raw_custom", raw), ("fused", operand),
            ("staged_native", operand), ("B3", b3), ("B4", b4))}
        timings["native"] = _time_mode("native", native, native_inputs)
        source = artifact.manifest["tensors"][name]
        records.append({"name": name, "shape": list(shapes[name]), "visits": 1,
                        "raw_pair_bytes": 4 * a_words.size,
                        "family_anchor_bytes": source["anchor"]["allocated_bytes"],
                        "family_target_bytes": source["target"]["allocated_bytes"],
                        "b3_resident_bytes": b3.allocated_bytes,
                        "b4_resident_bytes": b4.allocated_bytes,
                        "b3_transform": (b3.anchor.transform, b3.target.transform),
                        "b4_transform": b4.transform,
                        "maximum_decoded_matrix_bytes": 2 * a_words.size,
                        "paired_decoded_scratch_lower_bound_bytes": 4 * a_words.size,
                        **{mode + "_ns": duration for mode, duration in timings.items()}})
        del a_words, t_words, operand, target, native, raw, b3, b4, native_inputs
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "operator_profile.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    totals = {mode: sum(row[mode + "_ns"] for row in records) for mode in (
        "native", "raw_custom", "fused", "staged_native", "B3", "B4")}
    result = {"artifact_hash": artifact.manifest_hash,
              "revisions": artifact.manifest["revisions"],
              "fixture": input_identity, "scope": "synchronized representative per-operation profile",
              "names": list(names), "timing_repeats_per_operation": 3,
              "sums_are_not_a_complete_sweep": True,
              "per_operation_sum_ns": totals,
              "verified_b3_b4_tensors": list(names),
              "full_b3_b4_baseline_verified": max_names == len(artifact.manifest["tensors"]),
              "resident_b3_b4": "encoded arrays owned in RAM during each measured operation",
              "budget_bytes": budget.process_limit_bytes,
              "mlx_after": sample_mlx_memory(),
              "profile_csv": "results/continuation/operator_profile.csv"}
    write_json_atomic(OUT / "baseline_runtime.json", result)
    return result


def run_complete_sweep() -> dict:
    """Timed full original matrix list, with prebuilt operands and fresh outputs."""
    import mlx.core as mx
    import psutil

    from familytiles.convert import load_artifact
    from familytiles.measure import build_kernel_cells, compute_budget
    from familytiles.metal import load_operand
    from tools.audit_g3 import _captured_ring

    artifact = load_artifact(ROOT / "artifacts/family")
    selected = json.loads((ROOT / "results/survey/selected-family.json").read_text())
    pinned = json.loads((ROOT / "results/survey" / selected["pinned_manifest"]).read_text())
    names = tuple(build_kernel_cells(artifact, ROOT / "experiments/kernel-inputs.json")[0]
                  .extras["tensor_names"])
    shapes = {name: tuple(artifact.manifest["tensors"][name]["shape"]) for name in names}
    budget = compute_budget(psutil.virtual_memory().total, psutil.virtual_memory().available)
    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)
    family = {name: load_operand(artifact, name) for name in names}
    native = {}
    for name in names:
        _, target_words = _original_pair(artifact, pinned, name)
        target = mx.array(target_words, dtype=mx.uint16).view(mx.bfloat16)
        native[name] = mx.stack((family[name].anchor_words.view(mx.bfloat16), target), axis=0)
        mx.eval(native[name])
        del target_words, target
    inputs, native_inputs, identity = _captured_ring(names, shapes)

    def sweep(mode: str, iteration: int) -> int:
        start = time.perf_counter_ns()
        for name in names:
            item = (native[name] if mode == "native" else family[name])
            vector = (native_inputs[name] if mode == "native" else inputs[name])[iteration % 3]
            output = _run_pair(mode, item, vector)
            mx.eval(*output)
            mx.synchronize()
        return time.perf_counter_ns() - start

    for mode in ("native", "staged_native"):
        sweep(mode, 0)
    rows = []
    for trial in range(3):
        for mode in ("native", "staged_native", "staged_native", "native"):
            rows.append({"trial": trial, "mode": mode,
                         "duration_ns": sweep(mode, trial + 1),
                         "matrix_visits": len(names), "model_products": 2 * len(names)})
    from statistics import median
    times = {mode: median(row["duration_ns"] for row in rows if row["mode"] == mode)
             for mode in ("native", "staged_native")}
    result = {"artifact_hash": artifact.manifest_hash,
              "revisions": artifact.manifest["revisions"],
              "matrix_names": list(names), "activation_identity": identity,
              "biases": "absent in both modes, matching original G3 sweep",
              "native_weight_stack": "prebuilt once from distinct original matrices",
              "source": "tools/profile_execution.py:run_complete_sweep",
              "rows": rows, "median_ns": times,
              "staged_vs_native_ratio": times["staged_native"] / times["native"],
              "scope": "complete original matrix sweep; not end-to-end inference"}
    write_json_atomic(OUT / "staged_sweep.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-names", type=int, default=21)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.sweep:
        result = run_complete_sweep()
        print(json.dumps({"matrix_visits": len(result["matrix_names"]),
                          "median_ms": {key: value / 1e6
                                        for key, value in result["median_ns"].items()},
                          "staged_vs_native_ratio": result["staged_vs_native_ratio"]}))
    else:
        result = profile(max_names=args.max_names)
        print(json.dumps({"names": len(result["names"]),
                          "per_operation_sum_ms": {key: value / 1e6
                                                   for key, value in result["per_operation_sum_ns"].items()}}))
