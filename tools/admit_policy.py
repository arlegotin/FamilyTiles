"""Fresh-process 128/16 exact paired-runtime admission measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation"


def fixed_prompt(original: list[int], length: int) -> list[int]:
    if not original or length < len(original):
        raise ValueError("fixed prompt must preserve the full saved prefix")
    return (original * ((length + len(original) - 1) // len(original)))[:length]


def percentile_ns(values: list[int], percentile: int) -> float:
    if not values or not 0 <= percentile <= 100:
        raise ValueError("invalid latency sample or percentile")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _words(value) -> np.ndarray:
    import mlx.core as mx
    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).astype("<u2", copy=True)


def run(mode: str, *, prompt_length: int = 128, steps: int = 16) -> dict:
    import mlx.core as mx
    from familytiles.convert import load_artifact, canonical_ledger
    from familytiles.measure import (sample_memory, sample_mlx_memory,
                                     _swap_pressure)
    from familytiles.model import (load_runtime, loaded_weight_ledger, prefill,
                                   release_request, step_pair)

    if mode not in ("B1", "native_pair", "mixed10", "mixed15"):
        raise ValueError("unsupported admission mode")
    if prompt_length != 128 or steps != 16:
        raise ValueError("admission workload is frozen at 128 prompt and 16 decode steps")
    artifact = load_artifact(ROOT / "artifacts/family")
    original = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    prompts = {role: fixed_prompt(original[f"{role}_prompt"], prompt_length)
               for role in ("anchor", "target")}
    tokens = [(prompts["anchor"][index], prompts["target"][(index + 7) % prompt_length])
              for index in range(steps)]
    fixture_hash = hashlib.sha256(json.dumps({"prompts": prompts, "tokens": tokens},
                                             sort_keys=True).encode()).hexdigest()
    policy_path = None
    if mode.startswith("mixed"):
        policy_path = OUT / f"policy-{mode.removeprefix('mixed')}.json"
        policy = json.loads(policy_path.read_text())
        runtime = load_runtime(artifact, "mixed", execution_policy=policy)
    else:
        runtime = load_runtime(artifact, mode)
    weight_ledger = loaded_weight_ledger(runtime)
    load_memory = {"process": sample_memory(os.getpid(), "loaded"),
                   "mlx": sample_mlx_memory()}
    pressure_before = _swap_pressure()
    mx.reset_peak_memory()
    prefill_start = time.perf_counter_ns()
    states = tuple(prefill(runtime, role, prompts[role])[1]
                   for role in ("anchor", "target"))
    mx.synchronize()
    prefill_ns = time.perf_counter_ns() - prefill_start
    prefill_memory = {"process": sample_memory(os.getpid(), "prefill"),
                      "mlx": sample_mlx_memory()}
    mx.reset_peak_memory()
    latencies = []
    logit_words = {role: [] for role in ("anchor", "target")}
    for pair in tokens:
        start = time.perf_counter_ns()
        outputs = step_pair(runtime, states, pair)
        mx.eval(*outputs)
        mx.synchronize()
        latencies.append(time.perf_counter_ns() - start)
        for role, value in zip(("anchor", "target"), outputs):
            logit_words[role].append(_words(value))
    decode_memory = {"process": sample_memory(os.getpid(), "decode"),
                     "mlx": sample_mlx_memory()}
    pressure_after = _swap_pressure()
    for state in states:
        release_request(state)
    mx.synchronize()
    teardown_memory = {"process": sample_memory(os.getpid(), "teardown"),
                       "mlx": sample_mlx_memory()}
    arrays_path = OUT / f"admission-{mode.lower()}-logits.npz"
    np.savez_compressed(arrays_path, **{role: np.stack(values)
                                      for role, values in logit_words.items()})
    sampled_footprints = [phase["process"]["physical_footprint_bytes"]
                          for phase in (load_memory, prefill_memory, decode_memory)
                          if phase["process"]["physical_footprint_bytes"] is not None]
    record = {"mode": mode, "artifact_hash": artifact.manifest_hash,
              "policy_sha256": (hashlib.sha256(policy_path.read_bytes()).hexdigest()
                                if policy_path is not None else None),
              "source": "tools/admit_policy.py", "revisions": artifact.manifest["revisions"],
              "fixture_sha256": fixture_hash, "prompt_length": prompt_length,
              "decode_steps_per_stream": steps, "completed_model_steps": 2 * steps,
              "token_schedule_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
              "canonical_weight_bytes": (canonical_ledger(artifact)["family_bytes"] +
                  (policy["extra_resident_bytes"] if policy_path is not None else 0)
                  if mode.startswith("mixed") else canonical_ledger(artifact)["w1_bytes"]),
              "loaded_weights": weight_ledger,
              "prefill_ns": prefill_ns, "decode_round_ns": latencies,
              "p50_round_ns": percentile_ns(latencies, 50),
              "p95_per_request_decode_ns": percentile_ns(latencies, 95),
              "aggregate_model_tokens_per_second": 2 * steps / (sum(latencies) / 1e9),
              "memory": {"loaded": load_memory, "prefill": prefill_memory,
                         "decode": decode_memory, "teardown": teardown_memory,
                         "sampled_process_footprint_max_bytes": (
                             max(sampled_footprints) if sampled_footprints else None),
                         "sampled_not_instantaneous_peak": True},
              "pressure_before": pressure_before, "pressure_after": pressure_after,
              "arrays_file": arrays_path.name,
              "logit_hashes": {role: hashlib.sha256(np.stack(values).tobytes()).hexdigest()
                               for role, values in logit_words.items()}}
    write_json_atomic(OUT / f"admission-{mode.lower()}.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("B1", "native_pair", "mixed10", "mixed15"), required=True)
    args = parser.parse_args()
    result = run(args.mode)
    print(json.dumps({"mode": args.mode, "p95_ms": result["p95_per_request_decode_ns"] / 1e6,
                      "sampled_footprint_max_bytes": result["memory"]["sampled_process_footprint_max_bytes"]}))
