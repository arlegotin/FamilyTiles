"""Freeze the bounded admission verdict without upgrading phase samples to peaks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from familytiles.records import write_json_atomic
from tools.trace_decode import compare_words


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation"


def decide_candidate(*, native_p95_ns: float, candidate_p95_ns: float,
                     native_footprint: int | None, candidate_footprint: int | None,
                     logit_mismatches: int) -> dict:
    if native_p95_ns <= 0 or candidate_p95_ns <= 0 or logit_mismatches < 0:
        raise ValueError("invalid admission observations")
    ratio = candidate_p95_ns / native_p95_ns
    saving = (1 - candidate_footprint / native_footprint
              if native_footprint and candidate_footprint else None)
    return {"p95_ratio": ratio,
            "sampled_footprint_saving": saving,
            "short_logit_mismatches": logit_mismatches,
            "fast_admission": ratio <= 1.5 and logit_mismatches == 0,
            "sampled_memory_target": saving is not None and saving >= 0.10,
            "peak_memory_verified": False,
            "memory_measurement_scope": "maximum of load/prefill/decode physical-footprint phase samples"}


def summarize() -> dict:
    modes = ("B1", "native_pair", "mixed10", "mixed15")
    records = {mode: json.loads((OUT / f"admission-{mode.lower()}.json").read_text())
               for mode in modes}
    for field in ("artifact_hash", "fixture_sha256", "revisions",
                  "prompt_length", "decode_steps_per_stream"):
        if any(records[mode][field] != records["B1"][field] for mode in modes):
            raise ValueError(f"admission records disagree on {field}")
    arrays = {mode: np.load(OUT / records[mode]["arrays_file"], allow_pickle=False)
              for mode in modes}
    for mode in modes:
        for role in ("anchor", "target"):
            if hashlib.sha256(arrays[mode][role].tobytes()).hexdigest() != records[mode]["logit_hashes"][role]:
                raise ValueError(f"admission logit hash changed: {mode}:{role}")
    native = min(("B1", "native_pair"),
                 key=lambda mode: records[mode]["p95_per_request_decode_ns"])
    comparator = records[native]
    outcomes = {}
    for mode in modes:
        record = records[mode]
        mismatches = {role: sum(compare_words(arrays["B1"][role][step],
                                             arrays[mode][role][step])["bit_mismatches"]
                                for step in range(record["decode_steps_per_stream"]))
                      for role in ("anchor", "target")}
        outcomes[mode] = {**decide_candidate(
            native_p95_ns=comparator["p95_per_request_decode_ns"],
            candidate_p95_ns=record["p95_per_request_decode_ns"],
            native_footprint=comparator["memory"]["sampled_process_footprint_max_bytes"],
            candidate_footprint=record["memory"]["sampled_process_footprint_max_bytes"],
            logit_mismatches=sum(mismatches.values())),
            "by_model_logit_mismatches": mismatches,
            "canonical_weight_saving": 1 - record["canonical_weight_bytes"] / comparator["canonical_weight_bytes"],
            "loaded_unique_allocation_saving": 1 - record["loaded_weights"]["unique_weight_allocation_bytes"] /
                comparator["loaded_weights"]["unique_weight_allocation_bytes"]}
    rows = []
    for mode in modes:
        record = records[mode]
        rows.append({"mode": mode, "artifact_hash": record["artifact_hash"],
                     "policy_sha256": record["policy_sha256"] or "",
                     "fixture_sha256": record["fixture_sha256"],
                     "prompt_tokens_per_stream": record["prompt_length"],
                     "decode_steps_per_stream": record["decode_steps_per_stream"],
                     "prefill_ms": record["prefill_ns"] / 1e6,
                     "p50_round_ms": record["p50_round_ns"] / 1e6,
                     "p95_per_request_ms": record["p95_per_request_decode_ns"] / 1e6,
                     "aggregate_model_tokens_per_second": record["aggregate_model_tokens_per_second"],
                     "canonical_weight_bytes": record["canonical_weight_bytes"],
                     "loaded_unique_weight_bytes": record["loaded_weights"]["unique_weight_allocation_bytes"],
                     "sampled_footprint_max_bytes": record["memory"]["sampled_process_footprint_max_bytes"],
                     "sampled_not_instantaneous_peak": True,
                     "logit_mismatches": sum(outcomes[mode]["by_model_logit_mismatches"].values())})
    with (OUT / "runtime.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(OUT / "correctness.json", {
        "artifact_hash": comparator["artifact_hash"],
        "C1": "original complete CPU and bounded GPU checks retained; no codec changes",
        "C2": "eight-step fused versus raw custom logits bitwise equal for both models",
        "short_cached": {mode: outcomes[mode]["by_model_logit_mismatches"] for mode in modes},
        "original_C3_threshold": 0.001,
        "original_C3_corpus_status": "not run; short admission did not pass fast-runtime gate",
        "generation_status": "not run in continuation",
        "source_records": [f"results/continuation/admission-{mode.lower()}.json" for mode in modes]})
    selected = None
    write_json_atomic(OUT / "policy.json", {
        "artifact_hash": comparator["artifact_hash"],
        "selected_policy": selected,
        "candidate_policies": ["results/continuation/policy-10.json",
                               "results/continuation/policy-15.json"],
        "reason": "both candidate p95 latencies exceed 1.50 times the fastest matched native run; no final policy admitted",
        "original_G3": "historical fail remains unchanged"})
    summary = {"artifact_hash": comparator["artifact_hash"],
               "fixture_sha256": comparator["fixture_sha256"],
               "fastest_legitimate_native": native,
               "outcomes": outcomes,
               "decision": "NO-GO for tested fast live-inference policies",
               "capacity_only": "unproven: physical-footprint phase samples are not process high-water peaks, and full C3/B3/B4 runtime was not run",
               "full_evaluation_triggered": False,
               "records": [f"results/continuation/admission-{mode.lower()}.json" for mode in modes]}
    write_json_atomic(OUT / "admission_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = summarize()
    print(json.dumps({"decision": result["decision"],
                      "fastest_native": result["fastest_legitimate_native"],
                      "mixed10_p95_ratio": result["outcomes"]["mixed10"]["p95_ratio"],
                      "mixed15_p95_ratio": result["outcomes"]["mixed15"]["p95_ratio"]}))
