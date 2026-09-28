"""Fresh-process stock versus FamilyTiles materialized Qwen2 smoke check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from familytiles.convert import load_artifact
from familytiles.measure import sample_memory, sample_mlx_memory
from familytiles.model import _config_path, load_runtime, loaded_weight_ledger, prefill
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def run(mode: str) -> dict:
    artifact = load_artifact(ROOT / "artifacts/family")
    source = json.loads((ROOT / "results/inputs/target-activations.json").read_text())
    token_ids = source["token_ids"]
    if mode in ("A1", "B1"):
        runtime = load_runtime(artifact, mode)
        weight_ledger = loaded_weight_ledger(runtime)
        loaded = {"process": sample_memory(os.getpid(), "loaded"),
                  "mlx": sample_mlx_memory()}
        logits, state = prefill(runtime, "target", token_ids)
        mx.eval(logits)
        assert state.position == len(token_ids)
        decoded_bound = (max(module.last_prefill_decoded_matrix_bytes
                             for module in runtime.models["target"].modules()
                             if hasattr(module, "last_prefill_decoded_matrix_bytes"))
                         if mode == "A1" else 0)
    elif mode == "stock":
        model = load_model(_config_path(artifact, "target").parent, strict=True)[0]
        cache = make_prompt_cache(model)
        loaded = {"process": sample_memory(os.getpid(), "loaded"),
                  "mlx": sample_mlx_memory()}
        logits = model(mx.array([token_ids], dtype=mx.int32), cache=cache)[0, -1, :]
        mx.eval(logits, *(value for entry in cache for value in entry.state))
        decoded_bound = 0
        weight_ledger = None
    else:
        raise ValueError("expected stock, B1, or A1")
    words = np.asarray(logits.view(mx.uint16)).copy()
    out = ROOT / "results/correctness"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{mode.lower()}-target-prefill-logits.npy", words, allow_pickle=False)
    record = {"mode": mode, "artifact_hash": artifact.manifest_hash,
              "model_revision": artifact.manifest["revisions"]["target"],
              "token_ids_sha256": source["token_ids_sha256"],
              "output_words": int(words.size),
              "output_sha256": hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest(),
              "loaded_weight_ledger": weight_ledger,
              "loaded": loaded,
              "prefill": {"process": sample_memory(os.getpid(), "prefill"),
                          "mlx": sample_mlx_memory()},
              "maximum_one_matrix_decoded_bytes": decoded_bound}
    write_json_atomic(out / f"{mode.lower()}-target-prefill.json", record)
    return record


def compare() -> dict:
    out = ROOT / "results/correctness"
    a1_record = json.loads((out / "a1-target-prefill.json").read_text())
    stock_record = json.loads((out / "stock-target-prefill.json").read_text())
    b1_record = json.loads((out / "b1-target-prefill.json").read_text())
    for field in ("artifact_hash", "model_revision", "token_ids_sha256", "output_words"):
        if a1_record[field] != stock_record[field] or a1_record[field] != b1_record[field]:
            raise ValueError(f"incompatible model check records: {field}")
    words = {mode: np.load(out / f"{mode}-target-prefill-logits.npy", allow_pickle=False)
             for mode in ("a1", "stock", "b1")}
    for mode, record in (("a1", a1_record), ("stock", stock_record), ("b1", b1_record)):
        if hashlib.sha256(words[mode].astype("<u2", copy=False).tobytes()).hexdigest() != record["output_sha256"]:
            raise ValueError(f"{mode} logits file does not match its record")
    a = (words["a1"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (words["stock"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    difference = a - b
    result = {"artifact_hash": a1_record["artifact_hash"],
              "model_revision": a1_record["model_revision"],
              "token_ids_sha256": a1_record["token_ids_sha256"],
              "logit_words": int(a.size),
              "bit_mismatches": int(np.count_nonzero(words["a1"] != words["stock"])),
              "b1_stock_bit_mismatches": int(np.count_nonzero(words["b1"] != words["stock"])),
              "max_absolute_error": float(np.max(np.abs(difference))),
              "normalized_rms": float(np.sqrt(np.sum(difference * difference) /
                                              max(np.sum(b * b), 1e-24))),
              "argmax_agreement": bool(np.argmax(a) == np.argmax(b)),
              "two_model_weight_allocation_saving": (
                  1 - a1_record["loaded_weight_ledger"]["unique_weight_allocation_bytes"] /
                  b1_record["loaded_weight_ledger"]["unique_weight_allocation_bytes"]),
              "two_model_loaded_physical_footprint_saving": (
                  1 - a1_record["loaded"]["process"]["physical_footprint_bytes"] /
                  b1_record["loaded"]["process"]["physical_footprint_bytes"]),
              "one_request_prefill_sampled_physical_footprint_saving": (
                  1 - a1_record["prefill"]["process"]["physical_footprint_bytes"] /
                  b1_record["prefill"]["process"]["physical_footprint_bytes"]),
              "physical_footprints_are_phase_samples_not_instantaneous_peaks": True,
              "records": ["results/correctness/a1-target-prefill.json",
                          "results/correctness/b1-target-prefill.json",
                          "results/correctness/stock-target-prefill.json"]}
    write_json_atomic(out / "single-model-comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stock", "B1", "A1", "compare"), required=True)
    mode = parser.parse_args().mode
    result = compare() if mode == "compare" else run(mode)
    if mode == "compare":
        print(json.dumps({"mode": mode, "bit_mismatches": result["bit_mismatches"],
                          "normalized_rms": result["normalized_rms"]}))
    else:
        print(json.dumps({"mode": result["mode"], "output_words": result["output_words"],
                          "active_bytes": result["prefill"]["mlx"]["active_bytes"],
                          "physical_footprint_bytes": result["prefill"]["process"]["physical_footprint_bytes"]}))
