"""Real Qwen2 raw-pair runner versus separate raw and stock execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_tokenizer

from familytiles.convert import load_artifact
from familytiles.measure import sample_memory, sample_mlx_memory
from familytiles.model import (_config_path, load_runtime, prefill, step_pair,
                               step_single, loaded_weight_ledger)
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
SECOND_PROMPT = "A tank contains 60 liters of water. It loses 3 liters per minute. How much remains after 7 minutes?"


def _word_array(logits: mx.array) -> np.ndarray:
    mx.eval(logits)
    return np.asarray(logits.view(mx.uint16)).copy()


def _cache_hashes(state) -> list[dict[str, str]]:
    hashes = []
    for entry in state.cache:
        keys, values = entry.state
        mx.eval(keys, values)
        hashes.append({kind: hashlib.sha256(
            np.asarray(array.view(mx.uint16)).astype("<u2", copy=False).tobytes()
        ).hexdigest() for kind, array in (("keys", keys), ("values", values))})
    return hashes


def _input_spec(artifact) -> dict:
    first = json.loads((ROOT / "results/inputs/anchor-activations.json").read_text())
    tokenizer = load_tokenizer(_config_path(artifact, "target").parent)
    second_ids = tokenizer.encode(SECOND_PROMPT, add_special_tokens=False)
    if not second_ids:
        raise ValueError("second lockstep prompt produced no tokens")
    return {"anchor_prompt": first["token_ids"], "target_prompt": second_ids,
            "anchor_next": first["token_ids"][-1], "target_next": second_ids[-1],
            "first_prompt": first["prompt"], "second_prompt": SECOND_PROMPT,
            "tokenizer_revision": artifact.manifest["revisions"]["target"]}


def run(mode: str) -> dict:
    artifact = load_artifact(ROOT / "artifacts/family")
    inputs = _input_spec(artifact)
    runtime = load_runtime(artifact, mode)
    loaded = {"process": sample_memory(os.getpid(), "loaded"),
              "mlx": sample_mlx_memory(),
              "weights": loaded_weight_ledger(runtime)}
    anchor_prefill, anchor = prefill(runtime, "anchor", inputs["anchor_prompt"])
    target_prefill, target = prefill(runtime, "target", inputs["target_prompt"])
    prefill_words = (_word_array(anchor_prefill), _word_array(target_prefill))
    prefill_cache_hashes = {"anchor": _cache_hashes(anchor),
                            "target": _cache_hashes(target)}
    if mode == "raw_pair":
        _, anchor_control = prefill(runtime, "anchor", inputs["anchor_prompt"])
        _, target_control = prefill(runtime, "target", inputs["target_prompt"])
        outputs = step_pair(runtime, (anchor, target),
                            (inputs["anchor_next"], inputs["target_next"]))
        controls = (step_single(runtime, anchor_control, inputs["anchor_next"]),
                    step_single(runtime, target_control, inputs["target_next"]))
        pair_words = tuple(_word_array(item) for item in outputs)
        single_words = tuple(_word_array(item) for item in controls)
        control_mismatches = [int(np.count_nonzero(a != b))
                              for a, b in zip(pair_words, single_words)]
    elif mode == "B1":
        outputs = (step_single(runtime, anchor, inputs["anchor_next"]),
                   step_single(runtime, target, inputs["target_next"]))
        pair_words = tuple(_word_array(item) for item in outputs)
        control_mismatches = None
    else:
        raise ValueError("expected raw_pair or B1")
    out = ROOT / "results/correctness"
    out.mkdir(parents=True, exist_ok=True)
    arrays_path = out / f"{mode.lower()}-lockstep-logits.npz"
    np.savez(arrays_path, anchor=pair_words[0], target=pair_words[1],
             prefill_anchor=prefill_words[0], prefill_target=prefill_words[1])
    record = {"mode": mode, "artifact_hash": artifact.manifest_hash,
              "revisions": artifact.manifest["revisions"], "inputs": inputs,
              "input_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
              "logit_hashes": {role: hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()
                               for role, words in zip(("anchor", "target"), pair_words)},
              "prefill_logit_hashes": {role: hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()
                                       for role, words in zip(("anchor", "target"), prefill_words)},
              "prefill_cache_hashes": prefill_cache_hashes,
              "logit_words_per_model": [int(words.size) for words in pair_words],
              "pair_two_single_bit_mismatches": control_mismatches,
              "position_after_step": [anchor.position, target.position],
              "kv_cache_distinct": (anchor.cache is not target.cache and
                                    all(a is not b for a, b in zip(anchor.cache, target.cache))),
              "loaded": loaded,
              "after_step": {"process": sample_memory(os.getpid(), "after_step"),
                             "mlx": sample_mlx_memory()},
              "arrays_file": arrays_path.name}
    write_json_atomic(out / f"{mode.lower()}-lockstep.json", record)
    return record


def compare() -> dict:
    out = ROOT / "results/correctness"
    records = {mode: json.loads((out / f"{mode}-lockstep.json").read_text())
               for mode in ("raw_pair", "b1")}
    if (records["raw_pair"]["artifact_hash"] != records["b1"]["artifact_hash"] or
            records["raw_pair"]["input_sha256"] != records["b1"]["input_sha256"]):
        raise ValueError("raw and stock lockstep inputs disagree")
    arrays = {mode: np.load(out / records[mode]["arrays_file"], allow_pickle=False)
              for mode in ("raw_pair", "b1")}
    metrics = {}
    for role in ("anchor", "target"):
        raw_words = arrays["raw_pair"][role]
        stock_words = arrays["b1"][role]
        for mode, words in (("raw_pair", raw_words), ("b1", stock_words)):
            digest = hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()
            if digest != records[mode]["logit_hashes"][role]:
                raise ValueError(f"{mode}:{role} saved logits hash mismatch")
        raw = (raw_words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        stock = (stock_words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        difference = raw - stock
        metrics[role] = {"bit_mismatches": int(np.count_nonzero(raw_words != stock_words)),
                         "max_absolute_error": float(np.max(np.abs(difference))),
                         "normalized_rms": float(np.sqrt(np.sum(difference * difference) /
                                                         max(np.sum(stock * stock), 1e-24))),
                         "argmax_agreement": bool(np.argmax(raw) == np.argmax(stock))}
        for mode in ("raw_pair", "b1"):
            pwords = arrays[mode][f"prefill_{role}"]
            digest = hashlib.sha256(pwords.astype("<u2", copy=False).tobytes()).hexdigest()
            if digest != records[mode]["prefill_logit_hashes"][role]:
                raise ValueError(f"{mode}:{role} saved prefill logits hash mismatch")
        metrics[role]["prefill_bit_mismatches"] = int(np.count_nonzero(
            arrays["raw_pair"][f"prefill_{role}"] != arrays["b1"][f"prefill_{role}"]))
    result = {"artifact_hash": records["raw_pair"]["artifact_hash"],
              "input_sha256": records["raw_pair"]["input_sha256"],
              "pair_two_single_bit_mismatches": records["raw_pair"]["pair_two_single_bit_mismatches"],
              "kv_cache_distinct": records["raw_pair"]["kv_cache_distinct"],
              "different_prompt_lengths": (len(records["raw_pair"]["inputs"]["anchor_prompt"]) !=
                                           len(records["raw_pair"]["inputs"]["target_prompt"])),
              "prefill_cache_hash_mismatches": {
                  role: sum(
                      records["raw_pair"]["prefill_cache_hashes"][role][layer][kind] !=
                      records["b1"]["prefill_cache_hashes"][role][layer][kind]
                      for layer in range(len(records["raw_pair"]["prefill_cache_hashes"][role]))
                      for kind in ("keys", "values"))
                  for role in ("anchor", "target")},
              "versus_stock": metrics,
              "records": ["results/correctness/raw_pair-lockstep.json",
                          "results/correctness/b1-lockstep.json"]}
    write_json_atomic(out / "raw-lockstep-comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw_pair", "B1", "compare"), required=True)
    mode = parser.parse_args().mode
    record = compare() if mode == "compare" else run(mode)
    print(json.dumps({"mode": mode, "pair_two_single_bit_mismatches":
                      record["pair_two_single_bit_mismatches"] if mode != "B1" else None,
                      "versus_stock": record.get("versus_stock")}))
