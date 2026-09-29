"""Five-path cached Qwen2 differential using the existing exact family artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation"
MODES = ("B1", "native_pair", "raw_pair", "A1", "family_single")
MIXED_MODES = ("mixed15", "mixed10")


def token_schedule(saved: dict, steps: int) -> list[tuple[int, int]]:
    """Fixed, independent teacher-forced tokens; the saved first step is retained."""
    if steps < 1:
        raise ValueError("at least one cached step is required")
    prompts = (saved["anchor_prompt"], saved["target_prompt"])
    if not all(prompts):
        raise ValueError("both prompt histories are required")
    first = (int(saved["anchor_next"]), int(saved["target_next"]))
    return [first] + [tuple(int(prompt[-1 - (step % len(prompt))])
                             for prompt in prompts) for step in range(1, steps)]


def compare_words(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape or reference.dtype != np.dtype("<u2") or candidate.dtype != np.dtype("<u2"):
        raise ValueError("BF16 word arrays must have identical shapes and uint16 dtype")
    ref = (reference.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    got = (candidate.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    difference = got - ref
    return {"bit_mismatches": int(np.count_nonzero(reference != candidate)),
            "normalized_rms": float(np.sqrt(np.sum(difference * difference) /
                                            max(np.sum(ref * ref), 1e-24))),
            "max_absolute_error": float(np.max(np.abs(difference))) if difference.size else 0.0,
            "argmax_agreement": bool(np.argmax(ref) == np.argmax(got)) if ref.size else True}


def _words(value) -> np.ndarray:
    import mlx.core as mx
    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).astype("<u2", copy=True)


def _cache_digest(state) -> list[dict[str, str | int]]:
    import mlx.core as mx
    records = []
    for entry in state.cache:
        keys, values = entry.state
        mx.eval(keys, values)
        records.append({"offset": int(entry.offset),
                        "keys": hashlib.sha256(np.asarray(keys.view(mx.uint16)).tobytes()).hexdigest(),
                        "values": hashlib.sha256(np.asarray(values.view(mx.uint16)).tobytes()).hexdigest()})
    return records


def run_worker(mode: str, steps: int) -> dict:
    from familytiles.convert import load_artifact
    from familytiles.model import load_runtime, prefill, step_pair, step_single
    from familytiles.metal import kernel_config
    from familytiles.measure import sample_memory, sample_mlx_memory

    artifact = load_artifact(ROOT / "artifacts/family")
    saved = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    schedule = token_schedule(saved, steps)
    fixture_hash = hashlib.sha256(json.dumps({"inputs": saved, "tokens": schedule},
                                             sort_keys=True).encode()).hexdigest()
    if mode in MIXED_MODES:
        policy_path = OUT / f"policy-{mode.removeprefix('mixed')}.json"
        execution_policy = json.loads(policy_path.read_text())
        runtime = load_runtime(artifact, "mixed", execution_policy=execution_policy)
    else:
        execution_policy = None
        runtime = load_runtime(artifact, mode)
    states = tuple(prefill(runtime, role, saved[f"{role}_prompt"])[1]
                   for role in ("anchor", "target"))
    prefill_state = {role: {"position": state.position, "cache": _cache_digest(state)}
                     for role, state in zip(("anchor", "target"), states)}
    replay_states = None
    if mode == "B1":
        replay_states = tuple(prefill(runtime, role, saved[f"{role}_prompt"])[1]
                              for role in ("anchor", "target"))
        if any(_cache_digest(a) != _cache_digest(b)
               for a, b in zip(states, replay_states)):
            raise AssertionError("reference prefill is not self-reproducible")
    logit_words = {role: [] for role in ("anchor", "target")}
    post_steps = []
    for pair in schedule:
        if mode == "B1":
            output = tuple(step_single(runtime, state, token)
                           for state, token in zip(states, pair))
            replay = tuple(step_single(runtime, state, token)
                           for state, token in zip(replay_states, pair))
            if any(np.count_nonzero(_words(a) != _words(b))
                   for a, b in zip(output, replay)):
                raise AssertionError("reference logits are not self-reproducible")
            if any(_cache_digest(a) != _cache_digest(b)
                   for a, b in zip(states, replay_states)):
                raise AssertionError("reference cache is not self-reproducible")
        else:
            output = step_pair(runtime, states, pair)
        for role, value in zip(("anchor", "target"), output):
            logit_words[role].append(_words(value))
        post_steps.append({role: {"position": state.position, "cache": _cache_digest(state)}
                           for role, state in zip(("anchor", "target"), states)})
    array_path = OUT / f"numerical-{mode.lower()}-{steps}.npz"
    np.savez_compressed(array_path, **{role: np.stack(values)
                                      for role, values in logit_words.items()})
    record = {"mode": mode, "steps": steps, "artifact_hash": artifact.manifest_hash,
              "policy_hash": artifact.manifest.get("policy_sha256"),
              "revisions": artifact.manifest["revisions"], "fixture_sha256": fixture_hash,
              "input_lengths": [len(saved[f"{role}_prompt"]) for role in ("anchor", "target")],
              "tokens": schedule, "kernel_config": kernel_config(),
              "execution_policy_sha256": (hashlib.sha256(policy_path.read_bytes()).hexdigest()
                                          if execution_policy is not None else None),
              "prefill_state": prefill_state, "post_steps": post_steps,
              "logit_hashes": {role: hashlib.sha256(np.stack(values).tobytes()).hexdigest()
                               for role, values in logit_words.items()},
              "arrays_file": array_path.name,
              "after": {"process": sample_memory(__import__("os").getpid(), "after"),
                        "mlx": sample_mlx_memory()}}
    write_json_atomic(OUT / f"numerical-{mode.lower()}-{steps}.json", record)
    return {"mode": mode, "steps": steps, "arrays_file": array_path.name}


def compare(steps: int) -> dict:
    records = {mode: json.loads((OUT / f"numerical-{mode.lower()}-{steps}.json").read_text())
               for mode in MODES}
    identifiers = ("artifact_hash", "fixture_sha256", "revisions")
    for key in identifiers:
        if any(records[mode][key] != records["B1"][key] for mode in MODES):
            raise ValueError(f"numerical records disagree on {key}")
    arrays = {mode: np.load(OUT / records[mode]["arrays_file"], allow_pickle=False)
              for mode in MODES}
    for mode in MODES:
        for role in ("anchor", "target"):
            if hashlib.sha256(arrays[mode][role].tobytes()).hexdigest() != records[mode]["logit_hashes"][role]:
                raise ValueError(f"logit hash mismatch: {mode}:{role}")
    comparisons = {}
    for mode in MODES[1:]:
        comparisons[mode] = {role: [compare_words(arrays["B1"][role][step],
                                                 arrays[mode][role][step])
                                    for step in range(steps)]
                             for role in ("anchor", "target")}
    direct = {"A1_vs_native_pair": {role: [compare_words(
        arrays["native_pair"][role][step], arrays["A1"][role][step])
        for step in range(steps)] for role in ("anchor", "target")},
              "fused_vs_raw_pair": {role: [compare_words(
                  arrays["raw_pair"][role][step], arrays["family_single"][role][step])
                  for step in range(steps)] for role in ("anchor", "target")}}
    result = {"steps": steps, "artifact_hash": records["B1"]["artifact_hash"],
              "fixture_sha256": records["B1"]["fixture_sha256"],
              "reference_self_replay": "bitwise-identical",
              "prefill_cache_matches_reference": {mode: {
                  role: records[mode]["prefill_state"][role] == records["B1"]["prefill_state"][role]
                  for role in ("anchor", "target")} for mode in MODES[1:]},
              "versus_reference": comparisons, "same_schedule": direct,
              "records": [f"results/continuation/numerical-{mode.lower()}-{steps}.json" for mode in MODES]}
    write_json_atomic(OUT / f"numerical-trace-{steps}.json", result)
    if steps == 8:
        projection = json.loads((OUT / "projection_trace.json").read_text())
        write_json_atomic(OUT / "numerical_trace.json", {
            "artifact_hash": result["artifact_hash"],
            "fixture_sha256": result["fixture_sha256"],
            "reference_self_replay": result["reference_self_replay"],
            "prefill_cache_matches_reference": result["prefill_cache_matches_reference"],
            "per_model_max_nrmse": {mode: {role: max(
                step["normalized_rms"] for step in result["versus_reference"][mode][role])
                for role in ("anchor", "target")} for mode in MODES[1:]},
            "same_schedule_max_word_mismatch": {label: {role: max(
                step["bit_mismatches"] for step in roles[role])
                for role in ("anchor", "target")} for label, roles in direct.items()},
            "first_identified_raw_divergence": "layer 0 biased Q projection, reference activation replay",
            "projection_trace": "results/continuation/projection_trace.json",
            "projection_bias_experiment": [{"role": row["role"],
                "projection": row["projection"],
                "current_mismatches": row["stock_vs_current"]["bit_mismatches"],
                "two_round_mismatches": row["stock_vs_two_round"]["bit_mismatches"]}
                for row in projection["rows"]],
            "decision": "native-operator staging eligible; current fused arithmetic ineligible for C3",
            "scope": "eight cached teacher-forced steps, not the original 2048-prediction C3 gate",
            "records": result["records"]})
    return result


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, *MIXED_MODES, "all", "compare"), default="all")
    parser.add_argument("--steps", type=int, choices=(1, 8), default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        for mode in MODES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                            "--steps", str(args.steps), "--out", str(OUT)], check=True, cwd=ROOT)
        result = compare(args.steps)
        print(json.dumps({"steps": args.steps, "first_step_nrmse": {
            mode: {role: result["versus_reference"][mode][role][0]["normalized_rms"]
                   for role in ("anchor", "target")} for mode in MODES[1:]}}))
    elif args.mode == "compare":
        result = compare(args.steps)
        print(json.dumps({"steps": args.steps, "reference_self_replay": result["reference_self_replay"]}))
    else:
        print(json.dumps(run_worker(args.mode, args.steps)))


if __name__ == "__main__":
    main()
