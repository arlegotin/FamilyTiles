"""Predeclared, bounded BF16 family screen with frozen holdout policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import requests

from .baselines import encode_exact, encode_family_exact, exact_allocated_bytes
from .codec import CodecPolicy, allocated_bytes, encode_residual, encode_tensor
from .data import PinnedModel, TransferBudget, pin_model, read_range
from .records import GateRecord, RunContext, read_gate, publish_gate, write_json_atomic


ROLE_RE = re.compile(r"\.((?:q|k|v|o|gate|up|down)_proj)\.weight$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SEED = 20260928


def _hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SampleWindow:
    model_id: str
    tensor_name: str
    row: int
    start_column: int
    word_count: int
    stratum: str
    split: str

    @property
    def id(self) -> str:
        return f"{self.tensor_name}:{self.row}:{self.start_column}:{self.word_count}"


@dataclass(frozen=True)
class SamplePlan:
    seed: int
    windows: tuple[SampleWindow, ...]
    split_hash: str
    bytes_per_model: int


def _role(name: str) -> str | None:
    match = ROLE_RE.search(name)
    return match.group(1) if match else None


def make_sample_plan(models: tuple[PinnedModel, PinnedModel], seed: int,
                     limit_bytes: int) -> SamplePlan:
    if seed < 0 or limit_bytes <= 0:
        raise ValueError("invalid sample seed or limit")
    anchor, target = models
    common: dict[str, list[str]] = {role: [] for role in ROLES}
    for name, info in anchor.tensors.items():
        match = target.tensors.get(name)
        role = _role(name)
        if (role and match and info.dtype == match.dtype == "BF16" and len(info.shape) == 2
                and info.shape == match.shape and LAYER_RE.search(name)):
            common[role].append(name)
    rng = np.random.default_rng(seed)
    windows: list[SampleWindow] = []
    for role in ROLES:
        names = sorted(common[role], key=lambda n: int(LAYER_RE.search(n).group(1)))
        if not names:
            continue
        selected = [names[0], names[len(names) // 2], names[-1]]
        combinations = [(layer, quarter) for layer in range(3) for quarter in range(4)]
        rng.shuffle(combinations)
        chosen = combinations[:10]
        hold_indices = set(int(i) for i in rng.choice(10, size=3, replace=False))
        for index, (layer, quarter) in enumerate(chosen):
            name = selected[layer]
            rows, cols = anchor.tensors[name].shape
            segment_start, segment_end = rows * quarter // 4, rows * (quarter + 1) // 4
            take_rows = min(32, segment_end - segment_start)
            if take_rows <= 0:
                continue
            windows.append(SampleWindow("pair", name, segment_start, 0,
                                        take_rows * cols, role,
                                        "holdout" if index in hold_indices else "development"))
    if not windows:
        raise ValueError("no compatible BF16 projection windows")
    bytes_per_model = sum(w.word_count * 2 for w in windows)
    if bytes_per_model > limit_bytes:
        # Scale row spans without changing which parent window owns any block.
        factor = limit_bytes / bytes_per_model
        scaled = []
        for w in windows:
            cols = anchor.tensors[w.tensor_name].shape[1]
            rows = max(1, int(w.word_count // cols * factor))
            scaled.append(SampleWindow(w.model_id, w.tensor_name, w.row, 0,
                                       rows * cols, w.stratum, w.split))
        windows = scaled
        bytes_per_model = sum(w.word_count * 2 for w in windows)
    if bytes_per_model > limit_bytes:
        raise ValueError("sampling plan exceeds per-model cap")
    split_hash = _hash([(w.id, w.split) for w in windows])
    return SamplePlan(seed, tuple(windows), split_hash, bytes_per_model)


def choose_policy(development: dict[str, Any]) -> CodecPolicy:
    candidates = development.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("missing real-data candidate estimates")
    for item in candidates:
        if item.get("source", "real") != "real" or item.get("raw_family_bytes", 0) <= 0:
            raise ValueError("synthetic controls cannot select a family policy")
    singles = [x for x in candidates if len(x["transforms"]) == 1]
    duals = [x for x in candidates if len(x["transforms"]) == 2]
    if not singles:
        raise ValueError("single-transform policy missing")
    key = lambda x: (x["family_bytes"], abs(x["block"] - 128), x["transforms"])
    best_single = min(singles, key=key)
    best = best_single
    if duals:
        best_dual = min(duals, key=key)
        raw = best_single["raw_family_bytes"]
        if best_single["family_bytes"] - best_dual["family_bytes"] >= 0.02 * raw:
            best = best_dual
    return CodecPolicy(1, int(best["block"]), tuple(best["transforms"]))


def project_family(development: dict[str, Any], holdout: dict[str, Any],
                   inventory: dict[str, Any], policy: CodecPolicy) -> dict[str, Any]:
    del policy  # The supplied split metrics are already for the frozen policy.
    raw, b1 = inventory["raw_family_bytes"], inventory["b1_bytes"]
    if raw <= 0 or b1 <= 0 or b1 > raw:
        raise ValueError("invalid active-family accounting")
    b2, family = float(b1), float(b1)
    by_role = {}
    for role, size in inventory["target_role_bytes"].items():
        ratios = {}
        for metric in ("b2", "family"):
            values = []
            for split in (development, holdout):
                row = split.get("roles", {}).get(role)
                if row is None or row["raw"] <= 0:
                    values.append(1.0)
                else:
                    values.append(min(1.0, row[metric] / row["raw"]))
            ratios[metric] = max(values)
        b2 -= size * (1 - ratios["b2"])
        family -= size * (1 - ratios["family"])
        by_role[role] = {"active_target_bytes": size, "b2_ratio": ratios["b2"],
                         "family_ratio": ratios["family"]}
    b2_bytes, family_bytes = round(b2), round(family)
    saving_b1 = (b1 - family_bytes) / b1
    beyond_b2 = (b2_bytes - family_bytes) / raw
    return {"raw_family_bytes": raw, "b1_bytes": b1, "b2_bytes": b2_bytes,
            "family_bytes": family_bytes,
            "saving_vs_b1": saving_b1,
            "beyond_b2_fraction_raw": beyond_b2,
            "unresolved_alias_bytes": inventory.get("unresolved_alias_bytes", 0),
            "roles": by_role,
            "g1_pass": saving_b1 >= 0.15 - 1e-12 and beyond_b2 >= 0.05 - 1e-12}


def _inventory(anchor: PinnedModel, target: PinnedModel) -> dict[str, Any]:
    raw = sum(x.byte_length for x in anchor.tensors.values()) + sum(x.byte_length for x in target.tensors.values())
    role_bytes = {role: 0 for role in ROLES}
    eligible = set()
    for name, info in target.tensors.items():
        a = anchor.tensors.get(name)
        role = _role(name)
        if role and a and a.dtype == info.dtype == "BF16" and a.shape == info.shape and len(info.shape) == 2:
            role_bytes[role] += info.byte_length
            eligible.add(name)
    # Samples cannot prove a whole-tensor alias. B1 receives no speculative alias credit.
    unresolved = sum(info.byte_length for name, info in target.tensors.items()
                     if name in anchor.tensors and anchor.tensors[name].shape == info.shape
                     and anchor.tensors[name].dtype == info.dtype)
    return {"raw_family_bytes": raw, "b1_bytes": raw,
            "target_role_bytes": role_bytes, "eligible_tensor_count": len(eligible),
            "native_target_bytes": sum(x.byte_length for x in target.tensors.values()) - sum(role_bytes.values()),
            "unresolved_alias_bytes": unresolved}


def _read_window(models: tuple[PinnedModel, PinnedModel], w: SampleWindow,
                 budget: TransferBudget) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    arrays = []
    hashes = {}
    for model_id, model in zip(("anchor", "target"), models):
        info = model.tensors[w.tensor_name]
        cols = info.shape[1]
        start = info.data_start + info.byte_offset + 2 * (w.row * cols + w.start_column)
        data = read_range(model, info.shard, start, w.word_count * 2, budget)
        hashes[model_id] = hashlib.sha256(data).hexdigest()
        arrays.append(np.frombuffer(data, dtype="<u2").copy().reshape((-1, cols)))
    return arrays[0], arrays[1], hashes


def _score(a: np.ndarray, t: np.ndarray, policies: tuple[CodecPolicy, ...]) -> dict[str, Any]:
    raw = t.nbytes
    b3 = min(raw * 2, *(exact_allocated_bytes(encode_exact(a, transform)) +
                        exact_allocated_bytes(encode_exact(t, transform))
                        for transform in ("native", "byte_plane", "field")))
    b4 = min(exact_allocated_bytes(encode_family_exact(a, t, transform, anchor_form))
             for transform in ("xor", "ordered_delta") for anchor_form in ("raw", "compressed"))
    candidates = {}
    for policy in policies:
        key = f"{policy.block_values}:{','.join(policy.transforms)}"
        packed = encode_tensor(a, t, policy)
        tile = encode_tensor(a, t, policy, modes=("copy", "raw"))
        candidates[key] = {"family": allocated_bytes(packed), "b2": allocated_bytes(tile)}
    xor = encode_residual(a, t, "xor")
    ordered = encode_residual(a, t, "ordered_delta")
    return {"raw": raw, "candidates": candidates, "b3_pair": b3, "b4_pair": b4,
            "identical_words": int(np.count_nonzero(a == t)),
            "xor_max": int(xor.max(initial=0)), "ordered_max": int(ordered.max(initial=0))}


def _aggregate(records: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = {}
    totals = {"raw": 0, "b2": 0, "family": 0, "b3_pair": 0, "b4_pair": 0,
              "identical_words": 0, "xor_max": 0, "ordered_max": 0}
    for record in records:
        score = record["score"]
        role = record["window"]["stratum"]
        bucket = roles.setdefault(role, {"raw": 0, "b2": 0, "family": 0})
        for name in ("raw", "b2", "family"):
            value = score["candidates"][candidate][name] if name != "raw" else score["raw"]
            bucket[name] += value
            totals[name] += value
        for name in ("b3_pair", "b4_pair", "identical_words"):
            totals[name] += score[name]
        for name in ("xor_max", "ordered_max"):
            totals[name] = max(totals[name], score[name])
    return {"roles": roles, "totals": totals}


def _policies() -> tuple[CodecPolicy, ...]:
    return tuple(CodecPolicy(1, b, transforms) for b in (64, 128, 256)
                 for transforms in (("xor",), ("ordered_delta",), ("xor", "ordered_delta")))


def _model_record(model: PinnedModel) -> dict[str, Any]:
    return {"repo": model.repo, "revision": model.revision, "files": model.files,
            "metadata": model.metadata,
            "tensors": {name: asdict(info) for name, info in model.tensors.items()}}


def _controls(policy: CodecPolicy) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    anchor = rng.integers(0, 65536, (8, 256), dtype=np.uint16)
    random = rng.integers(0, 65536, (8, 256), dtype=np.uint16)
    shuffled = np.roll(anchor.reshape(-1), 111).reshape(anchor.shape).copy()
    crossings = np.bitwise_xor(anchor, np.uint16(0xFF80))
    identical = anchor.copy()
    sparse = anchor.copy()
    sparse[:, ::47] ^= np.uint16(1)
    return {name: {"raw": anchor.nbytes,
                   "encoded": allocated_bytes(encode_tensor(anchor, target, policy))}
            for name, target in (("independent_random", random), ("shuffled", shuffled),
                                 ("sign_exponent_crossings", crossings),
                                 ("identical", identical), ("sparse_changes", sparse))}


def survey(families: Path, sample_mib_per_model: int, seed: int,
           out: Path) -> dict[str, Any]:
    if seed != SEED or sample_mib_per_model <= 0 or sample_mib_per_model > 32:
        raise ValueError("survey requires the prespecified seed and at most 32 MiB per model")
    config = json.loads(families.read_text(encoding="utf-8"))
    enabled = [f for f in config["families"] if f["enabled"] and f["role"] in ("primary", "fallback-scale-check")]
    if len(enabled) != 2:
        raise ValueError("survey requires both prespecified Qwen families")
    out.mkdir(parents=True, exist_ok=True)
    for pointer in ("selected-family.json", "frozen-policy.json"):
        (out / pointer).unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[2]
    g0 = read_gate(root / "results/gates/G0.json")
    if g0.decision != "pass":
        raise ValueError("G0 has not passed")
    budget = TransferBudget(sample_mib_per_model * 2**20, 7 * 2**30,
                            out / "acquisition-ledger.json")
    summaries = []
    passing = []
    for family in enabled:
        family_out = out / family["id"]
        family_out.mkdir(parents=True, exist_ok=True)
        try:
            models = (pin_model(family["anchor"], None, budget),
                      pin_model(family["target"], None, budget))
            write_json_atomic(family_out / "pinned.json",
                              {"anchor": _model_record(models[0]), "target": _model_record(models[1])})
            inventory = _inventory(*models)
            plan = make_sample_plan(models, seed, budget.sample_limit_bytes)
            write_json_atomic(family_out / "sample-plan.json",
                              {"seed": seed, "split_hash": plan.split_hash,
                               "bytes_per_model": plan.bytes_per_model,
                               "windows": [asdict(w) for w in plan.windows]})
            if len(plan.windows) < 10 or not all(any(w.stratum == role for w in plan.windows) for role in ROLES):
                raise ValueError("insufficient compatible projection sample coverage")
            all_policies = _policies()
            dev_records = []
            for w in plan.windows:
                if w.split != "development":
                    continue
                a, t, hashes = _read_window(models, w, budget)
                dev_records.append({"window": asdict(w), "sha256": hashes,
                                    "score": _score(a, t, all_policies)})
            candidates = []
            for policy in all_policies:
                key = f"{policy.block_values}:{','.join(policy.transforms)}"
                agg = _aggregate(dev_records, key)
                projection = project_family(agg, agg, inventory, policy)
                candidates.append({"block": policy.block_values,
                                   "transforms": list(policy.transforms),
                                   "family_bytes": projection["family_bytes"],
                                   "raw_family_bytes": inventory["raw_family_bytes"],
                                   "source": "real"})
            development = {"candidates": candidates, "window_count": len(dev_records),
                           "split_hash": plan.split_hash}
            write_json_atomic(family_out / "development.json", development)
            policy = choose_policy(development)
            policy_record = asdict(policy)
            policy_record["development_hash"] = _hash(development)
            write_json_atomic(family_out / "frozen-policy.json", policy_record)
            key = f"{policy.block_values}:{','.join(policy.transforms)}"
            hold_records = []
            for w in plan.windows:
                if w.split != "holdout":
                    continue
                a, t, hashes = _read_window(models, w, budget)
                hold_records.append({"window": asdict(w), "sha256": hashes,
                                     "score": _score(a, t, (policy,))})
            write_json_atomic(family_out / "sample-records.json",
                              {"development": dev_records, "holdout": hold_records})
            dev_agg, hold_agg = _aggregate(dev_records, key), _aggregate(hold_records, key)
            projection = project_family(dev_agg, hold_agg, inventory, policy)
            projection.update({"family_id": family["id"],
                               "revisions": {"anchor": models[0].revision,
                                             "target": models[1].revision},
                               "policy": policy_record,
                               "sample_bytes_per_model": plan.bytes_per_model,
                               "sample_split_hash": plan.split_hash,
                               "development": dev_agg, "holdout": hold_agg,
                               "controls": _controls(policy), "inventory": inventory,
                               "limitations": "Sampled projection ratios; no sampled whole-tensor aliases declared. Unsupported tensors charged native."})
            write_json_atomic(family_out / "projection.json", projection)
            status = "pass" if projection["g1_pass"] else "fail"
            summaries.append({"family_id": family["id"], "status": status,
                              "projection": str((family_out / "projection.json").relative_to(out)),
                              "revisions": projection["revisions"]})
            if projection["g1_pass"]:
                passing.append((sum(models[0].files.values()) + sum(models[1].files.values()),
                                family, projection, policy))
        except (OSError, ValueError, requests.RequestException) as exc:
            summaries.append({"family_id": family["id"], "status": "blocked",
                              "reason": f"acquisition or metadata failed: {type(exc).__name__}: {exc}"})
            write_json_atomic(family_out / "failure.json", summaries[-1])
    decision = "pass" if passing else ("blocked" if any(x["status"] == "blocked" for x in summaries) else "fail")
    selected = None
    if passing:
        _, family, projection, policy = min(passing, key=lambda item: item[0])
        selected = family["id"]
        write_json_atomic(out / "selected-family.json",
                          {"family_id": selected, "anchor": family["anchor"],
                           "target": family["target"], "revisions": projection["revisions"],
                           "pinned_manifest": f"{selected}/pinned.json"})
        write_json_atomic(out / "frozen-policy.json",
                          {**asdict(policy), "family_id": selected,
                           "policy_sha256": _hash(asdict(policy))})
    summary = {"seed": seed, "sample_mib_per_model": sample_mib_per_model,
               "families": summaries, "decision": decision, "selected": selected,
               "transfer_ledger": budget.snapshot()}
    write_json_atomic(out / "summary.json", summary)
    gate = GateRecord(1, "G1", RunContext(selected,
                       next((x["revisions"] for x in summaries if x["family_id"] == selected), {}),
                       _hash(asdict(policy)) if passing else None, None,
                       hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       g0.context.environment_hash, _hash(summary)),
                      {"families": hashlib.sha256(families.read_bytes()).hexdigest(),
                       "summary": _hash(summary)},
                      {"minimum_b1_saving": 0.15, "minimum_beyond_b2_raw": 0.05},
                      {"families": summaries}, decision,
                      "one real family passed the conservative sample screen" if passing else
                      "both prescribed families failed the size screen" if decision == "fail" else
                      "at least one prescribed family could not be screened")
    publish_gate(root / "results/gates", gate)
    return summary
