"""Frozen whole-matrix exact execution policies; no online decisions."""

from __future__ import annotations

from collections.abc import Mapping


def packed_projections(tensors: Mapping[str, dict]) -> list[str]:
    projections = {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}
    return sorted(name for name, record in tensors.items()
                  if name.endswith(".weight") and name.split(".")[-2] in projections
                  and record["target"]["kind"] == "packed")


def promotion_cost(record: dict) -> int:
    if record["target"]["kind"] != "packed":
        raise ValueError("only packed target pairs can be promoted")
    raw_bytes = 2
    for dimension in record["shape"]:
        raw_bytes *= int(dimension)
    cost = raw_bytes - int(record["target"]["allocated_bytes"])
    if cost <= 0:
        raise ValueError("packed representation has no native promotion cost")
    return cost


def choose_native_promotions(tensors: Mapping[str, dict],
                             gains_by_role_ns: Mapping[str, int],
                             *, max_extra_bytes: int) -> list[str]:
    """Greedy measured time recovered per net resident byte, whole pairs only."""
    if max_extra_bytes < 0:
        raise ValueError("negative promotion budget")
    ranked = []
    for name in packed_projections(tensors):
        gain = max(0, int(gains_by_role_ns.get(name.split(".")[-2], 0)))
        if gain:
            cost = promotion_cost(tensors[name])
            ranked.append((gain / cost, gain, name, cost))
    ranked.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))
    chosen, spent = [], 0
    for _, _, name, cost in ranked:
        if spent + cost <= max_extra_bytes:
            chosen.append(name)
            spent += cost
    return sorted(chosen)


def validate_execution_policy(policy: Mapping, artifact_hash: str,
                              tensors: Mapping[str, dict]) -> None:
    if policy.get("artifact_hash") != artifact_hash:
        raise ValueError("policy artifact identity mismatch")
    native = policy.get("native_names")
    staged = policy.get("staged_names")
    if not isinstance(native, list) or not isinstance(staged, list):
        raise ValueError("policy needs explicit native and staged names")
    if native != sorted(set(native)) or staged != sorted(set(staged)):
        raise ValueError("policy names must be unique and sorted")
    all_packed = set(packed_projections(tensors))
    if set(native) & set(staged) or set(native) | set(staged) != all_packed:
        raise ValueError("policy loses or duplicates a packed tensor dependency")
    actual = sum(promotion_cost(tensors[name]) for name in native)
    if policy.get("extra_resident_bytes") != actual:
        raise ValueError("policy extra bytes omit native promotion or retained packet")
