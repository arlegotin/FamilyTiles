"""Freeze at most two native-first whole-projection policies from the profile."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

from familytiles.convert import canonical_ledger, load_artifact
from familytiles.policy import (choose_native_promotions, packed_projections,
                                promotion_cost, validate_execution_policy)
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation"


def select() -> dict:
    artifact = load_artifact(ROOT / "artifacts/family")
    ledger = canonical_ledger(artifact)
    csv_path = OUT / "operator_profile.csv"
    profile = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if len(profile) != 21:
        raise ValueError("selection requires the original complete representative profile")
    roles = sorted({row["name"].split(".")[-2] for row in profile})
    gains = {role: max(0, int(median(
        int(row["staged_native_ns"]) - int(row["native_ns"])
        for row in profile if row["name"].split(".")[-2] == role)))
        for role in roles}
    policies = []
    for saving in (0.15, 0.10):
        # Reserve space for the policy table, separate from the parent manifest.
        max_extra = int(ledger["w1_bytes"] * (1 - saving)) - ledger["family_bytes"] - 65536
        if max_extra <= 0:
            continue
        native = choose_native_promotions(artifact.manifest["tensors"], gains,
                                          max_extra_bytes=max_extra)
        packed = packed_projections(artifact.manifest["tensors"])
        extra = sum(promotion_cost(artifact.manifest["tensors"][name]) for name in native)
        policy = {"format_version": 1, "artifact_hash": artifact.manifest_hash,
                  "parent_frozen_policy_sha256": json.loads((ROOT / "results/survey/frozen-policy.json").read_text())["policy_sha256"],
                  "revisions": artifact.manifest["revisions"],
                  "profile_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  "selection_layers": [0, 14, 27], "confirmation_layers": [3, 11, 22],
                  "mode": "native-first", "native_names": native,
                  "staged_names": sorted(set(packed) - set(native)),
                  "fused_names": [], "extra_resident_bytes": extra,
                  "canonical_parent_bytes": ledger["family_bytes"],
                  "canonical_b1_bytes": ledger["w1_bytes"],
                  "policy_table_reserve_bytes": 65536,
                  "projected_weight_saving_fraction": (
                      ledger["w1_bytes"] - ledger["family_bytes"] - extra - 65536
                  ) / ledger["w1_bytes"],
                  "target_saving_fraction": saving,
                  "staging_scratch_bound_bytes": max((2 * int(artifact.manifest["tensors"][name]["shape"][0]) *
                    int(artifact.manifest["tensors"][name]["shape"][1])
                    for name in packed if name not in native), default=0),
                  "selection_gains_ns_per_role": gains,
                  "C3_status": "unverified; current fused mode excluded"}
        validate_execution_policy(policy, artifact.manifest_hash, artifact.manifest["tensors"])
        if policy["projected_weight_saving_fraction"] + 1e-12 < saving:
            raise AssertionError("selection exceeded its canonical byte budget")
        path = OUT / f"policy-{int(saving * 100)}.json"
        write_json_atomic(path, policy)
        policies.append({"path": str(path.relative_to(ROOT)),
                         "target_saving_fraction": saving,
                         "projected_weight_saving_fraction": policy["projected_weight_saving_fraction"],
                         "native_count": len(native), "staged_count": len(policy["staged_names"]),
                         "extra_resident_bytes": extra})
    decision = {"artifact_hash": artifact.manifest_hash,
                "original_G3": "unchanged historical failure",
                "eligible_modes": ["native resident", "exact staged native"],
                "fused_C3": "ineligible by eight-step diagnostic",
                "selection_profile": "results/continuation/operator_profile.csv",
                "profile_gains_ns_per_role": gains,
                "policies": policies,
                "note": "Projected active-weight bytes only; measured peak and runtime decide utility."}
    write_json_atomic(OUT / "selection.json", decision)
    return decision


if __name__ == "__main__":
    print(json.dumps(select()))
