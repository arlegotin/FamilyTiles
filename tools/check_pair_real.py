"""Untimed real-matrix C2 check for each deliberate pair-kernel revision."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from familytiles.convert import FamilyArtifact
from familytiles.metal import (decode_words, family_gemv, family_gemv_pair,
                               kernel_config, load_operand, raw_gemv,
                               raw_gemv_pair)
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/family"
NAMES = ("model.layers.0.self_attn.q_proj.weight",
         "model.layers.0.mlp.down_proj.weight")


def _mismatches(a, b):
    return int(np.count_nonzero(np.asarray(a.view(mx.uint16)) !=
                                np.asarray(b.view(mx.uint16))))


def main() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    artifact = FamilyArtifact(ARTIFACT, manifest,
                              (ARTIFACT / "manifest.sha256").read_text().strip())
    records = []
    totals = [0, 0]
    single_totals = [0, 0]
    for name in NAMES:
        operand = load_operand(artifact, name)
        raw_target = decode_words(operand)
        mx.eval(raw_target)
        seed = 20260928 ^ int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],
                                         "little")
        rng = np.random.default_rng(seed)
        x0_host = rng.normal(0, 0.25, operand.shape[1]).astype(np.float32)
        x1_host = rng.normal(0.7, 0.4, operand.shape[1]).astype(np.float32)
        x0 = mx.array(x0_host, dtype=mx.bfloat16)
        x1 = mx.array(x1_host, dtype=mx.bfloat16)
        b0 = mx.array(rng.normal(0, 0.1, operand.shape[0]).astype(np.float32),
                      dtype=mx.bfloat16)
        b1 = mx.array(rng.normal(0.5, 0.1, operand.shape[0]).astype(np.float32),
                      dtype=mx.bfloat16)
        actual = family_gemv_pair(operand, x0, x1, (b0, b1))
        raw = raw_gemv_pair(operand.anchor_words, raw_target, x0, x1, (b0, b1))
        singles = (raw_gemv(operand.anchor_words, x0, b0),
                   family_gemv(operand, x1, b1))
        mx.eval(*actual, *raw, *singles)
        pair_errors = [_mismatches(a, b) for a, b in zip(actual, raw)]
        single_errors = [_mismatches(a, b) for a, b in zip(actual, singles)]
        totals = [a + b for a, b in zip(totals, pair_errors)]
        single_totals = [a + b for a, b in zip(single_totals, single_errors)]
        records.append({"tensor": name, "shape": operand.shape,
                        "input_sha256": [hashlib.sha256(x0_host.tobytes()).hexdigest(),
                                         hashlib.sha256(x1_host.tobytes()).hexdigest()],
                        "pair_raw_bit_mismatches": pair_errors,
                        "pair_two_single_bit_mismatches": single_errors})
    result = {"artifact_hash": artifact.manifest_hash, "kernel_config": kernel_config(),
              "pair_raw_bit_mismatches": totals,
              "pair_two_single_bit_mismatches": single_totals,
              "tensors": records,
              "source_inspection": {"anchor_loads_per_coordinate": 1,
                                    "decoded_matrix_writes": 0,
                                    "output_vectors": 2}}
    write_json_atomic(ROOT / "results/correctness/pair-real.json", result)
    captured = check_captured(artifact)
    print(json.dumps({"pair_raw_bit_mismatches": totals,
                      "pair_two_single_bit_mismatches": single_totals,
                      "captured_pair_raw_bit_mismatches": captured["pair_raw_bit_mismatches"],
                      "layout_version": result["kernel_config"]["layout_version"]}))


def check_captured(artifact: FamilyArtifact) -> dict:
    """Use genuine, independent stock activations from both pinned models."""
    records = []
    mismatches = [0, 0]
    singles_mismatches = [0, 0]
    manifests = {}
    arrays = {}
    for role in ("anchor", "target"):
        path = ROOT / "results/inputs" / f"{role}-activations.json"
        manifests[role] = json.loads(path.read_text(encoding="utf-8"))
        arrays[role] = np.load(path.parent / manifests[role]["array_file"],
                               allow_pickle=False)
    for layer in (0, 14, 27):
        for projection in ("q_proj", "down_proj"):
            name = (f"model.layers.{layer}.self_attn.q_proj.weight"
                    if projection == "q_proj" else
                    f"model.layers.{layer}.mlp.down_proj.weight")
            operand = load_operand(artifact, name)
            raw_target = decode_words(operand)
            key = f"layer{layer}.{projection}"
            vectors = []
            hashes = []
            for role in ("anchor", "target"):
                words = arrays[role][key]
                got = hashlib.sha256(words.astype("<u2", copy=False).tobytes()).hexdigest()
                want = manifests[role]["arrays"][key]["sha256"]
                if got != want or words.shape != (operand.shape[1],):
                    raise ValueError(f"invalid captured vector {role}:{key}")
                vectors.append(mx.array(words, dtype=mx.uint16).view(mx.bfloat16))
                hashes.append(got)
            x0, x1 = vectors
            pair = family_gemv_pair(operand, x0, x1)
            raw = raw_gemv_pair(operand.anchor_words, raw_target, x0, x1)
            singles = (raw_gemv(operand.anchor_words, x0),
                       family_gemv(operand, x1))
            mx.eval(*pair, *raw, *singles)
            pair_errors = [_mismatches(a, b) for a, b in zip(pair, raw)]
            single_errors = [_mismatches(a, b) for a, b in zip(pair, singles)]
            mismatches = [a + b for a, b in zip(mismatches, pair_errors)]
            singles_mismatches = [a + b for a, b in zip(singles_mismatches, single_errors)]
            records.append({"tensor": name, "shape": operand.shape,
                            "input_sha256": hashes,
                            "pair_raw_bit_mismatches": pair_errors,
                            "pair_two_single_bit_mismatches": single_errors})
    result = {"artifact_hash": artifact.manifest_hash,
              "kernel_config": kernel_config(),
              "capture_manifests": [f"results/inputs/{role}-activations.json"
                                    for role in ("anchor", "target")],
              "pair_raw_bit_mismatches": mismatches,
              "pair_two_single_bit_mismatches": singles_mismatches,
              "tensors": records}
    write_json_atomic(ROOT / "results/correctness/pair-captured.json", result)
    return result


if __name__ == "__main__":
    main()
