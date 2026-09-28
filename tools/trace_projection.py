"""Replay recorded first-layer activations through stock and custom projections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from familytiles.convert import load_artifact
from familytiles.metal import raw_gemv
from familytiles.model import load_runtime, prefill
from familytiles.records import write_json_atomic
from trace_decode import compare_words


ROOT = Path(__file__).resolve().parents[1]


def _words(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.asarray(array.view(mx.uint16)).astype("<u2", copy=True)


def main() -> None:
    artifact = load_artifact(ROOT / "artifacts/family")
    fixture = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    runtime = load_runtime(artifact, "raw_pair")
    rows = []
    for role in ("anchor", "target"):
        _, state = prefill(runtime, role, fixture[f"{role}_prompt"])
        model = runtime.models[role]
        layer = model.model.layers[0]
        token = mx.array([[fixture[f"{role}_next"]]], dtype=mx.int32)
        x = layer.input_layernorm(model.model.embed_tokens(token))[0, 0, :]
        mx.eval(x)
        activation_hash = hashlib.sha256(_words(x).tobytes()).hexdigest()
        for projection in ("q_proj", "k_proj", "v_proj"):
            module = getattr(layer.self_attn, projection)
            operand = module.operand
            words = operand.anchor_words if role == "anchor" else operand.native_words
            stock = mx.addmm(module.bias, x.reshape(1, 1, -1),
                             words.view(mx.bfloat16).T)[0, 0, :]
            current = raw_gemv(words, x, module.bias)
            two_round = (raw_gemv(words, x, None) + module.bias).astype(mx.bfloat16)
            stock_words = _words(stock)
            rows.append({"role": role, "projection": projection,
                         "activation_sha256": activation_hash,
                         "stock_vs_current": compare_words(stock_words, _words(current)),
                         "stock_vs_two_round": compare_words(stock_words, _words(two_round))})
    result = {"artifact_hash": artifact.manifest_hash,
              "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
              "method": "reference first-layer input activation replay; two_round casts GEMV before bias",
              "rows": rows}
    write_json_atomic(ROOT / "results/continuation/projection_trace.json", result)
    print(json.dumps({"rows": [{"role": row["role"], "projection": row["projection"],
                                 "current_mismatches": row["stock_vs_current"]["bit_mismatches"],
                                 "two_round_mismatches": row["stock_vs_two_round"]["bit_mismatches"]}
                                for row in rows]}))


if __name__ == "__main__":
    main()
