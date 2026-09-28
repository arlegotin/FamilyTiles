"""Capture untimed operands from the pinned stock Qwen2 forward traversal.

Run once per model role in separate processes. The saved arrays are inputs to
the G3 C2 check, never part of the timed kernel sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask

from familytiles.measure import compute_budget
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain why binary search requires sorted input. Give a small example."
LAYERS = (0, 14, 27)


def capture(role: str, selected: Path, output_dir: Path) -> dict:
    import psutil

    selection = json.loads(selected.read_text(encoding="utf-8"))
    pinned_path = selected.parent / selection["pinned_manifest"]
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))[role]
    local = ROOT / "artifacts/downloaded" / pinned["repo"].replace("/", "--") / pinned["revision"]
    if not (local / "model.safetensors").is_file():
        raise FileNotFoundError(local / "model.safetensors")
    for filename, want in pinned["metadata"]["config_hashes"].items():
        path = local / filename
        if not path.is_file():
            path = Path(hf_hub_download(pinned["repo"], filename=filename,
                                        revision=pinned["revision"], local_dir=local))
        if path.stat().st_size > 16 * 2**20 or hashlib.sha256(path.read_bytes()).hexdigest() != want:
            raise ValueError(f"pinned metadata is oversized or mismatched: {filename}")
    budget = compute_budget(psutil.virtual_memory().total, psutil.virtual_memory().available)
    if budget.process_limit_bytes < 4 * 2**30:
        raise RuntimeError("insufficient safe budget for stock activation capture")
    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)

    model, tokenizer = load(str(local))
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if not ids:
        raise RuntimeError("prompt produced no tokens")
    input_ids = mx.array([ids], dtype=mx.int32)
    h = model.model.embed_tokens(input_ids)
    mask = create_attention_mask(h, None)
    arrays: dict[str, np.ndarray] = {}

    def save(layer_number: int, role_name: str, value: mx.array) -> None:
        vector = value[0, -1, :]
        mx.eval(vector)
        arrays[f"layer{layer_number}.{role_name}"] = np.asarray(
            vector.view(mx.uint16)
        ).copy()

    for layer_number, layer in enumerate(model.model.layers):
        attn_in = layer.input_layernorm(h)
        if layer_number in LAYERS:
            save(layer_number, "q_proj", attn_in)
        attn_out = layer.self_attn(attn_in, mask, None)
        h_after_attn = h + attn_out
        mlp_in = layer.post_attention_layernorm(h_after_attn)
        down_in = swiglu(layer.mlp.gate_proj(mlp_in), layer.mlp.up_proj(mlp_in))
        if layer_number in LAYERS:
            save(layer_number, "down_proj", down_in)
        h = h_after_attn + layer.mlp.down_proj(down_in)

    # The expanded traversal must have the stock model's exact arithmetic.
    expanded = model.model.norm(h)
    stock = model.model(input_ids)
    mx.eval(expanded, stock)
    mismatch_count = int(np.count_nonzero(
        np.asarray(expanded.view(mx.uint16)) != np.asarray(stock.view(mx.uint16))
    ))
    if mismatch_count:
        raise RuntimeError(f"expanded stock traversal disagrees at {mismatch_count} words")

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"{role}-activations.npz"
    np.savez(array_path, **arrays)
    record = {
        "role": role,
        "repo": pinned["repo"],
        "revision": pinned["revision"],
        "prompt": PROMPT,
        "token_ids": ids,
        "token_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest(),
        "source": "mlx-lm 0.31.3 Qwen2 stock modules; expanded traversal checked against model.model",
        "stock_expanded_bit_mismatches": mismatch_count,
        "array_file": array_path.name,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": "BF16 words as uint16",
                   "sha256": hashlib.sha256(value.astype("<u2", copy=False).tobytes()).hexdigest()}
            for name, value in arrays.items()
        },
    }
    write_json_atomic(output_dir / f"{role}-activations.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("anchor", "target"), required=True)
    parser.add_argument("--selected", type=Path,
                        default=ROOT / "results/survey/selected-family.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/inputs")
    args = parser.parse_args()
    result = capture(args.role, args.selected, args.out)
    print(json.dumps({"role": result["role"], "vectors": len(result["arrays"]),
                      "stock_expanded_bit_mismatches": result["stock_expanded_bit_mismatches"]}))
