"""Untimed per-linear diagnosis of raw custom GEMV versus stock MLX linear."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from familytiles.convert import load_artifact
from familytiles.metal import kernel_config
from familytiles.model import FamilyLinear, load_runtime, prefill, step_single
from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifact = load_artifact(ROOT / "artifacts/family")
    runtime = load_runtime(artifact, "raw_pair")
    inputs = json.loads((ROOT / "results/correctness/raw_pair-lockstep.json").read_text())["inputs"]
    labels = {id(module): f"{role}:{name}"
              for role, model in runtime.models.items()
              for name, module in model.named_modules()
              if isinstance(module, FamilyLinear)}
    records = []
    original = FamilyLinear.decode_step

    def traced(self: FamilyLinear, x: mx.array) -> mx.array:
        actual = original(self, x)
        words = (self.operand.anchor_words if self.model_id == "anchor"
                 else self.operand.native_words)
        weight = words.view(mx.bfloat16)
        x3 = x.reshape(1, 1, -1)
        stock = (mx.addmm(self.bias, x3, weight.T) if self.bias is not None
                 else x3 @ weight.T)[0, 0, :]
        mx.eval(actual, stock)
        actual_words = np.asarray(actual.view(mx.uint16))
        stock_words = np.asarray(stock.view(mx.uint16))
        difference = (np.asarray(actual.astype(mx.float32)).astype(np.float64) -
                      np.asarray(stock.astype(mx.float32)).astype(np.float64))
        reference = np.asarray(stock.astype(mx.float32)).astype(np.float64)
        mismatches = np.flatnonzero(actual_words != stock_words)
        exact_samples = []
        if self.bias is None and len(mismatches) and len(records) < 30:
            host_x = np.asarray(x.astype(mx.float32)).astype(np.float64)
            for row in mismatches[:2]:
                host_w = np.asarray(weight[int(row)].astype(mx.float32)).astype(np.float64)
                exact = float(np.dot(host_w, host_x))
                nearest = np.array([np.float32(exact)]).view(np.uint32)[0]
                bf16 = int((int(nearest) + 0x7fff + ((int(nearest) >> 16) & 1)) >> 16) & 0xffff
                exact_samples.append({"row": int(row), "exact_bf16": bf16,
                                      "stock_bf16": int(stock_words[row]),
                                      "custom_bf16": int(actual_words[row])})
        records.append({"linear": labels[id(self)],
                        "bias": self.bias is not None,
                        "bit_mismatches": int(len(mismatches)),
                        "exact_samples": exact_samples,
                        "normalized_rms": float(np.sqrt(np.sum(difference * difference) /
                                                        max(np.sum(reference * reference), 1e-24)))})
        return actual

    FamilyLinear.decode_step = traced
    try:
        for role in ("anchor", "target"):
            _, state = prefill(runtime, role, inputs[f"{role}_prompt"])
            step_single(runtime, state, inputs[f"{role}_next"])
    finally:
        FamilyLinear.decode_step = original
    result = {"artifact_hash": artifact.manifest_hash,
              "kernel_config": kernel_config(),
              "inputs": inputs,
              "linear_count": len(records),
              "first_mismatches": [record for record in records if record["bit_mismatches"]][:12],
              "total_bit_mismatches": sum(record["bit_mismatches"] for record in records),
              "records": records}
    write_json_atomic(ROOT / "results/correctness/raw-step-linear-trace.json", result)
    print(json.dumps({"linears": len(records),
                      "total_bit_mismatches": result["total_bit_mismatches"],
                      "first_mismatches": result["first_mismatches"][:4]}))


if __name__ == "__main__":
    main()
