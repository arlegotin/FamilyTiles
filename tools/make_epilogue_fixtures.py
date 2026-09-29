"""Freeze two additional cached-decoding prefixes before numerical inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from familytiles.records import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def _extend(words: list[int], length: int) -> list[int]:
    return (words * ((length + len(words) - 1) // len(words)))[:length]


def main() -> None:
    source = ROOT / "results/correctness/raw_pair-lockstep.json"
    saved = json.loads(source.read_text())["inputs"]
    anchor, target = saved["anchor_prompt"], saved["target_prompt"]
    pairs = []
    for name, lengths in (("distinct_extended", (21, 39)),
                          ("distinct_long", (128, 96))):
        prompts = {"anchor": _extend(anchor, lengths[0]),
                   "target": _extend(target, lengths[1])}
        tokens = [(prompts["anchor"][(7 * i + 3) % lengths[0]],
                   prompts["target"][(11 * i + 5) % lengths[1]])
                  for i in range(32)]
        case = {"name": name, "prompts": prompts, "tokens": tokens,
                "source": "separate pinned-model token histories in raw_pair-lockstep.json",
                "teacher_forced_steps": 32}
        case["sha256"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        pairs.append(case)
    record = {"version": 1,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "artifact_hash": "ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585",
              "pairs": pairs}
    write_json_atomic(ROOT / "results/epilogue-check/fixtures.json", record)
    print(json.dumps({"source_sha256": record["source_sha256"],
                      "pairs": [{"name": case["name"], "sha256": case["sha256"]}
                                for case in pairs]}))


if __name__ == "__main__":
    main()
