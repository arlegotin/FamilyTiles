"""Render supported research status from saved, source-bound records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .records import GATES, GateRecord, read_gate


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _early_details(results: Path, records: dict[str, GateRecord]) -> list[str]:
    if not all(key in records for key in ("G2", "G3")):
        return []
    g2, g3 = records["G2"], records["G3"]
    ledger = _optional_json(results / "conversion/ledger.json")
    summary = _optional_json(results / "kernel/summary.json")
    if ledger is None or summary is None:
        return []
    if ledger != g2.observed.get("ledger") or summary.get("gate") != g3.to_dict():
        raise ValueError("size ledger or kernel summary differs from gate evidence")
    if (g2.context.artifact_hash != g3.context.artifact_hash or
            g2.context.policy_hash != g3.context.policy_hash or
            g2.context.revisions != g3.context.revisions):
        raise ValueError("G2 and G3 artifact, policy or revisions disagree")
    single = _optional_json(results / "correctness/single-model-comparison.json")
    raw_pair = _optional_json(results / "correctness/raw-lockstep-comparison.json")
    gpu = _optional_json(results / "correctness/gpu.json")
    environment = _optional_json(results / "environment.json")
    small_projection = _optional_json(results / "survey/qwen2.5-0.5b/projection.json")
    selected_projection = _optional_json(results / "survey/qwen2.5-1.5b/projection.json")
    for optional in (single, raw_pair, gpu):
        if optional is not None and optional.get("artifact_hash") != g3.context.artifact_hash:
            raise ValueError("model or GPU evidence refers to a different artifact")
    if selected_projection is not None and selected_projection.get("revisions") != g3.context.revisions:
        raise ValueError("survey projection refers to different model revisions")
    observed = g3.observed
    median = observed["median_sweep_ns_by_mode"]
    best = observed.get("best_native_pair_mode") or min(
        (key for key in median if key.startswith("native_")), key=median.get)
    ratio = observed["pair_ratio_vs_best_native"]
    revisions = g3.context.revisions
    packages = environment.get("package_versions", {}) if environment else {}
    device = environment.get("device_info", {}) if environment else {}
    machine = (f"{device.get('device_name', 'unrecorded GPU')}, "
               f"{environment['physical_bytes'] / 2**30:.1f} GiB physical memory, "
               f"macOS {environment.get('macos', 'unrecorded')}, "
               f"Python {str(environment.get('python', 'unrecorded')).split()[0]}, "
               f"MLX {packages.get('mlx', 'unrecorded')}, "
               f"MLX-LM {packages.get('mlx-lm', 'unrecorded')}"
               if environment else "Machine record unavailable")
    lines = [
        "## Supported result", "",
        (f"On the pinned Qwen2.5-1.5B base/Instruct pair, FamilyTiles reconstructed "
         f"{g2.observed['audit']['verified_target_words']:,} "
         f"target BF16 words without a mismatch and used "
         f"{_percent(ledger['family_saving_vs_b1'])} fewer canonical active-weight bytes "
         f"than exact native sharing. The latest source-matched paired-kernel sweep "
         f"took {ratio:.2f}× the fastest native pair control, above the 1.50× G3 "
         f"ceiling. The full inference result remains unproven; G4/G5 were not run."),
        "", "## Machine and pinned data", "", machine, "",
        f"- Anchor: `Qwen/Qwen2.5-1.5B` at `{revisions.get('anchor', 'unrecorded')}`.",
        f"- Target: `Qwen/Qwen2.5-1.5B-Instruct` at `{revisions.get('target', 'unrecorded')}`.",
        "",
        f"Policy: `{g3.context.policy_hash}`; artifact: `{g3.context.artifact_hash}`. "
        "[Environment](results/environment.json), [dependency lock](results/requirements-lock.txt).",
        "", "## Exact-weight memory", "",
        "| Representation | Full active-weight bytes | Status |",
        "| --- | ---: | --- |",
        f"| B1 native exact sharing | {ledger['w0_bytes']:,} | measured allocation reference |",
        f"| B2 tile identity only | {ledger['b2_bytes']:,} | complete exact size audit |",
        f"| FamilyTiles anchor + patches + native leftovers | {ledger['family_bytes']:,} | complete exact size audit |",
        f"| B3 independent transformed Zstd | {ledger['b3_bytes']:,} | encoded size only |",
        f"| B4 family-delta Zstd with compressed anchor | {ledger['b4_compressed_anchor_bytes']:,} | encoded size only |",
        "",
        (f"FamilyTiles saved {_percent(ledger['family_beyond_b2_raw'])} of raw family "
         f"bytes beyond B2. B3/B4 are smaller encoded stores; their live "
         f"inference latency and transient memory were not measured. "
         "[Full tensor ledger](results/conversion/ledger.json)."),
    ]
    if small_projection is not None and selected_projection is not None:
        lines[4:4] = [(f"The bounded G1 sample screen rejected the prespecified "
                            f"0.5B pair at {_percent(small_projection['saving_vs_b1'])} "
                            f"projected full-family savings; the selected 1.5B pair "
                            f"projected {_percent(selected_projection['saving_vs_b1'])}. "
                            f"These were estimates, followed by the complete 1.5B audit. "
                            "[Survey](results/survey/summary.json), "
                            "[0.5B projection](results/survey/qwen2.5-0.5b/projection.json), "
                            "[1.5B projection](results/survey/qwen2.5-1.5b/projection.json)."), ""]
    if single is not None:
        lines += ["", (f"A one-request streamed target prefill matched stock logits "
                           f"bit for bit ({single['bit_mismatches']} mismatches across "
                           f"{single['logit_words']:,} words). Its two-model loaded "
                           f"weight allocation was {_percent(single['two_model_weight_allocation_saving'])} "
                           f"lower; loaded process physical footprint was "
                           f"{_percent(single['two_model_loaded_physical_footprint_saving'])} "
                           f"lower. These physical-footprint observations are phase samples, "
                           f"not instantaneous peaks. "
                           "[Single-model control](results/correctness/single-model-comparison.json).")]
    lines += ["", "## Paired-kernel evidence", "",
              "| Sweep mode | Median complete sweep | Interpretation |",
              "| --- | ---: | --- |",
              f"| FamilyTiles paired | {median['family_pair'] / 1e6:.2f} ms | latest source-matched run |",
              f"| {best} | {median[best] / 1e6:.2f} ms | fastest native pair control |",
              f"| Two FamilyTiles singles | {median['family_two_singles'] / 1e6:.2f} ms | grouping ablation |",
              "",
              (f"C1 GPU reconstruction and C2 same-schedule arithmetic were exact "
               f"({str(observed['c1_zero_mismatches']).lower()}, "
               f"{str(observed['c2_bitwise_equal']).lower()}). The latest paired sweep "
               f"was {ratio:.2f}× native. [Latest gate](results/gates/G3.json), "
               "[sweep summary](results/kernel/summary.json).")]
    history: list[tuple[str, GateRecord]] = []
    for path in (results / "gates").glob("*/G3.json"):
        candidate = read_gate(path)
        if (candidate.context.family_id == g3.context.family_id and
                candidate.context.revisions == g3.context.revisions and
                candidate.context.artifact_hash == g3.context.artifact_hash):
            history.append((path.parent.name, candidate))
    if history:
        lines += ["", "Saved G3 attempts are source-bound; different sources and run "
                  "conditions are not averaged together.", "",
                  "| Kernel source SHA prefix | Pair/native ratio | Decision | Record |",
                  "| --- | ---: | --- | --- |"]
        for identifier, candidate in sorted(history, key=lambda item: (
                item[1].context.implementation_hash or "", item[1].observed.get(
                    "pair_ratio_vs_best_native", 0))):
            lines.append(f"| `{(candidate.context.implementation_hash or '')[:10]}` | "
                         f"{candidate.observed['pair_ratio_vs_best_native']:.2f}× | "
                         f"{candidate.decision} | "
                         f"[raw gate](results/gates/{identifier}/G3.json) |")
    if raw_pair is not None:
        anchor = raw_pair["versus_stock"]["anchor"]
        target = raw_pair["versus_stock"]["target"]
        lines += ["", "## Exploratory model control after the kernel gate", "",
                  (f"Raw paired and separate custom-kernel calls agreed bit for bit "
                   f"for two distinct requests, and prefill logits/KV matched stock. "
                   f"The first decode-step logit normalized RMS error versus stock "
                   f"was {anchor['normalized_rms']:.4f} (anchor) and "
                   f"{target['normalized_rms']:.4f} (target), exceeding the "
                   f"predeclared 0.001 regression threshold on this diagnostic. "
                   f"This is not the full 2,048-prediction C3 corpus. "
                   "[Raw control](results/correctness/raw-lockstep-comparison.json).")]
    if gpu is not None:
        lines += ["", (f"GPU C1 verification covered {gpu['verified_encoded_tensors']} "
                           f"encoded tensors and {gpu['verified_encoded_words']:,} words "
                           f"with {gpu['hash_mismatched_tensors']} mismatched hashes. "
                           "[GPU record](results/correctness/gpu.json).")]
    lines += ["", "## Scope and reproduction", "",
              "Measured here: exact full-model representation, bounded GPU word decoding, "
              "source-bound matrix sweeps, one-request materialized prefill, and a short "
              "raw lockstep diagnostic. Core paired LLM throughput, process peak during "
              "a matched core workload, 2,048-prediction C3, quantization, streaming "
              "runtime baselines, switching and stress tests remain unmeasured. CUDA "
              "comparisons in `prior_art.md` are literature only.", "",
              "```bash",
              "python3 -m venv .venv && . .venv/bin/activate",
              "python -m pip install -e .",
              "python -m pytest tests/test_codec.py -q",
              "python -m familytiles.cli survey --families experiments/families.json --sample-mib-per-model 32 --seed 20260928 --out results/survey",
              "python -m familytiles.cli convert --family results/survey/selected-family.json --policy results/survey/frozen-policy.json --out artifacts/family",
              "python -m familytiles.cli verify --artifact artifacts/family --device cpu",
              "python -m familytiles.cli verify --artifact artifacts/family --device gpu",
              "python tools/check_pair_real.py",
              "python -m familytiles.cli bench --artifact artifacts/family --suite kernel",
              "python -m familytiles.cli report --results results --out RESULTS.md",
              "```", "",
              "The model downloads and converted artifact are excluded from Git. "
              "The saved raw records, pinned revisions, policy, and dependency lock "
              "identify the tested experiment."]
    return lines


def render_report(results: Path, out: Path) -> None:
    gate_dir = results / "gates"
    records: dict[str, GateRecord] = {}
    if gate_dir.exists():
        for path in sorted(gate_dir.glob("*.json")):
            record = read_gate(path)
            if path.stem.upper() != record.gate:
                raise ValueError("gate filename and record disagree")
            records[record.gate] = record
    identities = {
        (record.context.family_id, tuple(sorted(record.context.revisions.items())))
        for record in records.values() if record.context.family_id is not None
    }
    if len(identities) > 1:
        raise ValueError("gate records mix model identities")
    rows = []
    for gate in GATES:
        record = records.get(gate)
        decision = record.decision if record else "not_run"
        reason = record.reason.replace("|", "\\|").replace("\n", " ") if record else "No saved evidence"
        rows.append(f"| {gate} | {decision} | {reason} |")
    last = next((records[gate] for gate in reversed(GATES) if gate in records), None)
    if last is None:
        claim = "No research gate has been executed."
    elif last.decision == "fail":
        claim = f"The research stopped at {last.gate}: {last.reason}"
    elif last.decision == "blocked":
        claim = f"The research is blocked at {last.gate}: {last.reason}"
    elif last.gate == "G5" and last.decision == "pass":
        claim = "G5 passed according to the saved gate decision; inspect linked measurements before citing a result."
    else:
        claim = f"The latest saved gate is {last.gate} ({last.decision}); the research result is partial."
    lines = ["# FamilyTiles results", "", claim, "", "| Gate | Decision | Reason |",
             "| --- | --- | --- |", *rows, ""]
    if "G3" in records and records["G3"].decision == "fail":
        lines.extend(_early_details(results, records))
    else:
        lines.append("Measurements and reproduction commands are added as the corresponding gates run.")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
