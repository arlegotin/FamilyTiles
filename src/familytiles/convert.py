"""Stream a pinned BF16 family into a checked, depth-one exact artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterator

import numpy as np
import psutil
import requests

from .baselines import encode_exact, encode_family_exact, exact_allocated_bytes
from .codec import (CodecPolicy, EncodedTensor, allocated_bytes, decode_tensor,
                    encode_tensor, validate_encoded)
from .data import PinnedModel, TensorInfo, TransferBudget, download_selected
from .measure import compute_budget
from .records import GateRecord, RunContext, publish_gate, read_gate, write_json_atomic


ACTIVE_RE = re.compile(
    r"^(model\.embed_tokens\.weight|model\.norm\.weight|lm_head\.weight|"
    r"model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight|"
    r"model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|"
    r"mlp\.(?:gate|up|down)_proj)\.(?:weight|bias))$"
)
PROJECTION_RE = re.compile(r"\.((?:q|k|v|o|gate|up|down)_proj)\.weight$")
FORMAT_VERSION = 1


@dataclass(frozen=True)
class FamilyArtifact:
    root: Path
    manifest: dict[str, Any]
    manifest_hash: str


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_words(words: np.ndarray) -> str:
    return hashlib.sha256(memoryview(words).cast("B")).hexdigest()


def _safe_path(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("artifact path escapes its root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact path resolves outside its root")
    return resolved


def _from_record(record: dict[str, Any]) -> PinnedModel:
    return PinnedModel(record["repo"], record["revision"], record["files"],
                       {name: TensorInfo(**value) for name, value in record["tensors"].items()},
                       record["metadata"])


def _source_paths(models: tuple[PinnedModel, PinnedModel], survey_root: Path) -> dict[str, dict[str, Path]]:
    budget = TransferBudget(32 * 2**20, 7 * 2**30,
                            survey_root / "acquisition-ledger.json")
    repo_root = Path(__file__).resolve().parents[2]
    acquired = download_selected(models, repo_root / "artifacts/downloaded", budget)
    return {role: {filename: acquired[f"{model.repo}:{filename}"]
                   for filename in model.files}
            for role, model in zip(("anchor", "target"), models)}


def _require_g1(selected: dict[str, Any], policy: CodecPolicy) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gate = read_gate(repo_root / "results/gates/G1.json")
    policy_json = asdict(policy)
    digest = hashlib.sha256(json.dumps(policy_json, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    if (gate.decision != "pass" or gate.context.family_id != selected["family_id"]
            or gate.context.revisions != selected["revisions"]
            or gate.context.policy_hash != digest):
        raise ValueError("selected family/policy does not match passing G1 evidence")


def _config(model: PinnedModel) -> dict[str, Any]:
    embedded = model.metadata.get("config")
    if embedded is not None:
        return embedded
    url = f"https://huggingface.co/{model.repo}/resolve/{model.revision}/config.json"
    response = requests.get(url, headers={"Accept-Encoding": "identity"},
                            stream=True, timeout=(10, 30))
    try:
        if response.status_code != 200 or response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("pinned config acquisition failed")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if len(data) + len(chunk) > 2**20:
                raise ValueError("pinned config exceeds cap")
            data.extend(chunk)
    finally:
        response.close()
    expected = model.metadata.get("config_hashes", {}).get("config.json")
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("pinned config hash mismatch")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("invalid model config")
    return result


def _view(path: Path, info: TensorInfo) -> np.memmap:
    if info.dtype != "BF16":
        raise ValueError("selected active tensor is not stored as BF16")
    return np.memmap(path, mode="r", dtype="<u2",
                     offset=info.data_start + info.byte_offset, shape=info.shape)


def _best_independent(words: np.ndarray) -> int:
    return min(exact_allocated_bytes(encode_exact(words, transform))
               for transform in ("native", "byte_plane", "field"))


def _best_family(anchor: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    raw_anchor = min(exact_allocated_bytes(encode_family_exact(anchor, target, transform, "raw"))
                     for transform in ("xor", "ordered_delta"))
    compressed_anchor = min(exact_allocated_bytes(encode_family_exact(anchor, target, transform, "compressed"))
                            for transform in ("xor", "ordered_delta"))
    return raw_anchor, compressed_anchor


def _save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {"path": str(path.name if path.parent.name == "" else path.relative_to(path.parents[1])),
            "file_bytes": path.stat().st_size, "sha256": _sha(path)}


def convert_family(family: Path, policy: Path, out: Path) -> FamilyArtifact:
    selected = json.loads(family.read_text(encoding="utf-8"))
    frozen = json.loads(policy.read_text(encoding="utf-8"))
    if frozen.get("family_id", selected["family_id"]) != selected["family_id"]:
        raise ValueError("policy family mismatch")
    chosen = CodecPolicy(frozen["version"], frozen["block_values"], tuple(frozen["transforms"]))
    _require_g1(selected, chosen)
    pinned_path = _safe_path(family.parent, selected["pinned_manifest"])
    pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    models = (_from_record(pinned["anchor"]), _from_record(pinned["target"]))
    for role, model in zip(("anchor", "target"), models):
        if model.repo != selected[role] or model.revision != selected["revisions"][role]:
            raise ValueError("pinned source identity mismatch")
    memory = psutil.virtual_memory()
    budget = compute_budget(memory.total, memory.available)
    if budget.process_limit_bytes < 256 * 2**20:
        raise ValueError("safe conversion memory budget is too small")
    sources = _source_paths(models, family.parent)
    configs = {role: _config(model) for role, model in zip(("anchor", "target"), models)}
    for role, model in zip(("anchor", "target"), models):
        if not configs[role].get("tie_word_embeddings", False) and "lm_head.weight" not in model.tensors:
            raise ValueError("untied output head is missing from the checkpoint")
        if configs[role].get("tie_word_embeddings", False) and "lm_head.weight" in model.tensors:
            head = _view(sources[role][model.tensors["lm_head.weight"].shard], model.tensors["lm_head.weight"])
            embed = _view(sources[role][model.tensors["model.embed_tokens.weight"].shard],
                          model.tensors["model.embed_tokens.weight"])
            if head.shape != embed.shape or not np.array_equal(head, embed):
                raise ValueError("declared tied head conflicts with stored embedding")
    if out.exists():
        raise ValueError("artifact already exists; conversion is immutable")
    temporary = out.with_name(out.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    try:
        anchor_dir = temporary / "source/anchor"
        anchor_dir.mkdir(parents=True)
        for filename, path in sources["anchor"].items():
            destination = _safe_path(anchor_dir, filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
        anchor_names = {name for name in models[0].tensors if ACTIVE_RE.fullmatch(name)}
        target_names = {name for name in models[1].tensors if ACTIVE_RE.fullmatch(name)}
        if anchor_names != target_names:
            raise ValueError("served models have different active tensor inventories")
        excluded = sorted((set(models[0].tensors) | set(models[1].tensors)) - anchor_names)
        tensors = {}
        for index, name in enumerate(sorted(anchor_names)):
            a_info, t_info = models[0].tensors[name], models[1].tensors[name]
            if a_info.dtype != t_info.dtype or a_info.shape != t_info.shape:
                raise ValueError(f"active tensor shape/dtype mismatch: {name}")
            a = _view(sources["anchor"][a_info.shard], a_info)
            t = _view(sources["target"][t_info.shard], t_info)
            a_hash, t_hash = _hash_words(a), _hash_words(t)
            target: dict[str, Any]
            if a_hash == t_hash and np.array_equal(a, t):
                target = {"kind": "alias", "name": name, "sha256": t_hash,
                          "allocated_bytes": 0, "b2_bytes": 0}
            elif PROJECTION_RE.search(name) and len(a_info.shape) == 2:
                encoded = encode_tensor(a, t, chosen)
                if encoded.kind == "packed":
                    np.testing.assert_array_equal(decode_tensor(a, encoded), t)
                    descriptor_path = temporary / f"encoded/{index}.descriptors.npy"
                    payload_path = temporary / f"encoded/{index}.payload.npy"
                    _save_array(descriptor_path, encoded.descriptors)
                    _save_array(payload_path, encoded.payload)
                    target = {"kind": "packed", "sha256": t_hash,
                              "descriptor_path": str(descriptor_path.relative_to(temporary)),
                              "payload_path": str(payload_path.relative_to(temporary)),
                              "descriptor_sha256": _sha(descriptor_path),
                              "payload_sha256": _sha(payload_path),
                              "allocated_bytes": descriptor_path.stat().st_size + payload_path.stat().st_size}
                else:
                    target = {"kind": "native", "sha256": t_hash}
                b2 = encode_tensor(a, t, chosen, modes=("copy", "raw"))
                target["b2_bytes"] = (allocated_bytes(b2) + (256 if b2.kind == "packed" else 0))
            else:
                target = {"kind": "native", "sha256": t_hash, "b2_bytes": t_info.byte_length}
            if target["kind"] == "native":
                native_path = temporary / f"native/{index}.bin"
                native_path.parent.mkdir(parents=True, exist_ok=True)
                t.tofile(native_path)
                target.update({"path": str(native_path.relative_to(temporary)),
                               "file_sha256": _sha(native_path),
                               "allocated_bytes": native_path.stat().st_size})
            # B3/B4 are exact local size controls, including native leftovers.
            b3 = _best_independent(a) + _best_independent(t)
            b4_raw, b4_compressed = _best_family(a, t)
            tensors[name] = {"shape": list(a_info.shape), "dtype": a_info.dtype,
                             "anchor": {"path": f"source/anchor/{a_info.shard}",
                                        "offset": a_info.data_start + a_info.byte_offset,
                                        "sha256": a_hash, "allocated_bytes": a_info.byte_length},
                             "target": target,
                             "b3_pair_bytes": b3,
                             "b4_raw_anchor_pair_bytes": b4_raw,
                             "b4_compressed_anchor_pair_bytes": b4_compressed}
            del a, t
            peak_rss = max(peak_rss, process.memory_info().rss)
        manifest = {"format_version": FORMAT_VERSION,
                    "family_id": selected["family_id"],
                    "revisions": selected["revisions"],
                    "source_repos": {"anchor": models[0].repo, "target": models[1].repo},
                    "source_file_sha256": {filename: _sha(path) for filename, path in sources["anchor"].items()},
                    "source_file_bytes": {filename: path.stat().st_size for filename, path in sources["anchor"].items()},
                    "config_hashes": {role: model.metadata.get("config_hashes", {})
                                      for role, model in zip(("anchor", "target"), models)},
                    "tie_word_embeddings": {role: bool(configs[role].get("tie_word_embeddings", False))
                                            for role in ("anchor", "target")},
                    "policy": asdict(chosen), "tensors": tensors,
                    "excluded_tensors": excluded,
                    "conversion_seconds": time.perf_counter() - started,
                    "conversion_peak_rss_bytes": peak_rss,
                    "memory_budget_bytes": budget.process_limit_bytes}
        write_json_atomic(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(_sha(temporary / "manifest.json") + "\n")
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_artifact(out)


def load_artifact(root: Path) -> FamilyArtifact:
    path = root / "manifest.json"
    digest_path = root / "manifest.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise ValueError("incomplete family artifact")
    actual = _sha(path)
    if digest_path.read_text().strip() != actual:
        raise ValueError("artifact manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported family artifact version")
    if not isinstance(manifest.get("tensors"), dict) or not manifest["tensors"]:
        raise ValueError("missing active tensor inventory")
    policy = CodecPolicy(**{**manifest["policy"], "transforms": tuple(manifest["policy"]["transforms"])})
    for filename, expected in manifest["source_file_sha256"].items():
        source = _safe_path(root, f"source/anchor/{filename}")
        if not source.is_file() or _sha(source) != expected:
            raise ValueError("wrong or damaged anchor source")
    for name, record in manifest["tensors"].items():
        shape = tuple(record["shape"])
        if not ACTIVE_RE.fullmatch(name) or record["dtype"] != "BF16" or not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("invalid active tensor record")
        anchor_path = _safe_path(root, record["anchor"]["path"])
        anchor_length = int(np.prod(shape)) * 2
        if not anchor_path.is_file() or record["anchor"]["offset"] + anchor_length > anchor_path.stat().st_size:
            raise ValueError("truncated or missing anchor")
        target = record["target"]
        if target["kind"] == "alias":
            other = target["name"]
            if other != name or target["sha256"] != record["anchor"]["sha256"]:
                raise ValueError("invalid exact alias coordinate or hash")
        elif target["kind"] == "native":
            native = _safe_path(root, target["path"])
            if not native.is_file() or native.stat().st_size != anchor_length or _sha(native) != target["file_sha256"]:
                raise ValueError("invalid native target")
        elif target["kind"] == "packed":
            desc_path = _safe_path(root, target["descriptor_path"])
            payload_path = _safe_path(root, target["payload_path"])
            if _sha(desc_path) != target["descriptor_sha256"] or _sha(payload_path) != target["payload_sha256"]:
                raise ValueError("encoded tensor file checksum mismatch")
            desc = np.load(desc_path, mmap_mode="r", allow_pickle=False)
            payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
            a = _view(anchor_path, TensorInfo(name, shape, "BF16", anchor_path.name, 0,
                                              record["anchor"]["offset"], anchor_length))
            validate_encoded(a, EncodedTensor("packed", shape, policy, desc, payload, None))
        else:
            raise ValueError("unknown target tensor kind")
    return FamilyArtifact(root, manifest, actual)


def iter_tensor_words(artifact: FamilyArtifact, model_id: str, name: str,
                      chunk_rows: int) -> Iterator[np.ndarray]:
    if model_id not in ("anchor", "target") or chunk_rows <= 0:
        raise ValueError("invalid model or row chunk")
    record = artifact.manifest["tensors"].get(name)
    if record is None:
        raise ValueError("tensor is not active in artifact")
    shape = tuple(record["shape"])
    anchor_record, target = record["anchor"], record["target"]
    a_path = _safe_path(artifact.root, anchor_record["path"])
    a = np.memmap(a_path, mode="r", dtype="<u2", offset=anchor_record["offset"], shape=shape)
    if model_id == "anchor" or target["kind"] == "alias":
        values = a
    elif target["kind"] == "native":
        values = np.memmap(_safe_path(artifact.root, target["path"]), mode="r",
                           dtype="<u2", shape=shape)
    else:
        policy = CodecPolicy(**{**artifact.manifest["policy"],
                                "transforms": tuple(artifact.manifest["policy"]["transforms"])})
        desc = np.load(_safe_path(artifact.root, target["descriptor_path"]), mmap_mode="r", allow_pickle=False)
        payload = np.load(_safe_path(artifact.root, target["payload_path"]), mmap_mode="r", allow_pickle=False)
        values = decode_tensor(a, EncodedTensor("packed", shape, policy, desc, payload, None))
    for start in range(0, shape[0], chunk_rows):
        yield np.asarray(values[start:start + chunk_rows])


def verify_cpu(artifact: FamilyArtifact) -> dict[str, Any]:
    verified = 0
    words = 0
    mismatches = 0
    anchor_mismatches = 0
    for name, record in artifact.manifest["tensors"].items():
        anchor_digest = hashlib.sha256()
        for chunk in iter_tensor_words(artifact, "anchor", name, 64):
            anchor_digest.update(memoryview(chunk).cast("B"))
        if anchor_digest.hexdigest() != record["anchor"]["sha256"]:
            anchor_mismatches += 1
        digest = hashlib.sha256()
        for chunk in iter_tensor_words(artifact, "target", name, 64):
            digest.update(memoryview(chunk).cast("B"))
            words += chunk.size
        if digest.hexdigest() != record["target"]["sha256"]:
            mismatches += 1
        verified += 1
    return {"verified_active_tensors": verified, "verified_target_words": words,
            "mismatched_words": 0 if mismatches == 0 and anchor_mismatches == 0 else None,
            "hash_mismatched_target_tensors": mismatches,
            "hash_mismatched_anchor_tensors": anchor_mismatches,
            "representation_exact": mismatches == 0 and anchor_mismatches == 0,
            "artifact_hash": artifact.manifest_hash}


def canonical_ledger(artifact: FamilyArtifact) -> dict[str, Any]:
    records = artifact.manifest["tensors"].values()
    anchor = sum(r["anchor"]["allocated_bytes"] for r in records)
    raw_target = sum(int(np.prod(r["shape"])) * 2 for r in artifact.manifest["tensors"].values())
    alias = sum(int(np.prod(r["shape"])) * 2 for r in artifact.manifest["tensors"].values()
                if r["target"]["kind"] == "alias")
    b2_target = sum(r["target"]["b2_bytes"] for r in artifact.manifest["tensors"].values())
    family_target = sum(r["target"]["allocated_bytes"] for r in artifact.manifest["tensors"].values())
    manifest_bytes = (artifact.root / "manifest.json").stat().st_size + (artifact.root / "manifest.sha256").stat().st_size
    w0, w1 = anchor + raw_target, anchor + raw_target - alias
    return {"w0_bytes": w0, "w1_bytes": w1,
            "b2_bytes": anchor + b2_target + manifest_bytes,
            "family_bytes": anchor + family_target + manifest_bytes,
            "anchor_bytes": anchor, "raw_target_bytes": raw_target,
            "alias_bytes": alias, "native_target_bytes": sum(r["target"]["allocated_bytes"]
                                                    for r in artifact.manifest["tensors"].values()
                                                    if r["target"]["kind"] == "native"),
            "manifest_bytes": manifest_bytes,
            "b3_bytes": sum(r["b3_pair_bytes"] for r in artifact.manifest["tensors"].values()),
            "b4_raw_anchor_bytes": sum(r["b4_raw_anchor_pair_bytes"] for r in artifact.manifest["tensors"].values()),
            "b4_compressed_anchor_bytes": sum(r["b4_compressed_anchor_pair_bytes"] for r in artifact.manifest["tensors"].values()),
            "family_saving_vs_b1": (w1 - anchor - family_target - manifest_bytes) / w1,
            "family_beyond_b2_raw": (b2_target - family_target) / w0}


def evaluate_g2(artifact: FamilyArtifact, audit: dict[str, Any],
                ledger: dict[str, Any]) -> GateRecord:
    memory = psutil.virtual_memory()
    budget = compute_budget(memory.total, memory.available)
    matrices = [r for r in artifact.manifest["tensors"].values()
                if len(r["shape"]) == 2 and r["target"]["kind"] == "packed"]
    max_matrix = max((int(np.prod(r["shape"])) * 2 for r in matrices), default=0)
    layers = max((int(match.group(1)) + 1 for name in artifact.manifest["tensors"]
                  if (match := re.search(r"\.layers\.(\d+)\.", name))), default=0)
    embed = artifact.manifest["tensors"].get("model.embed_tokens.weight")
    hidden = embed["shape"][1] if embed else 0
    # Deliberately overestimates two 256-token BF16 KV states using full hidden width.
    kv_bound = 2 * 2 * layers * 256 * hidden * 2
    staging_bound = 2 * max_matrix + kv_bound + 2 * 256 * 2**20
    planned_peak = ledger["family_bytes"] + staging_bound
    observed = {"audit": audit, "ledger": ledger,
                "safe_process_budget_bytes": budget.process_limit_bytes,
                "largest_matrix_bytes": max_matrix,
                "kv_bound_bytes": kv_bound,
                "staging_and_cache_bound_bytes": staging_bound,
                "planned_peak_bytes": planned_peak,
                "download_reserved_bytes": TransferBudget(32 * 2**20, 7 * 2**30,
                    Path(__file__).resolve().parents[2] / "results/survey/acquisition-ledger.json").snapshot()["weight_reserved"]}
    success = (audit["representation_exact"]
               and ledger["family_saving_vs_b1"] >= 0.15
               and ledger["family_beyond_b2_raw"] >= 0.05
               and planned_peak <= budget.process_limit_bytes
               and observed["download_reserved_bytes"] <= 7 * 2**30)
    gate = read_gate(Path(__file__).resolve().parents[2] / "results/gates/G1.json")
    context = RunContext(artifact.manifest["family_id"], artifact.manifest["revisions"],
                         gate.context.policy_hash, artifact.manifest_hash,
                         hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         gate.context.environment_hash, None)
    return GateRecord(1, "G2", context,
                      {"artifact": artifact.manifest_hash,
                       "g1_summary": gate.input_hashes["summary"]},
                      {"minimum_b1_saving": 0.15,
                       "minimum_beyond_b2_raw": 0.05,
                       "maximum_process_budget_bytes": budget.process_limit_bytes},
                      observed, "pass" if success else "fail",
                      "complete CPU exactness, size and planned runtime budget passed" if success else
                      "complete CPU exactness, size or planned runtime budget failed")


def publish_g2(artifact: FamilyArtifact, audit: dict[str, Any]) -> GateRecord:
    repo_root = Path(__file__).resolve().parents[2]
    ledger = canonical_ledger(artifact)
    record = evaluate_g2(artifact, audit, ledger)
    write_json_atomic(repo_root / "results/conversion/conversion.json", {
        "family_id": artifact.manifest["family_id"],
        "revisions": artifact.manifest["revisions"],
        "artifact_hash": artifact.manifest_hash,
        "conversion_seconds_excluding_final_load": artifact.manifest["conversion_seconds"],
        "sampled_conversion_peak_rss_bytes": artifact.manifest["conversion_peak_rss_bytes"],
        "safe_budget_at_conversion_bytes": artifact.manifest["memory_budget_bytes"],
        "peak_rss_is_sampled_lower_bound": True,
    })
    write_json_atomic(repo_root / "results/conversion/cpu-audit.json", audit)
    write_json_atomic(repo_root / "results/conversion/ledger.json", ledger)
    publish_gate(repo_root / "results/gates", record)
    return record
