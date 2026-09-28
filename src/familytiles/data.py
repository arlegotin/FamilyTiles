"""Pinned safetensors metadata and strictly bounded checkpoint acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any

from huggingface_hub import HfApi
import requests

from .records import write_json_atomic


HEADER_LIMIT = 16 * 2**20
MAX_METADATA_FILE = 16 * 2**20
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}
REVISION_RE = re.compile(r"[0-9a-f]{40,64}\Z")
RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    data_start: int
    byte_offset: int
    byte_length: int


@dataclass(frozen=True)
class PinnedModel:
    repo: str
    revision: str
    files: dict[str, int]
    tensors: dict[str, TensorInfo]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ValueError("invalid model repository")
        if not REVISION_RE.fullmatch(self.revision):
            raise ValueError("model revision must be an immutable commit SHA")
        for name, size in self.files.items():
            if not _safe_name(name) or not isinstance(size, int) or size < 8:
                raise ValueError("invalid pinned file metadata")


def _safe_name(name: str) -> bool:
    path = Path(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and all(part not in ("", ".") for part in path.parts))


@dataclass(frozen=True)
class TransferBudget:
    sample_limit_bytes: int
    weight_limit_bytes: int
    ledger_path: Path
    DEFAULT_SAMPLE_BYTES = 32 * 2**20
    DEFAULT_WEIGHT_BYTES = 7 * 2**30

    def __post_init__(self) -> None:
        if self.sample_limit_bytes < 0 or self.weight_limit_bytes < 0:
            raise ValueError("negative transfer limit")

    def snapshot(self) -> dict[str, Any]:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        return {"sample_by_model": {}, "weight_reserved": 0, "metadata_reserved": 0}

    def reserve(self, kind: str, amount: int, model_id: str | None = None) -> None:
        if amount < 0 or kind not in ("sample", "weight", "metadata"):
            raise ValueError("invalid transfer reservation")
        if kind == "sample" and not model_id:
            raise ValueError("sample reservation needs a model identity")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.snapshot()
            if kind == "sample":
                current = state["sample_by_model"].get(model_id, 0)
                if current + amount > self.sample_limit_bytes:
                    raise ValueError("checkpoint sample budget exceeded")
                state["sample_by_model"][model_id] = current + amount
            elif kind == "weight":
                if state["weight_reserved"] + amount > self.weight_limit_bytes:
                    raise ValueError("model-weight download budget exceeded")
                state["weight_reserved"] += amount
            else:
                state["metadata_reserved"] += amount
            write_json_atomic(self.ledger_path, state)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def ensure_weight_capacity(self, amount: int) -> None:
        if amount < 0 or self.snapshot()["weight_reserved"] + amount > self.weight_limit_bytes:
            raise ValueError("selected shards exceed remaining model-weight budget")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors header key")
        result[key] = value
    return result


def parse_header(header: bytes, file_size: int, *, shard: str) -> list[TensorInfo]:
    if (not _safe_name(shard) or not isinstance(file_size, int) or file_size < 8
            or len(header) < 8):
        raise ValueError("invalid safetensors file metadata")
    length = struct.unpack("<Q", header[:8])[0]
    if length > HEADER_LIMIT or len(header) != 8 + length or len(header) > file_size:
        raise ValueError("invalid safetensors header length")
    try:
        obj = json.loads(header[8:].decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors header JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("safetensors header must be an object")
    tensors: list[TensorInfo] = []
    extents: list[tuple[int, int]] = []
    for name, item in obj.items():
        if name == "__metadata__":
            if not isinstance(item, dict) or not all(isinstance(v, str) for v in item.values()):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(name, str) or not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("invalid tensor entry")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not all(type(x) is int and 0 <= x < 2**63 for x in shape):
            raise ValueError("unsupported tensor dtype or shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int and 0 <= x < 2**63 for x in offsets):
            raise ValueError("invalid tensor offsets")
        start, end = offsets
        count = 1
        for dimension in shape:
            count *= dimension
            if count > 2**63:
                raise ValueError("tensor element count overflow")
        if end < start or end - start != count * DTYPE_BYTES[dtype] or 8 + length + end > file_size:
            raise ValueError("tensor extent does not match shape or file")
        tensors.append(TensorInfo(name, tuple(shape), dtype, shard, 8 + length, start, end - start))
        extents.append((start, end))
    for (_, previous_end), (next_start, _) in zip(sorted(extents), sorted(extents)[1:]):
        if next_start < previous_end:
            raise ValueError("overlapping tensor regions")
    return tensors


def _pinned_url(model: PinnedModel, filename: str) -> str:
    if not _safe_name(filename):
        raise ValueError("unsafe model filename")
    return f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{filename}"


def read_range(model: PinnedModel, shard: str, start: int, length: int,
               budget: TransferBudget, *, kind: str = "sample") -> bytes:
    if (shard not in model.files or type(start) is not int or type(length) is not int
            or start < 0 or length <= 0 or start + length > model.files[shard]):
        raise ValueError("range outside pinned shard")
    budget.reserve(kind, length, model.repo if kind == "sample" else None)
    response = requests.get(_pinned_url(model, shard),
                            headers={"Range": f"bytes={start}-{start + length - 1}",
                                     "Accept-Encoding": "identity"},
                            stream=True, timeout=(10, 30))
    try:
        # Reject a server that ignores Range before ever buffering its body.
        if response.status_code != 206 or response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("server did not return an uncompressed partial response")
        match = RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (start, start + length - 1, model.files[shard]):
            raise ValueError("range response does not match pinned request")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=min(64 * 1024, length + 1)):
            if len(data) + len(chunk) > length:
                raise ValueError("range body exceeded requested byte cap")
            data.extend(chunk)
        if len(data) != length:
            raise ValueError("short range response")
        return bytes(data)
    finally:
        response.close()


def _read_small(model: PinnedModel, filename: str, size: int,
                budget: TransferBudget) -> str:
    if size > MAX_METADATA_FILE:
        raise ValueError("metadata file exceeds bounded read")
    return hashlib.sha256(read_range(model, filename, 0, size, budget,
                                     kind="metadata")).hexdigest()


def pin_model(repo: str, revision: str | None, budget: TransferBudget) -> PinnedModel:
    """Resolve a mutable name once, then use only the immutable SHA in all URLs."""
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    sha = info.sha
    if not REVISION_RE.fullmatch(sha):
        raise ValueError("Hub returned no immutable commit revision")
    sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".safetensors")}
    if not sizes or any(size is None for size in sizes.values()):
        raise ValueError("Hub omitted checkpoint shard sizes")
    metadata: dict[str, Any] = {"license": getattr(info, "card_data", None).license if getattr(info, "card_data", None) else None,
                                "config_hashes": {}, "file_sha256": {}}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if oid:
            metadata["file_sha256"][sibling.rfilename] = oid
        if sibling.rfilename in ("config.json", "tokenizer.json", "tokenizer_config.json") and sibling.size is not None:
            sizes[sibling.rfilename] = sibling.size
    model = PinnedModel(repo, sha, sizes, {}, metadata)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if filename in sizes:
            metadata["config_hashes"][filename] = _read_small(model, filename, sizes[filename], budget)
    tensors: dict[str, TensorInfo] = {}
    for shard in sorted(name for name in sizes if name.endswith(".safetensors")):
        prefix = read_range(model, shard, 0, 8, budget, kind="metadata")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > HEADER_LIMIT:
            raise ValueError("checkpoint header exceeds cap")
        rest = read_range(model, shard, 8, header_len, budget, kind="metadata")
        for tensor in parse_header(prefix + rest, sizes[shard], shard=shard):
            if tensor.name in tensors:
                raise ValueError("duplicate tensor name across shards")
            tensors[tensor.name] = tensor
    return PinnedModel(repo, sha, {k: v for k, v in sizes.items() if k.endswith(".safetensors")}, tensors, metadata)


def _download_file(model: PinnedModel, filename: str, destination: Path) -> None:
    size = model.files[filename]
    expected_hash = model.metadata.get("file_sha256", {}).get(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    response = requests.get(_pinned_url(model, filename),
                            headers={"Accept-Encoding": "identity"}, stream=True,
                            timeout=(10, 60))
    try:
        if response.status_code != 200 or response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("invalid full-shard response")
        digest = hashlib.sha256()
        total = 0
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=2**20):
                total += len(chunk)
                if total > size:
                    raise ValueError("full-shard body exceeds pinned size")
                digest.update(chunk)
                output.write(chunk)
        if total != size or (expected_hash and digest.hexdigest() != expected_hash):
            raise ValueError("full-shard size/hash mismatch")
        os.replace(temporary, destination)
        write_json_atomic(destination.with_suffix(destination.suffix + ".sha256.json"),
                          {"repo": model.repo, "revision": model.revision,
                           "size": size, "sha256": digest.hexdigest()})
    finally:
        response.close()
        temporary.unlink(missing_ok=True)


def _valid_cached(path: Path, model: PinnedModel, filename: str) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file() or path.stat().st_size != model.files[filename]:
        return False
    try:
        entry = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if entry.get("repo") != model.repo or entry.get("revision") != model.revision or entry.get("size") != model.files[filename]:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = model.metadata.get("file_sha256", {}).get(filename)
    return digest == entry.get("sha256") and (not expected or digest == expected)


def download_selected(models: tuple[PinnedModel, PinnedModel], out: Path,
                      budget: TransferBudget) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    pending: list[tuple[PinnedModel, str, Path]] = []
    for model in models:
        for filename in sorted(model.files):
            path = out / model.repo.replace("/", "--") / model.revision / filename
            paths[f"{model.repo}:{filename}"] = path
            if not _valid_cached(path, model, filename):
                pending.append((model, filename, path))
    budget.ensure_weight_capacity(sum(model.files[filename] for model, filename, _ in pending))
    for model, filename, path in pending:
        budget.reserve("weight", model.files[filename])
        _download_file(model, filename, path)
    return paths
