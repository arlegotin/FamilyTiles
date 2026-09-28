"""Versioned research records and path/prerequisite checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal
from uuid import uuid4


JsonObject = dict[str, Any]
ModelId = Literal["anchor", "target"]
Decision = Literal["pass", "fail", "blocked", "not_run"]
DECISIONS = frozenset({"pass", "fail", "blocked", "not_run"})
GATES = tuple(f"G{i}" for i in range(6))
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunContext:
    family_id: str | None
    revisions: dict[str, str]
    policy_hash: str | None
    artifact_hash: str | None
    implementation_hash: str | None
    environment_hash: str | None
    input_hash: str | None

    @classmethod
    def from_dict(cls, value: JsonObject) -> RunContext:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid run context fields")
        if not isinstance(value["revisions"], dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in value["revisions"].items()
        ):
            raise ValueError("invalid revision map")
        for name in cls.__dataclass_fields__:
            if name != "revisions" and value[name] is not None and not isinstance(value[name], str):
                raise ValueError(f"invalid context field: {name}")
        return cls(**value)


@dataclass(frozen=True)
class GateRecord:
    schema_version: int
    gate: str
    context: RunContext
    input_hashes: dict[str, str]
    thresholds: JsonObject
    observed: JsonObject
    decision: Decision
    reason: str

    def to_dict(self) -> JsonObject:
        self.validate()
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.gate not in GATES:
            raise ValueError("unsupported gate record schema or gate")
        if self.decision not in DECISIONS or not isinstance(self.reason, str):
            raise ValueError("invalid gate decision or reason")
        if not isinstance(self.input_hashes, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.input_hashes.items()
        ):
            raise ValueError("invalid input hashes")
        if not isinstance(self.thresholds, dict) or not isinstance(self.observed, dict):
            raise ValueError("invalid threshold or observation")

    @classmethod
    def from_dict(cls, value: JsonObject) -> GateRecord:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid gate record fields")
        record = cls(**{**value, "context": RunContext.from_dict(value["context"])})
        record.validate()
        return record


def resolve_under_root(root: Path, value: str) -> Path:
    """Resolve a repository-relative path and reject traversal or link escapes."""
    requested = Path(value)
    if requested.is_absolute() or not value or ".." in requested.parts:
        raise ValueError("path must be repository-relative without parent traversal")
    resolved_root = root.resolve()
    resolved = (resolved_root / requested).resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("path escapes repository")
    return resolved


def write_json_atomic(path: Path, record: JsonObject) -> None:
    """Publish a complete UTF-8 JSON record in one filesystem rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def read_gate(path: Path) -> GateRecord:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return GateRecord.from_dict(data)


def publish_gate(gates_dir: Path, record: GateRecord) -> Path:
    """Retain immutable run evidence and update the stable latest pointer."""
    encoded = record.to_dict()
    run_dir = gates_dir / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    run_path = run_dir / f"{record.gate}.json"
    write_json_atomic(run_path, encoded)
    write_json_atomic(gates_dir / f"{record.gate}.json", encoded)
    return run_path


def require_gate(path: Path, gate: str, context: RunContext) -> GateRecord:
    record = read_gate(path)
    if record.gate != gate or record.context != context or record.decision != "pass":
        raise ValueError("prerequisite gate does not match or pass")
    return record
