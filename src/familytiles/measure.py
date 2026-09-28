"""Machine budget and G0 probe; later tasks add benchmark measurement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import psutil

from .records import GateRecord, JsonObject, RunContext, write_json_atomic


MIB = 2**20
GIB = 2**30
AUDIT_SOURCES = frozenset({"P01", "P02", "P04", "P08", "P09", "P12"})
PACKAGE_NAMES = ("numpy", "zstandard", "huggingface_hub", "safetensors",
                 "requests", "pytest", "psutil", "mlx", "mlx-lm")


@dataclass(frozen=True)
class MemoryBudget:
    physical_bytes: int
    available_bytes: int
    process_limit_bytes: int
    cache_limit_bytes: int


def compute_budget(physical_bytes: int, available_bytes: int) -> MemoryBudget:
    if physical_bytes <= 0 or available_bytes < 0:
        raise ValueError("invalid machine memory")
    limit = min(12 * GIB, int(0.40 * physical_bytes), int(0.60 * available_bytes))
    return MemoryBudget(physical_bytes, available_bytes, limit,
                        min(256 * MIB, limit // 4))


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _metal_probe(budget: MemoryBudget) -> JsonObject:
    """Run four-word integer reconstruction and a tiny stock BF16 matmul."""
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("MLX reports no Metal device")
    if budget.process_limit_bytes < 16 * MIB:
        raise RuntimeError("research budget below the 16 MiB G0 smoke minimum")
    mx.set_cache_limit(budget.cache_limit_bytes)
    mx.set_memory_limit(budget.process_limit_bytes)
    kernel = mx.fast.metal_kernel(
        name="familytiles_g0_xor_words", input_names=["anchor", "residual"],
        output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = anchor[i] ^ residual[i];",
        ensure_row_contiguous=False, compile_options={"math_mode": "safe"},
    )
    anchor = mx.array([0, 1, 65535, 32768], dtype=mx.uint16)
    residual = mx.array([0, 2, 65535, 32768], dtype=mx.uint16)
    decoded = kernel(
        inputs=[anchor, residual], template=[], grid=(4, 1, 1),
        threadgroup=(4, 1, 1), output_shapes=[(4,)], output_dtypes=[mx.uint16],
    )[0]
    mx.eval(decoded)
    words = decoded.tolist()
    if words != [0, 3, 0, 0]:
        raise RuntimeError(f"uint16 Metal mismatch: {words}")
    raw = mx.ones((2, 2), dtype=mx.bfloat16)
    product = mx.matmul(raw, raw)
    mx.eval(product)
    if product.tolist() != [[2.0, 2.0], [2.0, 2.0]]:
        raise RuntimeError("tiny BF16 stock matmul disagrees with expected result")
    return {
        "metal_available": True, "uint16_probe": True, "bf16_matmul": True,
        "device_info": mx.device_info(),
        "custom_kernel_api": {
            "constructor": "mlx.core.fast.metal_kernel",
            "ensure_row_contiguous": False,
            "compile_options": {"math_mode": "safe"},
            "output_dtype": "uint16", "tested_words": words,
        },
        "memory_api": {
            name: bool(hasattr(mx, name))
            for name in ("get_active_memory", "get_peak_memory",
                         "get_cache_memory", "set_memory_limit", "set_cache_limit")
        },
        "active_memory_after_probe": int(mx.get_active_memory()),
        "cache_memory_after_probe": int(mx.get_cache_memory()),
    }


def doctor(out: Path) -> JsonObject:
    available = psutil.virtual_memory()
    budget = compute_budget(available.total, available.available)
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    audit_path = root / "results/prior-art.json"
    audit_revisions = (
        json.loads(audit_path.read_text(encoding="utf-8")).get("platform_source_revisions", {})
        if audit_path.is_file() else {}
    )
    mlx_lm_qwen2 = Path(metadata.distribution("mlx-lm").locate_file("mlx_lm/models/qwen2.py"))
    record: JsonObject = {
        "schema_version": 1,
        "architecture": platform.machine(),
        "python": sys.version,
        "macos": platform.mac_ver()[0],
        "platform": platform.platform(),
        "physical_bytes": budget.physical_bytes,
        "available_bytes": budget.available_bytes,
        "process_limit_bytes": budget.process_limit_bytes,
        "cache_limit_bytes": budget.cache_limit_bytes,
        "package_versions": versions,
        "dependency_lock_hash": _hash_file(root / "results/requirements-lock.txt"),
        "platform_source_revisions": audit_revisions,
        "installed_mlx_lm_qwen2_sha256": _hash_file(mlx_lm_qwen2),
        "source_hashes": {
            path: _hash_file(root / path)
            for path in ("src/familytiles/measure.py", "src/familytiles/records.py")
        },
    }
    if record["architecture"] != "arm64" or sys.platform != "darwin":
        record.update(metal_available=False, uint16_probe=False, bf16_matmul=False,
                      smoke_error="native arm64 macOS required")
    else:
        try:
            record.update(_metal_probe(budget))
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            record.update(metal_available=False, uint16_probe=False, bf16_matmul=False,
                          smoke_error=f"{type(exc).__name__}: {exc}")
    write_json_atomic(out, record)
    return record


def evaluate_g0(environment: JsonObject, audit: JsonObject) -> GateRecord:
    audit_ids = set(audit.get("inspected_sources", []))
    clear_audit = (
        audit.get("collision_decision") in {"none_found", "reproduction"}
        and AUDIT_SOURCES <= audit_ids
        and isinstance(audit.get("source_revisions"), dict)
        and all(audit["source_revisions"].get(item) for item in AUDIT_SOURCES)
    )
    smoke = (
        environment.get("architecture") == "arm64"
        and environment.get("metal_available") is True
        and environment.get("uint16_probe") is True
        and environment.get("bf16_matmul") is True
        and bool(environment.get("dependency_lock_hash"))
        and environment.get("process_limit_bytes", 0) >= 16 * MIB
    )
    if audit.get("collision_decision") == "duplicate":
        decision, reason = "fail", "confirmed duplicate contribution; reclassify before continuation"
    elif not smoke:
        decision, reason = "blocked", "native Metal/dependency smoke did not pass"
    elif not clear_audit:
        decision, reason = "blocked", "closest-source collision audit is incomplete"
    else:
        decision, reason = "pass", "native smoke and scoped collision audit passed"
    root = Path(__file__).resolve().parents[2]
    code_hash = _hash_file(root / "src/familytiles/measure.py")
    env_hash = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    context = RunContext(None, {}, None, None, code_hash, env_hash, None)
    return GateRecord(1, "G0", context, {
        "dependency_lock": environment.get("dependency_lock_hash") or "",
        "prior_art": audit.get("audit_hash") or "",
    }, {"minimum_budget_bytes": 16 * MIB, "required_sources": sorted(AUDIT_SOURCES)},
        {"smoke": smoke, "audit_complete": clear_audit}, decision, reason)
