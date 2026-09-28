"""Machine budget and G0 probe; later tasks add benchmark measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib import metadata
import json
import mmap
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

import psutil

from .records import (GateRecord, JsonObject, RunContext, resolve_under_root,
                      validate_measurement_row, write_json_atomic)


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


@dataclass(frozen=True)
class CellSpec:
    suite: str
    mode: str
    requested_model_tokens: int
    timeout_seconds: float = 120.0
    estimated_peak_bytes: int = 0
    cache_limit_bytes: int = 256 * MIB
    trial: int = 0
    trace: bool = False
    persist: bool = False
    extras: JsonObject = field(default_factory=dict)

    def validate(self) -> None:
        if (self.suite not in {"_test", "kernel"} or not self.mode
                or self.requested_model_tokens < 0 or self.timeout_seconds <= 0
                or self.timeout_seconds > 120 or self.estimated_peak_bytes < 0
                or self.cache_limit_bytes < 0 or self.cache_limit_bytes > 256 * MIB
                or self.trial < 0 or not isinstance(self.extras, dict)):
            raise ValueError("invalid benchmark cell")
        if self.suite == "_test" and self.mode not in {"sleep", "touch"}:
            raise ValueError("unknown test cell mode")


def time_evaluated(step, synchronize, *, evaluate=None) -> int:
    """Include new work, lazy evaluation, and device synchronization in one span."""
    if evaluate is None:
        import mlx.core as mx

        def evaluate(value):
            mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    start = time.perf_counter_ns()
    result = step()
    evaluate(result)
    synchronize()
    return time.perf_counter_ns() - start


def _footprint_binary() -> Path | None:
    if sys.platform != "darwin":
        return None
    root = Path(__file__).resolve().parents[2]
    source = root / "tools/footprint.c"
    binary = root / "tools/footprint"
    if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        completed = subprocess.run(["xcrun", "clang", "-O2", "-Wall", "-Wextra",
                                    str(source), "-o", str(binary)],
                                   capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return None
    return binary


def read_footprint(pid: int) -> int | None:
    """Read ri_phys_footprint using the installed macOS SDK ABI."""
    if pid <= 0:
        return None
    helper = _footprint_binary()
    if helper is None:
        return None
    completed = subprocess.run([str(helper), str(pid)], capture_output=True,
                               text=True, check=False, timeout=2)
    if completed.returncode != 0:
        return None
    try:
        value = int(completed.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def sample_memory(pid: int, phase: str) -> JsonObject:
    """Keep RSS and physical footprint separate; neither includes MLX sums."""
    try:
        rss = int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        rss = None
    try:
        footprint = read_footprint(pid)
    except (OSError, subprocess.TimeoutExpired):
        footprint = None
    return {"timestamp_ns": time.perf_counter_ns(), "phase": phase,
            "rss_bytes": rss, "physical_footprint_bytes": footprint,
            "physical_probe_available": footprint is not None,
            "physical_probe_reason": (None if footprint is not None else
                                      "helper unavailable, denied, or process exited")}


def sample_mlx_memory() -> JsonObject:
    import mlx.core as mx

    return {"active_bytes": int(mx.get_active_memory()),
            "peak_active_bytes": int(mx.get_peak_memory()),
            "cache_bytes": int(mx.get_cache_memory())}


def physical_claim_verified(sample: JsonObject) -> bool:
    value = sample.get("physical_footprint_bytes")
    return isinstance(value, int) and value > 0


def _swap_pressure() -> JsonObject:
    virtual = psutil.virtual_memory()
    try:
        swap_used = int(psutil.swap_memory().used)
    except OSError:
        swap_used = None
    return {"available_bytes": int(virtual.available),
            "memory_percent": float(virtual.percent), "swap_used_bytes": swap_used}


def _read_events(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def run_cell(spec: CellSpec) -> JsonObject:
    """Run one bounded cell in a fresh process and preserve censored outcomes."""
    spec.validate()
    root = Path(__file__).resolve().parents[2]
    run_id = uuid4().hex
    memory = psutil.virtual_memory()
    budget = compute_budget(memory.total, memory.available)
    spec_hash = hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps({"cache_limit_bytes": spec.cache_limit_bytes,
                                            "trace": spec.trace, "extras": spec.extras},
                                           sort_keys=True).encode()).hexdigest()
    base = {"run_id": run_id, "suite": spec.suite, "mode": spec.mode,
            "trial": spec.trial, "requested_model_tokens": spec.requested_model_tokens,
            "cell_hash": spec_hash, "implementation_hash": _hash_file(Path(__file__)),
            "config_hash": config_hash, "budget_bytes": budget.process_limit_bytes,
            "cache_limit_bytes": spec.cache_limit_bytes, "trace": spec.trace}
    if spec.estimated_peak_bytes > budget.process_limit_bytes:
        row = {**base, "status": "budget_blocked", "pid": None,
               "completion_ns": None, "completed_model_tokens": 0,
               "latency_ns": [], "memory_samples": [], "child_reaped": True,
               "reason": "estimated peak exceeds freshly checked research budget"}
        validate_measurement_row(row)
        return row
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cell-", dir=results_dir) as temporary:
        cell_dir = Path(temporary)
        cell_path = cell_dir / "cell.json"
        write_json_atomic(cell_path, asdict(spec))
        output_path = cell_dir / "worker.jsonl"
        error_path = cell_dir / "worker.err"
        pressure_before = _swap_pressure()
        launched = time.perf_counter_ns()
        with output_path.open("wb") as output, error_path.open("wb") as error:
            child = subprocess.Popen(
                [sys.executable, "-m", "familytiles.measure", "--cell",
                 str(cell_path.relative_to(root))], cwd=root, stdout=output,
                stderr=error, close_fds=True)
            samples = []
            deadline = time.monotonic() + spec.timeout_seconds
            timed_out = False
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    child.kill()
                    timed_out = True
                    break
                if spec.trace:
                    samples.append(sample_memory(child.pid, "running"))
                time.sleep(0.05 if not spec.trace else 0.10)
            exit_code = child.wait()
        elapsed = time.perf_counter_ns() - launched
        events = _read_events(output_path)
        completed = [event for event in events if event.get("event") == "complete"]
        progress = [event for event in events if event.get("event") == "progress"]
        final = completed[-1] if completed and not timed_out and exit_code == 0 else None
        if timed_out:
            status = "timeout"
        elif final is not None:
            status = "complete"
        else:
            status = "error"
        row = {**base, "status": status, "pid": child.pid, "exit_code": exit_code,
               "child_reaped": True, "wall_ns": elapsed,
               "completion_ns": final.get("completion_ns") if final else None,
               "completed_model_tokens": (final.get("completed_model_tokens") if final else
                                          max((int(e.get("completed_model_tokens", 0))
                                               for e in progress), default=0)),
               "latency_ns": final.get("latency_ns", []) if final else [],
               "memory_samples": samples, "worker_events": events,
               "worker_stderr": error_path.read_text(encoding="utf-8")[-4096:],
               "pressure_before": pressure_before,
               "pressure_after": _swap_pressure(),
               "mlx_counters": final.get("mlx_counters") if final else None}
    validate_measurement_row(row)
    if spec.persist:
        write_json_atomic(root / "results/cells" / f"{run_id}.json", row)
    return row


def summarize_trials(rows: list[JsonObject]) -> JsonObject:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return {"complete_trials": 0, "median_completion_ns": None,
                "median_trial_p95_latency_ns": None}
    completions = [int(row["completion_ns"]) for row in complete]
    p95s = []
    for row in complete:
        timings = sorted(int(value) for value in row.get("latency_ns", []))
        if timings:
            p95s.append(timings[min(len(timings) - 1, int(0.95 * len(timings)))])
    return {"complete_trials": len(complete),
            "median_completion_ns": statistics.median(completions),
            "median_trial_p95_latency_ns": statistics.median(p95s) if p95s else None}


def _worker_emit(value: JsonObject) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def _worker_test(spec: CellSpec) -> None:
    if spec.mode == "sleep":
        delay = float(spec.extras.get("delay_seconds", 0))
        if delay < 0 or delay > 10:
            raise ValueError("invalid test delay")
        start = time.perf_counter_ns()
        latencies = []
        for index in range(spec.requested_model_tokens):
            step_start = time.perf_counter_ns()
            time.sleep(delay)
            latencies.append(time.perf_counter_ns() - step_start)
            _worker_emit({"event": "progress", "completed_model_tokens": index + 1})
        _worker_emit({"event": "complete", "completed_model_tokens": len(latencies),
                      "completion_ns": time.perf_counter_ns() - start,
                      "latency_ns": latencies, "mlx_counters": None})
    elif spec.mode == "touch":
        size = int(spec.extras.get("bytes", 0))
        if size <= 0 or size > 128 * MIB:
            raise ValueError("invalid touch allocation")
        time.sleep(float(spec.extras.get("prehold_seconds", 0)))
        data = mmap.mmap(-1, size)
        for offset in range(0, size, 4096):
            data[offset] = 1
        _worker_emit({"event": "progress", "completed_model_tokens": 0,
                      "touched_bytes": size, "timestamp_ns": time.perf_counter_ns(),
                      "mlx_counters": None})
        time.sleep(float(spec.extras.get("hold_seconds", 0.3)))
        data.close()
        _worker_emit({"event": "released", "timestamp_ns": time.perf_counter_ns()})
        time.sleep(float(spec.extras.get("posthold_seconds", 0)))
        _worker_emit({"event": "complete", "completed_model_tokens": 0,
                      "completion_ns": 1, "latency_ns": [], "mlx_counters": None})


def _worker_main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    value = json.loads(resolve_under_root(root, args.cell).read_text(encoding="utf-8"))
    spec = CellSpec(**value)
    spec.validate()
    memory = psutil.virtual_memory()
    budget = compute_budget(memory.total, memory.available)
    if spec.estimated_peak_bytes > budget.process_limit_bytes:
        raise RuntimeError("research budget fell below cell estimate before worker setup")
    if spec.suite == "_test":
        _worker_test(spec)
        return 0
    import mlx.core as mx
    mx.set_cache_limit(min(spec.cache_limit_bytes, budget.cache_limit_bytes))
    mx.set_memory_limit(budget.process_limit_bytes)
    raise ValueError("unimplemented worker suite/mode")


if __name__ == "__main__":
    raise SystemExit(_worker_main())
