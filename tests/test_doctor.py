import subprocess
import sys

from familytiles.measure import compute_budget, evaluate_g0


def test_budget_uses_minimum_and_keeps_cache_small():
    gib = 2**30
    assert compute_budget(64 * gib, 10 * gib).process_limit_bytes == 6 * gib
    assert compute_budget(64 * gib, 64 * gib).process_limit_bytes == 12 * gib
    assert compute_budget(2 * gib, 1 * gib).process_limit_bytes == 644245094
    assert compute_budget(64 * gib, 10 * gib).cache_limit_bytes <= 256 * 2**20


def test_doctor_does_not_pass_without_audit():
    environment = {
        "architecture": "arm64", "metal_available": True,
        "uint16_probe": True, "bf16_matmul": True,
        "dependency_lock_hash": "abc", "process_limit_bytes": 2**30,
    }
    incomplete_audit = {"collision_decision": "unresolved", "inspected_sources": []}
    assert evaluate_g0(environment, incomplete_audit).decision != "pass"


def test_doctor_rejects_missing_gpu_even_with_clear_audit():
    environment = {
        "architecture": "arm64", "metal_available": False,
        "uint16_probe": False, "bf16_matmul": False,
        "dependency_lock_hash": "abc", "process_limit_bytes": 2**30,
    }
    audit = {
        "collision_decision": "none_found",
        "inspected_sources": ["P01", "P02", "P04", "P08", "P09", "P12"],
    }
    assert evaluate_g0(environment, audit).decision == "blocked"


def test_cpu_import_never_initializes_mlx():
    script = "import familytiles.records, familytiles.measure, familytiles.report, sys; print('mlx.core' in sys.modules)"
    run = subprocess.run([sys.executable, "-c", script], check=True,
                         capture_output=True, text=True)
    assert run.stdout.strip() == "False"
