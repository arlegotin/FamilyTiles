# FamilyTiles Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the pinned exact family representation yields a useful live-inference memory/latency point, or publish a bounded no-go.

**Architecture:** Extend the existing measurement, model and exact-codec paths. Diagnose the original benchmark and numerical mismatch before adding one stock-operator staging route. Build resident B3/B4 controls and choose no more than two static whole-matrix policies only if the stage gates permit.

**Tech Stack:** Python 3.13, NumPy, MLX 0.32.2, MLX-LM 0.31.3, Zstandard, existing Metal kernels on the M3 Max.

**Spec:** [Continuation design](../specs/2026-09-28-familytiles-continuation-design.md), governed by the user's 28 September 2026 detailed continuation instruction and the [original design](../specs/2026-09-28-familytiles-design.md).

## Global constraints

- Preserve the two model revisions, original policy/artifact IDs, original failed G3 and all raw records.
- No new model download or reconversion while the artifact is valid; no more than two targeted kernel changes or two candidate policies.
- Keep original C1/C2/C3 meanings and the 0.001 C3 threshold. New live-memory and latency criteria belong only to the continuation.
- Record every unavailable or censored measurement as unavailable; no swap-dependent capacity claims.
- Make a meaningful commit after each verified stage and publish an early-stop report when a gate terminates work.

## Review focus

- Historical native batching must compute different model weights and activations; Task 1 checks output equality against two native calls.
- A timing loop must create fresh work and evaluate both outputs; Task 1 checks an operation counter and output digests.
- Cache replay must clone logical state; Task 2 checks REF against itself and independent offsets.
- Lazy output graphs must not retain decoded matrices; Task 2 checks scratch high water over repeated cached steps.
- Native promotion must release its old stream or charge it; Task 4 checks ownership bytes against compacted buffers.

---

### Task 1: Historical G3 audit and controlled reproduction

**Files:** Create `tools/audit_g3.py`, `tests/test_continuation_audit.py`, `results/continuation/audit.json` and per-block rows; use `measure.py` helpers without changing the old G3 files.

**Interfaces:** `audit_historical(results: Path) -> dict`, `run_interleaved(artifact: Path, sessions: int = 3) -> dict`. Output records identify source/header hashes, benchmark-code hash, tensor list, shapes, visits, input fixtures, math options, cache/power/pressure, output validation and unavailable historical fields.

- [ ] Write failing tests for same-source record reconciliation, two-distinct-products native control, and fresh evaluated work.
- [ ] Run focused tests; confirm expected failures.
- [ ] Audit old rows and current dispatch; implement the small three-session interleaved reproduction or paired fresh-process fallback if co-residency breaks budget.
- [ ] Run focused and full tests; save `audit.json` and raw blocks. Apply the 10% stability rule; do not select the best run.
- [ ] Commit `research: audit and reproduce paired kernel timings`.

### Task 2: Five-path cached-decoding differential

**Files:** Extend `model.py` with a native paired-operator route and cached exact-decode/native route; create `tools/trace_decode.py`, `tests/test_continuation_model.py`, `results/continuation/numerical_trace.json`.

**Interfaces:** `step_pair(..., execution="native" | "raw_custom" | "decode_native" | "fused")` may use a small explicit policy object rather than this exact parameter shape. Reuse the existing Qwen traversal, `FamilyLinear.prefill` stock arithmetic, and independent `RequestState` caches.

- [ ] Write failing tests for native lockstep versus separate stock, cached decode/native versus stock, separate KV, survivor state, and bounded decoded-matrix lifetime.
- [ ] Run focused tests red; implement the minimum native and staged routes, retaining the raw path as control.
- [ ] Run REF self-replay, five paths for the saved first step, and eight teacher-forced cached steps; trace earliest divergence with identical reference activations. Record bias/reduction findings and exact C1/C2/C3 scopes.
- [ ] If decode/native fails, reduce to one failing matrix and stop. Otherwise verify full tests and commit `feat: add exact decode with native cached operators`.

### Task 3: Representative cost profile and resident exact baselines

**Files:** Create `tools/profile_execution.py`, add thin decoding helpers in `baselines.py` only as needed, tests in `test_baselines.py`, and `results/continuation/operator_profile.csv` plus `baseline_runtime.json`.

**Interfaces:** Per-pair records include shape, visits, canonical/dependency bytes, native/raw/fused/staged combined time, scratch/copy bytes, format mode counts, source/fixture hashes. B3/B4 use the existing `ExactChunks`/`ExactFamilyBacking` transforms with resident encoded arrays.

- [ ] Test exact bounded reconstruction and resident buffer ownership; watch red.
- [ ] Implement thin B3/B4 matrix providers; verify decoded words against original tensors before timing. Avoid changing original encoded-size records.
- [ ] Time representative early/middle/late shapes and one whole sweep where promising, recording combined decode/native times. Compute required G3 cut and optimistic eligible-policy bound.
- [ ] If all valid execution routes are dominated or unsafe, publish early no-go. Otherwise run full tests and commit `research: profile exact inference alternatives`.

### Task 4: Static whole-matrix execution policy

**Files:** Create `src/familytiles/policy.py`, `tools/select_execution.py`, tests, at most two manifests under `results/continuation/`.

**Interfaces:** A fixed manifest maps each corresponding matrix pair to native, staged or eligible fused. It binds parent artifact, policy hash, source, profile prefix, dependency bytes, compact arena, shared scratch bound and projected savings. A comparator B3/B4 policy gets the same admission rule.

- [ ] Write failing ownership and policy-frozen tests, including a retained-stream case that must still be charged.
- [ ] Implement measured time-recovered-per-net-byte selection; produce at most a 15% and a 10% candidate, plus one fair comparator policy.
- [ ] Measure mixed sweep; allow one recorded correction for non-additive effects, then freeze. Verify the selected manifest and full tests; commit `feat: select bounded exact execution policy`.

### Task 5: Selected candidate admission, final evidence and report

**Files:** Extend `evaluate.py`/runner only as needed; create `results/continuation/{correctness.json,runtime.csv,CONTINUATION_RESULTS.md}` and report tests. If Task 2 or 3 stops early, create only the evidence reached and an explicit no-go report.

**Interfaces:** 128/16 two-request admission probe precedes one frozen 2,048-prediction C3 run and three repeated bounded workloads. Process records include loaded/warm/prefill/decode/teardown footprint samples, MLX counters, swap/pressure deltas, source/policy/fixture IDs, per-request p95 and total wall time.

- [ ] Write failing tests for verdict thresholds, unavailable peak, dominated B3/B4 and censored cells.
- [ ] Run admission probe and stop on numerical failure or dominated memory/latency point.
- [ ] Only for a survivor, run saved full C3 and generation checks, then three matched final workloads within the stated budgets.
- [ ] Generate the four-question verdict, verify all implemented tests and reproduction commands, commit `docs: report FamilyTiles continuation verdict`.

## Self-review

The five tasks preserve original records and cover the continuation stages A–E. Interfaces share `FamilyArtifact`, `RequestState`, `ExactChunks`, and `GateRecord`; no new format or source model is introduced. Every stage has a failing test before implementation, a bounded measurement, a commit, and an early-stop path. The final report is required even when an earlier stage fails.
