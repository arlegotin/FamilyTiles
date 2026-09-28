# FamilyTiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible evidence for or against exact anchor-coalesced inference on a real Qwen2 family on one M3 Max, stopping at the first justified gate failure.

**Architecture:** A small CLI coordinates immutable records, a CPU codec/survey/converter, validated Metal operands, and a minimal Qwen2 runner. Shared materialization and measurement code supports exact baselines and controls. Fresh benchmark subprocesses enforce ownership and distinguish canonical storage from physical footprint.

**Tech Stack:** Native arm64 Python, NumPy, Zstandard, Hugging Face Hub, safetensors, requests, pytest, psutil, MLX, MLX-LM, small Metal source, optional SDK C helper; dataset reader added at E3.

**Spec:** [FamilyTiles research specification](../specs/2026-09-28-familytiles-design.md). Read it before executing any task; its contracts and source catalog travel with this plan.

**Planning status:** No implementation, package installation, model acquisition, gate, or benchmark has been performed. This plan does not grant a positive research result. Work in the existing repository on the current `main` branch as requested; do not create a worktree, branch, or nested repository. Plan execution is a later task, with its execution method chosen at that handoff.

## Global Constraints

- Process budget: `B = min(12 GiB, 0.40 × installed physical memory, 0.60 × currently available memory)`; recheck before conversion/benchmarks.
- G1 sample cap: `32 MiB` per checkpoint; `160 MiB` total target across two Qwen pairs; separate metadata/overhead accounting.
- New model-weight acquisition: at most `7 GiB` across the implementation run, including retries/fallback; never automatically download both families.
- Seed `20260928`; disjoint `70%` development / `30%` holdout; block sizes `{64,128,256}` only in G1; widths `{0,2,4,8}`.
- Retain the second residual transform only if it earns at least `2%` of raw family active-weight bytes on development data.
- G1/G2: `(W1-WF)/W1 >= 0.15` and `(W2-WF)/W0 >= 0.05`, including anchor, native leftovers, metadata, aliases and alignment.
- G3: paired representative sweep time ≤ `1.5×` best native pair baseline; at most two layout changes after the first correct layout.
- C1/C2 have zero bit mismatches; C3 raw-stock logit normalized RMS ≤ `1e-3` and argmax agreement ≥ `99%`; no threshold widening.
- Memory cache cap initially `256 MiB`; cache-disabled diagnostic is separate. No wired/sysctl changes, forced swap or OOM.
- Benchmark cell cap `120 seconds`; three independent alternating-order trials; fixed teacher-forced work, new evaluated operations and synchronized boundaries.
- At least `20` request lifecycles; released live-memory allowance `max(16 MiB, 1% of canonical weights)`; investigate monotonic growth.
- Full G5: footprint and full active-weight allocation ≥ `15%` below B1; beyond B2 ≥ `5%` W0; throughput ≥ `80%` B1; per-stream p95 ≤ `1.25×` B1; throughput ≥ `1.10×` A2.
- Practical utility: ≥ `10%` lower memory with latency ≤ `1.10×`, or ≥ `20%` lower latency with memory ≤ `1.10×`, on predeclared applicable workloads.
- Little-endian original BF16 words, one served anchor, depth-one patches, separate request state; no training, PyTorch, new architecture, serving framework, GPU entropy decoder or alternate dtype conversion surface.
- No full decoded target in fused steady state, no `base_matmul + delta_matmul`, no per-token anchor copy, no unchecked `strict=False`, no remote code or pickle.
- Every research result has exact revision/policy/artifact/code/input identity. Skipped integration tests do not satisfy gates; acquisition failures/timeouts are distinct from measured failures.

## Review Focus

These implied failure cases receive explicit tests in the owning tasks below:

1. Interrupted/restarted acquisition or conversion must not reset download budgets or expose half-written artifacts — Tasks 5 and 8.
2. Symlink/path escapes and an alias to a same-shaped wrong tensor must be rejected before any GPU allocation — Tasks 1 and 8.
3. Reusing saved evidence after changing a model, kernel, policy or token fixture must fail prerequisite validation — Tasks 1, 13 and 23.
4. Switching from paired to single execution with unequal cache positions or cancellation must preserve the survivor's state — Tasks 15 and 17.
5. Timeout, missing physical probe or skipped tests must never become zero latency, infinite speedup or a positive report — Tasks 12, 19 and 23.

## Execution map and stop rules

| Tasks | Deliverable | Gate / stop |
| --- | --- | --- |
| 1–2 | Record/report spine; environment and collision audit | G0; stop unsupported platform or reclassify confirmed novelty collision |
| 3–7 | CPU format, bounded acquisition, exact comparison codecs, frozen real-data survey | G1; report both prescribed failures and stop |
| 8 | Complete selected-family conversion/CPU audit | G2; no runtime work if actual ratios or memory fail |
| 9–13 | GPU words, single/pair controls, memory/timing instrumentation, real sweep | G3; bounded tuning only |
| 14–17 | Strict single/lockstep model, corpus controls, independent state/lifetime | G4; narrow if integration fails |
| 18–19 | Core competitors and paired workload | Core criteria; stop expansion if core disproves required usefulness |
| 20–22 | Switching/cache, quantization/long context/generation, stress | Remaining G5 evidence if justified |
| 23 | Final decision, report, reproduction review | Always execute at the scope reached |

After any terminal gate failure, run the already available report path from Task 1, then the applicable documentation/reproduction checks of Task 23. Tasks downstream of the failed gate stay unchecked and `not_run`; do not build their code solely to complete the checklist. If one family's acquisition blocks, preserve that state while screening the other within budget. A selected G2 failure may use an already-frozen passing fallback only if the remaining aggregate budget allows it; never resample to tune away a failure.

Task steps specify a red/green cycle for functional implementation. A research measurement is not expected to pass; its success is an honest saved decision. Expected failures in tests mean the new contract is not implemented, not a broken environment. Record one meaningful commit per task (split a task further if a real independent deliverable emerges); stage named source and compact evidence files, never artifacts/weights. No remote push is required.

Representative assertion snippets use the fixtures described in their owning test steps; create those fixtures in that task. They pin the intended result without prescribing whole implementations. Verify relevant producer/dependency hashes when consuming evidence; keep the repository commit as separate provenance so documentation/new downstream code does not invalidate unchanged earlier gates.

## File map

| Create/extend during implementation | Ownership |
| --- | --- |
| `pyproject.toml`, `.gitignore`, `src/familytiles/__init__.py`, `cli.py` | Editable package, platform markers, explicit CLI surface |
| `src/familytiles/records.py`, `report.py` | Gate identities, JSON/CSV contracts, safe atomic paths, early/final report |
| `src/familytiles/data.py`, `survey.py` | Revision pinning, bounded network, sample/split/policy selection |
| `src/familytiles/codec.py`, `convert.py` | Pure integer format, validation, streaming complete artifact |
| `src/familytiles/baselines.py` | B3/B4 CPU codecs, materialization providers, B5/B6 integration |
| `src/familytiles/metal.py` | Metal accessors, raw/single/pair kernels, validated wrappers |
| `src/familytiles/model.py`, `evaluate.py` | Strict MLX model loading, Qwen2 lockstep, regression fixtures |
| `src/familytiles/measure.py`, `tools/footprint.c` | Subprocess supervisor, timing and memory evidence |
| `tests/test_records.py`, `test_doctor.py`, `test_data.py`, `test_survey.py` | Gate/environment/acquisition/survey unit tests |
| `tests/test_codec.py`, `test_baselines.py`, `test_convert.py` | Exactness, malformed inputs, streamed artifact tests |
| `tests/test_metal.py`, `test_measure.py`, `test_model.py`, `test_evaluate.py`, `test_report.py` | Runtime controls, methodology, regression and claim validation |
| `tests/conftest.py`, `tests/fixtures/` | Tiny generated fixtures and explicit `metal` / `model` markers; no real weights |
| `experiments/families.json`, `prompts.json` | Prespecified family definitions and the spec's eight verbatim prompts |
| `prior_art.md`, `results/`, root `RESULTS.md` | Inspected-source table, compact evidence and generated report |

CPU code imports no MLX at module load. Keep source in these modules while responsibilities remain understandable; no framework, database, task board or workflow engine. Tests may use stdlib mocks/local HTTP server; adding a test-only dependency requires a concrete need.

## Shared interfaces

Use `JsonObject = dict[str, Any]` only for versioned, validated serialized records; internal word operations use typed NumPy arrays. All paths are `pathlib.Path`; all byte sizes/offsets are Python integers until bounds-checked serialization. `ModelId = Literal['anchor', 'target']` and `Decision = Literal['pass','fail','blocked','not_run']` live in `records.py`.

The following type owners prevent later tasks from inventing incompatible interfaces:

| Owner task/module | Type and required fields |
| --- | --- |
| 1 / `records` | `RunContext(family_id, revisions, policy_hash, artifact_hash, implementation_hash, environment_hash, input_hash)`; unknown pre-stage fields are explicit nulls |
| 1 / `records` | `GateRecord(schema_version, gate, context, input_hashes, thresholds, observed, decision, reason)` |
| 2 / `measure` | `MemoryBudget(physical_bytes, available_bytes, process_limit_bytes, cache_limit_bytes)` |
| 3 / `codec` | `CodecPolicy(version, block_values, transforms)`; transform names `xor` / `ordered_delta` |
| 4 / `codec` | `EncodedTensor(kind, shape, policy, descriptors, payload, native_words)`; kind `native` / `packed`; empty streams for native; aliases are container-level |
| 5 / `data` | `TensorInfo(name, shape, dtype, shard, data_start, byte_offset, byte_length)`; byte_offset is relative to data_start, network ranges are absolute; `PinnedModel(repo, revision, files, tensors, metadata)` |
| 5 / `data` | `TransferBudget(sample_limit_bytes, weight_limit_bytes, ledger_path)` with atomic persisted accounting, metadata counters and reservation/reconciliation |
| 6 / `baselines` | `ExactChunks(shape, word_dtype, transform, chunk_index, payload, native_fallback)` and `ExactFamilyBacking(anchor, target, family_transform)` |
| 7 / `survey` | `SampleWindow(model_id, tensor_name, row, start_column, word_count, stratum, split)`; `SamplePlan(seed, windows, split_hash)` |
| 8 / `convert` | `FamilyArtifact(root, manifest, manifest_hash)`; only construct through full structural/identity validation |
| 9 / `metal` | `DeviceOperand(kind, shape, policy, anchor_words, descriptors, payload, native_words, storage_ids)`; alias resolves to existing storage |
| 12 / `measure` | `CellSpec(suite, mode, workload, trial, artifact_path, input_path, context, cache_limit_bytes, timeout_s)` |
| 14 / `model` | `RequestState(model_id, cache, position, token_ids, finished)`; `RuntimeFamily(models, operands, mode, context)` owns shared immutable weights and no global request KV |
| 16 / `evaluate` | `TokenFixture(dataset_revision, tokenizer_hash, input_ids, continuation_ids, source_hashes, fixture_hash)` |

Optional constructor fields may have empty defaults, but never implicit revision/policy substitutions. Runtime interfaces use `mx.array` only inside MLX modules and accept the exact validated types above. Statistical/evidence interfaces return validated `JsonObject` records whose fields are specified by the spec.

---

## G0: environment and evidence spine

### Task 1: Establish records, CLI prerequisites, and early-stop reporting

**Files:** Create `pyproject.toml`, `.gitignore`, `src/familytiles/{__init__,cli,records,report}.py`, `tests/{test_records,test_report}.py`, `results/.gitkeep`; update `README.md`.

**Interfaces:** Produce `RunContext`, `GateRecord`, `ModelId`, `Decision`; `resolve_under_root(root: Path, value: str) -> Path`, `write_json_atomic(path: Path, record: JsonObject) -> None`, `require_gate(path: Path, gate: str, context: RunContext) -> GateRecord`, `render_report(results: Path, out: Path) -> None`, `main(argv: list[str] | None = None) -> int`.

- [ ] Prepare the test environment with `python3 -m venv .venv`, `source .venv/bin/activate`, `python -m pip install --upgrade pip`, and `python -m pip install pytest`; verify `python -m pytest --version`. This makes the first red run a missing-contract failure rather than a missing-test-runner failure. Product dependencies are installed after the package metadata is written below.
- [ ] Write `test_gate_requires_matching_evidence`, asserting a saved pass is rejected for changed revision/policy/artifact and that missing/nonpass records cannot authorize work; `test_paths_reject_parent_absolute_and_symlink_escape`; `test_report_can_finish_after_g1_failure`, asserting G2–G5 show `not_run` and no positive claim or numeric throughput is invented.

```python
with pytest.raises(ValueError):
    require_gate(gate_path, 'G1', changed_policy_context)
render_report(early_failure_results, report_path)
assert 'not_run' in report_path.read_text()
```

- [ ] Run `python -m pytest tests/test_records.py tests/test_report.py -q`; confirm failure is missing implementation/imports for those contracts.
- [ ] Create the editable package and lazy CLI imports; define schema/version validation, evidence hashes, atomic record publication and documented exit codes `0/2/3/4/5`. Register only implemented actions; unimplemented commands fail explicitly. Add ignore rules for environments, model/sample bytes, artifacts, large caches and temporary build data. Implement the minimal record-based report now so every later gate can terminate honestly.
- [ ] Install the initial dependencies in `.venv` with MLX platform markers; run the same tests and `python -m familytiles.cli report --help`, expecting passed unit tests and documented arguments. The README labels future commands as unimplemented until their tasks land.
- [ ] Commit: `feat: add research records and early-stop reporting`.

### Task 2: Run the platform probe and source collision audit

**Files:** Create `src/familytiles/measure.py`, `tests/test_doctor.py`, `prior_art.md`; modify `cli.py`, `records.py`, `pyproject.toml`; save `results/{environment.json,requirements-lock.txt,prior-art.json}` and G0 record.

**Interfaces:** Consume Task 1 records; produce `MemoryBudget`, `compute_budget(physical_bytes: int, available_bytes: int) -> MemoryBudget`, `doctor(out: Path) -> JsonObject`, `evaluate_g0(environment: JsonObject, audit: JsonObject) -> GateRecord`.

- [ ] Write `test_budget_uses_minimum`, asserting 64 GiB physical/10 GiB available yields 6 GiB, and larger availability never exceeds 12 GiB; `test_doctor_does_not_pass_without_audit`, asserting hardware-only evidence is insufficient; `test_cpu_import_never_initializes_mlx`.

```python
assert compute_budget(64 * 2**30, 10 * 2**30).process_limit_bytes == 6 * 2**30
assert compute_budget(64 * 2**30, 64 * 2**30).process_limit_bytes == 12 * 2**30
assert evaluate_g0(smoke_pass_record, incomplete_audit).decision != 'pass'
```

- [ ] Run `python -m pytest tests/test_doctor.py -q`; confirm the new contracts fail before implementation.
- [ ] Implement architecture/Rosetta/Metal/device/API checks, the tiny uint16 reconstruction custom kernel and BF16 stock matmul, and lower MLX memory/cache limits within the budget. Record versions and the actual custom-kernel callable/contiguity options. Probe supported memory APIs without changing wired/OS limits.
- [ ] Inspect spec §15's six closest papers/code paths, recording immutable code SHA, paper version/hash, license, exact section/function and presence/absence/unknown for exact grouped independent-model execution with shared anchor loads. Check FM-Delta's primary BF16 Table 6. Source-only observations are not measured M3 results; missing code stays unknown. Pin the tested MLX/MLX-LM source and package versions, and write a scoped collision decision.
- [ ] Run `python -m pytest tests/test_doctor.py -q`, then `python -m familytiles.cli doctor --out results/environment.json` and `python -m pip freeze > results/requirements-lock.txt`. Expect unit tests pass; G0 passes only with real smoke evidence and completed scoped audit. On platform failure, render the early report. On confirmed novelty collision, document reproduction/extension scope before proceeding; do not preserve the original novelty assertion.
- [ ] Commit: `feat: probe M3 environment and record prior-art audit` (include failures honestly if G0 stops).

## G1: CPU proof and real-data eligibility

### Task 3: Implement reversible word transforms

**Files:** Create `src/familytiles/codec.py`, `tests/test_codec.py`.

**Interfaces:** Produce `CodecPolicy`; `ordered_key(words: np.ndarray) -> np.ndarray`, `inverse_key(keys: np.ndarray) -> np.ndarray`, `encode_residual(anchor: np.ndarray, target: np.ndarray, transform: str) -> np.ndarray`, `decode_residual(anchor: np.ndarray, residual: np.ndarray, transform: str) -> np.ndarray`. Words are little-endian u16; ordered residuals are u32.

- [ ] Write `test_all_words_key_bijection` with all 65,536 words; `test_extreme_delta_needs_17_bits` asserting `0xffff -> 0x7fff` encodes `131070` and reverse `131069`; `test_every_word_representative_anchors` using the seven anchors in spec §12; `test_million_seeded_pairs_roundtrip` with seed 20260928 for both transforms. Assert uint16 exact recovery, including ±0, NaN payloads, infinities and subnormals; malformed residuals with reconstructed key outside `[0,65535]` raise.

```python
words = np.arange(65536, dtype=np.uint16)
np.testing.assert_array_equal(inverse_key(ordered_key(words)), words)
a, t = np.array([0xffff], dtype='<u2'), np.array([0x7fff], dtype='<u2')
assert int(encode_residual(a, t, 'ordered_delta')[0]) == 131070
assert int(encode_residual(t, a, 'ordered_delta')[0]) == 131069
```

- [ ] Run `python -m pytest tests/test_codec.py -k 'key or delta or residual or pairs or anchors' -q`; expect unimplemented transform failures.
- [ ] Implement the spec §7 integer formulas with checked int32/u32 intermediates, no floating-point casts and no u16 truncation of zigzag. Validate policy version/block/menu.
- [ ] Run `python -m pytest tests/test_codec.py -q`; expect all transform cases pass bitwise without model downloads or MLX imports.
- [ ] Commit: `feat: add exact BF16 word transforms`.

### Task 4: Implement packed blocks, native fallback and malformed-stream rejection

**Files:** Modify `codec.py`, `tests/test_codec.py`.

**Interfaces:** Consume Task 3 transforms; produce `EncodedTensor`; `encode_tensor(anchor: np.ndarray, target: np.ndarray, policy: CodecPolicy, *, modes: tuple[str, ...] = ('copy','raw','packed')) -> EncodedTensor`, `validate_encoded(anchor: np.ndarray, encoded: EncodedTensor) -> None`, `decode_tensor(anchor: np.ndarray, encoded: EncodedTensor) -> np.ndarray`, `allocated_bytes(encoded: EncodedTensor) -> int` (array bytes; container metadata added in Task 8).

- [ ] Write parameterized `test_modes_widths_exceptions_and_tails` for B=64/128/256, b=0/2/4/8, exception counts 0/1/n and every lane position, dimensions 1/31/32/33/127/128/129 plus B±1 and zero-length codec inputs. Assert two-u32 descriptor fields exactly match spec §7, count 256 survives nine bits, rank crosses mask words correctly and 128/four-bit/two-exception cost is 92 bytes.
- [ ] Add `test_native_fallback_for_random_words` asserting no larger tile representation is chosen; `test_reject_corrupt_descriptors` covers invalid mode, reserved bits, disabled transforms, tail/mask/padding bits, placeholder bits, count/popcount mismatch, overlap, truncation and offset overflow without allocating 16 GiB. Write bounded deterministic mutation cases for each payload section.

```python
encoded = encode_tensor(anchor, target, CodecPolicy(1, 128, ('xor',)))
np.testing.assert_array_equal(decode_tensor(anchor, encoded), target)
assert allocated_bytes(four_bit_two_exception_tile) == 92
with pytest.raises(ValueError):
    validate_encoded(anchor, corrupted_reserved_bits)
```

- [ ] Run `python -m pytest tests/test_codec.py -q`; expect the new packing/validation cases fail while transforms remain green.
- [ ] Implement contiguous per-tensor descriptor/payload arrays, row-tail handling, literal exceptions, exact byte-cost mode selection and deterministic tie order. Validate before decoding; choose whole native when packed costs lose. Preserve support for modes=('copy','raw') for B2. Avoid per-tile objects in retained encoding.
- [ ] Run `python -m pytest tests/test_codec.py -q`; expect integer equality for every mode and rejection of every specified malformed stream. Save the executed E0 summary without representing it as real-family eligibility.
- [ ] Commit: `feat: add validated independently addressed patch blocks`.

### Task 5: Pin artifacts and enforce bounded acquisition

**Files:** Create `src/familytiles/data.py`, `tests/test_data.py`, `experiments/families.json`; modify `records.py` only for shared serialization needs.

**Interfaces:** Produce `TensorInfo`, `PinnedModel`, `TransferBudget`; `pin_model(repo: str, revision: str | None, budget: TransferBudget) -> PinnedModel`, `read_range(model: PinnedModel, shard: str, start: int, length: int, budget: TransferBudget) -> bytes`, `parse_header(header: bytes, file_size: int, *, shard: str) -> list[TensorInfo]` (header includes its eight-byte prefix), `download_selected(models: tuple[PinnedModel, PinnedModel], out: Path, budget: TransferBudget) -> dict[str, Path]`.

- [ ] Write a local HTTP-server fixture and `test_range_validation` for correct 206, wrong interval, compressed response, short body and ignored Range/200; assert hard read bounds even when advertised length is false. `test_header_limits_and_extents` checks eight-byte prefix, 16 MiB header cap, shape/dtype byte products, nonoverlap/overflow and duplicate names.
- [ ] Add `test_budget_persists_across_retry_and_restart`, asserting transferred/aborted/reserved bytes cannot be refunded into extra acquisition; `test_revision_is_immutable_across_slices`; `test_total_shard_sizes_block_download_before_io`. Fixtures use small numeric budgets while a separate assertion pins defaults to 32 MiB and 7 GiB.

```python
tensors = parse_header(prefix_and_header, file_size, shard='weights.safetensors')
assert tensors[0].data_start == len(prefix_and_header)
assert tensors[0].byte_offset == 0  # fixture starts at the data section
with pytest.raises(ValueError):
    parse_header(overlapping_header, file_size, shard='weights.safetensors')
```

- [ ] Run `python -m pytest tests/test_data.py -q`; expect missing bounded-reader and ledger contracts.
- [ ] Implement immutable Hub metadata resolution, streaming identity range reads, explicit caps, per-file/sample SHA-256, source/config/tokenizer/license provenance, validated cached-file reuse and atomic run-level ledger. Record metadata separately. Populate only the prescribed two Qwen pairs plus disabled optional SmolLM2; no full download in this task's verification.
- [ ] Run `python -m pytest tests/test_data.py -q`; expect all reader/budget cases pass offline. Inspect the family JSON against the exact six repository IDs in spec §6.
- [ ] Commit: `feat: add revision-pinned bounded model acquisition`.

### Task 6: Implement portable exact comparison codecs

**Files:** Create `src/familytiles/baselines.py`, `tests/test_baselines.py`; extend `tests/test_codec.py` only if shared word primitives are reused.

**Interfaces:** Produce `ExactChunks`, `ExactFamilyBacking`; `encode_exact(words: np.ndarray, transform: str, *, chunk_bytes: int = 262144, level: int = 3) -> ExactChunks`, `decode_exact(encoded: ExactChunks) -> Iterator[np.ndarray]`, `encode_family_exact(anchor: np.ndarray, target: np.ndarray, transform: str, anchor_form: str) -> ExactFamilyBacking`, `decode_family_exact(backing: ExactFamilyBacking) -> Iterator[tuple[np.ndarray, np.ndarray]]`, `exact_allocated_bytes(encoded: ExactChunks | ExactFamilyBacking) -> int`.

- [ ] Write `test_bf16_field_split_all_words` with the exact exponent/sign-fraction formulas in spec §10; `test_exact_chunks_roundtrip` for native/byte-plane/field, tensor tails and chunk boundaries; `test_b4_17bit_residual_roundtrip` for XOR u16 and ordered u32, each raw/compressed anchor form. Assert chunk index/headers/padding are charged and native fallback never expands the encoded operand unnecessarily.

```python
words = np.arange(65536, dtype='<u2')
encoded = encode_exact(words, 'field', chunk_bytes=262144, level=3)
recovered = np.concatenate(list(decode_exact(encoded)))
np.testing.assert_array_equal(recovered, words)
assert exact_allocated_bytes(encoded) >= len(encoded.payload)
```

- [ ] Run `python -m pytest tests/test_baselines.py -q`; expect missing codec interfaces.
- [ ] Implement level-3 Zstd and independently decodable approximately 256 KiB source-word chunks. Use byte planes over all four bytes of ordered residuals. Define stored indexes and field inverses explicitly; validate corrupt/truncated chunks. Keep compression/decode CPU-only and expose owned backing bytes for later RAM charging. No GPU entropy decoder or upstream CUDA port.
- [ ] Run `python -m pytest tests/test_baselines.py tests/test_codec.py -q`; expect bit-exact recovery for all transforms and no BF16 numerical conversion.
- [ ] Commit: `feat: add independent and family exact Zstd references`.

### Task 7: Freeze policies and run the conservative real-family screen

**Files:** Create `src/familytiles/survey.py`, `tests/test_survey.py`; modify `cli.py`; save `results/survey/` compact records and G1 decisions.

**Interfaces:** Consume Tasks 1/3–6; produce `SampleWindow`, `SamplePlan`; `make_sample_plan(models: tuple[PinnedModel, PinnedModel], seed: int, limit_bytes: int) -> SamplePlan`, `choose_policy(development: JsonObject) -> CodecPolicy`, `project_family(development: JsonObject, holdout: JsonObject, inventory: JsonObject, policy: CodecPolicy) -> JsonObject`, `survey(families: Path, sample_mib_per_model: int, seed: int, out: Path) -> JsonObject`.

- [ ] Write `test_split_precedes_selection`: parent windows are disjoint 70/30, all candidate subblocks stay in one split, and policy bytes are written/hashed before holdout access. `test_weighted_worse_stratum_projection` uses unequal tensor sizes and asserts the worse dev/holdout estimate per stratum, native unsampled cost and actual W0/W1/W2 denominators.
- [ ] Add `test_no_alias_from_samples`, checking conservative unresolved B1 alias bounds; `test_dual_transform_requires_two_percent_raw_family_gain`; `test_failure_removes_selection_pointer`; `test_controls_cannot_select_family`. Include exact 0.15/0.05 pass boundaries and just-below failures; acquisition failure remains blocked, never low compressibility.

```python
projection = project_family(dev, worse_holdout, inventory, frozen_policy)
assert projection['g1_pass'] is False  # fixture misses the 0.15 full-family threshold
assert set(dev_window_ids).isdisjoint(holdout_window_ids)
assert selection_pointer.exists() is False  # failed survey never publishes selection
```

- [ ] Run `python -m pytest tests/test_survey.py -q`; expect missing sampling/selection logic.
- [ ] Implement bounded role/layer/row sampling with common parent windows, all block-size candidates on development, family policies frozen before holdout, byte-weighted conservative estimates, B2/B3/B4 size accounting and required controls. Record native leftovers, largest residuals, exceptional block/mode frequencies and sample/window hashes. Do not form artificial Zstd chunks from disconnected ranges.
- [ ] Run `python -m pytest tests/test_survey.py tests/test_data.py -q`, then `python -m familytiles.cli survey --families experiments/families.json --sample-mib-per-model 32 --seed 20260928 --out results/survey`. Expected software tests pass; actual G1 may pass/fail/block. Select the smallest passing family with immutable revisions and policy pointers only after thresholds pass. Preserve both Qwen records; render and stop if neither qualifies.
- [ ] Commit: `feat: add frozen real-family compression survey` (include observed negative outcomes where applicable).

## G2: complete codec and memory audit

### Task 8: Stream the selected family into a verified artifact

**Files:** Create `src/familytiles/convert.py`, `tests/test_convert.py`; modify `cli.py`, `records.py`; save conversion records and G2.

**Interfaces:** Consume selected immutable family/policy and `EncodedTensor`; produce `FamilyArtifact`; `convert_family(family: Path, policy: Path, out: Path) -> FamilyArtifact`, `load_artifact(root: Path) -> FamilyArtifact`, `iter_tensor_words(artifact: FamilyArtifact, model_id: ModelId, name: str, chunk_rows: int) -> Iterator[np.ndarray]`, `verify_cpu(artifact: FamilyArtifact) -> JsonObject`, `canonical_ledger(artifact: FamilyArtifact) -> JsonObject`.

- [ ] Write tiny safetensors fixtures for a compatible family with projections, biases, norms, native head/embedding, a tied head and exact cross-model alias. `test_stream_conversion_accounts_every_active_tensor` asserts C1 hashes and W0/W1/W2/WF include native/alias/descriptor/padding/container costs and exclude unused checkpoint auxiliaries only with explicit records.
- [ ] Add `test_hash_lookup_still_compares_bytes` (force a lookup collision), `test_wrong_anchor_or_alias_coordinates_rejected`, `test_tied_head_conflict_is_error`, `test_interrupted_conversion_is_not_loadable`, `test_manifest_paths_do_not_escape` and truncated/mixed-revision/shape/version/reserved-bit mutation cases. Assert bounded chunk buffers and no two complete dictionaries during conversion using allocation spies on fixture inventories.

```python
artifact = convert_family(family_path, policy_path, output_path)
audit = verify_cpu(artifact)
assert audit['mismatched_words'] == 0
assert audit['verified_active_tensors'] == expected_active_tensor_count
with pytest.raises(ValueError):
    load_artifact(interrupted_output_path)
```

- [ ] Run `python -m pytest tests/test_convert.py -q`; expect missing converter/container contracts.
- [ ] Validate G1 and budget, inspect all selected shard sizes, then acquire only missing allowed bytes. Stream hashes/equality/encoding into a temporary manifest/array directory. Preserve shared storage IDs and publish only after complete CPU round trips. Save B2, B3, B4 actual size audits (including both B4 anchor forms); write their local backing artifacts for future providers. Charge actual serialized headers/metadata, close mappings, and plan runtime peaks with the largest staging/scratch/KV buffers.
- [ ] Run `python -m pytest tests/test_convert.py tests/test_codec.py tests/test_baselines.py -q`; after G1 pass run the spec's `convert` and `verify --device cpu` commands. Expected tests pass and every active tensor has exact hashes; G2 passes only if actual ratios and conversion/runtime budget hold. Save elapsed/peak conversion memory. Stop/narrow on failure; a frozen fallback is allowed only within the remaining 7 GiB ledger.
- [ ] Commit: `feat: stream and verify complete family artifacts`.

## G3: primitive viability

### Task 9: Validate device operands and reconstruct exact GPU words

**Files:** Create `src/familytiles/metal.py`, `tests/test_metal.py`, `tests/conftest.py`; modify `cli.py`, `convert.py` for bounded verification hooks.

**Interfaces:** Consume `FamilyArtifact`/`EncodedTensor`; produce `DeviceOperand`; `load_operand(artifact: FamilyArtifact, tensor_name: str) -> DeviceOperand`, `decode_words(operand: DeviceOperand, row_start: int = 0, row_count: int | None = None) -> mx.array`, `verify_gpu(artifact: FamilyArtifact) -> JsonObject`.

- [ ] Mark Metal tests explicitly; write `test_gpu_words_match_cpu_for_all_modes` with integer comparisons across every width/tail/mask boundary, 256 exceptions and exceptional BF16 payloads. `test_validation_precedes_gpu_launch` asserts corrupted manifests/descriptors cause zero launches. `test_anchor_view_has_no_full_copy` records unique allocations before/after uint16/BF16 views.

```python
decoded = decode_words(device_operand)
mx.eval(decoded)
np.testing.assert_array_equal(np.asarray(decoded), original_target_words)
assert allocation_probe.full_anchor_copy_bytes == 0
```

- [ ] Run `python -m pytest tests/test_metal.py -q`; on supported G0 hardware expect new implementation failures. On other hardware, skips are expected unit-suite behavior but cannot satisfy G3.
- [ ] Implement host validation plus the integer-only Metal accessor and bounded row-range decoder. Keep contiguous immutable descriptor/payload arrays and verified anchor/native views. Check row ranges and reject zero-sized GEMV independently of legal empty codec data. Record actual no-copy behavior and fail/rework any hidden full-anchor cast.
- [ ] Run `python -m pytest tests/test_metal.py -q` and `python -m familytiles.cli verify --artifact artifacts/family --device gpu`; require zero u16 mismatches on every encoded tensor in bounded chunks, each tied to its original target hash. Save raw comparison counts and allocation evidence, not only a Boolean.
- [ ] Commit: `feat: add exact Metal word reconstruction`.

### Task 10: Add same-schedule raw and fused single GEMV

**Files:** Modify `metal.py`, `tests/test_metal.py`.

**Interfaces:** Consume validated `DeviceOperand`; produce `raw_gemv(words: mx.array, x: mx.array, bias: mx.array | None = None) -> mx.array`, `family_gemv(operand: DeviceOperand, x: mx.array, bias: mx.array | None = None) -> mx.array`, `kernel_config() -> JsonObject`.

- [ ] Write `test_single_gemv_bitwise_raw_control` for finite BF16 activations/weights, all modes and irregular M/K tails; assert integer-view output equality, matched bias/cast and explicit rejection of zero/unsupported shapes/strides. `test_fused_single_allocates_no_matrix` asserts output/scratch-bound allocation and native target mode avoids anchor input use.

```python
packed, raw = family_gemv(operand, x, bias), raw_gemv(target_words, x, bias)
mx.eval(packed, raw)
np.testing.assert_array_equal(
    np.asarray(packed.view(mx.uint16)), np.asarray(raw.view(mx.uint16)))
```

- [ ] Run `python -m pytest tests/test_metal.py -k 'single or strides or zero' -q`; expect absent GEMV behavior.
- [ ] Generate raw and packed kernels from a shared arithmetic template, replacing only the accessor. Start FP32 accumulation and row-oriented fixed K traversal; retain identical reduction/compiler options and expansion `as_type<float>(uint(word) << 16)`. Use one bounded two-stage reduction only if required, charging partial sums in both controls. Record source/config hash and planned scratch size; no full decoded output buffer.
- [ ] Run the selected tests and compare raw against stock MLX with saved maximum absolute/normalized RMS errors. Require packed/raw finite-input bitwise equality; stock is a separately measured arithmetic reference. Do not change C2 to tolerance equality if it fails.
- [ ] Commit: `feat: fuse exact patch access into single GEMV`.

### Task 11: Reuse anchor words in paired GEMV

**Files:** Modify `metal.py`, `tests/test_metal.py`.

**Interfaces:** Consume Task 10 schedule; produce `raw_gemv_pair(anchor: mx.array, target: mx.array, x_anchor: mx.array, x_target: mx.array, biases: tuple[mx.array | None, mx.array | None] = (None, None)) -> tuple[mx.array, mx.array]`, `family_gemv_pair(operand: DeviceOperand, x_anchor: mx.array, x_target: mx.array, biases: tuple[mx.array | None, mx.array | None] = (None, None)) -> tuple[mx.array, mx.array]`.

- [ ] Write `test_pair_independent_inputs_and_biases`, asserting each packed output matches its raw-pair output bitwise with deliberately unrelated activation vectors and distinct biases. Include RAW/COPY/native fallback, tails, all exception modes, identical-input diagnostic and output alias rejection. `test_pair_has_no_decoded_matrix` checks bounded allocations.

```python
actual = family_gemv_pair(operand, x_anchor, x_target, biases)
expected = raw_gemv_pair(anchor_words, target_words, x_anchor, x_target, biases)
mx.eval(*actual, *expected)
for packed, raw in zip(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(packed.view(mx.uint16)), np.asarray(raw.view(mx.uint16)))
```

- [ ] Run `python -m pytest tests/test_metal.py -k pair -q`; expect unimplemented paired operations.
- [ ] Implement one anchor load in the shared accessor path feeding two independent products/accumulators, with literal target access for RAW. Implement matching raw pair and preserve compatibility of single/pair reduction schedules for survivor execution. Expose source/config for inspection and available Metal profiling; label inferred traffic as modeled.
- [ ] Run `python -m pytest tests/test_metal.py -q`; require C1/C2 all pass. Inspect generated source/available profiles for anchor reuse and no decoded writes; compare allocations against the scratch formula. Record two separate singles and raw pair as controls without making speed claims yet.
- [ ] Commit: `feat: add anchor-sharing paired GEMV controls`.

### Task 12: Make timing and unified-memory measurement trustworthy

**Files:** Modify `src/familytiles/measure.py`; create `tools/footprint.c`, `tests/test_measure.py`; modify `records.py` for measurement row validation.

**Interfaces:** Consume `MemoryBudget`/`RunContext`; produce `CellSpec`; `read_footprint(pid: int) -> int | None`, `sample_memory(pid: int, phase: str) -> JsonObject`, `run_cell(spec: CellSpec) -> JsonObject`, `time_evaluated(step: Callable[[], Any], synchronize: Callable[[], None]) -> int`, `summarize_trials(rows: list[JsonObject]) -> JsonObject`. Worker entry is `python -m familytiles.measure --cell <repo-relative-json>`; dispatch only explicitly implemented suites/modes.

- [ ] Write `test_timing_evaluates_new_work_before_stop` using an ordered fake evaluate/synchronize log; `test_timeout_is_censored_and_kills_child` with a small test-specific cap, asserting null completed latency and preserved partial count; `test_budget_rechecked_before_launch`; `test_counters_are_not_added`; `test_missing_footprint_cannot_pass_physical_claim`.

```python
row = run_cell(short_timeout_fixture)
assert row['status'] == 'timeout'
assert row['completion_ns'] is None
assert row['completed_model_tokens'] < requested_model_tokens
```

- [ ] Run `python -m pytest tests/test_measure.py -q`; expect missing supervisor/measurement contracts.
- [ ] Compile the optional helper with installed SDK headers, not hard-coded ABI offsets; check `proc_pid_rusage` return values. Implement fresh child lifecycle, memory/cache caps, pre/post pressure/swap records, run IDs, matched trace vs timing runs and cleanup on timeout. Store source/cell/config hashes and raw timing/trace records. Leave absent probes null with reasons; never backfill RSS as physical footprint.
- [ ] Run unit tests; manually validate the footprint helper against a child touching a known 64 MiB allocation and releasing it, recording before/during/after plus probe limitations. A probe that cannot reliably detect the allocation is unavailable for physical claims. Validate MLX active/peak/cache counters and cache-disabled diagnostic on small touched arrays. These checks do not claim model memory savings.
- [ ] Commit: `feat: add synchronized timing and process-footprint records`.

### Task 13: Run the representative kernel sweep and decide G3

**Files:** Modify `measure.py`, `cli.py`, `records.py`; extend `tests/test_measure.py`; save kernel correctness/timing/memory records and G3.

**Interfaces:** Consume Cells, artifact and GEMV operations; produce `build_kernel_cells(artifact: FamilyArtifact, inputs: Path) -> list[CellSpec]`, `evaluate_g3(rows: list[JsonObject], correctness: JsonObject) -> GateRecord`; add `bench --suite kernel`.

- [ ] Write `test_kernel_sweep_uses_summed_real_work` with a small/large role mix so an unweighted ratio would pass incorrectly; assert selection of the fastest legitimate native pair comparator and boundary `1.5`. `test_kernel_evidence_rejects_changed_source` and `test_layout_budget_is_two_after_first_correct` bind evidence to exact kernels and cap tuning.

```python
assert evaluate_g3(rows_at_1_5_ratio, c1_c2_pass).decision == 'pass'
assert evaluate_g3(rows_at_1_5001_ratio, c1_c2_pass).decision == 'fail'
assert evaluate_g3(fast_rows, c2_mismatch).decision == 'fail'
```

- [ ] Run `python -m pytest tests/test_measure.py -q`; expect new sweep/gate cases fail.
- [ ] Build the frozen early/middle/late, all-role inventory and input pool; capture a few native activations in an isolated untimed stock run with hashes, while retaining seeded random inputs. Prepare native batched operands once where compatible; otherwise use two native matvecs. Run raw single/pair, packed R=1/R=2 and two-single ablation in independent bounded processes, sweeping distinct real operands and separately a hot matrix. Include evaluation/synchronization and no debug reconstruction in timing.
- [ ] Run `python -m familytiles.cli bench --artifact artifacts/family --suite kernel`; require C1/C2 and pair sweep ≤1.5× native plus memory invariants to advance. Three alternating-order trials and allocation records must support the decision. If needed, try at most two documented layout changes, rerunning affected C2/native/control checks each time. If none passes, render a matrix/representation result and stop model integration.
- [ ] Commit: `feat: benchmark real projection sweeps and enforce G3`.

## G4: minimal complete model execution

### Task 14: Load strict model state and stream single-model prefill

**Files:** Create `src/familytiles/model.py`, `tests/test_model.py`; modify `convert.py` only for native inventory access; extend `measure.py` for named prefill phases.

**Interfaces:** Produce `RequestState`, `RuntimeFamily`, `FamilyLinear(operand: DeviceOperand, bias: mx.array | None, mode: str)` with `decode_step(x)`, `prefill(x)`, `reconstruct_words()`; `load_runtime(artifact: FamilyArtifact, mode: str) -> RuntimeFamily`, `prefill(runtime: RuntimeFamily, model_id: ModelId, token_ids: list[int]) -> tuple[mx.array, RequestState]`, `step_single(runtime: RuntimeFamily, state: RequestState, token_id: int) -> mx.array`, `release_request(state: RequestState) -> None`. Initially support B0/B1 stock, A1 materialized, and single raw/fused test modes; reject all unavailable modes explicitly.

- [ ] Write `test_active_inventory_is_strict` for missing/unexpected active weights, unexplained heads and incompatible ties; `test_native_and_alias_storage_are_unique`; `test_no_random_full_placeholder_is_evaluated`; `test_prefill_materializes_bounded_matrices` with a dependency-lifetime spy that would catch an unevaluated result retaining its weight.

```python
with pytest.raises(ValueError):
    load_runtime(artifact_with_conflicting_tied_head, 'A1')
assert loader_probe.evaluated_placeholder_bytes == 0
assert prefill_probe.max_live_decoded_matrices <= declared_staging_bound
```

- [ ] Run `python -m pytest tests/test_model.py -q`; expect new loader/wrapper failures on supported model fixtures, with explicit markers for real-artifact checks.
- [ ] Instantiate the pinned Qwen2 structure with supported linears replaced before placeholder evaluation, load immutable operands/native parameters directly and prove complete inventory coverage. Preserve module biases/dtypes and tied native storage. Implement the single-token fused path and multi-token stock-matmul materialization with evaluated outputs/cache before release; no complete decoded target or hidden source dictionary remains owned.
- [ ] Run unit/model fixture tests and one real single-model materialized/native forward comparison. Record unique arrays and prefill peak while processing multiple layers; verify source mappings and original target NumPy buffers are absent. If the actual staging bound exceeds one matrix, report/charge the measured bound and confirm budget rather than assuming deletion freed it.
- [ ] Commit: `feat: load exact family operands and stream prefill`.

### Task 15: Build the raw Qwen2 lockstep control

**Files:** Modify `model.py`, `tests/test_model.py`; add source attribution in `README.md` or model source comments.

**Interfaces:** Consume `RuntimeFamily`/`RequestState`; produce `step_pair(runtime: RuntimeFamily, states: tuple[RequestState, RequestState], token_ids: tuple[int, int]) -> tuple[mx.array, mx.array]`, `validate_pair_structure(runtime: RuntimeFamily) -> None`. Add raw-pair A3 and native B1 lockstep modes; packed pair remains disabled until Task 17.

- [ ] Write `test_raw_lockstep_matches_separate_execution` for independent histories, masks and cache positions; `test_same_shapes_incompatible_structure_rejected`; `test_unequal_states_do_not_share_kv`; `test_survivor_after_other_stream_finishes`, asserting the surviving state equals its isolated control after R=2→R=1.

```python
pair_outputs = step_pair(runtime, states, next_tokens)
mx.eval(*pair_outputs)
assert states[0].cache is not states[1].cache
assert states[0].position != states[1].position  # unequal-prefix fixture
```

- [ ] Run `python -m pytest tests/test_model.py -k 'lockstep or structure or states or survivor' -q`; expect no implemented pair runner.
- [ ] Adapt the smallest Qwen2 forward traversal that pairs matching linears while preserving each model's attention, rotary positions, normalization, activations, biases and KV. Reuse the pinned library's components and record the adapted source SHA/license. Prefill independently; pair only already-ready compatible operations; never share KV/config/tokenizer state. Implement raw controls first, including native comparison structure with stock operators.
- [ ] Run model fixture tests and short real distinct-prompt comparisons against separate stock/raw execution. Save max absolute/normalized RMS and exact raw-control pair-vs-single comparisons; the full C3 acceptance waits for Task 16's fixed corpus. Verify the raw runner has no shape-only semantic shortcuts.
- [ ] Commit: `feat: add independent-state Qwen2 lockstep control`.

### Task 16: Pin regression inputs and validate raw/materialized numerical behavior

**Files:** Create `src/familytiles/evaluate.py`, `tests/test_evaluate.py`, `experiments/prompts.json`; modify `pyproject.toml`, `model.py` only for forward/evaluation hooks; save data revision/token fixtures and raw/materialized E3 records.

**Interfaces:** Produce `TokenFixture`; `prepare_regression(artifact: FamilyArtifact, out: Path) -> dict[ModelId, TokenFixture]`, `prepare_workloads(artifact: FamilyArtifact, prompts: Path, out: Path) -> JsonObject`, `compare_logits(reference: Iterator[np.ndarray], test: Iterator[np.ndarray]) -> JsonObject`, `run_regression(artifact: FamilyArtifact, modes: tuple[str, ...], fixtures: Path) -> JsonObject`. `run_regression` supports stock teacher forcing and sequential post-prefill steps so the fused path can later be exercised on the same predictions.

- [ ] Write `test_exact_regression_window_selection`: pin test split/config, concatenate rows with newline, starts `0,257,...,1799`, exactly eight disjoint 257-token sequences and 2,048 predictions/model; tokenizer identity and special-token policy are explicit. Use small local mock text/tokenizer data in unit tests, not a full dataset download.
- [ ] Add `test_logits_statistics_accumulate_globally_in_fp64` asserting the spec's normalized-RMS formula over all entries, exact prediction count, chunk-invariant NLL/argmax statistics and thresholds 1e-3/0.99. `test_saved_inputs_bind_to_model_tokenizer_and_revision` rejects stale fixtures. `test_fixed_work_is_post_prefill` asserts prefill's predicted token is not counted as a decode forward.

```python
ref = np.array([[1.0, 2.0], [4.0, 3.0]], dtype=np.float64)
metrics = compare_logits(iter([ref]), iter([ref.copy()]))
assert metrics['normalized_rms'] == 0.0
assert metrics['argmax_agreement'] == 1.0
assert sum(len(ids) - 1 for ids in regression_ids) == 2048
```

- [ ] Run `python -m pytest tests/test_evaluate.py -q`; expect missing fixture/statistics behavior.
- [ ] Add the dataset reader and pin only `Salesforce/wikitext` / `wikitext-2-raw-v1` / test. Save attribution/revision/source hashes, per-model IDs and the eight exact prompt strings from spec §12. Implement native/raw/A1 comparisons in evaluation mode, chunked logits and FP64 statistics, deterministic prompt truncation/repetition and stock continuations for equal-work timing. Refresh the dependency lock.
- [ ] Run unit tests and the full raw-control/materialized E3 regression on real selected models. Require raw-stock aggregate logit RMS ≤1e-3 and argmax ≥99%; record NLL, low-margin disagreement and maximum error. Require A1 equals corresponding native stock-operator execution. If raw fails, debug without widening thresholds or retain the supported materialized result; do not enable packed pair to conceal a raw-runner bug.
- [ ] Commit: `feat: pin regression tokens and verify model controls`.

### Task 17: Enable packed lockstep and enforce G4 isolation/lifetime

**Files:** Modify `model.py`, `evaluate.py`, `cli.py`, `tests/test_model.py`, `tests/test_evaluate.py`; save packed E3 and G4 records.

**Interfaces:** Extend `load_runtime(..., mode='family_pair')` and `step_pair` using Task 11 without changing request interfaces; produce `verify_model(artifact: FamilyArtifact, fixtures: Path) -> JsonObject`, exposed as `verify --artifact ... --scope model`.

- [ ] Write `test_packed_runner_matches_raw_schedule` asserting bitwise fused/raw logits on the same teacher-forced steps; `test_cancel_restart_only_replaces_own_cache`; `test_alternating_single_pair_preserves_histories`; `test_lifecycle_live_memory_returns_to_idle` with ≥20 warmed short requests and the exact max(16 MiB,1% canonical) allowance, checking monotonic growth too.

```python
record = verify_model(artifact, fixture_path)
assert record['fused_raw_bit_mismatches'] == 0
assert record['lifecycle_count'] >= 20
assert record['released_live_growth_bytes'] <= max(16 * 2**20, canonical_bytes // 100)
```

- [ ] Run `python -m pytest tests/test_model.py tests/test_evaluate.py -q`; expect the new packed-pair/lifecycle behavior absent while raw controls remain green.
- [ ] Route paired linears through `family_gemv_pair` with each model's own input; use native heads/embeddings and independent nonlinear/cache operations. Handle uneven completion, cancellation and fresh-cache restart through the same single-step path. Evaluate/release request state at measured boundaries; assert no original dictionary, decoded target, numeric delta, source mapping or per-token anchor copy in the treatment child.
- [ ] Run `python -m pytest tests/test_model.py -q` and `python -m familytiles.cli verify --artifact artifacts/family --scope model`. Require every applicable C1/C2 dependency, raw/materialized C3, fused/raw equality, different prompt lengths/histories/positions, survivor behavior, ownership and ≥20 lifecycles before G4 passes. Reuse E3 only when relevant implementation/input evidence matches; otherwise recompute affected comparisons. Save actual prefill/idle/steady bounds and return reasons for unavailable probes.
- [ ] Commit: `feat: verify packed lockstep inference and request lifetimes`.

## G5: competitive and practical utility

### Task 18: Connect exact baselines to the shared runner

**Files:** Modify `baselines.py`, `model.py`, `tests/test_baselines.py`, `tests/test_model.py`.

**Interfaces:** Consume actual B2/B3/B4 backing from Task 8; produce `materialize_linear(backing: ExactChunks | ExactFamilyBacking, model_id: ModelId) -> mx.array`, `load_exact_baseline(artifact: FamilyArtifact, mode: str) -> RuntimeFamily`. Extend `load_runtime` to B2/B3/B4/A2/A3 while preserving `prefill`, `step_single`, and `step_pair` signatures. A2 calls separate fused singles; A3 uses raw pair.

- [ ] Write `test_baseline_native_semantics_and_aliases` comparing complete active inventories for B0/B1/B2; `test_streamed_provider_is_exact_and_releases_staging` for B3/B4 including independently compressed anchor reconstruction; `test_owned_backing_is_touched_and_charged`; `test_a2_is_two_real_fused_calls`, asserting unrelated inputs and evaluated work.

```python
matrix = materialize_linear(exact_family_backing, 'target')
mx.eval(matrix)
np.testing.assert_array_equal(np.asarray(matrix.view(mx.uint16)), target_words)
del matrix
assert provider_probe.retained_decoded_matrices == 0  # after evaluated use/release
```

- [ ] Run `python -m pytest tests/test_baselines.py tests/test_model.py -q`; expect missing runtime provider/mode behavior.
- [ ] Reuse the streamed prefill/linear boundaries for B3/B4 in decode, retaining their encoded store in ordinary owned RAM. Charge all backing/native copies and bounded staging; choose B4's smaller practical full representation and retain audit of both anchor forms. B2 shares the exact COPY/RAW accessor/container semantics. Ensure B1 is normal stock BF16 with actual aliases, with no padded tied head or stacked weight allocation per iteration. Keep native library execution as B0/B1's reference.
- [ ] Run unit tests and one short matched-input forward for every implemented exact mode in fresh subprocesses. Require exact reconstruction, corresponding stock-operator numerical behavior and trustworthy loaded/canonical ledgers. Report modes unavailable for a concrete budget/platform reason; unavailable baseline cells cannot be counted as wins.
- [ ] Commit: `feat: run competitive exact baselines with bounded staging`.

### Task 19: Measure core paired inference and decide whether to expand

**Files:** Modify `measure.py`, `cli.py`, `records.py`, `tests/test_measure.py`; save core timing/memory/correctness records.

**Interfaces:** Produce `build_core_cells(artifact: FamilyArtifact, inputs: Path) -> list[CellSpec]`, `evaluate_core(rows: list[JsonObject]) -> JsonObject`; expose `bench --suite core` with B1/B2/B3/B4/A1/A2/A3/family_pair.

- [ ] Write `test_core_counts_64_model_token_steps` asserting two streams ×32 post-prefill forwards, fixed saved 256-token prompts and no EOS work reduction. `test_core_metrics_include_each_stream` asserts independent p95 and total completed steps/wall time; `test_censored_core_has_no_ratio`; `test_memory_run_matches_timing_workload`; `test_core_thresholds` covers 0.15 allocation/footprint, 0.80 throughput, 1.25 p95 and 1.10 grouping boundaries.

```python
summary = evaluate_core(completed_core_rows)
assert summary['completed_model_tokens_per_trial'] == 64
assert summary['core_pass'] is True
assert evaluate_core(censored_core_rows)['core_pass'] is False
```

- [ ] Run `python -m pytest tests/test_measure.py -q`; expect missing core workload/decision behavior.
- [ ] Build matched ready-pair cells and separate identically configured memory traces, require G4 evidence, pre-touch stores, recheck memory/pressure and record cold/steady phases. Run slow baselines first on one core trial with the 120-second cap; censor explicitly and collect a clearly distinct short comparable case if useful. Continue three alternating-order full trials only where practical. Compute whole-core timings and median trial-level ratios with spread, A2/A3 ablations and maximum comparable sampled footprint.
- [ ] Run `python -m familytiles.cli bench --artifact artifacts/family --suite core`. If correctness or any necessary core memory/latency/grouping criterion fails, preserve evidence, report the narrower result and stop optional larger workloads. If strong exact alternatives censor, utility remains unverified; do not substitute an infinite baseline time. Proceed to remaining suites only when core correctness/usefulness remains plausible.
- [ ] Commit: `feat: record matched core memory and paired latency`.

### Task 20: Measure one-active-model caching and switch workloads

**Files:** Modify `baselines.py`, `measure.py`, `cli.py`, `tests/test_baselines.py`, `tests/test_measure.py`; save switch records and official-adapter applicability audit.

**Interfaces:** Produce `HotModelCache(artifact: FamilyArtifact, backing: dict[str, ExactFamilyBacking] | dict[ModelId, dict[str, ExactChunks]], budget: MemoryBudget)` with `activate(model_id: ModelId) -> Any` and `release() -> None`; the string keys are active tensor names, and the artifact supplies native/alias inventory and model configuration. Produce `build_switch_cells(artifact: FamilyArtifact, inputs: Path) -> list[CellSpec]`; expose `bench --suite switch` for B1/B5/family_single.

- [ ] Write `test_hot_cache_charges_retained_backing` and `test_miss_releases_old_decoded_model_before_replacement`, asserting both retained encoded copies and decode transients are counted. `test_switch_orders_and_queue_time` asserts exactly eight 64-token requests with 16 measured forwards each for AAAABBBB and ABABABAB, and waiting/reconstruction contribute to total completion/TTFT.

```python
cache.activate('anchor')
cache.activate('target')
assert cache_probe.maximum_decoded_models == 1
assert cache_probe.retained_backing_bytes == expected_complete_backing_bytes
cache.release()
```

- [ ] Run `python -m pytest tests/test_baselines.py tests/test_measure.py -q`; expect missing hot-cache/switch contracts.
- [ ] Implement one active decoded model over the best practical implemented exact backing chosen from the complete audit/core results; preserve configuration/tokenizer semantics on each switch. Release before replacement when needed; no full decoded helper process. Audit official adapter availability with the correct base and pinned sources. If available, add the native representation to applicable measurements; otherwise save `not_applicable`, without synthesizing or training an adapter.
- [ ] Run unit tests and `python -m familytiles.cli bench --artifact artifacts/family --suite switch` for both orders, with matched memory traces, 120-second caps and three trials for completed cells. Save switch peaks, reconstruction cost, serialization/queue consequences and total request latency. Apply utility comparison only on the predeclared switch workload; sticky and alternating results both remain visible.
- [ ] Commit: `feat: compare hot-model cache and request switching`.

### Task 21: Add quantization, longer prefill and generation evidence

**Files:** Modify `baselines.py`, `evaluate.py`, `measure.py`, `cli.py`, `tests/test_baselines.py`, `tests/test_evaluate.py`; save eval/generation records.

**Interfaces:** Produce `prepare_quantized(artifact: FamilyArtifact, bits: int, out: Path) -> JsonObject`, `run_generation(artifact: FamilyArtifact, prompts: Path, max_tokens: int = 64) -> JsonObject`, `build_eval_cells(artifact: FamilyArtifact, inputs: Path) -> list[CellSpec]`; expose `bench --suite eval`.

- [ ] Write `test_quantized_storage_counts_codes_scales_biases` and reject bits outside 4/8 for this interface; assert B6 metadata labels the mode lossy and includes group size/mode/native exclusions. `test_generation_records_first_divergence_and_full_ids` tests no-divergence, native-only divergence and raw divergence. `test_eval_reuses_only_matching_regression` and `test_long_prefill_uses_1024_tokens_32_forwards` bind inputs and counts.

```python
quantized = prepare_quantized(artifact, 4, quantized_path)
assert quantized['bits'] == 4
assert quantized['parameter_exact'] is False
assert quantized['loaded_bytes'] == packed_codes_bytes + scales_biases_bytes + native_bytes + metadata_bytes
```

- [ ] Run `python -m pytest tests/test_baselines.py tests/test_evaluate.py -q`; expect missing quantization/generation/eval dispatch.
- [ ] Use pinned native MLX-LM conversion on the selected checkpoints to actually packed 8/4-bit artifacts in ignored local storage; no expanded pseudo-codes. Run E3 NLL on the exact saved 2,048 predictions/model and the same core 256/32 performance inputs. Measure 1,024/32 prefill/TTFT/peak/decode for B1, FamilyTiles and the best completed exact core streaming baseline, respecting timeout/censoring rules.
- [ ] Run `python -m familytiles.cli bench --artifact artifacts/family --suite eval`; save up to 64 greedy tokens for all eight prompts/model with full IDs and first native/raw divergence, plus separate documented-chat-template smoke checks. Reuse equivalent E3 results by identity. Report stock reduction effects separately from codec errors and label the small corpus as regression evidence, not broad capability parity.
- [ ] Commit: `feat: record quantization tradeoffs and extended evaluation`.

### Task 22: Confirm stress behavior under matched request lifecycles

**Files:** Modify `evaluate.py`, `measure.py`, `cli.py`, `tests/test_model.py`; save stress trace/correctness records.

**Interfaces:** Produce `run_stress(artifact: FamilyArtifact, inputs: Path) -> JsonObject`; expose `bench --suite stress`. Reuse `release_request`, `step_single`, `step_pair` and G4 lifecycle logic; do not implement a second runner.

- [ ] Extend `test_stress_preserves_each_isolated_stream` for 64/256-token prompts, different positions, one early EOS, cancellation/restart, R=1/R=2 alternation and released state. Assert ≥20 cycles after warmed idle, spec allowance, independent state and exact raw-reference token/logit behavior; physical/MLX counters are interpreted independently.

```python
record = run_stress(artifact, input_path)
assert record['lifecycle_count'] >= 20
assert record['isolated_stream_mismatches'] == 0
assert record['monotonic_live_growth'] is False
```

- [ ] Run `python -m pytest tests/test_model.py -k 'stress or lifecycle or cancel or survivor' -q`; expect new stress dispatch cases fail before implementation.
- [ ] Wire the preserved G4 checks into the explicit suite with full per-phase traces and repeated request release. Do not force OOM, increase system limits, or widen leak thresholds. Reuse matching evidence only when workload/iterations/code/cache policy match.
- [ ] Run `python -m familytiles.cli bench --artifact artifacts/family --suite stress`; require independent correctness and stable live storage after allocator warmup. Any monotonic growth invalidates the full result pending repair. Save sampled-peak limitations and pressure observations.
- [ ] Commit: `feat: record uneven-stream isolation and lifetime evidence`.

### Task 23: Generate the decision and make the reached outcome reproducible

**Files:** Modify `report.py`, `records.py`, `tests/test_report.py`, `README.md`, `prior_art.md`; generate root `RESULTS.md` and final gate records. At an early stop, extend only report behavior necessary for evidence already collected.

**Interfaces:** Extend `render_report(results: Path, out: Path) -> None`; produce `evaluate_g5(records: list[JsonObject]) -> GateRecord`, `validate_evidence_set(results: Path) -> JsonObject`. Reporting reads files only; it does not start downloads or measurements.

- [ ] Write table-driven `test_final_outcome_is_supported_by_evidence` for full pass, G1 negative, G2 loss of savings, kernel-only, failed C3, missing physical probe, grouped-speed failure, dominated exact alternatives, unavailable adapter and censored required baseline. Assert exact G5 thresholds, proper workload-specific B3/B4/B5 utility comparison, unavailable values stay null and missing/skipped gates are never passing.
- [ ] Add `test_report_rejects_mixed_revision_policy_or_inputs`, `test_report_labels_measured_vs_literature`, `test_all_claims_link_raw_records` and `test_report_does_not_change_when_record_order_changes`. Early-stop tests need only synthetic record fixtures and no unimplemented runtime modules.

```python
assert evaluate_g5(complete_passing_records).decision == 'pass'
assert evaluate_g5(records_without_physical_probe).decision != 'pass'
assert evaluate_g5(records_with_censored_required_competitor).decision != 'pass'
assert evaluate_g5(records_with_failing_g1).decision != 'pass'
```

- [ ] Run `python -m pytest tests/test_records.py tests/test_report.py -q`; expect missing final decision/report coverage, then implement only the spec's supported claim paragraph, machine/revisions, gates, whole-model ledger, core/ablation/numerical tables, negative/censored evidence, measured/literature distinctions and exact reproduction commands. Use median matched trial-level metrics and report spread; do not infer absent experiments. List tested capacity budgets only.
- [ ] Run `python -m familytiles.cli report --results results --out RESULTS.md`; inspect every success/partial/negative claim against its saved record and the adversarial checklist in spec §16. Run all implemented CPU tests; run GPU/model tests only at the reached stage with markers/artifact paths recorded. Ensure report references survive a clean checkout and ignores exclude model/sample bytes.
- [ ] Rehearse reproduction in a fresh environment from the committed lock using the pinned public artifacts and saved token fixtures. If the full path was reached, reproduce CPU tests, conversion verification and core cells (reuse already cached verified public bytes without duplicating downloads); otherwise reproduce the highest completed gate and early-stop report. Give another agent the spec/plan/README and evidence paths only when execution review/delegation is authorized, and have it report missing assumptions. A metadata-only review is not claimed as a rerun. Fix any reproducibility gaps and record what was actually repeated.
- [ ] Commit: `docs: report reproducible FamilyTiles research outcome`. Verify clean status and summarize the supported claim with its terminal gate; make no new performance assertions beyond saved measurements.

## Spec coverage and handoff

| Spec requirement | Owning tasks |
| --- | --- |
| Purpose, scope, exactness, alternatives, prior-art boundary | 1–3, 9–11, 16–17, 23 |
| Gates, immutable evidence, early stops and predeclared thresholds | 1–2, 7–8, 13, 17, 19, 23 |
| Environment, resource limits, platform/dependency pins | 2, 5, 8, 12 |
| Family pinning, headers, sample splits, controls and conservative projections | 5–7 |
| Ordered/XOR words, 17-bit residuals, block layout, exceptions/native fallback | 3–4 |
| Container/integrity/aliases/streaming complete audit | 8–9 |
| Single/pair Metal, controls, real sweep, bounded tuning | 9–13 |
| Strict model loading, stock components, prefill, raw-first lockstep | 14–17 |
| Baselines B0–B6, A1–A3 and official-adapter opportunity cost | 6–8, 10–11, 14–15, 18, 20–21 |
| Unified memory, physical probe, timing, pressure, cache and ownership | 8–9, 12–14, 17–22 |
| E0/E1/E2/E3/E4 and numerical/generation regressions | 3–4 / 7–9 / 10–13 / 16–17,21 / 17,21–22 |
| Fixed workloads, caps/trials/token accounting and raw records | 12–13, 16, 19–22 |
| Full-positive/partial/rejection and reproducible generated report | 1, 19, 23 |
| Source catalog, no fabricated CUDA results, adversarial review | 2, 20, 23 |

Self-review before plan execution: verify every task's consumed type is defined above or by an earlier task, each gate's required evidence is available before the next stage, CPU tests do not initialize Metal, and every Review Focus case has an owning test. Confirm policies are frozen before holdout, model numerical checks precede core, and physical-memory absence cannot pass G5.

This document completes the requested planning deliverable. For a later implementation handoff, review the spec and plan and select native execution or subagent-driven execution; neither is presumed here. Native execution can maintain continuity across the tightly coupled codec/kernel/model interfaces, while independent reviews should concentrate on each gate's evidence and the numerical/ownership contracts. Preserve the instruction to work on the current branch unless the user changes it.
