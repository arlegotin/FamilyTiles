# FamilyTiles epilogue check: stop the fused consumer

**Decision (29 September 2026):** The BF16-before-bias correction works in the compiled kernels and removes the saved first-layer Q/K/V projection mismatches, but corrected **raw-weight paired model execution still fails** the unchanged short cached-decode stock-logit criterion. Its eight-step aggregate normalized RMS error is **0.010115 anchor** and **0.014015 target**, against **0.001**. Under the predeclared stopping rule, this closes the fused-consumer attempt on the pinned M3 Max pair. The full C3, corrected fused model check, end-to-end latency, and physical-memory peak were **not run**. This is a numerical failure of the tested consumer, not a failure of the verified exact representation.

## Identity and what changed

| Item | Identity |
|---|---|
| Machine | Apple M3 Max, 36 GiB unified memory; MLX 0.32.2; MLX-LM 0.31.3 |
| Anchor | `Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323` |
| Target | `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Codec policy | SHA-256 `53628dd18b85be5a969284737dfb1e0f1243365fe431c3bf8e18d75b702281bd` |
| Converted artifact | Manifest hash `ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585` |
| Correction | Commit `7467e00`; `src/familytiles/metal.py` SHA-256 `e4c5fd850183f7c369930fe20775d2e579a6f6b4c3ed761a5bea955dd7f38a6c` |

The [initial audit](decision.json) found the two-round epilogue only in a diagnostic helper, with no source-bound full corrected-path result. The correction now rounds the FP32 dot to BF16 before adding BF16 bias in the active single, generic paired, tiled paired, and ordered paired raw/compressed source variants. The decoder, reduction loop, codec, layout, and fixture selection did not change. Bias-free paths still cast the dot once. The [compiled fixture test](../../tests/test_epilogue.py) exercises the BF16 midpoint counterexample, positive and negative biases, cancellation, zero bias, distinct model operands, and the saved real Q/K/V inputs. The focused tests and the full repository suite passed; the compiled real Q/K/V comparison had zero output-word mismatches in both models.

The [dispatch trace](remaining-trace.json) confirms the corrected raw cached path invoked `familytiles_pair_gemv_v1` with `math_mode=safe`, no contiguity copy requested, source SHA-256 `c879ab82a47b2671891143f2f68339a1c23f1ae35f2c8cd7495edf5f3ffb22ca`, and the per-shape header hashes recorded there. These hashes bind the result more tightly than a Python source hash alone.

## Corrected cached decoding

The saved two-request fixture, first-step token, and eight teacher-forced tokens match the prior [stock B1 record](../continuation/numerical-b1-8.json): artifact, revisions, fixture hash `98ec11e0a5a2d292f7893ec71cfd793cc87d90928633f2da7d44336d5df5205c`, and token sequence were checked before comparison. Each worker owned and advanced its own KV state. The old stock reference includes bitwise-identical self-replay. The metric expands BF16 logits to FP64 and computes `sqrt(sum((test-reference)^2)/max(sum(reference^2),1e-24))` across all eight steps for each model.

| Model | Eight-step aggregate NRMSE | Threshold | BF16 logit-word mismatches | Gate |
|---|---:|---:|---:|---|
| Anchor | 0.010115 | 0.001 | 1,019,870 | Fail |
| Target | 0.014015 | 0.001 | 1,073,708 | Fail |

The [full per-step metrics](regression.json), [raw run metadata](numerical-raw_pair-8.json), and [raw logit words](numerical-raw_pair-8.npz) preserve the result. The first-step NRMSE is 0.016707 anchor and 0.016687 target. These are short-gate measurements, **not** the 2,048-prediction C3 corpus.

On identical captured stock activations, corrected raw pairing matches all first-layer Q/K/V, O, and gate outputs bit for bit. The [first residual projection](remaining-trace.json) is the bias-free `model.layers.0.mlp.up_proj.weight`: one differing BF16 word in the anchor output and two in the target output. Their normalized RMS errors are `7.05e-10` and `2.27e-6`. This locates the earliest observed word discrepancy; it does not establish that those few words alone cause the later logit error. Differences in the unchanged reduction/arithmetic path remain a plausible explanation, but were not established by a reduction-order experiment. The [captured inputs and outputs](layer0-reference.npz) and their [hashes](layer0-reference.json) permit reproduction without comparing different upstream activations.

Two additional 32-step pairs were [frozen before the eight-step result](fixtures.json), but were not run because the first gate failed. C2 against corrected fused output, full C3, model-level fused latency, final workload suite, and load-to-finish process high-water are all `not_run`. The corrected fused epilogue is implemented and passes its small compiled fixture; **it was not validated as a full-model consumer**. The prior controlled 21-matrix ratios of 1.359/1.338/1.372 remain a scoped, uncorrected-model timing result, not an end-to-end result for this correction.

The original complete representation remains 4,909,619,493 canonical active-weight bytes versus 6,174,857,216 for native exact sharing, a 20.49% saving. No new live-memory measurement was made. The earlier mixed-policy phase samples and latency failures remain in the [continuation report](../continuation/CONTINUATION_RESULTS.md); none is promoted to a process-peak claim here. For this pinned deployment, use the validated stock/native execution path. Further FamilyTiles consumer tuning is outside this closed experiment.

## Reproduce this check

These commands require the pinned local `artifacts/family` and the locked arm64 Metal environment; no model download or reconversion is part of this check. From the repository root:

```bash
python -m pytest tests/test_epilogue.py -q
python tools/trace_decode.py --mode raw_pair --steps 8 --out results/epilogue-check
python -m tools.trace_remaining --mode capture
python -m tools.trace_remaining --mode compare
python -m tools.trace_remaining --mode summarize
python -m pytest -q
```

The `summarize` command verifies artifact, revision, fixture, token, source, and saved array hashes before writing `regression.json` and the final outcome in `decision.json`. The [runtime CSV](runtime.csv) has no data rows because the raw gate stopped the experiment before timing admission.
