# FamilyTiles continuation: fast live inference no-go for the tested policies

**Verdict:** The pinned Qwen2.5-1.5B base/Instruct family remains an exact and smaller representation. Exact reconstruction followed by stock MLX operators reproduced stock logits bit for bit in the saved eight-step diagnostic and the new 16-step admission workload. The two frozen native-first policies reduced *phase-sampled* process footprint by 10.6% and 15.2%, but their p95 cached-decode latency was 3.91× and 5.25× the fastest native exact-sharing control. Neither passes the continuation's 1.50× fast-runtime ceiling. No policy advanced to the 2,048-prediction C3 corpus or final workload suite. This is a **no-go for the tested fast consumer**, not a rejection of the verified exact codec or a measured process-peak capacity result.

## Fixed identity and scope

| Item | Identity |
|---|---|
| Machine | Apple M3 Max, 36 GiB unified memory; arm64 macOS; MLX 0.32.2; MLX-LM 0.31.3 |
| Anchor | `Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323` |
| Target | `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Original codec policy | `53628dd18b85be5a969284737dfb1e0f1243365fe431c3bf8e18d75b702281bd` |
| Original artifact | `ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585` |
| Continuation policy | Static whole-matrix native promotion plus exact reconstruct-then-stock for remaining projections; no fused custom arithmetic admitted |

The [environment lock](../environment.json), [original report](../../RESULTS.md), [continuation design](../../docs/superpowers/specs/2026-09-28-familytiles-continuation-design.md), and all records linked below remain in the repository. Original G3 and C3 criteria were not changed.

## 1. Why did one source appear to pass and fail G3?

The historical source hash `0cba3b5a56` produced **76.00 ms paired / 62.39 ms native batched (1.22×)** in one record and **181.96 ms / 64.77 ms (2.81×)** in another. The saved benchmark-code, configuration, tensor inventory, fixture, and activation hashes match. Historical compiled headers, git dirty state, and warmup duration were not saved. Swap was present at different fixed levels without recorded growth. The cause is **unresolved**; selecting either historical result would be unjustified.

A new controlled 21-matrix sweep used distinct model weights and activations, evaluated fresh work, prebuilt the native weight stack, and interleaved native / FamilyTiles / FamilyTiles / native blocks in three fresh sessions. The session-median ratios were **1.359, 1.338, 1.372** (1.6% relative spread). This source-bound rerun is stable and falls below 1.50×, while the original failed G3 record remains a failed historical gate. The rerun has no model-level numerical approval. See [audit.json](audit.json) and [per-block records](blocks/session-0.json).

## 2. What caused the first-decode numerical error?

The five-way replay used independent KV states and the same teacher-forced tokens. Reference self-replay, the native lockstep runner, and exact decode followed by stock MLX operators had **zero BF16 logit-word mismatches** for both models over eight cached steps. Raw custom and fused compressed paths matched each other bit for bit, but their first-step stock-logit normalized RMS errors were **0.00980 anchor** and **0.01826 target**, above the unchanged 0.001 C3 threshold. The largest eight-step errors were 0.01424 and 0.02219. Matching raw and fused results establishes same-schedule transparency; it does not establish stock compatibility.

The earliest saved per-linear divergence was layer 0's biased Q projection. Replaying the *same reference input activation* through stock and custom Q/K/V operators found 434/57/91 mismatched anchor output words and 416/54/71 target words. Rounding the raw GEMV result to BF16 before adding bias eliminated those **first-layer projection** mismatches. This identifies the first bias-epilogue discrepancy; it does **not** establish that all later raw custom reduction differences are repaired. We left fused execution ineligible and used the validated native-operator route. The [numerical summary](numerical_trace.json), [projection trace](projection_trace.json), and [eight-step comparisons](numerical-trace-8.json) preserve the evidence. The original 2,048-prediction C3 gate was not run.

## 3. Which exact memory/latency point survived?

The original complete active-weight ledger remains **6,174,857,216 bytes B1** versus **4,909,619,493 bytes FamilyTiles** (20.49% saving). The fixed [15% policy](policy-15.json) promotes 99 pairs; the [10% policy](policy-10.json) promotes 123. Promoted target matrices replace their packets in the live runner. The remaining encoded pairs are reconstructed exactly into temporary BF16 matrices and consumed by stock MLX. The policy loader hashes promoted original words and does not keep their encoded packets. The two candidate manifests were frozen before admission.

The [representative operator profile](operator_profile.csv) showed the staged native route cost **36.18 ms versus 15.54 ms** for the sum of 21 individually synchronized operations. These sums are attribution data, not complete-sweep latency. On a separate complete 21-matrix sweep, staging every matrix cost **19.37 ms versus 7.30 ms native (2.65×)**. Confirmation-layer mixed sweeps cost **1.82× native at the 10% policy** and **2.14× at the 15% policy**. Different layer weights were used for confirmation; activation vectors were fixed captured Qwen vectors of the matching widths. See [staged sweep](staged_sweep.json) and [mixed confirmation records](mixed-sweep-10-confirmation.json).

The 128-token prompt / 16 teacher-forced cached-step admission used two distinct requests, fresh processes, the same token fixture, and no swap growth. Each per-request delay is the full paired round time. The native stock path was the fastest valid control. Values below are **one short admission run per mode**, not three-trial final benchmarks.

| Mode | Loaded unique weights | Max sampled process footprint | Footprint saving vs B1 | Prefill | p95 per-request decode | p95 / B1 | Logit-word mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 stock exact sharing | 6,174.9 MB | 6,637.2 MB | — | 273.7 ms | 31.8 ms | 1.00× | 0 |
| Native lockstep | 6,174.9 MB | 6,645.2 MB | −0.1% | 285.6 ms | 35.1 ms | 1.10× | 0 |
| Mixed 10% | 5,555.1 MB | 5,935.9 MB | 10.6% | 331.6 ms | 124.4 ms | 3.91× | 0 |
| Mixed 15% | 5,240.0 MB | 5,630.3 MB | 15.2% | 355.1 ms | 166.9 ms | 5.25× | 0 |

MB is decimal. The physical-footprint figures are the maximum of loaded, prefill, and decode **phase samples**, not instantaneous or load-to-finish process peaks. MLX active/peak/cache counters and the phase samples are in the [admission records](admission-b1.json); the [CSV](runtime.csv) and [decision](admission_summary.json) bind the same fixture and artifact. System swap occupancy remained constant; that is not a proof of zero memory compression during the run. Neither candidate meets 1.50×, including after its first decode step, so a longer performance campaign would not change this short gate's decision without a different implementation.

### B3/B4 and the capacity limit

The existing exact B3 and B4 codecs were instantiated as **resident encoded arrays per measured matrix**, decoded and checked against original BF16 words. Their 21-matrix encoded byte totals match the original per-tensor ledger exactly: **399,320,277 B3** and **316,002,046 B4**. Their combined decode-plus-native *sums of individually synchronized operations* were **790.52 ms B3** and **2,211.76 ms B4**, versus 36.18 ms for FamilyTiles staging on those operations. The [baseline runtime record](baseline_runtime.json) explicitly limits this to 21 matrices. There was no all-active-tensor resident B3/B4 runner, no matched process-peak comparison, and no full end-to-end timing for them. Their smaller whole-family encoded stores, **4,382,169,185** and **3,482,853,051** bytes respectively, cannot be equated with live memory or latency. The local probe argues against investing in their full runtime for this failed admission, but it does not establish a whole-model dominance ranking.

## 4. Build next, or stop?

**Stop this FamilyTiles consumer work on the pinned M3 Max pair.** The exact format and decoder remain useful research outputs. The tested stock-operator escape route is numerically sound but costs several times native decode latency for only 10–15% sampled process-footprint saving. Fused execution is faster at the matrix level, yet its raw arithmetic still violates stock compatibility. Neither fixed mixed policy passed the short fast gate; the full C3 corpus, generation fixtures, final lifecycle suite, and reliable process high-water check were deliberately not run. A capacity-only claim is therefore unverified, and the original G3 failure remains on record.

The next justified work, if deployment needs this exact pair, is to use the stock native representation or a separate representation with an explicitly accepted numerical contract. A future FamilyTiles attempt would need a validated consumer that removes the measured staging cost while preserving stock arithmetic, then a new bounded admission. More codec screening or random kernel tuning does not address the observed blocker.

## Reproduce the continuation

Use the existing pinned public model downloads and converted `artifacts/family` from [the original report](../../RESULTS.md). These artifacts are intentionally not committed. From the repository root in the locked virtual environment:

```bash
python -m pytest -q
python tools/audit_g3.py --artifact artifacts/family --out results/continuation
python tools/trace_decode.py --mode all --steps 1
python tools/trace_decode.py --mode all --steps 8
python tools/trace_projection.py
python -m tools.profile_execution --max-names 21
python -m tools.profile_execution --sweep
python -m tools.select_execution
python -m tools.profile_execution --sweep --policy results/continuation/policy-10.json --confirmation
python -m tools.profile_execution --sweep --policy results/continuation/policy-15.json --confirmation
python -m tools.admit_policy --mode B1
python -m tools.admit_policy --mode native_pair
python -m tools.admit_policy --mode mixed10
python -m tools.admit_policy --mode mixed15
python -m tools.summarize_admission
```

The admission scripts require local Metal and the pinned artifact. The CPU policy, codec, and report checks run without model downloads. Reproduction times and sampled footprints can vary with memory pressure; use the recorded raw rows and identities, not one selected timing ratio.
