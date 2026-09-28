# FamilyTiles live-inference continuation

The user's 28 September 2026 continuation instruction is the binding brief. This document records the implementation choices needed to execute it in the existing repository. The original [research spec](2026-09-28-familytiles-design.md), artifact, gates, and raw records remain unchanged.

## Objective and fixed identities

Determine whether exact shared weights can produce a useful resident inference point on the existing M3 Max. Keep Qwen2.5-1.5B base `8faed761d45a263340a0528343f099c05c9a4323`, Instruct `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, frozen policy `53628dd18b85be5a969284737dfb1e0f1243365fe431c3bf8e18d75b702281bd`, and artifact `ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585`. The existing CPU and GPU C1 evidence can be reused while codec inputs stay unchanged. Work on the current branch in reviewable commits, per the original user preference; store new records under `results/continuation/`.

## Stage gates

1. Audit the contradictory same-source G3 records and current harness. Reproduce only the complete 21-matrix sweep in three sessions with interleaved blocks and record absolute times, code/config/workload identities, memory pressure, and missing fields. Preserve the old gate.
2. Recreate two independent cached states and compare REF, native lockstep, raw custom lockstep, exact-decode/native lockstep, and fused lockstep. A native runner discrepancy is repaired before kernel work. Exact-decode/native must pass the 0.001 short cached-step diagnostic before performance work. Keep C1, C2, and stock C3 separate.
3. Profile representative matrix pairs and combined decode/native operations. Build thin resident B3/B4 adapters from the already implemented exact codecs and verify their inverse transforms. A stable, validated fused route is optional; no more than two hypothesis-driven kernel changes.
4. Select at most two whole-matrix-pair policies, offline, from native resident, compressed/native-staged, and numerically eligible compressed/fused. Each manifest replaces promoted streams or charges the retained bytes, tracks anchor dependencies and scratch, and targets at least 15% then 10% projected weight-side saving where feasible. Give B3/B4 one comparable selective policy.
5. Admit one frozen candidate through a short two-request 128-prompt/16-step probe. Only a correct, non-dominated point proceeds to the original 2,048-prediction C3 check and the bounded final workloads. The new live-memory target is at least 10% below matched B1 peak; fast-runtime p95 is at most 1.50× matched native. These are continuation targets and do not rewrite G3.

Stop at the first specified early exit. Numerical mismatch, unstable timings, missing reliable peak, timeout, or a better B3/B4 point must remain explicit. Publish `CONTINUATION_RESULTS.md` with a verdict even after an early stop. No new models, codec, serving framework, allocator, or training.

## Runtime ownership

One served anchor, one target representation, native leftovers, separate KV caches, and bounded decoded scratch are the only planned live stores. Native promotion replaces a compressed stream when physically compacted; otherwise the stream remains charged. A temporary decoded matrix must be evaluated and released before the next one. Use fresh process memory runs, matched allocator policy and token work, physical footprint samples labeled as such, and the same safe process budget.

## Evidence and interpretation

Every continuation record identifies artifact, policy, source revision, fixture, workload, and resident representation. Historical source SHA equality is insufficient experimental identity. Native controls must compute both distinct model products; all declared outputs must be evaluated. The original matrix sweep omits biases for every mode; report that limitation and include real biases in model-level checks. Published CUDA numbers remain literature-only. The final outcome is one of recovered original path, useful native-first/hybrid, capacity-only, simpler exact baseline preferred, or no-go/inconclusive.
