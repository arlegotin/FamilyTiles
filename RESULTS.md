# FamilyTiles results

The research stopped at G3: paired primitive exceeded 1.5x the fastest native pair sweep

| Gate | Decision | Reason |
| --- | --- | --- |
| G0 | pass | native smoke and scoped collision audit passed |
| G1 | pass | one real family passed the conservative sample screen |
| G2 | pass | complete CPU exactness, size and planned runtime budget passed |
| G3 | fail | paired primitive exceeded 1.5x the fastest native pair sweep |
| G4 | not_run | No saved evidence |
| G5 | not_run | No saved evidence |

## Supported result

On the pinned Qwen2.5-1.5B base/Instruct pair, FamilyTiles reconstructed 1,543,714,304 target BF16 words without a mismatch and used 20.49% fewer canonical active-weight bytes than exact native sharing. The latest source-matched paired-kernel sweep took 2.81× the fastest native pair control, above the 1.50× G3 ceiling. The full inference result remains unproven; G4/G5 were not run.

The bounded G1 sample screen rejected the prespecified 0.5B pair at 12.06% projected full-family savings; the selected 1.5B pair projected 20.21%. These were estimates, followed by the complete 1.5B audit. [Survey](results/survey/summary.json), [0.5B projection](results/survey/qwen2.5-0.5b/projection.json), [1.5B projection](results/survey/qwen2.5-1.5b/projection.json).

## Machine and pinned data

Apple M3 Max, 36.0 GiB physical memory, macOS 26.6.2, Python 3.13.5, MLX 0.32.2, MLX-LM 0.31.3

- Anchor: `Qwen/Qwen2.5-1.5B` at `8faed761d45a263340a0528343f099c05c9a4323`.
- Target: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Policy: `53628dd18b85be5a969284737dfb1e0f1243365fe431c3bf8e18d75b702281bd`; artifact: `ed6da10e3efbe6eea5b6b0bde6f0c5536cc97dd884ed85ca6060ff7c4dc33585`. [Environment](results/environment.json), [dependency lock](results/requirements-lock.txt).

## Exact-weight memory

| Representation | Full active-weight bytes | Status |
| --- | ---: | --- |
| B1 native exact sharing | 6,174,857,216 | measured allocation reference |
| B2 tile identity only | 6,175,162,549 | complete exact size audit |
| FamilyTiles anchor + patches + native leftovers | 4,909,619,493 | complete exact size audit |
| B3 independent transformed Zstd | 4,382,169,185 | encoded size only |
| B4 family-delta Zstd with compressed anchor | 3,482,853,051 | encoded size only |

FamilyTiles saved 20.50% of raw family bytes beyond B2. B3/B4 are smaller encoded stores; their live inference latency and transient memory were not measured. [Full tensor ledger](results/conversion/ledger.json).

A one-request streamed target prefill matched stock logits bit for bit (0 mismatches across 151,936 words). Its two-model loaded weight allocation was 20.50% lower; loaded process physical footprint was 19.99% lower. These physical-footprint observations are phase samples, not instantaneous peaks. [Single-model control](results/correctness/single-model-comparison.json).

## Paired-kernel evidence

| Sweep mode | Median complete sweep | Interpretation |
| --- | ---: | --- |
| FamilyTiles paired | 181.96 ms | latest source-matched run |
| native_batched | 64.77 ms | fastest native pair control |
| Two FamilyTiles singles | 215.05 ms | grouping ablation |

C1 GPU reconstruction and C2 same-schedule arithmetic were exact (true, true). The latest paired sweep was 2.81× native. [Latest gate](results/gates/G3.json), [sweep summary](results/kernel/summary.json).

Saved G3 attempts are source-bound; different sources and run conditions are not averaged together.

| Kernel source SHA prefix | Pair/native ratio | Decision | Record |
| --- | ---: | --- | --- |
| `0cba3b5a56` | 1.22× | pass | [raw gate](results/gates/79f303f318b14f65aafd28e367d05c0d/G3.json) |
| `0cba3b5a56` | 2.81× | fail | [raw gate](results/gates/e18d8163e6a6405d93b3da179a28904d/G3.json) |
| `1651c482a4` | 1.58× | fail | [raw gate](results/gates/f34e4ead50964a1391c922813de28c4d/G3.json) |
| `ed5f46ddd1` | 1.63× | fail | [raw gate](results/gates/4c0f4a62f80d41579344b0c0ec9cfb63/G3.json) |
| `f376e3f0ae` | 3.29× | fail | [raw gate](results/gates/71ea57bdee3b493b8fabe3b75239bace/G3.json) |
| `f376e3f0ae` | 4.10× | fail | [raw gate](results/gates/61dc30a89d154587b66d082b26e58c24/G3.json) |

## Exploratory model control after the kernel gate

Raw paired and separate custom-kernel calls agreed bit for bit for two distinct requests, and prefill logits/KV matched stock. The first decode-step logit normalized RMS error versus stock was 0.0098 (anchor) and 0.0183 (target), exceeding the predeclared 0.001 regression threshold on this diagnostic. This is not the full 2,048-prediction C3 corpus. [Raw control](results/correctness/raw-lockstep-comparison.json).

GPU C1 verification covered 196 encoded tensors and 1,310,195,712 words with 0 mismatched hashes. [GPU record](results/correctness/gpu.json).

## Scope and reproduction

Measured here: exact full-model representation, bounded GPU word decoding, source-bound matrix sweeps, one-request materialized prefill, and a short raw lockstep diagnostic. Core paired LLM throughput, process peak during a matched core workload, 2,048-prediction C3, quantization, streaming runtime baselines, switching and stress tests remain unmeasured. CUDA comparisons in `prior_art.md` are literature only.

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e .
python -m pytest tests/test_codec.py -q
python -m familytiles.cli survey --families experiments/families.json --sample-mib-per-model 32 --seed 20260928 --out results/survey
python -m familytiles.cli convert --family results/survey/selected-family.json --policy results/survey/frozen-policy.json --out artifacts/family
python -m familytiles.cli verify --artifact artifacts/family --device cpu
python -m familytiles.cli verify --artifact artifacts/family --device gpu
python tools/check_pair_real.py
python -m familytiles.cli bench --artifact artifacts/family --suite kernel
python -m familytiles.cli report --results results --out RESULTS.md
```

The model downloads and converted artifact are excluded from Git. The saved raw records, pinned revisions, policy, and dependency lock identify the tested experiment.
