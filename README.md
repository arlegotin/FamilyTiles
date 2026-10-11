# FamilyTiles

Related language models share a lot of weight structure. FamilyTiles tests whether two models can share an anchor checkpoint and store the second as small, **bit-exact patches**, while keeping unrelated weights native.

**For the tested pair, the representation used 20.49% less active-weight storage and reconstructed all 1.54 billion target BF16 words exactly.** Using it for inference proved harder: the custom execution failed its model-output requirement, and the stock-operator fallbacks were too slow. The consumer experiment is closed.

The repository preserves the exact codec, Metal decoder and kernel work, numerical fixtures, and the measurements that separated a storage win from a usable inference system.

## How it works

One model supplies the anchor weights. Eligible tensors in the second model are encoded as lossless residuals against that anchor; tensors that don't benefit stay native. The proposed Metal kernel reconstructs patched BF16 weights during multiplication and reuses anchor loads for two requests with different inputs.

That design could be useful for comparing related checkpoints or serving model variants when their original weights must survive unchanged. It also imposes two different requirements: reconstruct the right bits, then execute the model within its numerical and latency contract. Passing the first doesn't establish the second.

## What the experiment found

The study used Qwen2.5-1.5B base and Instruct checkpoints on an Apple M3 Max with 36 GiB unified memory.

| Measurement | Result |
|---|---|
| Exact reconstruction | **1,543,714,304** target BF16 words checked on CPU, zero mismatches; GPU decoding verified 196 encoded tensors. |
| Active-weight representation | **6.17 GB → 4.91 GB**, 20.49% smaller than native exact sharing. Storage accounting, not a measured process-memory peak. |
| Paired matrix kernel | **1.34–1.37× native latency** in three controlled 21-matrix sweeps: slower than native, before validating the corrected full-model fused path. |
| Corrected raw-weight model execution | Eight-step logit NRMSE **0.0101 / 0.0140** for base / Instruct, above the **0.001** requirement. This measures disagreement with stock logits, not task-accuracy loss. |
| Stock-operator fallbacks | **3.91× / 5.25× native p95 decode latency**. Their 10.6% / 15.2% lower footprints were phase samples, not process peaks. |

The numerical failure occurred even in the corrected **raw-weight** custom path, before compressed full-model execution was admitted. A bias-rounding correction passed compiled fixtures and removed the saved first-layer Q/K/V mismatches, but the cached model output still failed the short gate. The earliest remaining projection discrepancy was located; its contribution to the final logit error wasn't established.

The corrected fused full-model run, 2,048-prediction regression, end-to-end fused latency, and physical-memory peak were **not run** after that failure. No tested path delivers validated memory-saving inference. For this deployment, use the original checkpoints with stock MLX-LM.

[Closure and evidence](CLOSURE.md) · [Initial results](RESULTS.md) · [Final numerical decision](results/epilogue-check/FINAL_DECISION.md)

## Explore the reusable parts

- [CPU codec](src/familytiles/codec.py): exact residual encoding, literal fallback, and reconstruction checks, including unusual BF16 words.
- [Metal implementation](src/familytiles/metal.py): GPU reconstruction and paired multiplication kernels, with explicit numerical fixtures.
- [Conversion ledger](results/conversion/ledger.json): complete active-weight storage accounting for the selected model pair.
- [Saved regression](results/epilogue-check/regression.json) and [captured layer fixtures](results/epilogue-check/layer0-reference.json): the numerical failure and the inputs needed to investigate it.

For a small CPU-only check, use Python 3.12 and an isolated environment. These are the recorded codec/test dependency versions; this installs no model weights or MLX runtime.

```bash
git clone https://github.com/arlegotin/FamilyTiles.git
cd FamilyTiles
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.3 zstandard==0.25.0 pytest==9.1.1
PYTHONPATH=src python -m pytest tests/test_codec.py -q
```

The full Apple Silicon environment is recorded in [the lockfile](results/requirements-lock.txt). Model weights and converted artifacts aren't committed. [The final report](results/epilogue-check/FINAL_DECISION.md#reproduce-this-check) gives the diagnostic commands for a machine with the pinned local artifacts; the CLI has no validated compressed-model inference command.

The weight saving remains useful evidence. A separate consumer attempt should first pass its raw-model numerical contract, then test compression, full-model latency, and live-memory peaks. An application willing to tolerate output changes needs its own quality requirement; these logit errors don't answer that question.

## Credit and reuse

By **Artem Legotin**. Original code is licensed under [AGPL-3.0-only](LICENSE); original prose and reports under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Preserve author credit when sharing covered material. [Licensing scope](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md) retain upstream rights, including the adapted MLX-LM attention code.

If you build on the codec, kernels, findings, or ideas, please cite [CITATION.cff](CITATION.cff), link this repository, and identify the commit you used. Citation of independently reused ideas is a scholarly request, separate from the license obligations.
