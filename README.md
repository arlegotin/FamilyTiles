# FamilyTiles

FamilyTiles explores a way to keep two related language models in memory without keeping two full copies of their weights. One model is the **anchor**; eligible weights in the other use small, bit-exact patches, while the rest stay native. The proposed Metal kernel reconstructs patched BF16 weights as it multiplies them and reuses anchor weights for two requests with different inputs. This could help local model comparisons, regression tests, or services that need the original checkpoint weights.

**Status:** The exact encoding works for the tested pair. The inference implementation did **not** meet its numerical and latency requirements, so use the original checkpoints with stock MLX-LM for this deployment.

## Results

Tested on an Apple M3 Max with 36 GiB unified memory, using Qwen2.5-1.5B base and Instruct checkpoints. The figures below have different scopes:

| Check | Result |
|---|---|
| Weight accuracy | All **1.54 billion** target BF16 words reconstructed exactly in the CPU audit; GPU decoding verified 196 encoded tensors. |
| Active-weight storage | **6.17 GB → 4.91 GB**, or **20.49% less** than native exact sharing. This is a representation size, not measured peak process RAM. |
| Paired kernel | **1.34–1.37×** native latency across three controlled 21-matrix sweeps. Encouraging as a matrix result, but the corrected full-model fused path was not validated. |
| Cached model output | After a bias-rounding fix, the raw custom path still had logit NRMSE **0.0101 / 0.0140** for base / Instruct, above the **0.001** requirement. This measures difference from stock logits, not task accuracy. |
| Stock-operator fallback | Two staged policies had **3.91× / 5.25×** native p95 decode latency. Their 10.6% / 15.2% lower footprints were phase samples, not process peaks. |

The compression result is useful research evidence, but no tested execution path delivers validated memory-saving inference. The full 2,048-prediction regression, corrected fused model run, and process-peak measurement were stopped after the short numerical gate failed.

## Future work

This consumer experiment is closed. A separate attempt should first show that its **raw-weight** execution meets the intended model-output contract, then test compression, full-model latency, and live-memory peaks. A quality-tolerant application would need its own stated quality and latency requirements; the NRMSE values here do not answer that question.

See the [closure summary](CLOSURE.md) for evidence and limits, the [original results](RESULTS.md) for the initial gates, and the [final arithmetic check](results/epilogue-check/FINAL_DECISION.md) for manual reproduction. The package has no validated inference command; model weights and converted artifacts are not committed.
