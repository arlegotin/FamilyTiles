# FamilyTiles

Research implementation for exact shared-weight inference on one Apple M3 Max. It stores two related BF16 checkpoints as one served anchor plus directly addressable patches, then tests paired execution of their different matrix–vector products.

The real Qwen2.5-1.5B base/Instruct pair passed exact conversion: all 1.54 billion target BF16 words were verified, and canonical active-weight storage was 20.49% below two native models. A streamed prefill matched stock logits bit for bit, with 19.99% lower sampled loaded process footprint. The paired kernel passed its 1.5× native-speed ceiling once but failed a same-source rerun at 2.81×. The current G3 decision is **fail**. A short raw paired-model diagnostic also exceeded the stock-logit regression threshold. The complete inference claim is unproven; see [RESULTS.md](RESULTS.md) for the measurements and limits.

Install the local package in a Python virtual environment, then run `python -m familytiles.cli report --results results --out RESULTS.md` to render the saved gate status. The [G1 survey](results/survey/summary.json), [G2 ledger](results/conversion/ledger.json), [G3 raw trials](results/kernel/summary.json), [frozen policy](results/survey/frozen-policy.json), and [scoped prior-art audit](prior_art.md) are committed. The model weights and converted artifact are excluded from Git.

- [Research specification](docs/superpowers/specs/2026-09-28-familytiles-design.md)
- [Gated implementation plan](docs/superpowers/plans/2026-09-28-familytiles.md)

The project stopped at G3 according to the prespecified gate. Later CLI suites and the full C3/G5 evaluation are not implemented or measured. The specification fixes the resource limits, numerical checks, baselines, and acceptance thresholds before measurement.

A bounded [continuation](results/continuation/CONTINUATION_RESULTS.md) tested exact reconstruction with stock MLX operators and two fixed mixed native/compressed policies. Cached logits matched stock in the short checks, but the mixed policies missed the continuation's 1.50× decode-latency ceiling by a wide margin. The original G3 failure and unrun full C3 gate remain unchanged.
