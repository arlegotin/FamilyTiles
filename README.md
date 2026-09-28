# FamilyTiles

Research design for exact shared-weight inference on one Apple M3 Max: keep two related BF16 checkpoints as one served anchor plus directly addressable patches, and test whether a paired kernel can reuse the anchor while preserving each model's original weight words.

G0 passed on the local M3 Max using pinned MLX and MLX-LM packages. G1 screened two real Qwen pairs with bounded samples: the 0.5B pair projected 12.1% full-family savings and failed; the 1.5B pair projected 20.2% and passed. G2 then converted the full 1.5B pair and verified every active BF16 tensor. Its canonical active-weight representation is 20.49% smaller than native exact sharing. Process-memory and inference-speed results remain unmeasured. The scoped [prior-art audit](prior_art.md) records the inspected papers and source revisions.

Implementation is in progress. Install the local package in a Python virtual environment, then run `python -m familytiles.cli report --results results --out RESULTS.md` to render the currently supported gate status. The [G1 survey](results/survey/summary.json) and [G2 ledger](results/conversion/ledger.json) are saved; the [frozen policy](results/survey/frozen-policy.json) selects the 1.5B pair. Commands for later gates become available as their implementations are verified.

- [Research specification](docs/superpowers/specs/2026-09-28-familytiles-design.md)
- [Gated implementation plan](docs/superpowers/plans/2026-09-28-familytiles.md)

Implementation follows gates G0–G5 in order. An evidence-backed early rejection is a valid outcome. The specification fixes the resource limits, numerical checks, baselines, and acceptance thresholds before measurement.
