# FamilyTiles

Research design for exact shared-weight inference on one Apple M3 Max: keep two related BF16 checkpoints as one served anchor plus directly addressable patches, and test whether a paired kernel can reuse the anchor while preserving each model's original weight words.

G0 passed on the local M3 Max using pinned MLX and MLX-LM packages. The real-checkpoint compression and inference gates have not yet been executed, so no memory or performance result is claimed. The scoped [prior-art audit](prior_art.md) records the inspected papers and source revisions.

Implementation is in progress. Install the local package in a Python virtual environment, then run `python -m familytiles.cli report --results results --out RESULTS.md` to render the currently supported gate status. Commands for later gates become available as their implementations are verified.

- [Research specification](docs/superpowers/specs/2026-09-28-familytiles-design.md)
- [Gated implementation plan](docs/superpowers/plans/2026-09-28-familytiles.md)

Implementation follows gates G0–G5 in order. An evidence-backed early rejection is a valid outcome. The specification fixes the resource limits, numerical checks, baselines, and acceptance thresholds before measurement.
