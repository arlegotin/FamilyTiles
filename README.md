# FamilyTiles

Research design for exact shared-weight inference on one Apple M3 Max: keep two related BF16 checkpoints as one served anchor plus directly addressable patches, and test whether a paired kernel can reuse the anchor while preserving each model's original weight words.

The repository currently contains the specification and implementation plan. The research gates have not been executed, and no compression, correctness, memory, or performance result is claimed.

- [Research specification](docs/superpowers/specs/2026-09-28-familytiles-design.md)
- [Gated implementation plan](docs/superpowers/plans/2026-09-28-familytiles.md)

Implementation follows gates G0–G5 in order. An evidence-backed early rejection is a valid outcome. The specification fixes the resource limits, numerical checks, baselines, and acceptance thresholds before measurement.
