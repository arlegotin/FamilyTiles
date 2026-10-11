# Third-party notices

Original FamilyTiles material is licensed as described in [LICENSING.md](LICENSING.md).

## Adapted MLX-LM attention code

`_attention_finish` in `src/familytiles/model.py` is adapted from
`mlx_lm/models/qwen2.py` in MLX-LM 0.31.3, Copyright Apple Inc., under MIT.
The source attribution remains in the function and the full upstream notice
is retained in [LICENSES/MLX-LM-MIT.txt](LICENSES/MLX-LM-MIT.txt). The adaptation
and surrounding original work do not remove the upstream MIT grant.

Source: <https://github.com/ml-explore/mlx-lm>. The installed source hash and
package versions are recorded in [results/environment.json](results/environment.json).

## Models and dependencies

The Qwen base and Instruct checkpoints remain subject to their upstream model
licenses. Revision and license metadata are recorded in the pinned survey
records under `results/survey/`. Model weights and converted artifacts are
not distributed in this repository.

MLX, MLX-LM, NumPy, Zstandard, Hugging Face libraries, and other dependencies
retain their own licenses. Installing them does not relicense them under
FamilyTiles' license. Preserve their applicable notices when redistributing
them or adapted source.
