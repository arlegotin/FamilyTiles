"""Strict Qwen2 runtime for one anchor and one exact patched checkpoint.

Qwen2 traversal and cache behavior come from the pinned mlx-lm implementation;
only its seven linear projections are replaced for FamilyTiles modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.qwen2 import Model, ModelArgs
from mlx_lm.utils import load_model

from .convert import FamilyArtifact, _safe_path
from .measure import compute_budget
from .metal import DeviceOperand, decode_words, family_gemv, load_operand, raw_gemv


ModelId = Literal["anchor", "target"]
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
MODES = frozenset({"B1", "A1", "family_single"})


def expected_active_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Active parameter shapes of the pinned MLX-LM Qwen2 implementation."""
    integers = ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size")
    if (config.get("model_type") != "qwen2" or
            any(type(config.get(key)) is not int or config[key] <= 0 for key in integers) or
            config["hidden_size"] % config["num_attention_heads"] or
            config["num_attention_heads"] % config["num_key_value_heads"]):
        raise ValueError("unsupported Qwen2 config")
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    kv_dim = hidden // config["num_attention_heads"] * config["num_key_value_heads"]
    vocab = config["vocab_size"]
    shapes = {"model.embed_tokens.weight": (vocab, hidden),
              "model.norm.weight": (hidden,)}
    if not config.get("tie_word_embeddings", True):
        shapes["lm_head.weight"] = (vocab, hidden)
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
        for projection in PROJECTIONS:
            branch = "self_attn" if projection in PROJECTIONS[:4] else "mlp"
            output, input_dim = {
                "q_proj": (hidden, hidden),
                "k_proj": (kv_dim, hidden),
                "v_proj": (kv_dim, hidden),
                "o_proj": (hidden, hidden),
                "gate_proj": (intermediate, hidden),
                "up_proj": (intermediate, hidden),
                "down_proj": (hidden, intermediate),
            }[projection]
            shapes[prefix + f"{branch}.{projection}.weight"] = (output, input_dim)
            if projection in ("q_proj", "k_proj", "v_proj"):
                shapes[prefix + f"{branch}.{projection}.bias"] = (output,)
    return shapes


def expected_active_names(config: dict[str, Any]) -> set[str]:
    return set(expected_active_shapes(config))


def validate_active_inventory(manifest: dict[str, Any],
                              configs: dict[str, dict[str, Any]]) -> None:
    if set(configs) != {"anchor", "target"}:
        raise ValueError("both model configurations are required")
    shapes = {role: expected_active_shapes(config) for role, config in configs.items()}
    inventories = {role: set(shape) for role, shape in shapes.items()}
    for role in ("anchor", "target"):
        if bool(configs[role].get("tie_word_embeddings", True)) != bool(
                manifest["tie_word_embeddings"][role]):
            raise ValueError(f"{role} tied-head declaration disagrees with artifact")
    actual = set(manifest["tensors"])
    if actual != inventories["anchor"] or actual != inventories["target"]:
        raise ValueError("active tensor inventory disagrees with pinned Qwen2 structure")
    for name in actual:
        actual_shape = tuple(manifest["tensors"][name]["shape"])
        if actual_shape != shapes["anchor"][name] or actual_shape != shapes["target"][name]:
            raise ValueError(f"active tensor shape disagrees with pinned Qwen2: {name}")
    projection_weights = {name for name in actual if name.endswith(".weight") and
                          any(name.endswith(f".{p}.weight") for p in PROJECTIONS)}
    if any(manifest["tensors"][name]["target"]["kind"] == "packed"
           for name in actual - projection_weights):
        raise ValueError("nonlinear/native tensor unexpectedly encoded as a projection")


class FamilyLinear(nn.Module):
    """Exact matrix accessor with bounded stock prefill and fused single decode."""

    def __init__(self, operand: DeviceOperand, model_id: ModelId,
                 bias: mx.array | None, mode: str):
        super().__init__()
        if model_id not in ("anchor", "target") or mode not in ("A1", "family_single"):
            raise ValueError("unsupported FamilyLinear role or mode")
        self.operand = operand
        self.model_id = model_id
        self.bias = bias
        self.mode = mode
        self.last_prefill_decoded_matrix_bytes = 0
        self.live_decoded_matrices = 0

    def reconstruct_words(self) -> mx.array:
        if self.model_id == "anchor":
            return self.operand.anchor_words
        return decode_words(self.operand)

    def decode_step(self, x: mx.array) -> mx.array:
        if x.ndim != 1 or x.dtype != mx.bfloat16:
            raise ValueError("decode_step expects one BF16 activation vector")
        if self.mode == "A1":
            return self.prefill(x)
        if self.model_id == "anchor":
            return raw_gemv(self.operand.anchor_words, x, self.bias)
        return family_gemv(self.operand, x, self.bias)

    def prefill(self, x: mx.array) -> mx.array:
        if x.dtype != mx.bfloat16 or x.shape[-1] != self.operand.shape[1]:
            raise ValueError("prefill activation shape/dtype mismatch")
        owns_decoded = self.model_id == "target" and self.operand.kind == "packed"
        words = self.reconstruct_words()
        self.live_decoded_matrices += int(owns_decoded)
        self.last_prefill_decoded_matrix_bytes = (
            self.operand.shape[0] * self.operand.shape[1] * 2 if owns_decoded else 0
        )
        try:
            weight = words.view(mx.bfloat16)
            # Match mlx.nn.Linear's single-rounding bias path exactly.
            result = (mx.addmm(self.bias, x, weight.T) if self.bias is not None
                      else x @ weight.T)
            # Break the lazy dependency on the full decoded matrix before the
            # next projection is materialized.
            mx.eval(result)
            return result
        finally:
            del words
            self.live_decoded_matrices -= int(owns_decoded)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 1:
            return self.decode_step(x)
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1 and self.mode != "A1":
            return self.decode_step(x[0, 0, :]).reshape(1, 1, -1)
        return self.prefill(x)


@dataclass
class RequestState:
    model_id: ModelId
    cache: list[Any]
    position: int
    active: bool = True


@dataclass
class RuntimeFamily:
    artifact: FamilyArtifact
    mode: str
    models: dict[ModelId, Model]
    configs: dict[ModelId, dict[str, Any]]
    operands: dict[str, DeviceOperand]
    native_arrays: dict[tuple[str, str], mx.array]
    placeholder_evaluated_bytes: int = 0


def loaded_weight_ledger(runtime: RuntimeFamily) -> dict[str, int]:
    """Count each owned MLX array once; aliases are shared object references."""
    if runtime.mode == "B1":
        from mlx.utils import tree_flatten
        arrays = [array for model in runtime.models.values()
                  for _, array in tree_flatten(model.parameters())]
    else:
        arrays = list(runtime.native_arrays.values())
        for operand in runtime.operands.values():
            arrays.extend(array for array in (operand.anchor_words, operand.descriptors,
                                               operand.payload, operand.native_words)
                          if array is not None)
    unique = {id(array): array for array in arrays}
    return {"unique_weight_allocation_bytes": sum(int(array.nbytes)
                                                   for array in unique.values()),
            "unique_weight_allocation_count": len(unique),
            "duplicate_references": len(arrays) - len(unique),
            "random_placeholder_evaluated_bytes": runtime.placeholder_evaluated_bytes}


def _config_path(artifact: FamilyArtifact, role: ModelId) -> Path:
    repo = artifact.manifest["source_repos"][role]
    revision = artifact.manifest["revisions"][role]
    if (len(repo.split("/")) != 2 or any(part in ("", ".", "..") for part in repo.split("/"))
            or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("invalid pinned model identity")
    root = Path(__file__).resolve().parents[2]
    local = root / "artifacts/downloaded" / repo.replace("/", "--") / revision
    path = local / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo, filename="config.json", revision=revision,
                                    local_dir=local))
    if path.stat().st_size > 2**20:
        raise ValueError("model config exceeds bounded metadata cap")
    want = artifact.manifest["config_hashes"][role]["config.json"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise ValueError("pinned model config hash mismatch")
    return path


def _load_configs(artifact: FamilyArtifact) -> dict[ModelId, dict[str, Any]]:
    return {role: json.loads(_config_path(artifact, role).read_text(encoding="utf-8"))
            for role in ("anchor", "target")}


def _attribute(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    node = root
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node, parts[-1]


def _read_native(artifact: FamilyArtifact, role: ModelId, name: str,
                 already: dict[tuple[str, str], mx.array]) -> mx.array:
    record = artifact.manifest["tensors"][name]
    target = record["target"]
    if role == "target" and target["kind"] == "alias":
        return already[("anchor", name)]
    shape = tuple(record["shape"])
    if role == "anchor":
        path = _safe_path(artifact.root, record["anchor"]["path"])
        offset = record["anchor"]["offset"]
        want = record["anchor"]["sha256"]
    else:
        if target["kind"] != "native":
            raise ValueError("packed projection is not a native parameter")
        path = _safe_path(artifact.root, target["path"])
        offset = 0
        want = target["sha256"]
    words = np.memmap(path, mode="r", dtype="<u2", offset=offset, shape=shape)
    if hashlib.sha256(memoryview(words).cast("B")).hexdigest() != want:
        raise ValueError(f"native tensor hash mismatch: {role}:{name}")
    value = mx.array(words, dtype=mx.uint16).view(mx.bfloat16)
    mx.eval(value)
    del words
    return value


def load_runtime(artifact: FamilyArtifact, mode: str) -> RuntimeFamily:
    if mode not in MODES:
        raise ValueError(f"unsupported runtime mode {mode!r}")
    available = psutil.virtual_memory()
    budget = compute_budget(available.total, available.available)
    if budget.process_limit_bytes < 4 * 2**30:
        raise RuntimeError("safe process budget is insufficient for the selected family")
    mx.set_memory_limit(budget.process_limit_bytes)
    mx.set_cache_limit(budget.cache_limit_bytes)
    configs = _load_configs(artifact)
    validate_active_inventory(artifact.manifest, configs)
    if mode == "B1":
        models = {role: load_model(_config_path(artifact, role).parent, strict=True)[0]
                  for role in ("anchor", "target")}
        for name, record in artifact.manifest["tensors"].items():
            if record["target"]["kind"] == "alias":
                anchor_parent, attr = _attribute(models["anchor"], name)
                target_parent, _ = _attribute(models["target"], name)
                setattr(target_parent, attr, getattr(anchor_parent, attr))
        return RuntimeFamily(artifact, mode, models, configs, {}, {})

    # Qwen2 constructors create lazy random placeholders. Every active leaf is
    # replaced before any model parameters are evaluated.
    models = {role: Model(ModelArgs.from_dict(configs[role]))
              for role in ("anchor", "target")}
    operands: dict[str, DeviceOperand] = {}
    native: dict[tuple[str, str], mx.array] = {}
    for name in sorted(artifact.manifest["tensors"]):
        if name.endswith(".weight") and any(
                name.endswith(f".{projection}.weight") for projection in PROJECTIONS):
            operand = load_operand(artifact, name)
            operands[name] = operand
            module_name = name.removesuffix(".weight")
            bias_name = module_name + ".bias"
            for role in ("anchor", "target"):
                bias = None
                if bias_name in artifact.manifest["tensors"]:
                    bias = native.get((role, bias_name))
                    if bias is None:
                        bias = _read_native(artifact, role, bias_name, native)
                        native[(role, bias_name)] = bias
                parent, attribute = _attribute(models[role], module_name)
                setattr(parent, attribute, FamilyLinear(operand, role, bias, mode))
        elif not (name.endswith(".weight") and any(
                name.endswith(f".{projection}.weight") for projection in PROJECTIONS)):
            for role in ("anchor", "target"):
                if (role, name) not in native:
                    native[(role, name)] = _read_native(artifact, role, name, native)
                # Projection biases live in FamilyLinear, not a stock module.
                if not any(name.endswith(f".{projection}.bias") for projection in PROJECTIONS):
                    parent, attr = _attribute(models[role], name)
                    setattr(parent, attr, native[(role, name)])
    if set(operands) != {n for n in artifact.manifest["tensors"]
                         if n.endswith(".weight") and any(n.endswith(f".{p}.weight")
                                                              for p in PROJECTIONS)}:
        raise ValueError("projection inventory incomplete")
    for role in ("anchor", "target"):
        for name in artifact.manifest["tensors"]:
            if name in operands:
                parent, attr = _attribute(models[role], name.removesuffix(".weight"))
                if not isinstance(getattr(parent, attr), FamilyLinear):
                    raise ValueError(f"unreplaced linear placeholder: {role}:{name}")
            elif name.endswith(".bias") and any(
                    name.endswith(f".{projection}.bias") for projection in PROJECTIONS):
                parent, attr = _attribute(models[role], name.removesuffix(".bias"))
                if getattr(parent, attr).bias is not native[(role, name)]:
                    raise ValueError(f"unreplaced linear bias placeholder: {role}:{name}")
            else:
                parent, attr = _attribute(models[role], name)
                if getattr(parent, attr) is not native[(role, name)]:
                    raise ValueError(f"unreplaced native placeholder: {role}:{name}")
    return RuntimeFamily(artifact, mode, models, configs, operands, native)


def _evaluate_cache(cache: list[Any], logits: mx.array) -> None:
    state = [array for entry in cache for array in entry.state if array is not None]
    mx.eval(logits, *state)


def prefill(runtime: RuntimeFamily, model_id: ModelId,
            token_ids: list[int]) -> tuple[mx.array, RequestState]:
    if model_id not in runtime.models or not token_ids:
        raise ValueError("prefill needs an active model and at least one token")
    model = runtime.models[model_id]
    cache = make_prompt_cache(model)
    inputs = mx.array([token_ids], dtype=mx.int32)
    logits = model(inputs, cache=cache)[0, -1, :]
    _evaluate_cache(cache, logits)
    return logits, RequestState(model_id, cache, len(token_ids))


def step_single(runtime: RuntimeFamily, state: RequestState,
                token_id: int) -> mx.array:
    if not state.active or state.model_id not in runtime.models:
        raise ValueError("request state is inactive or belongs to another runtime")
    logits = runtime.models[state.model_id](
        mx.array([[token_id]], dtype=mx.int32), cache=state.cache
    )[0, -1, :]
    _evaluate_cache(state.cache, logits)
    state.position += 1
    return logits


def release_request(state: RequestState) -> None:
    state.cache.clear()
    state.active = False
