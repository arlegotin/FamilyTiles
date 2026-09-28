"""Stock operator and exact staged decode paths for independent Qwen2 streams."""

import numpy as np
import pytest


pytestmark = pytest.mark.metal


@pytest.fixture
def tiny_stock_runtime():
    import mlx.core as mx
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from familytiles.model import RuntimeFamily

    config = {"model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
              "num_hidden_layers": 1, "num_attention_heads": 2,
              "num_key_value_heads": 1, "vocab_size": 32,
              "tie_word_embeddings": True, "rms_norm_eps": 1e-6}
    models = {role: Model(ModelArgs.from_dict(config)) for role in ("anchor", "target")}
    for model in models.values():
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
    return RuntimeFamily(None, "native_pair", models,
                         {role: config.copy() for role in models}, {}, {})


def _words(value):
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.view(mx.uint16)).copy()


def test_native_lockstep_matches_separate_stock_with_unequal_cache_positions(tiny_stock_runtime):
    from familytiles.model import prefill, step_pair, step_single

    runtime = tiny_stock_runtime
    _, anchor = prefill(runtime, "anchor", [1, 2, 3, 4])
    _, target = prefill(runtime, "target", [5, 6])
    _, anchor_ref = prefill(runtime, "anchor", [1, 2, 3, 4])
    _, target_ref = prefill(runtime, "target", [5, 6])
    actual = step_pair(runtime, (anchor, target), (7, 8))
    expected = (step_single(runtime, anchor_ref, 7),
                step_single(runtime, target_ref, 8))
    assert [np.count_nonzero(_words(a) != _words(b)) for a, b in zip(actual, expected)] == [0, 0]
    assert anchor.position == anchor_ref.position == 5
    assert target.position == target_ref.position == 3
    assert anchor.cache is not target.cache


def test_packed_decode_native_replays_cached_stock_without_retained_matrices(tiny_stock_runtime):
    import mlx.core as mx
    from familytiles.codec import CodecPolicy, encode_tensor
    from familytiles.metal import device_operand_from_words
    from familytiles.model import (FamilyLinear, PROJECTIONS, _attribute, prefill,
                                   release_request, step_pair, step_single)

    runtime = tiny_stock_runtime
    prompts = ([1, 2, 3], [4, 5])
    tokens = [(6, 7), (8, 9), (10, 11)]
    _, stock_anchor = prefill(runtime, "anchor", prompts[0])
    _, stock_target = prefill(runtime, "target", prompts[1])
    expected = []
    for pair in tokens:
        expected.append(tuple(_words(step_single(runtime, state, token))
                              for state, token in zip((stock_anchor, stock_target), pair)))
    release_request(stock_anchor)
    release_request(stock_target)

    policy = CodecPolicy(1, 64, ("ordered_delta",))
    for name in list(runtime.models["anchor"].named_modules()):
        module_name, module = name
        if module_name.rsplit(".", 1)[-1] not in PROJECTIONS:
            continue
        modules = {role: getattr(*_attribute(runtime.models[role], module_name))
                   for role in ("anchor", "target")}
        anchor_words = _words(modules["anchor"].weight)
        target_words = _words(modules["target"].weight)
        operand = device_operand_from_words(anchor_words, encode_tensor(
            anchor_words, target_words, policy, modes=("packed",)))
        runtime.operands[module_name + ".weight"] = operand
        for role in ("anchor", "target"):
            parent, attribute = _attribute(runtime.models[role], module_name)
            setattr(parent, attribute, FamilyLinear(operand, role,
                                                   modules[role].bias if hasattr(modules[role], "bias") else None,
                                                   "A1"))
    runtime.mode = "A1"
    _, anchor = prefill(runtime, "anchor", prompts[0])
    _, target = prefill(runtime, "target", prompts[1])
    for pair, reference in zip(tokens, expected):
        actual = step_pair(runtime, (anchor, target), pair)
        assert [np.count_nonzero(_words(a) != b) for a, b in zip(actual, reference)] == [0, 0]
        assert all(module.live_decoded_matrices == 0
                   for role in ("anchor", "target")
                   for _, module in runtime.models[role].named_modules()
                   if isinstance(module, FamilyLinear))
    release_request(target)
    survivor = step_single(runtime, anchor, 12)
    assert survivor.shape == (32,)
    mx.eval(survivor)


def test_fused_lockstep_dispatch_matches_raw_pair_schedule(tiny_stock_runtime):
    from familytiles.model import (prefill, step_pair, wrap_raw_pair_models)

    runtime = tiny_stock_runtime
    runtime.operands, runtime.native_arrays = wrap_raw_pair_models(runtime.models, runtime.configs)
    runtime.mode = "raw_pair"
    _, a = prefill(runtime, "anchor", [1, 2, 3])
    _, t = prefill(runtime, "target", [4, 5])
    raw = tuple(_words(value) for value in step_pair(runtime, (a, t), (6, 7)))
    runtime.mode = "family_single"
    _, a = prefill(runtime, "anchor", [1, 2, 3])
    _, t = prefill(runtime, "target", [4, 5])
    fused = tuple(_words(value) for value in step_pair(runtime, (a, t), (6, 7)))
    assert [np.count_nonzero(x != y) for x, y in zip(raw, fused)] == [0, 0]


def test_mixed_native_and_exact_staged_dispatch_matches_stock(tiny_stock_runtime):
    import mlx.core as mx
    from familytiles.codec import CodecPolicy, encode_tensor
    from familytiles.metal import device_operand_from_words
    from familytiles.model import (FamilyLinear, PROJECTIONS, _attribute,
                                   prefill, step_pair, step_single)

    runtime = tiny_stock_runtime
    prompts = ([1, 2, 3], [4, 5])
    _, stock_a = prefill(runtime, "anchor", prompts[0])
    _, stock_t = prefill(runtime, "target", prompts[1])
    expected = tuple(_words(step_single(runtime, state, token))
                     for state, token in zip((stock_a, stock_t), (6, 7)))
    selected = "model.layers.0.self_attn.q_proj.weight"
    module_name = selected.removesuffix(".weight")
    modules = {role: getattr(*_attribute(runtime.models[role], module_name))
               for role in ("anchor", "target")}
    a_words, t_words = (_words(modules[role].weight) for role in ("anchor", "target"))
    operand = device_operand_from_words(a_words, encode_tensor(
        a_words, t_words, CodecPolicy(1, 64, ("ordered_delta",)), modes=("packed",)))
    runtime.operands[selected] = operand
    for role in ("anchor", "target"):
        parent, attribute = _attribute(runtime.models[role], module_name)
        setattr(parent, attribute, FamilyLinear(operand, role, modules[role].bias, "A1"))
    runtime.mode = "mixed"
    runtime.native_promoted = frozenset(
        f"model.layers.0.{branch}.{projection}.weight"
        for projection in PROJECTIONS if projection != "q_proj"
        for branch in (("self_attn",) if projection in PROJECTIONS[:4] else ("mlp",)))
    _, a = prefill(runtime, "anchor", prompts[0])
    _, t = prefill(runtime, "target", prompts[1])
    actual = tuple(_words(value) for value in step_pair(runtime, (a, t), (6, 7)))
    assert [np.count_nonzero(x != y) for x, y in zip(actual, expected)] == [0, 0]
    mx.eval(*[value for state in (a, t) for entry in state.cache for value in entry.state])
