"""Validated MLX Metal operands and exact uint16 reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .codec import CodecPolicy, EncodedTensor, validate_encoded
from .convert import FamilyArtifact, _hash_words, _safe_path
from .records import write_json_atomic


DECODE_HELPERS = r"""
template <typename P>
inline uint ft_read_u32(P payload, uint pos) {
    return uint(payload[pos]) | (uint(payload[pos + 1]) << 8) |
           (uint(payload[pos + 2]) << 16) | (uint(payload[pos + 3]) << 24);
}
template <typename P>
inline uint ft_read_u16(P payload, uint pos) {
    return uint(payload[pos]) | (uint(payload[pos + 1]) << 8);
}
template <typename D, typename P>
inline uint ft_target_word_from_anchor(uint a, D descriptors, P payload,
                                       uint row, uint column) {
    uint tile = row * TILES_PER_ROW + column / BLOCK_VALUES;
    uint lane = column % BLOCK_VALUES;
    uint offset = descriptors[2 * tile] * 4;
    uint field = descriptors[2 * tile + 1];
    uint mode = field & 3;
    if (mode == 1) return ft_read_u16(payload, offset + 2 * lane);
    if (mode == 0) return a;
    uint width_selector = (field >> 2) & 3;
    uint width = width_selector == 0 ? 0 : (1u << width_selector);
    uint count = (field >> 5) & 511;
    uint valid = min(uint(BLOCK_VALUES), uint(K_VALUES) - (column / BLOCK_VALUES) * BLOCK_VALUES);
    uint low_bytes = ((valid * width + 31) / 32) * 4;
    uint code = 0;
    if (width != 0) {
        uint bit = lane * width;
        code = (uint(payload[offset + bit / 8]) >> (bit % 8)) & ((1u << width) - 1u);
    }
    if (count != 0) {
        uint mask_base = offset + low_bytes;
        uint mask_word = ft_read_u32(payload, mask_base + (lane / 32) * 4);
        uint lower = lane % 32;
        if ((mask_word >> lower) & 1u) {
            uint rank = 0;
            for (uint j = 0; j < lane / 32; ++j)
                rank += popcount(ft_read_u32(payload, mask_base + 4 * j));
            if (lower != 0) rank += popcount(mask_word & ((1u << lower) - 1u));
            uint mask_bytes = ((BLOCK_VALUES + 31) / 32) * 4;
            return ft_read_u16(payload, mask_base + mask_bytes + 2 * rank);
        }
    }
    if (((field >> 4) & 1u) == 0) return a ^ code;
    uint key = (a & 0x8000u) != 0 ? ((~a) & 0xFFFFu) : (a ^ 0x8000u);
    int delta = (code & 1u) != 0 ? -int((code + 1u) / 2u) : int(code / 2u);
    uint target_key = uint(int(key) + delta);
    return (target_key & 0x8000u) != 0 ? (target_key ^ 0x8000u) : ((~target_key) & 0xFFFFu);
}
template <typename A, typename D, typename P>
inline uint ft_target_word(A anchor, D descriptors, P payload,
                           uint row, uint column) {
    uint tile = row * TILES_PER_ROW + column / BLOCK_VALUES;
    uint field = descriptors[2 * tile + 1];
    if ((field & 3u) == 1u)
        return ft_read_u16(payload, descriptors[2 * tile] * 4 +
                           2 * (column % BLOCK_VALUES));
    uint a = uint(anchor[row * K_VALUES + column]);
    return ft_target_word_from_anchor(a, descriptors, payload, row, column);
}
"""


@dataclass(frozen=True)
class DeviceOperand:
    kind: str
    shape: tuple[int, int]
    policy: CodecPolicy
    anchor_words: Any
    descriptors: Any | None
    payload: Any | None
    native_words: Any | None
    storage_ids: dict[str, int]


def device_operand_from_words(anchor: np.ndarray, encoded: EncodedTensor) -> DeviceOperand:
    """Validate host bytes before allocating any GPU operand."""
    validate_encoded(anchor, encoded)
    if len(encoded.shape) != 2 or min(encoded.shape) <= 0:
        raise ValueError("Metal decoder requires positive two-dimensional shape")
    import mlx.core as mx

    a = mx.array(np.asarray(anchor), dtype=mx.uint16)
    if encoded.kind == "native":
        native = mx.array(encoded.native_words, dtype=mx.uint16)
        desc = payload = None
    else:
        desc = mx.array(encoded.descriptors, dtype=mx.uint32)
        payload = mx.array(encoded.payload, dtype=mx.uint8)
        native = None
    mx.eval(*(x for x in (a, desc, payload, native) if x is not None))
    return DeviceOperand(encoded.kind, encoded.shape, encoded.policy, a, desc, payload,
                         native, {"anchor": id(a), "descriptors": id(desc),
                                  "payload": id(payload), "native": id(native)})


def load_operand(artifact: FamilyArtifact, tensor_name: str) -> DeviceOperand:
    """Load one checked operand without retaining source NumPy matrices."""
    record = artifact.manifest["tensors"].get(tensor_name)
    if record is None or len(record["shape"]) != 2:
        raise ValueError("operand is not an active linear matrix")
    shape = tuple(record["shape"])
    a_info, target = record["anchor"], record["target"]
    anchor_path = _safe_path(artifact.root, a_info["path"])
    a = np.memmap(anchor_path, mode="r", dtype="<u2", offset=a_info["offset"], shape=shape)
    if _hash_words(a) != a_info["sha256"]:
        raise ValueError("anchor tensor hash mismatch before Metal allocation")
    policy = CodecPolicy(**{**artifact.manifest["policy"],
                            "transforms": tuple(artifact.manifest["policy"]["transforms"])})
    if target["kind"] == "alias":
        # COPY all tensor words without a second allocation.
        import mlx.core as mx
        anchor_device = mx.array(a, dtype=mx.uint16)
        mx.eval(anchor_device)
        return DeviceOperand("alias", shape, policy, anchor_device, None, None, None,
                             {"anchor": id(anchor_device)})
    if target["kind"] == "native":
        path = _safe_path(artifact.root, target["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != target["file_sha256"]:
            raise ValueError("native target hash mismatch before Metal allocation")
        words = np.memmap(path, mode="r", dtype="<u2", shape=shape)
        return device_operand_from_words(a, EncodedTensor("native", shape, policy, None, None, words))
    if target["kind"] != "packed":
        raise ValueError("unsupported target operand kind")
    descriptor_path = _safe_path(artifact.root, target["descriptor_path"])
    payload_path = _safe_path(artifact.root, target["payload_path"])
    with descriptor_path.open("rb") as stream:
        desc_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with payload_path.open("rb") as stream:
        payload_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if desc_hash != target["descriptor_sha256"] or payload_hash != target["payload_sha256"]:
        raise ValueError("packed operand checksum mismatch before Metal allocation")
    desc = np.load(descriptor_path, mmap_mode="r", allow_pickle=False)
    payload = np.load(payload_path, mmap_mode="r", allow_pickle=False)
    return device_operand_from_words(a, EncodedTensor("packed", shape, policy, desc, payload, None))


def decode_words(operand: DeviceOperand, row_start: int = 0,
                 row_count: int | None = None):
    import mlx.core as mx

    rows, columns = operand.shape
    if row_count is None:
        row_count = rows - row_start
    if (row_start < 0 or row_count <= 0 or row_start + row_count > rows
            or columns <= 0 or operand.kind not in ("alias", "native", "packed")):
        raise ValueError("invalid Metal row range or operand")
    if operand.kind == "alias":
        return operand.anchor_words[row_start:row_start + row_count]
    if operand.kind == "native":
        return operand.native_words[row_start:row_start + row_count]
    header = (f"#define K_VALUES {columns}\n#define BLOCK_VALUES {operand.policy.block_values}\n"
              f"#define TILES_PER_ROW {(columns + operand.policy.block_values - 1) // operand.policy.block_values}\n"
              f"#define ROW_START {row_start}\n#define OUT_VALUES {row_count * columns}\n"
              + DECODE_HELPERS)
    source = """
        uint i = thread_position_in_grid.x;
        if (i >= OUT_VALUES) return;
        uint row = ROW_START + i / K_VALUES;
        uint column = i % K_VALUES;
        out[i] = ushort(ft_target_word(anchor, descriptors, payload, row, column));
    """
    kernel = mx.fast.metal_kernel(
        name="familytiles_decode_words_v1", input_names=["anchor", "descriptors", "payload"],
        output_names=["out"], source=source, header=header,
        ensure_row_contiguous=False, compile_options={"math_mode": "safe"},
    )
    return kernel(inputs=[operand.anchor_words, operand.descriptors, operand.payload],
                  template=[], grid=(row_count * columns, 1, 1),
                  threadgroup=(min(256, row_count * columns), 1, 1),
                  output_shapes=[(row_count, columns)], output_dtypes=[mx.uint16])[0]


def probe_anchor_view() -> dict[str, int | bool]:
    """Check whether the uint16/BF16 bit view retains only one full allocation."""
    import mlx.core as mx

    anchor = mx.zeros((4096, 2048), dtype=mx.uint16)
    mx.eval(anchor)
    before = int(mx.get_active_memory())
    bf16 = anchor.view(mx.bfloat16)
    mx.eval(bf16)
    after = int(mx.get_active_memory())
    additional = after - before
    return {"anchor_bytes": int(anchor.nbytes), "active_before_bytes": before,
            "active_after_bytes": after, "additional_bytes": additional,
            "no_full_copy": additional < 2**20}


GEMV_LANES = 32
GEMV_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint row = index / GEMV_LANES;
    uint lane = index % GEMV_LANES;
    float sum = 0.0f;
    for (uint column = lane; column < K_VALUES; column += GEMV_LANES) {
        uint weight_word = WEIGHT_ACCESS;
        float weight = as_type<float>(weight_word << 16);
        float activation = float(x[column * x_strides[0]]);
        sum += weight * activation;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        if (HAS_BIAS) sum += float(bias[row * bias_strides[0]]);
        out[row] = bfloat16_t(sum);
    }
"""
PAIR_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint row = index / GEMV_LANES;
    uint lane = index % GEMV_LANES;
    float sum_anchor = 0.0f;
    float sum_target = 0.0f;
    for (uint column = lane; column < K_VALUES; column += GEMV_LANES) {
        uint anchor_word = ANCHOR_ACCESS;
        uint target_word = TARGET_ACCESS;
        float anchor_weight = as_type<float>(anchor_word << 16);
        float target_weight = as_type<float>(target_word << 16);
        float activation_anchor = float(x_anchor[column * x_anchor_strides[0]]);
        float activation_target = float(x_target[column * x_target_strides[0]]);
        sum_anchor += anchor_weight * activation_anchor;
        sum_target += target_weight * activation_target;
    }
    sum_anchor = simd_sum(sum_anchor);
    sum_target = simd_sum(sum_target);
    if (lane == 0) {
        if (HAS_ANCHOR_BIAS)
            sum_anchor += float(bias_anchor[row * bias_anchor_strides[0]]);
        if (HAS_TARGET_BIAS)
            sum_target += float(bias_target[row * bias_target_strides[0]]);
        out_anchor[row] = bfloat16_t(sum_anchor);
        out_target[row] = bfloat16_t(sum_target);
    }
"""


def _gemv_kernel(*, packed: bool, rows: int, columns: int,
                 block_values: int, has_bias: bool):
    import mlx.core as mx

    if packed:
        input_names = ["anchor", "descriptors", "payload", "x", "bias"]
        accessor = "ft_target_word(anchor, descriptors, payload, row, column)"
        helpers = DECODE_HELPERS
    else:
        input_names = ["weights", "x", "bias"]
        accessor = "uint(weights[row * weights_strides[0] + column * weights_strides[1]])"
        helpers = ""
    header = (f"#define K_VALUES {columns}\n#define BLOCK_VALUES {block_values}\n"
              f"#define TILES_PER_ROW {(columns + block_values - 1) // block_values}\n"
              f"#define GEMV_LANES {GEMV_LANES}\n#define HAS_BIAS {int(has_bias)}\n"
              + helpers)
    kernel = mx.fast.metal_kernel(
        name="familytiles_single_gemv_v1", input_names=input_names,
        output_names=["out"], source=GEMV_SOURCE.replace("WEIGHT_ACCESS", accessor),
        header=header,
        ensure_row_contiguous=False, compile_options={"math_mode": "safe"},
    )
    return kernel


def _check_gemv_inputs(shape: tuple[int, int], x: Any, bias: Any | None) -> None:
    import mlx.core as mx

    if (len(shape) != 2 or min(shape) <= 0 or x.ndim != 1
            or x.shape[0] != shape[1] or x.dtype != mx.bfloat16):
        raise ValueError("GEMV requires a positive matrix and one BF16 activation vector")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != shape[0]
                             or bias.dtype != mx.bfloat16):
        raise ValueError("GEMV bias must be a BF16 output vector")


def raw_gemv(words: Any, x: Any, bias: Any | None = None):
    """BF16 matvec control with the packed kernel's reduction schedule."""
    import mlx.core as mx

    if words.ndim != 2 or words.dtype != mx.uint16:
        raise ValueError("raw GEMV weights must be a uint16 matrix")
    _check_gemv_inputs(tuple(words.shape), x, bias)
    rows, columns = words.shape
    kernel = _gemv_kernel(packed=False, rows=rows, columns=columns,
                          block_values=128, has_bias=bias is not None)
    return kernel(inputs=[words, x, bias if bias is not None else x], template=[],
                  grid=(rows * GEMV_LANES, 1, 1), threadgroup=(GEMV_LANES, 1, 1),
                  output_shapes=[(rows,)], output_dtypes=[mx.bfloat16])[0]


def family_gemv(operand: DeviceOperand, x: Any, bias: Any | None = None):
    """Consume exact encoded words in a single output-sized Metal operation."""
    import mlx.core as mx

    _check_gemv_inputs(operand.shape, x, bias)
    if operand.kind in ("native", "alias"):
        words = operand.native_words if operand.kind == "native" else operand.anchor_words
        return raw_gemv(words, x, bias)
    if operand.kind != "packed":
        raise ValueError("unknown GEMV operand kind")
    rows, columns = operand.shape
    kernel = _gemv_kernel(packed=True, rows=rows, columns=columns,
                          block_values=operand.policy.block_values,
                          has_bias=bias is not None)
    return kernel(inputs=[operand.anchor_words, operand.descriptors, operand.payload,
                          x, bias if bias is not None else x], template=[],
                  grid=(rows * GEMV_LANES, 1, 1), threadgroup=(GEMV_LANES, 1, 1),
                  output_shapes=[(rows,)], output_dtypes=[mx.bfloat16])[0]


def _pair_kernel(*, kind: str, columns: int, block_values: int,
                 has_anchor_bias: bool, has_target_bias: bool):
    import mlx.core as mx

    if kind == "raw":
        names = ["anchor", "target", "x_anchor", "x_target",
                 "bias_anchor", "bias_target"]
        anchor_access = "uint(anchor[row * anchor_strides[0] + column * anchor_strides[1]])"
        target_access = "uint(target[row * target_strides[0] + column * target_strides[1]])"
        helpers = ""
    elif kind == "packed":
        names = ["anchor", "descriptors", "payload", "x_anchor", "x_target",
                 "bias_anchor", "bias_target"]
        anchor_access = "uint(anchor[row * K_VALUES + column])"
        target_access = "ft_target_word_from_anchor(anchor_word, descriptors, payload, row, column)"
        helpers = DECODE_HELPERS
    elif kind == "alias":
        names = ["anchor", "x_anchor", "x_target", "bias_anchor", "bias_target"]
        anchor_access = "uint(anchor[row * K_VALUES + column])"
        target_access = "anchor_word"
        helpers = ""
    else:
        raise ValueError("unknown paired GEMV kind")
    header = (f"#define K_VALUES {columns}\n#define BLOCK_VALUES {block_values}\n"
              f"#define TILES_PER_ROW {(columns + block_values - 1) // block_values}\n"
              f"#define GEMV_LANES {GEMV_LANES}\n"
              f"#define HAS_ANCHOR_BIAS {int(has_anchor_bias)}\n"
              f"#define HAS_TARGET_BIAS {int(has_target_bias)}\n" + helpers)
    source = (PAIR_SOURCE.replace("ANCHOR_ACCESS", anchor_access)
             .replace("TARGET_ACCESS", target_access))
    return mx.fast.metal_kernel(
        name="familytiles_pair_gemv_v1", input_names=names,
        output_names=["out_anchor", "out_target"], source=source, header=header,
        ensure_row_contiguous=False, compile_options={"math_mode": "safe"},
    )


def _check_pair_inputs(shape: tuple[int, int], x_anchor: Any, x_target: Any,
                       biases: tuple[Any | None, Any | None]) -> None:
    if not isinstance(biases, tuple) or len(biases) != 2:
        raise ValueError("paired GEMV requires exactly two biases or None values")
    _check_gemv_inputs(shape, x_anchor, biases[0])
    _check_gemv_inputs(shape, x_target, biases[1])


def raw_gemv_pair(anchor: Any, target: Any, x_anchor: Any, x_target: Any,
                  biases: tuple[Any | None, Any | None] = (None, None)):
    """Grouped raw control with exactly the paired packed arithmetic schedule."""
    import mlx.core as mx

    if (anchor.ndim != 2 or anchor.dtype != mx.uint16 or target.ndim != 2
            or target.dtype != mx.uint16 or anchor.shape != target.shape):
        raise ValueError("paired raw weights must be equal-shape uint16 matrices")
    shape = tuple(anchor.shape)
    _check_pair_inputs(shape, x_anchor, x_target, biases)
    rows, columns = shape
    kernel = _pair_kernel(kind="raw", columns=columns, block_values=128,
                          has_anchor_bias=biases[0] is not None,
                          has_target_bias=biases[1] is not None)
    return tuple(kernel(inputs=[anchor, target, x_anchor, x_target,
                                biases[0] if biases[0] is not None else x_anchor,
                                biases[1] if biases[1] is not None else x_target],
                        template=[], grid=(rows * GEMV_LANES, 1, 1),
                        threadgroup=(GEMV_LANES, 1, 1),
                        output_shapes=[(rows,), (rows,)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16]))


def family_gemv_pair(operand: DeviceOperand, x_anchor: Any, x_target: Any,
                     biases: tuple[Any | None, Any | None] = (None, None)):
    """Reuse each loaded anchor word for independent anchor/target outputs."""
    import mlx.core as mx

    _check_pair_inputs(operand.shape, x_anchor, x_target, biases)
    if operand.kind == "native":
        return raw_gemv_pair(operand.anchor_words, operand.native_words,
                             x_anchor, x_target, biases)
    if operand.kind not in ("packed", "alias"):
        raise ValueError("unknown paired operand kind")
    rows, columns = operand.shape
    kernel = _pair_kernel(kind=operand.kind, columns=columns,
                          block_values=operand.policy.block_values,
                          has_anchor_bias=biases[0] is not None,
                          has_target_bias=biases[1] is not None)
    if operand.kind == "packed":
        inputs = [operand.anchor_words, operand.descriptors, operand.payload,
                  x_anchor, x_target]
    else:
        inputs = [operand.anchor_words, x_anchor, x_target]
    inputs.extend((biases[0] if biases[0] is not None else x_anchor,
                   biases[1] if biases[1] is not None else x_target))
    return tuple(kernel(inputs=inputs, template=[], grid=(rows * GEMV_LANES, 1, 1),
                        threadgroup=(GEMV_LANES, 1, 1),
                        output_shapes=[(rows,), (rows,)],
                        output_dtypes=[mx.bfloat16, mx.bfloat16]))


def kernel_config() -> dict[str, Any]:
    return {"layout_version": 1, "lanes_per_row": GEMV_LANES,
            "accumulation": "fp32", "output": "bf16",
            "math_mode": "safe", "partial_sum_bytes": 0,
            "single_source_sha256": hashlib.sha256(
                (GEMV_SOURCE + DECODE_HELPERS +
                 "ft_target_word(anchor, descriptors, payload, row, column)" +
                 "weights[row * weights_strides[0] + column * weights_strides[1]]").encode()
            ).hexdigest(),
            "pair_source_sha256": hashlib.sha256(
                (PAIR_SOURCE + DECODE_HELPERS +
                 "ft_target_word_from_anchor(anchor_word, descriptors, payload, row, column)" +
                 "target[row * target_strides[0] + column * target_strides[1]]").encode()
            ).hexdigest()}


def verify_gpu(artifact: FamilyArtifact) -> dict[str, Any]:
    import mlx.core as mx
    import psutil
    from .measure import compute_budget

    memory = psutil.virtual_memory()
    budget = compute_budget(memory.total, memory.available)
    mx.set_cache_limit(budget.cache_limit_bytes)
    mx.set_memory_limit(budget.process_limit_bytes)
    tensor_count = word_count = mismatches = 0
    for name, record in artifact.manifest["tensors"].items():
        if record["target"]["kind"] != "packed":
            continue
        operand = load_operand(artifact, name)
        rows, columns = operand.shape
        chunk_rows = max(1, min(rows, (16 * 2**20) // (2 * columns)))
        digest = hashlib.sha256()
        for start in range(0, rows, chunk_rows):
            output = decode_words(operand, start, min(chunk_rows, rows - start))
            mx.eval(output)
            values = np.asarray(output)
            digest.update(memoryview(values).cast("B"))
            word_count += values.size
        if digest.hexdigest() != record["target"]["sha256"]:
            mismatches += 1
        tensor_count += 1
    result = {"artifact_hash": artifact.manifest_hash,
              "verified_encoded_tensors": tensor_count,
              "verified_encoded_words": word_count,
              "hash_mismatched_tensors": mismatches,
              "representation_exact": mismatches == 0,
              "anchor_view_probe": probe_anchor_view(),
              "safe_process_budget_bytes": budget.process_limit_bytes}
    root = Path(__file__).resolve().parents[2]
    write_json_atomic(root / "results/correctness/gpu.json", result)
    return result
