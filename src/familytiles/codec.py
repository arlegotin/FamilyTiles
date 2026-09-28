"""Reversible BF16 word transforms and independently addressed patch blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WORD_DTYPE = np.dtype("<u2")
TRANSFORMS = frozenset({"xor", "ordered_delta"})
WIDTHS = (0, 2, 4, 8)
MODE_IDS = {"copy": 0, "raw": 1, "packed": 2}


@dataclass(frozen=True)
class CodecPolicy:
    version: int
    block_values: int
    transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != 1 or self.block_values not in (64, 128, 256):
            raise ValueError("unsupported codec version or block size")
        if (not isinstance(self.transforms, tuple) or not self.transforms
                or len(set(self.transforms)) != len(self.transforms)
                or not set(self.transforms) <= TRANSFORMS):
            raise ValueError("invalid transform menu")


@dataclass(frozen=True)
class EncodedTensor:
    """One row-major uint16 tensor; packed arrays are the only retained tile state."""

    kind: str
    shape: tuple[int, int]
    policy: CodecPolicy
    descriptors: np.ndarray | None
    payload: np.ndarray | None
    native_words: np.ndarray | None


def _words(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != WORD_DTYPE:
        raise ValueError("expected little-endian uint16 words")
    return array


def ordered_key(words: np.ndarray) -> np.ndarray:
    original = _words(words)
    return np.where((original & 0x8000) != 0, (~original) & 0xFFFF,
                    original ^ 0x8000).astype(WORD_DTYPE, copy=False)


def inverse_key(keys: np.ndarray) -> np.ndarray:
    coded = _words(keys)
    return np.where((coded & 0x8000) != 0, coded ^ 0x8000,
                    (~coded) & 0xFFFF).astype(WORD_DTYPE, copy=False)


def encode_residual(anchor: np.ndarray, target: np.ndarray, transform: str) -> np.ndarray:
    a, t = _words(anchor), _words(target)
    if a.shape != t.shape:
        raise ValueError("anchor/target shape mismatch")
    if transform == "xor":
        return np.bitwise_xor(a, t).astype(WORD_DTYPE, copy=False)
    if transform != "ordered_delta":
        raise ValueError("unsupported residual transform")
    difference = ordered_key(t).astype(np.int32) - ordered_key(a).astype(np.int32)
    return np.where(difference >= 0, 2 * difference,
                    -2 * difference - 1).astype(np.uint32)


def decode_residual(anchor: np.ndarray, residual: np.ndarray, transform: str) -> np.ndarray:
    a = _words(anchor)
    coded = np.asarray(residual)
    if a.shape != coded.shape or not np.issubdtype(coded.dtype, np.integer):
        raise ValueError("residual shape or dtype mismatch")
    if transform == "xor":
        if np.any(coded < 0) or np.any(coded > 0xFFFF):
            raise ValueError("XOR residual outside uint16 range")
        return np.bitwise_xor(a, coded.astype(WORD_DTYPE))
    if transform != "ordered_delta":
        raise ValueError("unsupported residual transform")
    if np.any(coded < 0) or np.any(coded > 131070):
        raise ValueError("ordered residual outside 17-bit range")
    z = coded.astype(np.int64)
    difference = np.where((z & 1) == 0, z // 2, -(z + 1) // 2)
    reconstructed_key = ordered_key(a).astype(np.int64) + difference
    if np.any(reconstructed_key < 0) or np.any(reconstructed_key > 0xFFFF):
        raise ValueError("ordered residual reconstructs an invalid key")
    return inverse_key(reconstructed_key.astype(WORD_DTYPE))


def _matrix(words: np.ndarray) -> np.ndarray:
    array = _words(words)
    if array.ndim != 2 or not array.flags.c_contiguous:
        raise ValueError("expected a contiguous row-major matrix")
    return array


def _pad4(data: bytes) -> bytes:
    return data + bytes((-len(data)) % 4)


def _tile_payload(residual: np.ndarray, target: np.ndarray, width: int,
                  block: int) -> tuple[bytes, int]:
    exceptional = residual >= (1 << width)
    count = int(np.count_nonzero(exceptional))
    parts = []
    if width:
        lanes_per_byte = 8 // width
        padded_lanes = ((len(target) + lanes_per_byte - 1) // lanes_per_byte) * lanes_per_byte
        codes = np.zeros(padded_lanes, dtype=np.uint8)
        codes[:len(target)] = np.where(exceptional, 0, residual).astype(np.uint8)
        grouped = codes.reshape((-1, lanes_per_byte)).astype(np.uint16)
        packed_bytes = np.bitwise_or.reduce(
            grouped << (np.arange(lanes_per_byte, dtype=np.uint16) * width), axis=1
        ).astype(np.uint8)
        parts.append(_pad4(packed_bytes.tobytes()))
    if count:
        mask_lanes = np.zeros(block, dtype=np.uint8)
        mask_lanes[:len(target)] = exceptional
        parts.extend((np.packbits(mask_lanes, bitorder="little").tobytes(),
                      target[exceptional].astype(WORD_DTYPE).tobytes()))
    return _pad4(b"".join(parts)), count


def encode_tensor(anchor: np.ndarray, target: np.ndarray, policy: CodecPolicy, *,
                  modes: tuple[str, ...] = ("copy", "raw", "packed")) -> EncodedTensor:
    """Choose the cheapest exact block mode, then a whole-tensor native fallback."""
    a, t = _matrix(anchor), _matrix(target)
    if a.shape != t.shape:
        raise ValueError("anchor/target shape mismatch")
    if not isinstance(policy, CodecPolicy) or not modes or len(set(modes)) != len(modes) or not set(modes) <= MODE_IDS.keys():
        raise ValueError("invalid codec policy or mode menu")
    descriptors: list[tuple[int, int]] = []
    payload = bytearray()
    for row in range(a.shape[0]):
        for start in range(0, a.shape[1], policy.block_values):
            stop = min(start + policy.block_values, a.shape[1])
            aa, tt = a[row, start:stop], t[row, start:stop]
            choices: list[tuple[int, int, int, bytes]] = []
            if "copy" in modes and np.array_equal(aa, tt):
                choices.append((8, 0, 0, b""))
            if "raw" in modes:
                raw = _pad4(tt.tobytes())
                choices.append((8 + len(raw), 1, 1, raw))
            if "packed" in modes:
                for transform_id, transform in enumerate(("xor", "ordered_delta")):
                    if transform not in policy.transforms:
                        continue
                    residual = encode_residual(aa, tt, transform)
                    for width_id, width in enumerate(WIDTHS):
                        data, count = _tile_payload(residual, tt, width, policy.block_values)
                        field = 2 | (width_id << 2) | (transform_id << 4) | (count << 5)
                        choices.append((8 + len(data), 2 + 4 * transform_id + width_id, field, data))
            if not choices:
                raise ValueError("no available tile mode")
            _, _, field, data = min(choices, key=lambda item: (item[0], item[1]))
            if len(payload) // 4 > 0xFFFFFFFF or len(payload) + len(data) > (1 << 34):
                raise ValueError("tensor payload exceeds descriptor address range")
            descriptors.append((len(payload) // 4, field))
            payload.extend(data)
    desc = np.asarray(descriptors, dtype="<u4").reshape((-1, 2))
    body = np.frombuffer(bytes(payload), dtype=np.uint8).copy()
    encoded = EncodedTensor("packed", a.shape, policy, desc, body, None)
    # A packed-only mode is useful for testing otherwise uneconomic bit patterns.
    if modes != ("packed",) and allocated_bytes(encoded) >= t.nbytes:
        return EncodedTensor("native", a.shape, policy, None, None, t.copy())
    return encoded


def allocated_bytes(encoded: EncodedTensor) -> int:
    if encoded.kind == "native":
        if encoded.native_words is None:
            raise ValueError("native tensor has no words")
        return int(encoded.native_words.nbytes)
    if encoded.kind != "packed" or encoded.descriptors is None or encoded.payload is None:
        raise ValueError("malformed tensor representation")
    return int(encoded.descriptors.nbytes + encoded.payload.nbytes)


def _sections(encoded: EncodedTensor, anchor: np.ndarray):
    """Yield validated tile records; all bounds are checked before any slice use."""
    if encoded.kind != "packed" or encoded.descriptors is None or encoded.payload is None:
        raise ValueError("malformed packed tensor")
    desc, payload = encoded.descriptors, encoded.payload
    blocks_per_row = (anchor.shape[1] + encoded.policy.block_values - 1) // encoded.policy.block_values
    if (desc.dtype != np.dtype("<u4") or desc.shape != (anchor.shape[0] * blocks_per_row, 2)
            or not desc.flags.c_contiguous or payload.dtype != np.dtype("uint8")
            or payload.ndim != 1 or not payload.flags.c_contiguous or len(payload) % 4):
        raise ValueError("invalid descriptor or payload array")
    cursor = 0
    for tile_index, (offset_word, field_word) in enumerate(desc):
        offset, field = int(offset_word), int(field_word)
        row, block_index = divmod(tile_index, blocks_per_row)
        start = block_index * encoded.policy.block_values
        n = min(encoded.policy.block_values, anchor.shape[1] - start)
        mode, width_id, transform_id = field & 3, (field >> 2) & 3, (field >> 4) & 1
        count = (field >> 5) & 0x1FF
        if offset != cursor // 4 or field >> 14 or mode == 3 or count > n:
            raise ValueError("invalid descriptor field or offset")
        if mode == 0:
            if field != 0:
                raise ValueError("noncanonical COPY descriptor")
            size = 0
        elif mode == 1:
            if field != 1:
                raise ValueError("noncanonical RAW descriptor")
            size = (2 * n + 3) & ~3
        else:
            transform = ("xor", "ordered_delta")[transform_id]
            if transform not in encoded.policy.transforms:
                raise ValueError("disabled transform")
            width = WIDTHS[width_id]
            low_size = 4 * ((n * width + 31) // 32)
            mask_size = 4 * ((encoded.policy.block_values + 31) // 32) if count else 0
            size = (low_size + mask_size + 2 * count + 3) & ~3
        if cursor + size > len(payload):
            raise ValueError("truncated payload")
        yield row, start, n, mode, width_id, transform_id, count, cursor, size
        cursor += size
    if cursor != len(payload):
        raise ValueError("trailing payload bytes")


def _decode_validated(anchor: np.ndarray, encoded: EncodedTensor) -> np.ndarray:
    if encoded.kind == "native":
        return encoded.native_words.copy()
    result = np.empty(encoded.shape, dtype=WORD_DTYPE)
    payload = encoded.payload
    for row, start, n, mode, width_id, transform_id, count, offset, size in _sections(encoded, anchor):
        aa = anchor[row, start:start + n]
        data = payload[offset:offset + size]
        if mode == 0:
            result[row, start:start + n] = aa
        elif mode == 1:
            result[row, start:start + n] = np.frombuffer(data[:2 * n].tobytes(), dtype=WORD_DTYPE)
        else:
            width = WIDTHS[width_id]
            low_size = 4 * ((n * width + 31) // 32)
            codes = np.zeros(n, dtype=np.uint32)
            if width:
                low = np.frombuffer(data[:low_size].tobytes(), dtype="<u4")
                for lane in range(n):
                    codes[lane] = (int(low[lane * width // 32]) >> ((lane * width) % 32)) & ((1 << width) - 1)
            exceptions = np.zeros(n, dtype=bool)
            if count:
                mask_size = 4 * ((encoded.policy.block_values + 31) // 32)
                masks = np.frombuffer(data[low_size:low_size + mask_size].tobytes(), dtype="<u4")
                for lane in range(n):
                    exceptions[lane] = bool(int(masks[lane // 32]) & (1 << (lane % 32)))
                literals = np.frombuffer(data[low_size + mask_size:low_size + mask_size + 2 * count].tobytes(), dtype=WORD_DTYPE)
            normal = ~exceptions
            transform = ("xor", "ordered_delta")[transform_id]
            values = np.empty(n, dtype=WORD_DTYPE)
            values[normal] = decode_residual(aa[normal], codes[normal], transform)
            if count:
                values[exceptions] = literals
            result[row, start:start + n] = values
    return result


def validate_encoded(anchor: np.ndarray, encoded: EncodedTensor) -> None:
    a = _matrix(anchor)
    if not isinstance(encoded, EncodedTensor) or not isinstance(encoded.policy, CodecPolicy) or encoded.shape != a.shape:
        raise ValueError("invalid tensor metadata or anchor shape")
    if encoded.kind == "native":
        if encoded.descriptors is not None or encoded.payload is not None:
            raise ValueError("native tensor contains packed arrays")
        words = _matrix(encoded.native_words)
        if words.shape != a.shape:
            raise ValueError("native shape mismatch")
        return
    if encoded.kind != "packed" or encoded.native_words is not None:
        raise ValueError("invalid packed tensor fields")
    payload = encoded.payload
    for row, start, n, mode, width_id, transform_id, count, offset, size in _sections(encoded, a):
        data = payload[offset:offset + size]
        if mode == 1:
            if np.any(data[2 * n:]):
                raise ValueError("nonzero RAW padding")
            continue
        if mode == 0:
            continue
        width = WIDTHS[width_id]
        low_size = 4 * ((n * width + 31) // 32)
        mask_size = 4 * ((encoded.policy.block_values + 31) // 32) if count else 0
        if low_size and n * width % 32:
            low_words = np.frombuffer(data[:low_size].tobytes(), dtype="<u4")
            if int(low_words[-1]) >> (n * width % 32):
                raise ValueError("nonzero unused low bits")
        masks = np.frombuffer(data[low_size:low_size + mask_size].tobytes(), dtype="<u4")
        bits = sum(int(x).bit_count() for x in masks)
        if bits != count or any(int(masks[lane // 32]) & (1 << (lane % 32)) for lane in range(n, encoded.policy.block_values) if count):
            raise ValueError("exception mask/count or tail mismatch")
        if count and width:
            low = np.frombuffer(data[:low_size].tobytes(), dtype="<u4")
            for lane in range(n):
                if int(masks[lane // 32]) & (1 << (lane % 32)):
                    if (int(low[lane * width // 32]) >> ((lane * width) % 32)) & ((1 << width) - 1):
                        raise ValueError("exception low-code placeholder is nonzero")
        end = low_size + mask_size + 2 * count
        if np.any(data[end:]):
            raise ValueError("nonzero PACKED padding")
    # Also checks ordered-key reconstruction bounds for normal lanes.
    _decode_validated(a, encoded)


def decode_tensor(anchor: np.ndarray, encoded: EncodedTensor) -> np.ndarray:
    validate_encoded(anchor, encoded)
    return _decode_validated(_matrix(anchor), encoded)
