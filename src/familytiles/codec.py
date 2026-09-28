"""Reversible BF16 word transforms. No floating-point arithmetic is used."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WORD_DTYPE = np.dtype("<u2")
TRANSFORMS = frozenset({"xor", "ordered_delta"})


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
