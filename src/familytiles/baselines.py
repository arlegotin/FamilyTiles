"""Portable exact Zstd references with independently decodable chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import zstandard as zstd

from .codec import decode_residual, encode_residual


@dataclass(frozen=True)
class ExactChunks:
    shape: tuple[int, ...]
    word_dtype: str
    transform: str
    chunk_index: np.ndarray  # uint64 rows: payload offset, byte length, value count
    payload: np.ndarray      # one owned uint8 allocation
    native_fallback: bool


@dataclass(frozen=True)
class ExactFamilyBacking:
    anchor: ExactChunks
    target: ExactChunks
    family_transform: str
    anchor_form: str


def _words(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype("<u2"), np.dtype("<u4")):
        raise ValueError("exact chunks require little-endian unsigned words")
    if not array.flags.c_contiguous:
        raise ValueError("exact chunks require contiguous words")
    return array


def _forward(words: np.ndarray, transform: str) -> bytes:
    flattened = words.reshape(-1)
    if transform == "native":
        return flattened.tobytes()
    if transform == "byte_plane":
        bytes_per_word = words.dtype.itemsize
        planes = flattened.view(np.uint8).reshape((-1, bytes_per_word)).T
        return planes.copy(order="C").tobytes()
    if transform == "field" and words.dtype == np.dtype("<u2"):
        exponent = ((flattened >> 7) & 0xFF).astype(np.uint8)
        sign_fraction = (((flattened >> 15) << 7) | (flattened & 0x7F)).astype(np.uint8)
        return exponent.tobytes() + sign_fraction.tobytes()
    raise ValueError("unsupported exact byte transform")


def _inverse(data: bytes, count: int, dtype: np.dtype, transform: str) -> np.ndarray:
    width = dtype.itemsize
    if len(data) != count * width:
        raise ValueError("decoded chunk size mismatch")
    if transform == "native":
        return np.frombuffer(data, dtype=dtype).copy()
    if transform == "byte_plane":
        planes = np.frombuffer(data, dtype=np.uint8).reshape((width, count))
        return np.ascontiguousarray(planes.T).view(dtype).reshape(-1)
    if transform == "field" and width == 2:
        bytes_ = np.frombuffer(data, dtype=np.uint8)
        exponent, sign_fraction = bytes_[:count].astype(np.uint16), bytes_[count:].astype(np.uint16)
        return ((exponent << 7) | ((sign_fraction >> 7) << 15) | (sign_fraction & 0x7F)).astype(dtype)
    raise ValueError("unsupported exact byte transform")


def encode_exact(words: np.ndarray, transform: str, *, chunk_bytes: int = 262144,
                 level: int = 3) -> ExactChunks:
    array = _words(words)
    if transform not in ("native", "byte_plane", "field") or (transform == "field" and array.dtype.itemsize != 2):
        raise ValueError("unsupported exact transform")
    if type(chunk_bytes) is not int or chunk_bytes < array.dtype.itemsize or level != 3:
        raise ValueError("unsupported chunk size or Zstd level")
    native = array.reshape(-1).tobytes()
    if transform == "native" or not native:
        return ExactChunks(array.shape, array.dtype.str, "native", np.empty((0, 3), dtype="<u8"),
                           np.frombuffer(native, dtype=np.uint8).copy(), True)
    compressor = zstd.ZstdCompressor(level=level)
    chunk_values = max(1, chunk_bytes // array.dtype.itemsize)
    data = bytearray()
    index: list[tuple[int, int, int]] = []
    flat = array.reshape(-1)
    for start in range(0, flat.size, chunk_values):
        chunk = flat[start:start + chunk_values]
        coded = compressor.compress(_forward(chunk, transform))
        index.append((len(data), len(coded), len(chunk)))
        data.extend(coded)
    table = np.asarray(index, dtype="<u8").reshape((-1, 3))
    if table.nbytes + len(data) >= len(native):
        return ExactChunks(array.shape, array.dtype.str, "native", np.empty((0, 3), dtype="<u8"),
                           np.frombuffer(native, dtype=np.uint8).copy(), True)
    return ExactChunks(array.shape, array.dtype.str, transform, table,
                       np.frombuffer(bytes(data), dtype=np.uint8).copy(), False)


def decode_exact(encoded: ExactChunks) -> Iterator[np.ndarray]:
    if not isinstance(encoded, ExactChunks) or encoded.word_dtype not in ("<u2", "<u4"):
        raise ValueError("invalid exact chunk metadata")
    dtype = np.dtype(encoded.word_dtype)
    count = 1
    for dim in encoded.shape:
        if type(dim) is not int or dim < 0:
            raise ValueError("invalid exact chunk shape")
        count *= dim
    if (encoded.payload.dtype != np.dtype("uint8") or encoded.payload.ndim != 1
            or not encoded.payload.flags.c_contiguous or encoded.chunk_index.dtype != np.dtype("<u8")
            or encoded.chunk_index.ndim != 2 or encoded.chunk_index.shape[1] != 3
            or not encoded.chunk_index.flags.c_contiguous):
        raise ValueError("invalid exact chunk arrays")
    if encoded.native_fallback:
        if encoded.transform != "native" or len(encoded.chunk_index) or len(encoded.payload) != count * dtype.itemsize:
            raise ValueError("invalid native fallback")
        if count:
            yield np.frombuffer(encoded.payload, dtype=dtype)
        return
    if encoded.transform not in ("byte_plane", "field") or (encoded.transform == "field" and dtype.itemsize != 2):
        raise ValueError("invalid encoded transform")
    offset = total = 0
    for entry in encoded.chunk_index:
        start, length, n = map(int, entry)
        if start != offset or length <= 0 or n <= 0 or start + length > len(encoded.payload):
            raise ValueError("invalid exact chunk index")
        offset += length
        total += n
    if offset != len(encoded.payload) or total != count:
        raise ValueError("exact chunk index does not cover payload and shape")
    decompressor = zstd.ZstdDecompressor()
    for start, length, n in encoded.chunk_index:
        data = encoded.payload[int(start):int(start + length)].tobytes()
        try:
            raw = decompressor.decompress(data, max_output_size=int(n) * dtype.itemsize)
        except zstd.ZstdError as exc:
            raise ValueError("corrupt exact Zstd chunk") from exc
        yield _inverse(raw, int(n), dtype, encoded.transform)


def encode_family_exact(anchor: np.ndarray, target: np.ndarray, transform: str,
                        anchor_form: str) -> ExactFamilyBacking:
    a, t = _words(anchor), _words(target)
    if a.dtype != np.dtype("<u2") or t.dtype != a.dtype or a.shape != t.shape:
        raise ValueError("family exact baseline requires paired uint16 tensors")
    if transform not in ("xor", "ordered_delta") or anchor_form not in ("raw", "compressed"):
        raise ValueError("unsupported family exact mode")
    residual = encode_residual(a, t, transform)
    # A u16 anchor chunk and a u32 residual chunk cover the same value count.
    anchor_chunks = encode_exact(a, "native" if anchor_form == "raw" else "byte_plane",
                                 chunk_bytes=131072)
    target_chunks = encode_exact(residual, "byte_plane", chunk_bytes=262144)
    return ExactFamilyBacking(anchor_chunks, target_chunks, transform, anchor_form)


def decode_family_exact(backing: ExactFamilyBacking) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(backing, ExactFamilyBacking) or backing.family_transform not in ("xor", "ordered_delta"):
        raise ValueError("invalid family backing")
    if backing.anchor.shape != backing.target.shape or backing.anchor.word_dtype != "<u2":
        raise ValueError("family anchor/target shape mismatch")
    if backing.target.word_dtype != ("<u2" if backing.family_transform == "xor" else "<u4"):
        raise ValueError("family residual dtype mismatch")
    anchor_iter, residual_iter = iter(decode_exact(backing.anchor)), iter(decode_exact(backing.target))
    aa = rr = None
    a_pos = r_pos = 0
    while True:
        if aa is None or a_pos == len(aa):
            aa = next(anchor_iter, None)
            a_pos = 0
        if rr is None or r_pos == len(rr):
            rr = next(residual_iter, None)
            r_pos = 0
        if aa is None or rr is None:
            if aa is not None or rr is not None:
                raise ValueError("family chunk lengths differ")
            break
        n = min(len(aa) - a_pos, len(rr) - r_pos)
        anchor_chunk = aa[a_pos:a_pos + n]
        target_chunk = decode_residual(anchor_chunk, rr[r_pos:r_pos + n], backing.family_transform)
        yield anchor_chunk, target_chunk
        a_pos += n
        r_pos += n


def exact_allocated_bytes(encoded: ExactChunks | ExactFamilyBacking) -> int:
    if isinstance(encoded, ExactFamilyBacking):
        return exact_allocated_bytes(encoded.anchor) + exact_allocated_bytes(encoded.target)
    if not isinstance(encoded, ExactChunks):
        raise ValueError("invalid exact backing")
    return int(encoded.chunk_index.nbytes + encoded.payload.nbytes)
