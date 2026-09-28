import json
import struct
from pathlib import Path

import pytest

from familytiles.data import (PinnedModel, TransferBudget, download_selected,
                              parse_header, read_range)


def _header(tensors):
    body = json.dumps(tensors).encode()
    return struct.pack("<Q", len(body)) + body


def test_header_extents_and_limits():
    header = _header({"x": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}})
    tensor, = parse_header(header, len(header) + 12, shard="weights.safetensors")
    assert tensor.data_start == len(header)
    assert tensor.byte_offset == 0
    assert tensor.byte_length == 12
    for bad in (
        _header({"x": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 11]}}),
        _header({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [100, 102]}}),
        _header({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                 "y": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]}}),
        _header({"x": {"dtype": "BF16", "shape": [-1], "data_offsets": [0, 2]}}),
        _header({"x": {"dtype": "BF16", "shape": [2**64], "data_offsets": [0, 2]}}),
        b"\xff\xff\xff\xff\xff\xff\xff\xff",
        header[:-1],
    ):
        with pytest.raises(ValueError):
            parse_header(bad, len(header) + 12, shard="weights.safetensors")


def test_range_validation_and_hard_stream_limit(tmp_path, monkeypatch):
    model = PinnedModel("Qwen/test", "a" * 40, {"w.safetensors": 100}, {}, {})
    budget = TransferBudget(32, 100, tmp_path / "ledger.json")

    class Response:
        def __init__(self, status, headers, chunks):
            self.status_code, self.headers, self.chunks = status, headers, chunks
        def iter_content(self, chunk_size):
            yield from self.chunks
        def close(self):
            pass

    responses = [
        Response(206, {"Content-Range": "bytes 4-7/100", "Content-Encoding": "identity"}, [b"abcd"]),
        Response(206, {"Content-Range": "bytes 3-6/100"}, [b"abcd"]),
        Response(206, {"Content-Range": "bytes 4-7/100", "Content-Encoding": "gzip"}, [b"abcd"]),
        Response(206, {"Content-Range": "bytes 4-7/100"}, [b"abc"]),
        Response(200, {"Content-Length": "100"}, [b"x" * 100]),
    ]
    monkeypatch.setattr("familytiles.data.requests.get", lambda *a, **kw: responses.pop(0))
    assert read_range(model, "w.safetensors", 4, 4, budget) == b"abcd"
    for _ in range(4):
        with pytest.raises(ValueError):
            read_range(model, "w.safetensors", 4, 4, budget)
    assert budget.snapshot()["sample_by_model"][model.repo] == 20


def test_budget_persists_across_restart_and_limits_models(tmp_path):
    ledger = tmp_path / "ledger.json"
    budget = TransferBudget(8, 20, ledger)
    budget.reserve("sample", 5, "a")
    budget.reserve("sample", 2, "a")
    with pytest.raises(ValueError):
        TransferBudget(8, 20, ledger).reserve("sample", 2, "a")
    TransferBudget(8, 20, ledger).reserve("sample", 8, "b")
    budget.reserve("weight", 15)
    with pytest.raises(ValueError):
        TransferBudget(8, 20, ledger).reserve("weight", 6)
    assert budget.snapshot()["weight_reserved"] == 15


def test_download_preflight_rejects_over_budget_before_network(tmp_path, monkeypatch):
    model = PinnedModel("Qwen/test", "a" * 40, {"w.safetensors": 12}, {}, {})
    budget = TransferBudget(8, 20, tmp_path / "ledger.json")
    monkeypatch.setattr("familytiles.data.requests.get", lambda *a, **kw: pytest.fail("network was used"))
    with pytest.raises(ValueError):
        download_selected((model, model), tmp_path / "weights", budget)
    assert budget.snapshot()["weight_reserved"] == 0


def test_revision_is_immutable_and_defaults(tmp_path):
    assert TransferBudget.DEFAULT_SAMPLE_BYTES == 32 * 2**20
    assert TransferBudget.DEFAULT_WEIGHT_BYTES == 7 * 2**30
    with pytest.raises(ValueError):
        PinnedModel("Qwen/test", "main", {}, {}, {})
