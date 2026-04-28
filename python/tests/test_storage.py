"""Tests for the shadow.storage abstraction.

Verifies that FileStore and InMemoryStore both implement the Storage
protocol correctly, return identical results for the same operations,
and can be substituted for one another at the API boundary. This is
the property cloud backends will rely on when they plug in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow.storage import FileStore, InMemoryStore, Storage, StorageError


def _record(
    payload: dict[str, Any],
    *,
    kind: str = "chat_response",
    session_tag: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if session_tag is not None:
        meta["session_tag"] = session_tag
    return {
        "version": "0.1",
        "id": "sha256:" + "0" * 64,
        "kind": kind,
        "ts": "2026-04-28T00:00:00.000Z",
        "parent": None,
        "meta": meta,
        "payload": payload,
    }


@pytest.fixture(params=["file", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Storage:
    """Run every contract test against both backends."""
    if request.param == "file":
        return FileStore(root=tmp_path / "store")
    return InMemoryStore()


# ---------------------------------------------------------------------------
# Protocol contract — every backend must satisfy these
# ---------------------------------------------------------------------------


class TestStorageContract:
    def test_implements_protocol(self, store: Storage) -> None:
        assert isinstance(store, Storage)

    def test_put_returns_content_id(self, store: Storage) -> None:
        rec = _record({"a": 1})
        cid = store.put(rec)
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_put_then_get_round_trip(self, store: Storage) -> None:
        rec = _record({"a": 1, "b": [2, 3]})
        cid = store.put(rec)
        fetched = store.get(cid)
        assert fetched is not None
        assert fetched["payload"] == {"a": 1, "b": [2, 3]}

    def test_get_missing_returns_none(self, store: Storage) -> None:
        # Random ID that was never put.
        fake_id = "deadbeef" * 8
        assert store.get(fake_id) is None

    def test_put_is_idempotent(self, store: Storage) -> None:
        """Same payload twice → same content_id."""
        rec1 = _record({"x": 42})
        rec2 = _record({"x": 42})
        cid1 = store.put(rec1)
        cid2 = store.put(rec2)
        assert cid1 == cid2

    def test_query_filters_by_kind(self, store: Storage) -> None:
        store.put(_record({"a": 1}, kind="chat_response"))
        store.put(_record({"b": 2}, kind="chat_request"))
        store.put(_record({"c": 3}, kind="chat_response"))
        results = list(store.query(kind="chat_response"))
        assert len(results) == 2

    def test_query_filters_by_session_tag(self, store: Storage) -> None:
        store.put(_record({"a": 1}, session_tag="alpha"))
        store.put(_record({"b": 2}, session_tag="beta"))
        store.put(_record({"c": 3}, session_tag="alpha"))
        results = list(store.query(session_tag="alpha"))
        assert len(results) == 2

    def test_query_limit(self, store: Storage) -> None:
        for i in range(10):
            store.put(_record({"i": i}))
        results = list(store.query(limit=3))
        assert len(results) == 3

    def test_query_no_filter_returns_all(self, store: Storage) -> None:
        store.put(_record({"a": 1}))
        store.put(_record({"b": 2}))
        store.put(_record({"c": 3}))
        results = list(store.query())
        assert len(results) == 3

    def test_delete_existing_returns_true(self, store: Storage) -> None:
        cid = store.put(_record({"a": 1}))
        assert store.delete(cid) is True
        assert store.get(cid) is None

    def test_delete_missing_returns_false(self, store: Storage) -> None:
        assert store.delete("deadbeef" * 8) is False


# ---------------------------------------------------------------------------
# FileStore-specific
# ---------------------------------------------------------------------------


class TestFileStore:
    def test_root_dir_created_on_init(self, tmp_path: Path) -> None:
        root = tmp_path / "new_store"
        FileStore(root=root)
        assert root.is_dir()

    def test_records_sharded_by_id_prefix(self, tmp_path: Path) -> None:
        store = FileStore(root=tmp_path / "store")
        cid = store.put(_record({"a": 1}))
        clean = cid.removeprefix("sha256:")
        expected = tmp_path / "store" / clean[:2] / f"{clean[2:]}.json"
        assert expected.exists()

    def test_corrupted_file_raises_storage_error(self, tmp_path: Path) -> None:
        store = FileStore(root=tmp_path / "store")
        cid = store.put(_record({"a": 1}))
        path = store._path_for(cid)
        path.write_text("not valid json {{{")
        with pytest.raises(StorageError):
            store.get(cid)

    def test_short_id_raises(self, tmp_path: Path) -> None:
        store = FileStore(root=tmp_path / "store")
        with pytest.raises(StorageError, match="too short"):
            store._path_for("a")


# ---------------------------------------------------------------------------
# InMemoryStore-specific
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_len_reports_record_count(self) -> None:
        store = InMemoryStore()
        assert len(store) == 0
        store.put(_record({"a": 1}))
        assert len(store) == 1

    def test_records_independent_after_put(self) -> None:
        """Mutations to the input dict must not leak into stored copy."""
        store = InMemoryStore()
        rec = _record({"a": 1})
        cid = store.put(rec)
        rec["payload"]["a"] = 999
        fetched = store.get(cid)
        assert fetched is not None
        assert fetched["payload"] == {"a": 1}


# ---------------------------------------------------------------------------
# Cross-backend equivalence
# ---------------------------------------------------------------------------


def test_two_backends_return_same_id_for_same_payload(tmp_path: Path) -> None:
    fs = FileStore(root=tmp_path / "fs")
    mem = InMemoryStore()
    rec = _record({"shared": [1, 2, {"nested": "value"}]})
    fs_id = fs.put(rec)
    mem_id = mem.put(rec)
    assert fs_id == mem_id, "content-addressing must be backend-independent"
