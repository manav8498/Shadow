"""In-memory storage backend for tests and short-lived processes."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from shadow import _core


class InMemoryStore:
    """Records held in a dict; lost on process exit.

    Useful for tests and CLI sessions that don't need persistence.
    Implements the :class:`shadow.storage.Storage` Protocol.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._insert_order: list[str] = []

    def put(self, record: dict[str, Any]) -> str:
        # Use the record's payload content_id as the storage key —
        # same convention as FileStore, so IDs are portable across
        # backends.
        content_id = _core.content_id(record.get("payload"))
        if content_id not in self._records:
            self._insert_order.append(content_id)
        # Deep copy so caller mutations don't leak into stored state.
        self._records[content_id] = copy.deepcopy(record)
        return content_id

    def get(self, record_id: str) -> dict[str, Any] | None:
        rec = self._records.get(record_id)
        return copy.deepcopy(rec) if rec is not None else None

    def query(
        self,
        *,
        kind: str | None = None,
        session_tag: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        count = 0
        for cid in self._insert_order:
            if limit is not None and count >= limit:
                return
            rec = self._records[cid]
            if kind is not None and rec.get("kind") != kind:
                continue
            if session_tag is not None:
                meta = rec.get("meta") or {}
                if not isinstance(meta, dict) or meta.get("session_tag") != session_tag:
                    continue
            yield copy.deepcopy(rec)
            count += 1

    def delete(self, record_id: str) -> bool:
        if record_id not in self._records:
            return False
        del self._records[record_id]
        self._insert_order.remove(record_id)
        return True

    def close(self) -> None:
        # Nothing to release.
        pass

    def __len__(self) -> int:
        return len(self._records)
