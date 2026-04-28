"""Storage Protocol — the contract every backend implements."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from shadow.errors import ShadowError


class StorageError(ShadowError):
    """Generic storage backend failure (I/O, permission, network)."""


@runtime_checkable
class Storage(Protocol):
    """Read/write interface for agentlog records.

    Implementations: :class:`shadow.storage.FileStore`,
    :class:`shadow.storage.InMemoryStore`. Cloud adds Postgres / S3 /
    ClickHouse stores against the same interface.

    All methods are synchronous. Async backends should expose a
    sync facade here (or subclass with `async def` overrides — the
    cloud impls do this).
    """

    def put(self, record: dict[str, Any]) -> str:
        """Store ``record``; return its content-addressable ID.

        IDs are stable: calling ``put`` twice with the same record
        returns the same ID and is idempotent (the second put may
        no-op or refresh metadata, but the ID does not change).
        """
        ...

    def get(self, record_id: str) -> dict[str, Any] | None:
        """Fetch by content ID; return None if not present."""
        ...

    def query(
        self,
        *,
        kind: str | None = None,
        session_tag: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Enumerate records matching the filter.

        Filters are conjunctive (AND): a record must satisfy every
        non-None filter to be yielded. ``limit`` caps results.
        Iteration order is implementation-defined; FileStore yields
        in insertion order, cloud backends may use indexes.
        """
        ...

    def delete(self, record_id: str) -> bool:
        """Remove ``record_id``; return True if it existed.

        Optional — backends MAY raise :class:`NotImplementedError` if
        deletion is not supported (e.g. immutable archive backends).
        Callers should check the implementation's docstring.
        """
        ...

    def close(self) -> None:
        """Release backend resources (file handles, connections)."""
        ...
