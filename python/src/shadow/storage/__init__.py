"""Storage abstraction for Shadow traces and artifacts.

Defines the :class:`Storage` Protocol that abstracts how Shadow reads
and writes ``.agentlog`` traces, replay artifacts, and the SQLite
index. The OSS distribution ships :class:`FileStore` (the current
on-disk behavior) and :class:`InMemoryStore` (for tests). The cloud
distribution plugs in :class:`PostgresStore` / :class:`S3Store` /
:class:`ClickHouseStore` against the same interface.

**Status: foundation.** This commit defines the interface and ships
the file-based implementation. Existing callers continue to use
direct file I/O (via ``shadow._core.parse_agentlog`` etc.) — they
are migrated to this interface incrementally. The benefit: future
multi-tenant cloud work plugs in cleanly without a second refactor.

Design
------
A ``Storage`` is keyed by **content-addressable IDs** (sha256 hashes
from canonical JSON), so the interface is symmetric across backends:
``put(record)`` returns the ID, ``get(id)`` retrieves the record,
``query(filter)`` enumerates records by metadata.

The trade-off vs direct file I/O: a Storage call may incur a network
round-trip (Postgres) or an S3 GET, so callers should batch where
possible. The FileStore implementation is zero-cost — it's a thin
wrapper over the existing code paths.

Usage
-----
    from shadow.storage import FileStore

    store = FileStore(root="./.shadow/traces")
    record_id = store.put(record)
    fetched = store.get(record_id)
    matches = store.query(kind="chat_response", limit=100)
"""

from shadow.storage.base import Storage, StorageError
from shadow.storage.file_store import FileStore
from shadow.storage.memory_store import InMemoryStore

__all__ = [
    "FileStore",
    "InMemoryStore",
    "Storage",
    "StorageError",
]
