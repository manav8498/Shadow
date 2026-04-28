"""File-backed storage — content-addressable on disk.

Records are written one per file under ``<root>/<id[:2]>/<id[2:]>.json``
(the standard "object-store sharding" pattern used by git, ipfs,
docker, etc.). The first two hex chars of the ID become a directory,
preventing the root from accumulating tens of thousands of entries
in one folder.

Reads are cheap (one file open per `get`). Writes are atomic via
write-to-tmp + rename.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.storage.base import StorageError


class FileStore:
    """Content-addressable record store on the local filesystem.

    Implements the :class:`shadow.storage.Storage` Protocol.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, record_id: str) -> Path:
        # Strip 'sha256:' prefix if present so the on-disk layout is clean.
        clean = record_id.removeprefix("sha256:")
        if len(clean) < 2:
            raise StorageError(f"record_id {record_id!r} too short to shard")
        return self._root / clean[:2] / f"{clean[2:]}.json"

    def put(self, record: dict[str, Any]) -> str:
        content_id = _core.content_id(record.get("payload"))
        path = self._path_for(content_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file + rename.
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=path.parent,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            json.dump(record, tmp)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return content_id

    def get(self, record_id: str) -> dict[str, Any] | None:
        path = self._path_for(record_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (OSError, json.JSONDecodeError) as e:
            raise StorageError(f"failed to read {path}: {e}") from e

    def query(
        self,
        *,
        kind: str | None = None,
        session_tag: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        # FileStore has no native index — iterate the full tree.
        # Cloud backends override this with index lookups.
        count = 0
        for path in sorted(self._root.rglob("*.json")):
            if limit is not None and count >= limit:
                return
            try:
                with open(path, encoding="utf-8") as f:
                    record: dict[str, Any] = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue  # skip unreadable files
            if kind is not None and record.get("kind") != kind:
                continue
            if session_tag is not None:
                meta = record.get("meta") or {}
                if not isinstance(meta, dict) or meta.get("session_tag") != session_tag:
                    continue
            yield record
            count += 1

    def delete(self, record_id: str) -> bool:
        path = self._path_for(record_id)
        if not path.exists():
            return False
        try:
            path.unlink()
            # Try to remove the empty shard directory; OK if it's not empty.
            with contextlib.suppress(OSError):
                path.parent.rmdir()
            return True
        except OSError as e:
            raise StorageError(f"failed to delete {path}: {e}") from e

    def close(self) -> None:
        # Nothing to release for FileStore.
        pass

    @property
    def root(self) -> Path:
        return self._root
