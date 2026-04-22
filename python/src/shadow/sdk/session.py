"""Session — the recording context manager that writes an `.agentlog` file.

Usage:

```python
from shadow.sdk import Session
from shadow.llm import MockLLM

async def run(agent_fn, backend):
    with Session(output_path="trace.agentlog", tags={"env": "dev"}) as session:
        # every call to session.record_chat(request, response) writes a
        # chat_request + chat_response pair into the .agentlog file.
        request = {"model": "claude-opus-4-7", "messages": [...], "params": {}}
        response = await backend.complete(request)
        session.record_chat(request, response)
```

In v0.1 the Session is a manual recorder — agents call `record_chat` after
each LLM round-trip. Monkey-patch-based auto-instrumentation of the
anthropic/openai Python clients is deferred to v0.2 per the plan (and
because the two SDKs have different streaming/non-streaming surfaces
that aren't worth unifying in v0.1).
"""

from __future__ import annotations

import datetime
import os
import platform
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from shadow import __version__, _core
from shadow.redact import Redactor


def _now_iso() -> str:
    """RFC 3339 UTC timestamp with millisecond precision, ending in `Z`."""
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class Session:
    """Record LLM interactions into a `.agentlog` file.

    Parameters
    ----------
    output_path:
        Path to write the `.agentlog` file on exit. Parent directories
        are created if missing.
    tags:
        Key→value string tags placed in the metadata record's
        `payload.tags`. Indexed by the SQLite store in CLI workflows.
    session_tag:
        Free-form label placed in `envelope.meta.session_tag`. Used for
        the SQLite index's `session_tag` column.
    redactor:
        Optional custom redactor. Defaults to [`Redactor()`] (SPEC §9
        defaults). Pass `None` to disable (NOT recommended).
    """

    def __init__(
        self,
        output_path: Path | str,
        tags: dict[str, str] | None = None,
        session_tag: str | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._output_path = Path(output_path)
        self._tags = dict(tags or {})
        self._session_tag = session_tag
        self._redactor: Redactor | None = redactor if redactor is not None else Redactor()
        self._records: list[dict[str, Any]] = []
        self._root_id: str | None = None

    def __enter__(self) -> Self:
        meta_payload: dict[str, Any] = {
            "sdk": {"name": "shadow", "version": __version__},
            "runtime": {
                "python": sys.version.split()[0],
                "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
            },
        }
        if self._tags:
            meta_payload["tags"] = dict(self._tags)
        meta_payload = self._redact(meta_payload)
        meta_id = _core.content_id(meta_payload)
        self._root_id = meta_id
        self._records.append(self._envelope("metadata", meta_payload, meta_id, parent=None))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_out = _core.write_agentlog(self._records)
        self._output_path.write_bytes(bytes_out)

    @property
    def root_id(self) -> str | None:
        """Root record id, available after __enter__."""
        return self._root_id

    def record_chat(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        parent_id: str | None = None,
    ) -> tuple[str, str]:
        """Append a chat_request + chat_response pair to the trace.

        Returns `(request_id, response_id)`. The caller can use them for
        subsequent `record_tool_call` / `record_tool_result` parent refs.
        """
        if self._root_id is None:
            raise RuntimeError("Session not entered; use `with Session(...) as s:`")
        parent = parent_id if parent_id is not None else self._last_id()
        redacted_req = self._redact(request)
        req_id = _core.content_id(redacted_req)
        req_rec = self._envelope("chat_request", redacted_req, req_id, parent=parent)
        self._records.append(req_rec)
        redacted_resp = self._redact(response)
        resp_id = _core.content_id(redacted_resp)
        resp_rec = self._envelope("chat_response", redacted_resp, resp_id, parent=req_id)
        self._records.append(resp_rec)
        return req_id, resp_id

    def record_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        parent_id: str | None = None,
    ) -> str:
        payload = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": arguments,
        }
        payload = self._redact(payload)
        record_id = _core.content_id(payload)
        parent = parent_id if parent_id is not None else self._last_id()
        self._records.append(self._envelope("tool_call", payload, record_id, parent=parent))
        return record_id

    def record_tool_result(
        self,
        tool_call_id: str,
        output: str | dict[str, Any],
        is_error: bool = False,
        latency_ms: int = 0,
        parent_id: str | None = None,
    ) -> str:
        payload = {
            "tool_call_id": tool_call_id,
            "output": output,
            "is_error": is_error,
            "latency_ms": latency_ms,
        }
        payload = self._redact(payload)
        record_id = _core.content_id(payload)
        parent = parent_id if parent_id is not None else self._last_id()
        self._records.append(self._envelope("tool_result", payload, record_id, parent=parent))
        return record_id

    def _envelope(
        self, kind: str, payload: dict[str, Any], record_id: str, parent: str | None
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            "version": _core.SPEC_VERSION,
            "id": record_id,
            "kind": kind,
            "ts": _now_iso(),
            "parent": parent,
            "payload": payload,
        }
        if self._session_tag is not None or (self._redactor and self._redactor.last_modified):
            meta: dict[str, Any] = {}
            if self._session_tag is not None:
                meta["session_tag"] = self._session_tag
            if self._redactor and self._redactor.last_modified:
                meta["redacted"] = True
            env["meta"] = meta
        return env

    def _redact(self, value: dict[str, Any]) -> dict[str, Any]:
        if self._redactor is None:
            return value
        redacted: dict[str, Any] = self._redactor.redact_value(value)
        return redacted

    def _last_id(self) -> str | None:
        return self._records[-1]["id"] if self._records else None


# Convenience: the SHADOW_SESSION_OUTPUT environment variable is honored by
# `shadow record -- <cmd>` to point the wrapped process at a trace path.
def output_path_from_env() -> Path | None:
    """Return `$SHADOW_SESSION_OUTPUT` if set, else None."""
    env = os.environ.get("SHADOW_SESSION_OUTPUT")
    return Path(env) if env else None
