"""Session — the recording context manager that writes an `.agentlog` file.

Usage (auto-instrumentation, default):

```python
from shadow.sdk import Session
import anthropic

with Session(output_path="trace.agentlog", tags={"env": "dev"}):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hello"}],
    )
    # Shadow records this call automatically; no record_chat needed.
```

The auto-instrumentor monkey-patches `anthropic.resources.messages.{,Async}Messages.create`
and `openai.resources.chat.completions.{,Async}Completions.create` at
`Session.__enter__` and restores them at `__exit__`. If an SDK isn't
installed, it's skipped cleanly. Streaming calls pass through un-recorded.

You can also record manually — useful for custom backends or reshaping:

```python
with Session(output_path="trace.agentlog") as session:
    session.record_chat(request_dict, response_dict)
```

Pass `auto_instrument=False` to disable the monkey-patching.
"""

from __future__ import annotations

import contextlib
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
        auto_instrument: bool = True,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        from shadow.sdk.tracing import current_parent_span_id, current_trace_id, new_trace_id

        self._output_path = Path(output_path)
        self._tags = dict(tags or {})
        self._session_tag = session_tag
        self._redactor: Redactor | None = redactor if redactor is not None else Redactor()
        self._records: list[dict[str, Any]] = []
        self._root_id: str | None = None
        self._auto_instrument = auto_instrument
        self._instrumentor: Any | None = None
        # Distributed-trace propagation: inherit trace_id from the env if a
        # parent set it, else mint a new one. span_id is this session's own.
        self._trace_id = trace_id or current_trace_id() or new_trace_id()
        self._parent_span_id = parent_span_id or current_parent_span_id()

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
        if self._auto_instrument:
            from shadow.sdk.instrumentation import Instrumentor

            # If install() fails mid-way, uninstall whatever patches ARE
            # recorded so we don't leave the global SDK classes in a
            # half-patched state across the session boundary.
            instr = Instrumentor(self)
            try:
                instr.install()
            except Exception:
                with contextlib.suppress(Exception):
                    instr.uninstall()
                raise
            self._instrumentor = instr
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._instrumentor is not None:
            self._instrumentor.uninstall()
            self._instrumentor = None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_out = _core.write_agentlog(self._records)
        self._output_path.write_bytes(bytes_out)

    @property
    def root_id(self) -> str | None:
        """Root record id, available after __enter__."""
        return self._root_id

    @property
    def trace_id(self) -> str:
        """Distributed trace id. Same across all sessions linked via env vars."""
        return self._trace_id

    def env_for_child(self) -> dict[str, str]:
        """Env dict to pass to a child process so it joins this trace."""
        from shadow.sdk.tracing import env_for_child

        # Use the root record's id as this session's span_id for clarity.
        span_id = (self._root_id or "").replace("sha256:", "")[:16]
        return env_for_child(self._trace_id, span_id or "0" * 16)

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
        meta: dict[str, Any] = {}
        if self._session_tag is not None:
            meta["session_tag"] = self._session_tag
        if self._redactor and self._redactor.last_modified:
            meta["redacted"] = True
        meta["trace_id"] = self._trace_id
        if self._parent_span_id is not None:
            meta["parent_span_id"] = self._parent_span_id
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
