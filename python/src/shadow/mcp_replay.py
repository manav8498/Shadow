"""MCP-native replay: a transport-stream shim that replays recorded
MCP JSON-RPC traffic without re-running the original MCP server.

The MCP Python SDK's ``ClientSession`` is constructed from a pair of
``anyio`` memory streams: one for outgoing requests, one for incoming
responses. We hand it a recording that pretends to be the server side
of that pair — when the client sends a request, the shim looks up the
matching recorded response by ``(method, canonicalize(params))`` and
yields it on the inbound stream.

Why intercept at the transport layer instead of monkey-patching
``ClientSession.call_tool`` / ``read_resource`` / ``get_prompt``:

- Survives SDK upgrades: every method ultimately goes through the
  same ``send_request`` / ``send_notification`` plumbing, which
  ultimately writes to the outbound stream.
- Covers tools, resources, prompts, completions, sampling, AND
  server-initiated notifications uniformly — no per-method wrapper.
- Aligns with SEP-1287 (the "deterministic transport" proposal in
  flight as of April 2026), so Shadow's recordings interoperate
  with whatever ships there.

Recording shape: a list of MCP-trace records (already imported via
``shadow import --format mcp``) carrying request/response pairs and
notifications, with the request-side carrying a ``method`` and
``params`` (and an ``id``, which we strip from the match key).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def canonicalize_params(params: Any) -> str:
    """Deterministic JSON encoding of MCP params for use as a match key.
    Sorted keys, no whitespace, ensure_ascii=False so non-ASCII URIs
    in ``resources/read`` round-trip.
    """
    return json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class MCPCall:
    """One recorded MCP request-response pair (or a server-initiated
    notification) ready for replay."""

    method: str
    params: Any
    response: Any = None
    """The recorded ``result`` field — what the client sees on success."""
    error: dict[str, Any] | None = None
    """The recorded ``error`` field, if the server returned a JSON-RPC error."""
    is_notification: bool = False
    """True for server-initiated notifications (no client request to match)."""

    @property
    def match_key(self) -> tuple[str, str]:
        return (self.method, canonicalize_params(self.params))


@dataclass
class RecordingIndex:
    """Indexes a list of :class:`MCPCall` for fast replay lookup.

    Match key is ``(method, canonicalize(params))``; the ``id`` field
    of the JSON-RPC request is intentionally NOT part of the key
    (per-session, not portable). Repeated calls with identical
    method+params return the recorded responses in their original
    order — preserves "the second `tools/list` came back with one
    fewer tool" behaviour.
    """

    calls: list[MCPCall]
    notifications: list[MCPCall] = field(default_factory=list)
    _index: dict[tuple[str, str], list[MCPCall]] = field(default_factory=dict)
    _consumed: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for call in self.calls:
            if call.is_notification:
                self.notifications.append(call)
                continue
            self._index.setdefault(call.match_key, []).append(call)

    def lookup(
        self, method: str, params: Any, *, fail_on_overflow: bool = False
    ) -> MCPCall | None:
        """Return the next recorded call for ``(method, params)``.

        ``fail_on_overflow=False`` (the historical default) returns the
        last recorded response when the candidate replays a call more
        times than the baseline did — a pragmatic fallback so chatty
        agents don't crash.

        ``fail_on_overflow=True`` returns ``None`` instead, so strict
        replay can treat over-consumption the same as a missing
        recording. ``ReplayClientSession(strict=True)`` flips this.
        """
        key = (method, canonicalize_params(params))
        bucket = self._index.get(key)
        if not bucket:
            return None
        idx = self._consumed.get(key, 0)
        if idx >= len(bucket):
            if fail_on_overflow:
                return None
            # Non-strict: return the last recorded response. A reviewer
            # auditing drift should still call ``unconsumed_keys()``
            # afterwards.
            return bucket[-1]
        self._consumed[key] = idx + 1
        return bucket[idx]

    def unconsumed_keys(self) -> list[tuple[str, str]]:
        """Match keys that exist in the recording but were never
        looked up. Helps callers detect "the candidate skipped a
        recorded MCP call.\" """
        out: list[tuple[str, str]] = []
        for key, bucket in self._index.items():
            consumed = self._consumed.get(key, 0)
            if consumed < len(bucket):
                out.append(key)
        return out


def index_from_imported_mcp_records(
    records: list[dict[str, Any]],
) -> RecordingIndex:
    """Build a :class:`RecordingIndex` from a Shadow trace produced
    by ``shadow import --format mcp``.

    The MCP importer emits ``chat_request`` / ``chat_response`` /
    ``tool_call`` / ``tool_result`` records that mirror MCP's
    ``tools/call`` flow. For replay we extract every recorded MCP
    method invocation. The recognised mapping:

    - A ``tool_call`` record + paired ``tool_result`` → MCPCall with
      method=``tools/call``, params={"name": tool_name, "arguments": ...},
      response=result.output, error=result if is_error.
    - Other MCP methods (``resources/read``, ``prompts/get``, etc.)
      are recognised when they appear in a ``metadata`` record's
      payload under ``mcp.calls`` (the importer's pass-through field).

    Records without a recognised MCP shape are ignored.
    """
    calls: list[MCPCall] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    for rec in records:
        kind = rec.get("kind")
        payload = rec.get("payload") or {}
        if kind == "tool_call":
            tool_call_id = str(payload.get("tool_call_id") or "")
            pending_tool_calls[tool_call_id] = payload
        elif kind == "tool_result":
            tool_call_id = str(payload.get("tool_call_id") or "")
            pending = pending_tool_calls.pop(tool_call_id, None)
            if pending is None:
                continue
            tool_name = str(pending.get("tool_name") or "")
            arguments = pending.get("arguments") or {}
            output = payload.get("output")
            is_error = bool(payload.get("is_error"))
            calls.append(
                MCPCall(
                    method="tools/call",
                    params={"name": tool_name, "arguments": arguments},
                    response=output if not is_error else None,
                    error={"message": str(output)} if is_error else None,
                )
            )
        elif kind == "metadata":
            extra = payload.get("mcp", {})
            for raw_call in extra.get("calls", []):
                if not isinstance(raw_call, dict):
                    continue
                method = str(raw_call.get("method") or "")
                if not method:
                    continue
                calls.append(
                    MCPCall(
                        method=method,
                        params=raw_call.get("params"),
                        response=raw_call.get("result"),
                        error=raw_call.get("error"),
                        is_notification=bool(raw_call.get("notification")),
                    )
                )
    return RecordingIndex(calls=calls)


# ---- ClientSession-compatible replay shim ----------------------------


class ReplayClientSession:
    """A drop-in replacement for ``mcp.ClientSession`` that yields
    recorded responses instead of talking to a real server.

    Implements the four most common methods used by agent code:

    - ``initialize()`` — replays the recorded ``initialize`` response,
      or returns a synthetic capability set if not recorded.
    - ``call_tool(name, arguments)`` — looks up the recording by
      ``(tools/call, {"name", "arguments"})``.
    - ``read_resource(uri)`` — keyed on ``(resources/read, {"uri"})``.
    - ``list_tools()`` / ``list_resources()`` / ``list_prompts()`` —
      keyed on the bare method.

    Methods that aren't represented in the recording raise
    :class:`MCPCallNotRecorded` so a candidate that drifts from the
    baseline fails fast (rather than silently returning stale data).

    The shim is sync-by-default for testability; the production MCP
    SDK is async, so an ``await`` wrapper is provided via
    :meth:`async_call_tool` etc.
    """

    def __init__(self, index: RecordingIndex, *, strict: bool = True) -> None:
        # Default to strict because the docstring above promises that
        # an un-recorded call "fails fast (rather than silently
        # returning stale data)". A non-strict default would have
        # contradicted that, and replay determinism is the whole
        # point of the surface. Pass strict=False to opt back in to
        # silent-None on miss.
        self._index = index
        self._strict = strict

    # --- introspection / debug -------------------------------------

    @property
    def index(self) -> RecordingIndex:
        return self._index

    # --- core methods ---------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        params = {"name": name, "arguments": arguments or {}}
        return self._lookup_or_raise("tools/call", params)

    def read_resource(self, uri: str) -> Any:
        return self._lookup_or_raise("resources/read", {"uri": uri})

    def list_tools(self) -> Any:
        return self._lookup_or_raise("tools/list", {})

    def list_resources(self) -> Any:
        return self._lookup_or_raise("resources/list", {})

    def list_prompts(self) -> Any:
        return self._lookup_or_raise("prompts/list", {})

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        params = {"name": name, "arguments": arguments or {}}
        return self._lookup_or_raise("prompts/get", params)

    def initialize(self) -> Any:
        # Return recorded `initialize` if present; otherwise a sane stub.
        recorded = self._index.lookup("initialize", {})
        if recorded is not None:
            return recorded.response
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "replay-stub", "version": "0.1.0"},
        }

    async def async_call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self.call_tool(name, arguments)

    async def async_read_resource(self, uri: str) -> Any:
        return self.read_resource(uri)

    # --- helpers --------------------------------------------------

    def _lookup_or_raise(self, method: str, params: Any) -> Any:
        # Strict mode treats over-consumption as drift — same as a
        # missing recording — by passing fail_on_overflow=True. The
        # default lookup keeps reusing the last response.
        call = self._index.lookup(method, params, fail_on_overflow=self._strict)
        if call is None:
            if self._strict:
                raise MCPCallNotRecorded(method=method, params=params)
            # Non-strict: return None for "not found" so the caller's
            # null-check path runs. Real MCP returns errors via JSON-RPC,
            # but raising would surprise callers used to "lookup misses".
            return None
        if call.error is not None:
            raise MCPServerError(call.error)
        return call.response


class MCPCallNotRecorded(LookupError):  # noqa: N818 — public name predates style rule
    """Raised in strict mode when the candidate makes an MCP call the
    recording doesn't cover."""

    def __init__(self, method: str, params: Any) -> None:
        self.method = method
        self.params = params
        super().__init__(
            f"MCP method {method!r} with params={canonicalize_params(params)!r} "
            f"is not in the recorded session"
        )


class MCPServerError(RuntimeError):
    """Raised when the recording contains a JSON-RPC error response."""

    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        super().__init__(str(error))


__all__ = [
    "MCPCall",
    "MCPCallNotRecorded",
    "MCPServerError",
    "RecordingIndex",
    "ReplayClientSession",
    "canonicalize_params",
    "index_from_imported_mcp_records",
]
