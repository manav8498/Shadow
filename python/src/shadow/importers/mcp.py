"""MCP (Model Context Protocol) log → Shadow `.agentlog` converter.

MCP is Anthropic's open JSON-RPC-2.0 protocol for agents to talk
to external tools, data sources, and context (spec 2025-06-18,
v1.0). Widely adopted by Claude Desktop, Cursor, Windsurf, Zed,
VS Code, and every major Anthropic-flavoured IDE by early 2026.

This importer turns an MCP session log into a partial `.agentlog`:

- **Perfectly captured**: tool-call trajectory (what tools the agent
  invoked, in what order, with what arguments) and tool-schema
  (what tools the server advertised). These are Shadow's
  `trajectory` and `conformance` axes — the importer gives full
  signal on both.

- **Partially captured**: tool results (present in most but not all
  MCP sessions). Wired as `tool_result` content blocks attached to
  the corresponding response record.

- **Not captured**: the LLM completion that drove each tool call.
  MCP is the tool protocol, not the LLM protocol; the model's
  chat_request / chat_response lives outside the MCP session. The
  semantic / verbosity / safety axes will show zero deltas on
  imports produced by this path — that's honest, not a bug.

## Accepted input shapes

1. **JSONL stream** — one JSON-RPC message per line. What
   `mcp-server --log`-style tools emit.

2. **JSON array** — the session packaged as `[msg, msg, ...]`.
   What the MCP Inspector's "export session" produces.

3. **Wrapped log** — `{"messages": [...], "metadata": {...}}` where
   metadata carries a session id, a timestamp, or a server name.
   The top-level object's `metadata` is merged into the Shadow
   metadata record; `messages` is processed as case (2).

All three are auto-detected by the first non-whitespace byte.

## Round-trip shape

Every `tools/call` JSON-RPC request becomes one Shadow record pair:

- A synthesised `chat_request` record carrying the tool name + args
  under `payload.tool_call` so the replay machinery has something to
  work with (the real LLM prompt is unknown, so we stub a single-
  turn request with the tool call as-if it were the assistant turn).
- A `chat_response` record with a `tool_use` content block that
  exactly mirrors Anthropic's content-block shape (the format the
  rest of the Rust differ expects).

`tools/list` responses are folded into the metadata record's
`payload.tools` field so schema-watch and the trajectory axis see
the server's advertised schemas.

Other methods (`prompts/*`, `resources/*`, `ping`, `initialize`) are
logged into `metadata.payload.mcp_events` for forensics but don't
produce per-turn records — they're infrastructure, not behaviour.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

MCP_FORMAT = "mcp"


def mcp_to_agentlog(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed MCP session log → Shadow records.

    `data` is the already-parsed Python structure (dict / list) —
    the CLI is responsible for the JSON parsing. Pass either a list
    of JSON-RPC messages, a dict wrapping `{"messages": [...]}`, or
    (as a convenience) a single message dict.
    """
    messages, session_meta = _normalise_input(data)
    if not messages:
        raise ShadowConfigError(
            "MCP input contained no JSON-RPC messages — check the file format.\n"
            "hint: `shadow import --format mcp` expects either a JSONL stream "
            "or a JSON array of JSON-RPC 2.0 messages."
        )

    # Pair requests to responses by id; JSON-RPC guarantees id uniqueness
    # within a session, so a single-pass dict is sufficient.
    pending: dict[Any, dict[str, Any]] = {}
    matched: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    tool_listings: list[dict[str, Any]] = []
    extra_events: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # Requests have a `method`. Responses have a `result` (or
        # `error`) and an id but no method.
        if "method" in msg:
            pending[msg.get("id")] = msg
        elif "id" in msg and (msg["id"] in pending):
            req = pending.pop(msg["id"])
            matched.append((req, msg))
        else:
            # Notifications (no id) or orphan responses — archive them.
            extra_events.append(msg)

    # Requests that never got a paired response.
    for orphan_req in pending.values():
        matched.append((orphan_req, None))

    # Build the `.agentlog` record sequence.
    out: list[dict[str, Any]] = []
    meta_tools: list[dict[str, Any]] = []

    # First pass: pull tool schemas from any tools/list response so
    # the metadata record reflects what the server advertised.
    for req, resp in matched:
        if req.get("method") == "tools/list" and resp and "result" in resp:
            tools = (resp["result"] or {}).get("tools") or []
            for t in tools:
                if isinstance(t, dict) and "name" in t:
                    meta_tools.append(
                        {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "input_schema": t.get("inputSchema") or t.get("input_schema", {}),
                        }
                    )
            tool_listings.append(resp)

    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow-import-mcp", "version": "0.4"},
        "source": {"format": "mcp", "spec": "2025-06-18"},
    }
    if meta_tools:
        meta_payload["tools"] = meta_tools
    if session_meta:
        meta_payload["mcp_session_metadata"] = session_meta
    if extra_events:
        meta_payload["mcp_events"] = extra_events[:50]  # cap for readability

    meta_record = _make_record("metadata", meta_payload, parent=None, ts=_now_iso())
    out.append(meta_record)
    last_parent = meta_record["id"]

    # Second pass: emit chat_request / chat_response pairs for each
    # tools/call invocation, in the order they appeared in the log.
    for req, resp in matched:
        if req.get("method") != "tools/call":
            continue
        params = req.get("params") or {}
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        tool_call_id = req.get("id")

        # Synthesise a minimal chat_request. The "real" LLM prompt
        # isn't in an MCP log — we record the tool invocation itself
        # as the request so replay machinery has a payload to hash
        # against.
        request_payload: dict[str, Any] = {
            "model": "mcp-imported",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Invoke MCP tool `{tool_name}` with arguments: "
                        f"{json.dumps(tool_args, sort_keys=True)}"
                    ),
                },
            ],
            "params": {},
            "tools": meta_tools,  # tools available to this session
            "mcp_source": {"method": req.get("method"), "jsonrpc_id": tool_call_id},
        }
        req_record = _make_record(
            "chat_request", request_payload, parent=last_parent, ts=_now_iso()
        )
        out.append(req_record)

        # Build the response payload. The Rust differ expects
        # Anthropic-style content blocks: a list of {type: tool_use, ...}
        # and {type: tool_result, ...} items.
        content: list[dict[str, Any]] = [
            {
                "type": "tool_use",
                "id": f"mcp_{tool_call_id}",
                "name": tool_name,
                "input": tool_args,
            }
        ]
        stop_reason = "tool_use"
        usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

        if resp is not None:
            if "error" in resp:
                # MCP error response — map into a tool_result with is_error.
                err = resp["error"] or {}
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": f"mcp_{tool_call_id}",
                        "content": [
                            {
                                "type": "text",
                                "text": f"error {err.get('code', '?')}: "
                                f"{err.get('message', '')}",
                            }
                        ],
                        "is_error": True,
                    }
                )
                stop_reason = "end_turn"
            elif "result" in resp:
                tool_result_blob = resp["result"]
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": f"mcp_{tool_call_id}",
                        "content": _coerce_result_content(tool_result_blob),
                    }
                )
                stop_reason = "end_turn"

        response_payload = {
            "model": "mcp-imported",
            "content": content,
            "stop_reason": stop_reason,
            "latency_ms": 0,  # not available from MCP logs
            "usage": usage,
        }
        resp_record = _make_record(
            "chat_response", response_payload, parent=req_record["id"], ts=_now_iso()
        )
        out.append(resp_record)
        last_parent = resp_record["id"]

    if len(out) == 1:
        # We have only the metadata record — no tools/call events in
        # the log. That's a valid but mostly-empty trace.
        # Emit a warning via the metadata payload so consumers can
        # detect it without raising.
        meta_record["payload"]["warnings"] = [
            "no tools/call events in the MCP log; trace contains metadata only."
        ]

    return out


# ---- helpers ------------------------------------------------------------


def _normalise_input(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract the message list + optional session metadata from input."""
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)], {}
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            meta = {k: v for k, v in data.items() if k != "messages"}
            return [m for m in messages if isinstance(m, dict)], meta
        # Single message dict — treat as one-message session.
        return [data], {}
    raise ShadowConfigError(
        f"MCP input must be a JSON-RPC message list or an object wrapping "
        f"one; got {type(data).__name__}."
    )


def _coerce_result_content(result: Any) -> list[dict[str, Any]]:
    """Convert an MCP `result` field into Anthropic-style tool_result content.

    MCP spec v1 allows `result` to be arbitrary JSON; the common shape for
    tool responses is `{"content": [{"type": "text"|"image"|..., ...}]}`.
    We pass that through verbatim; anything else gets wrapped as a single
    text block containing the JSON-serialised payload.
    """
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        # Already in the expected block shape.
        out: list[dict[str, Any]] = []
        for block in result["content"]:
            if isinstance(block, dict):
                out.append(block)
        if out:
            return out
    return [{"type": "text", "text": json.dumps(result, sort_keys=True, default=str)}]


def _make_record(kind: str, payload: dict[str, Any], parent: str | None, ts: str) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": _core.content_id(payload),
        "kind": kind,
        "ts": ts,
        "parent": parent,
        "payload": payload,
    }


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["MCP_FORMAT", "mcp_to_agentlog"]
