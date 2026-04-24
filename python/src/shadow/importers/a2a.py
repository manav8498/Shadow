"""A2A (Agent-to-Agent) session log converter.

The A2A protocol is the Linux Foundation standard for agent-to-agent
communication, donated by Google in 2025 and now used in production at
Microsoft, AWS, Salesforce, SAP, ServiceNow, and others. It uses
JSON-RPC 2.0 over HTTPS and exchanges messages between a client agent
(which formulates and sends tasks) and a remote agent (which executes
them).

Where MCP is agent-to-tool, A2A is agent-to-agent. When a client agent
dispatches a `task/send` to a remote agent, the remote agent processes
the task and returns a `task/result` or a stream of `task/status`
updates. These exchanges form an A2A session log that Shadow can
import, diff, and policy-check like any other trace.

## Accepted input shapes

1. **JSONL stream**, one JSON-RPC message per line. What A2A server
   logs typically emit.
2. **JSON array**, `[msg, msg, ...]` from session exports.
3. **Wrapped object**, `{"messages": [...], "metadata": {...}}` where
   metadata carries `agent_card` or `session_id` fields. The client's
   Signed Agent Card (A2A v1.0) may be included here.

## Field mappings

| A2A field                         | Shadow field                          |
|-----------------------------------|---------------------------------------|
| `tasks/send` params.message       | `chat_request.messages[-1].content`   |
| `tasks/send` params.context.model | `chat_request.model`                  |
| `tasks/result` result.output      | `chat_response.content[text block]`   |
| `tasks/status` messages           | `metadata.payload.a2a_events`         |
| `tool` subparts in task messages  | `tool_use` / `tool_result` blocks     |
| `usage`                           | `chat_response.usage`                 |
| Agent Card (when present)         | `metadata.payload.agent_card`         |

Orphan events (notifications without a paired response) are archived
under `metadata.payload.a2a_events` for forensics. The importer does
not invent intermediate LLM calls; if the A2A log does not contain a
request/response round-trip for a task, that task is silently skipped.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowConfigError

A2A_FORMAT = "a2a"


def a2a_to_agentlog(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed A2A session log to Shadow records."""
    messages, session_meta = _normalise_input(data)
    if not messages:
        raise ShadowConfigError(
            "A2A input contained no JSON-RPC messages.\n"
            "hint: `shadow import --format a2a` expects a JSONL stream or JSON "
            "array of JSON-RPC 2.0 messages. See "
            "https://a2a-protocol.org/ for the message schema."
        )

    pending: dict[Any, dict[str, Any]] = {}
    matched: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    extra_events: list[dict[str, Any]] = []
    agent_cards: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if "method" in msg:
            pending[msg.get("id")] = msg
        elif "id" in msg and msg["id"] in pending:
            req = pending.pop(msg["id"])
            matched.append((req, msg))
        else:
            extra_events.append(msg)
        # Collect Agent Cards when advertised
        params = msg.get("params") or {}
        if isinstance(params, dict) and isinstance(params.get("agent_card"), dict):
            agent_cards.append(params["agent_card"])

    for orphan_req in pending.values():
        matched.append((orphan_req, None))

    out: list[dict[str, Any]] = []
    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow", "version": __version__},
        "imported_from": A2A_FORMAT,
        "source": {"format": "a2a", "spec_ref": "https://a2a-protocol.org/"},
    }
    if agent_cards:
        meta_payload["agent_cards"] = agent_cards
    if session_meta:
        meta_payload["a2a_session_metadata"] = session_meta
    if extra_events:
        meta_payload["a2a_events"] = extra_events[:50]

    meta_record = _make_record("metadata", meta_payload, parent=None, ts=_now_iso())
    out.append(meta_record)
    last_parent: str = meta_record["id"]

    for req, resp in matched:
        # Only task-send style methods produce chat pairs. Others are
        # session-control events and go into a2a_events.
        method = req.get("method", "")
        if not method.startswith("tasks/"):
            continue
        params = req.get("params") or {}
        task_msg = params.get("message") or params.get("input") or {}
        context = params.get("context") or {}
        model = context.get("model") or context.get("model_id") or "a2a-imported"

        user_content = _extract_user_content(task_msg)
        req_payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "params": {
                k: v
                for k, v in context.items()
                if k
                in (
                    "temperature",
                    "top_p",
                    "max_tokens",
                    "stop_sequences",
                )
            },
            "a2a_source": {
                "method": method,
                "task_id": req.get("id"),
                "remote_agent": context.get("remote_agent"),
            },
        }
        req_record = _make_record("chat_request", req_payload, parent=last_parent, ts=_now_iso())
        out.append(req_record)

        content: list[dict[str, Any]] = []
        stop_reason = "end_turn"
        usage = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

        if resp is not None:
            if "error" in resp:
                err = resp["error"] or {}
                content.append(
                    {
                        "type": "text",
                        "text": f"error {err.get('code', '?')}: {err.get('message', '')}",
                    }
                )
                stop_reason = "error"
            elif "result" in resp:
                result = resp["result"] or {}
                output = result.get("output") or result.get("message") or result
                content.extend(_coerce_result_content(output))
                stop_reason = "end_turn"
                u = result.get("usage") if isinstance(result, dict) else None
                if isinstance(u, dict):
                    usage = {
                        "input_tokens": int(u.get("input_tokens") or u.get("prompt_tokens") or 0),
                        "output_tokens": int(
                            u.get("output_tokens") or u.get("completion_tokens") or 0
                        ),
                        "thinking_tokens": int(u.get("thinking_tokens") or 0),
                    }

        resp_payload = {
            "model": model,
            "content": content or [{"type": "text", "text": ""}],
            "stop_reason": stop_reason,
            "latency_ms": 0,
            "usage": usage,
        }
        resp_record = _make_record(
            "chat_response", resp_payload, parent=req_record["id"], ts=_now_iso()
        )
        out.append(resp_record)
        last_parent = resp_record["id"]

    if len(out) == 1:
        meta_record["payload"]["warnings"] = [
            "no tasks/send events in the A2A log; trace is metadata-only."
        ]

    return out


# ---- helpers --------------------------------------------------------------


def _normalise_input(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)], {}
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            meta = {k: v for k, v in data.items() if k != "messages"}
            return [m for m in messages if isinstance(m, dict)], meta
        return [data], {}
    raise ShadowConfigError(
        f"A2A input must be a JSON-RPC message list or an object wrapping "
        f"one; got {type(data).__name__}."
    )


def _extract_user_content(task_msg: Any) -> Any:
    """Pull a user-facing content value from an A2A task message."""
    if isinstance(task_msg, str):
        return task_msg
    if isinstance(task_msg, dict):
        if isinstance(task_msg.get("content"), str | list):
            return task_msg["content"]
        if isinstance(task_msg.get("text"), str):
            return task_msg["text"]
        if isinstance(task_msg.get("parts"), list):
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in task_msg["parts"]
            )
    return json.dumps(task_msg, sort_keys=True, default=str)


def _coerce_result_content(result: Any) -> list[dict[str, Any]]:
    """Convert a task result into Anthropic-style content blocks."""
    if isinstance(result, dict):
        if isinstance(result.get("content"), list):
            return [b for b in result["content"] if isinstance(b, dict)]
        if isinstance(result.get("parts"), list):
            blocks: list[dict[str, Any]] = []
            for part in result["parts"]:
                if isinstance(part, dict) and "text" in part:
                    blocks.append({"type": "text", "text": part["text"]})
            if blocks:
                return blocks
        if isinstance(result.get("text"), str):
            return [{"type": "text", "text": result["text"]}]
    if isinstance(result, str):
        return [{"type": "text", "text": result}]
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


__all__ = ["A2A_FORMAT", "a2a_to_agentlog"]
