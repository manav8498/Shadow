"""PydanticAI message-log → Shadow `.agentlog` converter.

[PydanticAI](https://ai.pydantic.dev) is a Python agent framework
(`pydantic-ai` on PyPI) that stores every agent run's message history
in a structured list of `ModelRequest` / `ModelResponse` objects,
each carrying a list of typed `parts`. The serialised form (what
`agent.run_sync(...).all_messages_json()` emits, or what Logfire
captures via PydanticAI's built-in instrumentation) is a JSON array
of those messages.

This importer accepts two shapes:

1. **PydanticAI message-history dump** — a top-level list of
   `{kind: "request"|"response", parts: [...], ...}` objects. The
   PydanticAI-native shape.

2. **Logfire span export** — a JSON array of Logfire spans where the
   `pydantic_ai.agent` spans carry the message history under
   `attributes.events` / `attributes.all_messages_json`. This is what
   falls out of the Logfire `"export JSON"` button or the
   `logfire backfill` CLI.

Field conventions in both shapes follow PydanticAI's types, documented
at https://ai.pydantic.dev/api/messages/. The part kinds we map:

| Part kind                | Becomes                                    |
|--------------------------|--------------------------------------------|
| `system-prompt`          | `messages[0]` with role `"system"`         |
| `user-prompt`            | `messages[N]` with role `"user"`           |
| `text`                   | response `content` text block              |
| `tool-call`              | response `content` tool_use block          |
| `tool-return`            | subsequent request `tool_result` content   |
| `retry-prompt`           | folded into request as a user turn         |

Usage surfaces from each `ModelResponse.usage` — PydanticAI names the
fields `request_tokens` / `response_tokens` / `total_tokens`.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

PYDANTIC_AI_FORMAT = "pydantic-ai"


def pydantic_ai_to_agentlog(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed PydanticAI message log → Shadow records."""
    messages, trace_meta = _normalise_input(data)
    if not messages:
        raise ShadowConfigError(
            "PydanticAI input contained no messages — check the file format.\n"
            "hint: `shadow import --format pydantic-ai` expects the output of "
            "`agent_result.all_messages_json()` or a Logfire span export."
        )

    out: list[dict[str, Any]] = []
    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow-import-pydantic-ai", "version": "1.0"},
        "source": {"format": "pydantic-ai"},
    }
    if trace_meta:
        meta_payload["pydantic_ai_metadata"] = trace_meta
    meta_record = _make_record("metadata", meta_payload, parent=None, ts=_now_iso())
    out.append(meta_record)
    last_parent: str = meta_record["id"]

    # PydanticAI message history is a strictly alternating sequence:
    # request (role-ish turns from the user + tool returns) → response
    # (model output) → request → response → ...
    # We pair each request with the next response. Dangling requests
    # (an empty agent call) still get a chat_request record.
    pending_request: dict[str, Any] | None = None
    accumulated_user: list[dict[str, Any]] = []
    system_prompt: str | None = None
    model_name = _extract_default_model(messages)
    tool_schemas = _extract_tool_schemas(messages)

    for msg in messages:
        kind = msg.get("kind") or msg.get("role")
        parts = msg.get("parts") or []
        if not isinstance(parts, list):
            parts = []
        if kind in ("request", "ModelRequest"):
            sys_part, user_parts = _partition_request_parts(parts)
            if sys_part is not None and system_prompt is None:
                system_prompt = sys_part
            accumulated_user.extend(user_parts)
            # Stash — we can't emit the chat_request yet because we
            # don't know the model yet (that comes from the response).
            pending_request = {
                "system_prompt": system_prompt,
                "user_parts": list(accumulated_user),
                "model_hint": msg.get("model_name") or model_name,
                "params": _extract_params(msg),
            }
        elif kind in ("response", "ModelResponse"):
            if pending_request is None:
                # Response with no preceding request — synthesise a minimal one.
                pending_request = {
                    "system_prompt": system_prompt,
                    "user_parts": [],
                    "model_hint": msg.get("model_name") or model_name,
                    "params": _extract_params(msg),
                }
            req_payload = _assemble_request_payload(pending_request, tool_schemas)
            if msg.get("model_name"):
                req_payload["model"] = msg["model_name"]
            req_record = _make_record(
                "chat_request",
                req_payload,
                parent=last_parent,
                ts=_iso_or_now(msg.get("timestamp")),
            )
            out.append(req_record)
            resp_payload = _response_payload(msg, req_payload["model"])
            resp_record = _make_record(
                "chat_response",
                resp_payload,
                parent=req_record["id"],
                ts=_iso_or_now(msg.get("timestamp")),
            )
            out.append(resp_record)
            last_parent = resp_record["id"]
            # After a response, accumulated user context is consumed.
            # But any *following* tool-return parts need to land as a
            # new chat_request's user-turn (pattern: req → resp with
            # tool_use → req with tool_result → resp).
            pending_request = None
            accumulated_user = []

    # If a request was queued without a response, flush it so the trace
    # isn't silently lossy.
    if pending_request is not None:
        req_payload = _assemble_request_payload(pending_request, tool_schemas)
        out.append(_make_record("chat_request", req_payload, parent=last_parent, ts=_now_iso()))

    return out


# ---- normalisation --------------------------------------------------------


def _normalise_input(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the flat message list + optional trace metadata."""
    if isinstance(data, list):
        # Could be a plain message list or a Logfire spans array.
        if data and _looks_like_logfire_span(data[0]):
            return _extract_messages_from_logfire(data)
        return [m for m in data if isinstance(m, dict)], {}
    if isinstance(data, dict):
        if isinstance(data.get("messages"), list):
            meta = {k: v for k, v in data.items() if k != "messages"}
            return [m for m in data["messages"] if isinstance(m, dict)], meta
        if isinstance(data.get("all_messages"), list):
            meta = {k: v for k, v in data.items() if k != "all_messages"}
            return [m for m in data["all_messages"] if isinstance(m, dict)], meta
        if isinstance(data.get("spans"), list) and any(
            _looks_like_logfire_span(s) for s in data["spans"]
        ):
            messages, meta = _extract_messages_from_logfire(data["spans"])
            # Keep any top-level metadata too.
            meta.update({k: v for k, v in data.items() if k != "spans"})
            return messages, meta
        if data.get("kind") in ("request", "response"):
            return [data], {}
    raise ShadowConfigError(
        f"PydanticAI input must be a message list, a Logfire span array, or an "
        f"object wrapping one; got {type(data).__name__}."
    )


def _looks_like_logfire_span(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    attrs = obj.get("attributes") or {}
    if not isinstance(attrs, dict):
        return False
    return (
        "pydantic_ai" in (obj.get("span_name") or obj.get("name") or "")
        or "all_messages_json" in attrs
        or ("events" in attrs and any("pydantic_ai" in str(k) for k in attrs))
    )


def _extract_messages_from_logfire(
    spans: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pull the PydanticAI message history out of Logfire spans."""
    messages: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"logfire_span_count": len(spans)}
    for span in spans:
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes") or {}
        raw = attrs.get("all_messages_json") or attrs.get("pydantic_ai.all_messages")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
        else:
            parsed = raw
        if isinstance(parsed, list):
            messages.extend(m for m in parsed if isinstance(m, dict))
    return messages, meta


# ---- per-message helpers --------------------------------------------------


def _partition_request_parts(
    parts: list[Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull out a system prompt + return user/tool-return parts as turns."""
    system_prompt: str | None = None
    turns: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("part_kind") or part.get("kind") or part.get("type")
        if kind in ("system-prompt", "SystemPromptPart"):
            content = part.get("content")
            if isinstance(content, str):
                system_prompt = content
        elif kind in ("user-prompt", "UserPromptPart"):
            content = part.get("content")
            if isinstance(content, str):
                turns.append({"role": "user", "content": content})
            elif isinstance(content, list):
                turns.append({"role": "user", "content": _stringify_content(content)})
        elif kind in ("tool-return", "ToolReturnPart"):
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": part.get("tool_call_id")
                            or f"pydai_{part.get('tool_name', 'tool')}",
                            "content": _tool_return_content(part.get("content")),
                        }
                    ],
                }
            )
        elif kind in ("retry-prompt", "RetryPromptPart"):
            content = part.get("content") or part.get("tool_name") or "retry"
            turns.append({"role": "user", "content": f"[retry] {content}"})
    return system_prompt, turns


def _stringify_content(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(json.dumps(item, sort_keys=True, default=str))
    return "\n".join(parts)


def _tool_return_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [{"type": "text", "text": json.dumps(content, sort_keys=True, default=str)}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and "type" in item:
                blocks.append(item)
            else:
                blocks.append(
                    {"type": "text", "text": json.dumps(item, sort_keys=True, default=str)}
                )
        return blocks or [{"type": "text", "text": ""}]
    return [{"type": "text", "text": json.dumps(content, default=str)}]


def _assemble_request_payload(
    pending: dict[str, Any], tool_schemas: list[dict[str, Any]]
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    sys_prompt = pending.get("system_prompt")
    if isinstance(sys_prompt, str):
        messages.append({"role": "system", "content": sys_prompt})
    messages.extend(pending.get("user_parts") or [])
    payload: dict[str, Any] = {
        "model": pending.get("model_hint") or "pydantic-ai-unknown",
        "messages": messages,
        "params": pending.get("params") or {},
    }
    if tool_schemas:
        payload["tools"] = tool_schemas
    return payload


def _response_payload(msg: dict[str, Any], model: str) -> dict[str, Any]:
    parts = msg.get("parts") or []
    content_blocks: list[dict[str, Any]] = []
    text_buf: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("part_kind") or part.get("kind") or part.get("type")
        if kind in ("text", "TextPart"):
            content = part.get("content")
            if isinstance(content, str):
                text_buf.append(content)
        elif kind in ("tool-call", "ToolCallPart"):
            if text_buf:
                content_blocks.append({"type": "text", "text": "".join(text_buf)})
                text_buf = []
            args = part.get("args") or part.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": part.get("tool_call_id") or f"pydai_{part.get('tool_name', 'tool')}",
                    "name": part.get("tool_name") or part.get("name") or "",
                    "input": args if isinstance(args, dict) else {},
                }
            )
    if text_buf:
        content_blocks.append({"type": "text", "text": "".join(text_buf)})

    usage = msg.get("usage") or {}
    stop_reason = _normalise_stop_reason(msg, content_blocks)
    return {
        "model": msg.get("model_name") or model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "latency_ms": int(msg.get("duration_ms") or 0),
        "usage": {
            "input_tokens": int(usage.get("request_tokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("response_tokens") or usage.get("output_tokens") or 0),
            "thinking_tokens": int(usage.get("thinking_tokens") or 0),
        },
    }


def _normalise_stop_reason(msg: dict[str, Any], blocks: list[dict[str, Any]]) -> str:
    raw = msg.get("finish_reason") or msg.get("stop_reason")
    if isinstance(raw, str):
        lower = raw.lower()
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "tool-calls": "tool_use",
            "content_filter": "content_filter",
            "content-filter": "content_filter",
        }
        return mapping.get(lower, lower)
    # Infer from content: tool_use blocks indicate tool_use stop.
    if any(b.get("type") == "tool_use" for b in blocks):
        return "tool_use"
    return "end_turn"


def _extract_params(msg: dict[str, Any]) -> dict[str, Any]:
    raw = msg.get("model_request_parameters") or msg.get("params") or {}
    if not isinstance(raw, dict):
        return {}
    # Keep only well-known keys.
    params: dict[str, Any] = {}
    for k in (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ):
        if k in raw and raw[k] is not None:
            params[k] = raw[k]
    return params


def _extract_default_model(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict):
            name = msg.get("model_name")
            if isinstance(name, str):
                return name
    return "pydantic-ai-unknown"


def _extract_tool_schemas(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk messages for any advertised tool definitions."""
    seen: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        tools = msg.get("model_request_parameters", {})
        if isinstance(tools, dict):
            fn_tools = tools.get("function_tools") or tools.get("tools") or []
            if isinstance(fn_tools, list):
                for t in fn_tools:
                    if isinstance(t, dict) and t.get("name"):
                        seen[t["name"]] = {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "input_schema": t.get("parameters_json_schema")
                            or t.get("parameters")
                            or {},
                        }
    return list(seen.values())


# ---- low-level helpers ----------------------------------------------------


def _iso_or_now(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return _now_iso()


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


__all__ = ["PYDANTIC_AI_FORMAT", "pydantic_ai_to_agentlog"]
