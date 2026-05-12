"""LlamaIndex instrumentation events → Shadow `.agentlog` converter.

LlamaIndex publishes agent traces via the
`llama_index.core.instrumentation` event bus. Subscribers receive
typed `Event` objects (subclasses of `BaseEvent`) — LLM lifecycle
events, tool-call lifecycle events, and agent step events — each
carrying `id_`, `span_id`, `timestamp`, and event-specific payload
fields. The serialised form (what `event.dict()` / `event.json()`
produces, or what a `BaseEventHandler.handle` callback receives as
a `model_dump()`) is a plain dict.

This importer accepts the raw dict-list shape so callers stay free
of any LlamaIndex install requirement — emit the events with
whatever instrumentation handler you prefer, json-dump them, then
feed them here. Two shapes are supported:

1. **Flat event list** — `[{class_name, id_, span_id, timestamp,
   ...event fields}, ...]`. The natural shape of
   `[handler.event(e) for e in stream]`.

2. **Wrapped object** — `{"events": [...]}` (or `{"trace": {...}}`).
   The default `BaseEventHandler` dump from
   `llama_index.core.instrumentation.dispatcher` when materialised
   to disk.

Event types we pair up:

| Start event              | End event              | Becomes                        |
|--------------------------|------------------------|--------------------------------|
| `LLMChatStartEvent`      | `LLMChatEndEvent`      | `chat_request` + `chat_response` |
| `LLMCompletionStartEvent`| `LLMCompletionEndEvent`| `chat_request` + `chat_response` |
| `ToolCallStartEvent`     | `ToolCallEndEvent`     | `tool_call` + `tool_result`    |
| `AgentToolCallEvent`     | (single)               | `tool_call`                    |
| `AgentRunStepStartEvent` | `AgentRunStepEndEvent` | (folded into chain, no record) |

Pairing key: events share an `id_` (LlamaIndex's per-span correlation
id) when they belong to the same operation. If `id_` is missing we
fall back to FIFO matching by event family.

## Design notes

- We use the per-span `id_` as the pairing key, mirroring how
  LlamaIndex's own `SpanHandler` correlates start/end events.
- Unknown event classes are ignored silently so this importer keeps
  working when LlamaIndex adds new event types in future versions —
  the metadata record carries an `unknown_event_count` field so
  callers can detect drift.
- Tool-call payload shape matches the rest of Shadow
  (`{tool_name, tool_call_id, arguments}` / `{tool_call_id, output,
  is_error}`) so the MCP replayer and counterfactual loop can
  consume imported traces without per-format branching.
- `model` is always present on `chat_response` payloads as required
  by Shadow's schema; falls back to an empty string when the event
  doesn't expose it.
"""

from __future__ import annotations

import contextlib
import datetime
import json
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

LLAMAINDEX_FORMAT = "llamaindex"

# Event-class names we recognise. We tolerate both the canonical
# `class_name` field and a few common alternates (`event_type`,
# `type`) because different LlamaIndex serialisers spell it
# differently.
_LLM_CHAT_START = {"LLMChatStartEvent", "LLMChatInProgressEvent"}
_LLM_CHAT_END = {"LLMChatEndEvent"}
_LLM_COMPLETION_START = {"LLMCompletionStartEvent", "LLMCompletionInProgressEvent"}
_LLM_COMPLETION_END = {"LLMCompletionEndEvent"}
_TOOL_CALL_START = {"ToolCallStartEvent", "AgentToolCallEvent"}
_TOOL_CALL_END = {"ToolCallEndEvent"}
_STEP_START = {"AgentRunStepStartEvent"}
_STEP_END = {"AgentRunStepEndEvent"}


def llamaindex_to_agentlog(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed LlamaIndex instrumentation event log → Shadow records."""
    events, trace_meta = _normalise_input(data)
    if not events:
        raise ShadowConfigError(
            "LlamaIndex input contained no events — check the file format.\n"
            "hint: `shadow import --format llamaindex` expects a JSON array "
            "of `llama_index.core.instrumentation` event dicts, or an object "
            'wrapping that array under `"events"`.'
        )

    return import_llamaindex_events(events, trace_meta=trace_meta)


def import_llamaindex_events(
    events: list[dict[str, Any]],
    *,
    trace_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert a flat list of LlamaIndex event dicts → Shadow records.

    This is the public-facing API; `llamaindex_to_agentlog` is a
    thin wrapper that also accepts a wrapped `{"events": [...]}`
    shape.
    """
    out: list[dict[str, Any]] = []
    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow-import-llamaindex", "version": "1.0"},
        "source": {"format": LLAMAINDEX_FORMAT},
        "event_count": len(events),
    }
    if trace_meta:
        meta_payload["llamaindex_metadata"] = _jsonify(trace_meta)

    # First pass: build pairing tables so we can match start↔end
    # events regardless of interleaving with unrelated events.
    chat_starts: dict[str, dict[str, Any]] = {}
    chat_pending_fifo: list[dict[str, Any]] = []
    tool_starts: dict[str, dict[str, Any]] = {}
    tool_pending_fifo: list[dict[str, Any]] = []
    unknown = 0

    # Each emitted record (apart from metadata) — built up in event order.
    emitted: list[tuple[str, dict[str, Any], str | None, str | None]] = []
    # Map from a logical "operation id" (the start-event id_) to the
    # response record id, so child tool_calls inside the same span can
    # parent off the chat_response.
    span_to_response: dict[str, str] = {}

    for event in events:
        if not isinstance(event, dict):
            unknown += 1
            continue
        class_name = _class_name(event)
        if class_name in _LLM_CHAT_START or class_name in _LLM_COMPLETION_START:
            key = _event_key(event)
            chat_starts[key] = event
            chat_pending_fifo.append(event)
        elif class_name in _LLM_CHAT_END or class_name in _LLM_COMPLETION_END:
            start = _pop_pair(chat_starts, chat_pending_fifo, event)
            req_payload = _chat_request_payload(start, event)
            req_id = _core.content_id(req_payload)
            emitted.append(
                (
                    "chat_request",
                    {
                        "payload": req_payload,
                        "id": req_id,
                        "ts": _ts_of(start) or _ts_of(event) or _now_iso(),
                    },
                    _event_key(start) if start else None,
                    None,
                )
            )
            resp_payload = _chat_response_payload(start, event)
            resp_id = _core.content_id(resp_payload)
            emitted.append(
                (
                    "chat_response",
                    {
                        "payload": resp_payload,
                        "id": resp_id,
                        "ts": _ts_of(event) or _now_iso(),
                    },
                    _event_key(event),
                    req_id,
                )
            )
            if start is not None:
                span_to_response[_event_key(start)] = resp_id
            span_to_response[_event_key(event)] = resp_id
        elif class_name in _TOOL_CALL_START:
            key = _event_key(event)
            tool_starts[key] = event
            tool_pending_fifo.append(event)
            # `AgentToolCallEvent` is a single, fire-and-forget event in
            # some LlamaIndex versions — emit a tool_call right away in
            # case no matching end event arrives.
            if class_name == "AgentToolCallEvent":
                call_payload = _tool_call_payload(event)
                call_id = _core.content_id(call_payload)
                emitted.append(
                    (
                        "tool_call",
                        {
                            "payload": call_payload,
                            "id": call_id,
                            "ts": _ts_of(event) or _now_iso(),
                        },
                        key,
                        None,
                    )
                )
        elif class_name in _TOOL_CALL_END:
            start = _pop_pair(tool_starts, tool_pending_fifo, event)
            # If we already emitted a fire-and-forget tool_call for the
            # matching AgentToolCallEvent, skip the duplicate call.
            already_emitted = False
            if start is not None:
                start_class = _class_name(start)
                already_emitted = start_class == "AgentToolCallEvent"
            call_id = None
            if not already_emitted:
                call_payload = _tool_call_payload(start or event)
                call_id = _core.content_id(call_payload)
                emitted.append(
                    (
                        "tool_call",
                        {
                            "payload": call_payload,
                            "id": call_id,
                            "ts": _ts_of(start) or _ts_of(event) or _now_iso(),
                        },
                        _event_key(start) if start else None,
                        None,
                    )
                )
            else:
                # Find the previously emitted tool_call id for parenting.
                start_key = _event_key(start) if start else None
                for kind, info, owner_key, _parent in reversed(emitted):
                    if kind == "tool_call" and owner_key == start_key:
                        call_id = info["id"]
                        break
            result_payload = _tool_result_payload(start, event)
            result_id = _core.content_id(result_payload)
            emitted.append(
                (
                    "tool_result",
                    {
                        "payload": result_payload,
                        "id": result_id,
                        "ts": _ts_of(event) or _now_iso(),
                    },
                    _event_key(event),
                    call_id,
                )
            )
        elif class_name in _STEP_START or class_name in _STEP_END:
            # Agent step boundaries are useful for tracing but don't
            # map to a Shadow record kind. Drop them — they're already
            # captured implicitly by the parent chain of chat/tool
            # records.
            continue
        else:
            unknown += 1

    if unknown:
        meta_payload["unknown_event_count"] = unknown

    meta_id = _core.content_id(meta_payload)
    out.append(
        {
            "version": "0.1",
            "id": meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": meta_payload,
        }
    )
    last_parent: str = meta_id
    for kind, info, _owner, parent_hint in emitted:
        parent = parent_hint or last_parent
        out.append(
            {
                "version": "0.1",
                "id": info["id"],
                "kind": kind,
                "ts": info["ts"],
                "parent": parent,
                "payload": info["payload"],
            }
        )
        last_parent = info["id"]
    return out


# ---- normalisation --------------------------------------------------------


def _normalise_input(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a flat event list + optional trace metadata."""
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)], {}
    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            meta = {k: v for k, v in data.items() if k != "events"}
            return [e for e in data["events"] if isinstance(e, dict)], meta
        if isinstance(data.get("trace"), dict):
            inner = data["trace"]
            if isinstance(inner.get("events"), list):
                meta = {k: v for k, v in data.items() if k != "trace"}
                return [e for e in inner["events"] if isinstance(e, dict)], meta
        # Single-event convenience.
        if _class_name(data):
            return [data], {}
    raise ShadowConfigError(
        f"LlamaIndex input must be a list of events, an object wrapping "
        f"`events`, or a single event dict; got {type(data).__name__}."
    )


# ---- event helpers --------------------------------------------------------


def _class_name(event: dict[str, Any]) -> str:
    """Return the class-name discriminator for a LlamaIndex event dict."""
    for key in ("class_name", "event_type", "type", "name"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _event_key(event: dict[str, Any] | None) -> str:
    if not isinstance(event, dict):
        return ""
    for key in ("id_", "span_id", "id", "event_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _pop_pair(
    table: dict[str, dict[str, Any]],
    fifo: list[dict[str, Any]],
    end_event: dict[str, Any],
) -> dict[str, Any] | None:
    """Find and remove the matching start event for a given end event."""
    key = _event_key(end_event)
    if key and key in table:
        start = table.pop(key)
        with contextlib.suppress(ValueError):
            fifo.remove(start)
        return start
    if fifo:
        start = fifo.pop(0)
        start_key = _event_key(start)
        if start_key:
            table.pop(start_key, None)
        return start
    return None


def _ts_of(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    value = event.get("timestamp") or event.get("ts")
    if isinstance(value, str) and value:
        return value
    return None


# ---- chat payload builders ------------------------------------------------


def _chat_request_payload(start: dict[str, Any] | None, end: dict[str, Any]) -> dict[str, Any]:
    base = start if isinstance(start, dict) else end
    model = _extract_model(base) or _extract_model(end)
    messages = _extract_messages(base)
    params = _extract_params(base)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "params": params,
    }
    return payload


def _chat_response_payload(start: dict[str, Any] | None, end: dict[str, Any]) -> dict[str, Any]:
    model = _extract_model(end) or _extract_model(start) or ""
    content = _extract_response_content(end)
    usage_raw = _find_usage(end) or _find_usage(start) or {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
        "output_tokens": int(
            usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0
        ),
        "thinking_tokens": int(usage_raw.get("reasoning_tokens", 0) or 0),
    }
    stop_reason = _normalise_stop_reason(end, content)
    latency_ms = _duration_ms(_ts_of(start), _ts_of(end))
    return {
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _extract_model(event: dict[str, Any] | None) -> str:
    if not isinstance(event, dict):
        return ""
    for key in ("model", "model_name"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    # LLMChat events nest model metadata under `model_dict` /
    # `additional_kwargs` on some LlamaIndex versions.
    nested = event.get("model_dict") or event.get("additional_kwargs") or {}
    if isinstance(nested, dict):
        for key in ("model", "model_name"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    # Try to pull from a nested response object (LLMChatEndEvent has
    # `response.raw.model`).
    response = event.get("response")
    if isinstance(response, dict):
        for key in ("model", "model_name"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        raw = response.get("raw")
        if isinstance(raw, dict):
            value = raw.get("model")
            if isinstance(value, str) and value:
                return value
    return ""


def _extract_messages(event: dict[str, Any]) -> list[dict[str, Any]]:
    raw = event.get("messages")
    out: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, dict):
                out.append(_normalise_message(m))
            elif isinstance(m, str):
                out.append({"role": "user", "content": m})
    elif isinstance(raw, dict):
        out.append(_normalise_message(raw))
    elif isinstance(event.get("prompt"), str):
        out.append({"role": "user", "content": event["prompt"]})
    return out


def _normalise_message(m: dict[str, Any]) -> dict[str, Any]:
    role = m.get("role")
    if not isinstance(role, str):
        role = str(m.get("type") or "user")
    role = role.lower()
    role = {
        "human": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
    }.get(role, role)
    content = m.get("content")
    if content is None:
        # Some LlamaIndex ChatMessage variants put text in
        # `blocks: [{block_type: "text", text: "..."}]`.
        blocks = m.get("blocks")
        if isinstance(blocks, list):
            parts: list[str] = []
            for b in blocks:
                if isinstance(b, dict):
                    text = b.get("text") or b.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            content = "\n".join(parts)
        else:
            content = ""
    return {"role": role, "content": content if content is not None else ""}


def _extract_params(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("additional_kwargs") or event.get("params") or {}
    if not isinstance(raw, dict):
        return {}
    params: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ):
        if key in raw and raw[key] is not None:
            params[key] = raw[key]
    return params


def _extract_response_content(event: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    response = event.get("response")
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            text = message.get("content")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            tool_calls = message.get("additional_kwargs", {}).get("tool_calls") or []
            blocks.extend(_normalise_tool_use_blocks(tool_calls))
        text = response.get("text") or response.get("content")
        if isinstance(text, str) and text and not blocks:
            blocks.append({"type": "text", "text": text})
        raw_blocks = response.get("blocks")
        if isinstance(raw_blocks, list) and not blocks:
            for b in raw_blocks:
                if isinstance(b, dict):
                    text = b.get("text") or b.get("content")
                    if isinstance(text, str):
                        blocks.append({"type": "text", "text": text})
    elif isinstance(response, str):
        blocks.append({"type": "text", "text": response})
    return blocks


def _normalise_tool_use_blocks(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    blocks: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        args = (
            call.get("arguments")
            or call.get("args")
            or call.get("function", {}).get("arguments", {})
            if isinstance(call.get("function"), dict)
            else call.get("arguments") or call.get("args") or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        name = call.get("name") or call.get("tool_name")
        if not name and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or call.get("tool_call_id") or f"li_{len(blocks)}",
                "name": name or "",
                "input": args if isinstance(args, dict) else {},
            }
        )
    return blocks


def _find_usage(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    direct = event.get("usage")
    if isinstance(direct, dict):
        return dict(direct)
    response = event.get("response")
    if isinstance(response, dict):
        resp_usage = response.get("usage")
        if isinstance(resp_usage, dict):
            return dict(resp_usage)
        raw = response.get("raw")
        if isinstance(raw, dict):
            raw_usage = raw.get("usage")
            if isinstance(raw_usage, dict):
                return dict(raw_usage)
    return None


def _normalise_stop_reason(event: dict[str, Any], content: list[dict[str, Any]]) -> str:
    raw: Any = None
    response = event.get("response")
    if isinstance(response, dict):
        raw = response.get("finish_reason") or response.get("stop_reason")
        if raw is None:
            raw_obj = response.get("raw")
            if isinstance(raw_obj, dict):
                raw = raw_obj.get("finish_reason") or raw_obj.get("stop_reason")
                if raw is None:
                    choices = raw_obj.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            raw = first.get("finish_reason")
    if raw is None:
        raw = event.get("finish_reason") or event.get("stop_reason")
    if isinstance(raw, str):
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "tool-calls": "tool_use",
            "tool_use": "tool_use",
            "content_filter": "content_filter",
            "content-filter": "content_filter",
        }
        return mapping.get(raw.lower(), raw.lower())
    if any(b.get("type") == "tool_use" for b in content):
        return "tool_use"
    return "end_turn"


# ---- tool payload builders ------------------------------------------------


def _tool_call_payload(event: dict[str, Any]) -> dict[str, Any]:
    args = (
        event.get("arguments")
        or event.get("tool_kwargs")
        or event.get("input")
        or event.get("args")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_value": _jsonify(args)}
    return {
        "tool_name": str(event.get("tool_name") or event.get("name") or ""),
        "tool_call_id": str(
            event.get("tool_call_id") or event.get("id_") or event.get("span_id") or ""
        ),
        "arguments": args,
    }


def _tool_result_payload(start: dict[str, Any] | None, end: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = str(
        end.get("tool_call_id")
        or (start or {}).get("tool_call_id")
        or end.get("id_")
        or (start or {}).get("id_")
        or end.get("span_id")
        or ""
    )
    output_raw = (
        end.get("output") or end.get("response") or end.get("result") or end.get("tool_output")
    )
    is_error = bool(end.get("is_error") or end.get("error"))
    if isinstance(output_raw, dict):
        text = output_raw.get("content") or output_raw.get("output") or output_raw.get("text")
        if isinstance(text, str):
            output: Any = text
        else:
            output = _jsonify(output_raw)
    elif output_raw is None and is_error:
        err = end.get("error")
        output = str(err) if err is not None else ""
    else:
        output = _jsonify(output_raw) if output_raw is not None else ""
    return {
        "tool_call_id": tool_call_id,
        "output": output,
        "is_error": is_error,
    }


# ---- low-level helpers ----------------------------------------------------


def _duration_ms(start_ts: str | None, end_ts: str | None) -> int:
    if not start_ts or not end_ts:
        return 0
    try:
        s = datetime.datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        e = datetime.datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        if s.tzinfo is None:
            s = s.replace(tzinfo=datetime.UTC)
        if e.tzinfo is None:
            e = e.replace(tzinfo=datetime.UTC)
        return max(0, int((e - s).total_seconds() * 1000))
    except ValueError:
        return 0


def _jsonify(obj: Any) -> Any:
    """Coerce arbitrary values into a JSON-stable shape for content-addressing."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = [
    "LLAMAINDEX_FORMAT",
    "import_llamaindex_events",
    "llamaindex_to_agentlog",
]
