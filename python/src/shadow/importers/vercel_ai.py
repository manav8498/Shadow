"""Vercel AI SDK trace-export → Shadow `.agentlog` converter.

The Vercel AI SDK (`ai` npm package) emits traces via its OpenTelemetry
integration — each `generateText` / `generateObject` / `streamText`
call produces a span with `ai.*` attributes. Their `AISDKExporter`
captures these spans and ships them over OTLP or dumps them to JSON.

This importer accepts two shapes in the wild:

1. **OTLP-style spans** — `{"spans": [{...attributes: {"ai.prompt.messages": ...}}]}`
   The export format produced by `@vercel/otel` + the AI SDK telemetry
   hook. Attributes follow Vercel's convention (namespace `ai.*`).

2. **Event-log shape** — `{"events": [{"type": "generation", ...}]}`
   Emitted by the Vercel AI Observability dashboard's "export JSON"
   button. Simpler schema, no OTel plumbing.

Both shapes round-trip to Shadow's chat_request/chat_response pair per
generation. Tool calls surface as Anthropic-style `tool_use` /
`tool_result` content blocks so the rest of the differ just works.

## Field mappings

| Vercel AI field             | Shadow field                           |
|-----------------------------|----------------------------------------|
| `ai.model.id`               | `payload.model` (both request + response) |
| `ai.prompt.messages`        | `payload.messages`                     |
| `ai.prompt.system`          | injected as `messages[0]` system turn  |
| `ai.response.text`          | response `content[{type:text,text}]`  |
| `ai.response.toolCalls`     | response `content[{type:tool_use,...}]` |
| `ai.usage.promptTokens`     | `payload.usage.input_tokens`          |
| `ai.usage.completionTokens` | `payload.usage.output_tokens`         |
| `ai.finishReason`           | `payload.stop_reason` (normalised)    |
| `durationMs` / end-start    | `payload.latency_ms`                  |
| `ai.settings.*`             | `payload.params.*`                    |

Unknown `ai.*` keys are preserved under `payload.source.extras` so
nothing is silently dropped.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

VERCEL_AI_FORMAT = "vercel-ai"


def vercel_ai_to_agentlog(data: Any) -> list[dict[str, Any]]:
    """Convert a parsed Vercel AI SDK trace export → Shadow records."""
    spans, trace_meta = _normalise_input(data)
    if not spans:
        raise ShadowConfigError(
            "Vercel AI input contained no generation spans — check the file format.\n"
            "hint: `shadow import --format vercel-ai` expects either an "
            "OTLP-style `{spans: [...]}` object or the Vercel AI dashboard's "
            "`{events: [...]}` export."
        )

    out: list[dict[str, Any]] = []
    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow-import-vercel-ai", "version": "1.2"},
        "source": {"format": "vercel-ai"},
    }
    if trace_meta:
        meta_payload["vercel_trace_metadata"] = trace_meta
    meta_record = _make_record("metadata", meta_payload, parent=None, ts=_now_iso())
    out.append(meta_record)
    last_parent: str = meta_record["id"]

    for span in spans:
        req_payload, resp_payload = _span_to_payloads(span)
        if req_payload is None:
            # Non-generation span (e.g. `embed`, `speech`) — we don't
            # model these in Shadow's axes yet. Skip silently; the
            # metadata record already attributes the source.
            continue
        req_record = _make_record(
            "chat_request", req_payload, parent=last_parent, ts=_span_start(span)
        )
        out.append(req_record)
        if resp_payload is not None:
            resp_record = _make_record(
                "chat_response", resp_payload, parent=req_record["id"], ts=_span_end(span)
            )
            out.append(resp_record)
            last_parent = resp_record["id"]
        else:
            last_parent = req_record["id"]

    return out


# ---- normalisation --------------------------------------------------------


def _normalise_input(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract the generation-span list + optional trace metadata."""
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)], {}
    if not isinstance(data, dict):
        raise ShadowConfigError(
            f"Vercel AI input must be a list of spans or an object wrapping "
            f"`spans`/`events`; got {type(data).__name__}."
        )
    # OTLP-style: top-level `spans` list.
    if isinstance(data.get("spans"), list):
        meta = {k: v for k, v in data.items() if k != "spans"}
        return [s for s in data["spans"] if isinstance(s, dict)], meta
    # Dashboard-style: top-level `events` list.
    if isinstance(data.get("events"), list):
        meta = {k: v for k, v in data.items() if k != "events"}
        return [e for e in data["events"] if isinstance(e, dict)], meta
    # Single-span convenience.
    if _looks_like_span(data):
        return [data], {}
    raise ShadowConfigError(
        "Vercel AI input had no recognisable `spans` or `events` list.\n"
        "hint: run `vercel ai traces export --format json` or dump "
        "`aiSdkExporter.exported()`, and pass that file to "
        "`shadow import --format vercel-ai`."
    )


def _looks_like_span(obj: dict[str, Any]) -> bool:
    """Heuristic: a dict is a span if it has ai.* attributes anywhere."""
    attrs = obj.get("attributes") or obj
    if not isinstance(attrs, dict):
        return False
    return any(k.startswith("ai.") for k in attrs)


# ---- span → payload -------------------------------------------------------


def _span_to_payloads(
    span: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Convert one Vercel AI span → (chat_request_payload, chat_response_payload)."""
    attrs = _collect_attrs(span)
    # Only generation-shaped spans produce request/response pairs. Probe
    # by presence of either the input prompt or the output text.
    has_input = any(k in attrs for k in ("ai.prompt.messages", "ai.prompt"))
    has_output = any(
        k in attrs
        for k in (
            "ai.response.text",
            "ai.response.toolCalls",
            "ai.response.object",
            "ai.response.finishReason",
        )
    )
    if not has_input and not has_output:
        return None, None

    model = (
        attrs.get("ai.model.id")
        or attrs.get("ai.model")
        or attrs.get("gen_ai.request.model")
        or "vercel-ai-unknown"
    )
    messages = _extract_messages(attrs)
    params = _extract_params(attrs)
    tools = _extract_tools(attrs)

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "params": params,
    }
    if tools:
        request_payload["tools"] = tools

    # Stash any unrecognised ai.* keys for forensics.
    extras = {
        k: v
        for k, v in attrs.items()
        if k.startswith("ai.")
        and k
        not in {
            "ai.model.id",
            "ai.model",
            "ai.prompt.messages",
            "ai.prompt",
            "ai.prompt.system",
            "ai.response.text",
            "ai.response.toolCalls",
            "ai.response.object",
            "ai.response.finishReason",
            "ai.usage.promptTokens",
            "ai.usage.completionTokens",
            "ai.usage.totalTokens",
            "ai.settings.temperature",
            "ai.settings.topP",
            "ai.settings.topK",
            "ai.settings.maxTokens",
            "ai.settings.stopSequences",
            "ai.tools",
        }
    }
    if extras:
        request_payload.setdefault("source", {})["extras"] = _jsonify(extras)

    content_blocks = _build_response_content(attrs)
    stop_reason = _normalise_finish_reason(
        attrs.get("ai.response.finishReason") or attrs.get("gen_ai.response.finish_reasons")
    )
    latency_ms = _span_duration_ms(span)
    usage = {
        "input_tokens": int(attrs.get("ai.usage.promptTokens") or 0),
        "output_tokens": int(attrs.get("ai.usage.completionTokens") or 0),
        "thinking_tokens": int(attrs.get("ai.usage.reasoningTokens") or 0),
    }
    response_payload: dict[str, Any] = {
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }
    # Error spans — map to an error-ish stop_reason and a text block.
    status = (span.get("status") or {}).get("code") or span.get("statusCode")
    if isinstance(status, str) and status.upper() in {"ERROR", "FAILED"}:
        response_payload["stop_reason"] = "error"
        msg = (span.get("status") or {}).get("message") or "unknown error"
        response_payload["content"] = [{"type": "text", "text": f"error: {msg}"}]

    return request_payload, response_payload


def _collect_attrs(span: dict[str, Any]) -> dict[str, Any]:
    """Merge `span.attributes` with top-level keys starting with `ai.`.

    The OTLP shape nests attributes; the dashboard shape lays them flat.
    We accept both.
    """
    merged: dict[str, Any] = {}
    attrs = span.get("attributes")
    if isinstance(attrs, dict):
        merged.update(attrs)
    for k, v in span.items():
        if k == "attributes":
            continue
        if isinstance(k, str) and k.startswith("ai."):
            merged[k] = v
    return merged


def _extract_messages(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the prompt as a Shadow `messages` list.

    Vercel AI serialises messages as a JSON-string attribute (OTel
    can't hold arrays natively on some exporters). We accept either
    a string or a pre-parsed list.
    """
    raw = attrs.get("ai.prompt.messages")
    messages: list[dict[str, Any]] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            messages = [m for m in parsed if isinstance(m, dict)]
    elif isinstance(raw, list):
        messages = [m for m in raw if isinstance(m, dict)]
    elif raw is None:
        # Some Vercel AI calls use `ai.prompt` (a single prompt string).
        prompt = attrs.get("ai.prompt")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]

    system = attrs.get("ai.prompt.system")
    if isinstance(system, str):
        if messages and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": system}
        else:
            messages = [{"role": "system", "content": system}, *messages]
    return messages


def _extract_params(attrs: dict[str, Any]) -> dict[str, Any]:
    """Pull the settings.* knobs into a Shadow `params` block."""
    params: dict[str, Any] = {}
    mapping = {
        "ai.settings.temperature": "temperature",
        "ai.settings.topP": "top_p",
        "ai.settings.topK": "top_k",
        "ai.settings.maxTokens": "max_tokens",
        "ai.settings.stopSequences": "stop_sequences",
        "ai.settings.presencePenalty": "presence_penalty",
        "ai.settings.frequencyPenalty": "frequency_penalty",
    }
    for src, dst in mapping.items():
        if src in attrs and attrs[src] is not None:
            params[dst] = attrs[src]
    return params


def _extract_tools(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Vercel AI serialises the advertised tool set as a JSON-string attribute."""
    raw = attrs.get("ai.tools") or attrs.get("ai.prompt.tools")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        parsed = raw
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for t in parsed:
        if not isinstance(t, dict):
            continue
        # Vercel AI tools: {name, description, parameters (JSON-schema)}.
        out.append(
            {
                "name": t.get("name") or t.get("toolName", ""),
                "description": t.get("description", ""),
                "input_schema": t.get("parameters") or t.get("inputSchema") or {},
            }
        )
    return out


def _build_response_content(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Assemble Anthropic-style content blocks from the response attrs."""
    blocks: list[dict[str, Any]] = []
    text = attrs.get("ai.response.text")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    obj = attrs.get("ai.response.object")
    if obj is not None and not isinstance(obj, str):
        blocks.append({"type": "text", "text": json.dumps(obj, sort_keys=True, default=str)})
    elif isinstance(obj, str) and obj and not text:
        blocks.append({"type": "text", "text": obj})

    raw_calls = attrs.get("ai.response.toolCalls")
    if isinstance(raw_calls, str):
        try:
            raw_calls = json.loads(raw_calls)
        except json.JSONDecodeError:
            raw_calls = None
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args") or call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("toolCallId") or call.get("id") or f"vercel_{len(blocks)}",
                    "name": call.get("toolName") or call.get("name", ""),
                    "input": args if isinstance(args, dict) else {},
                }
            )
    return blocks


# ---- small helpers --------------------------------------------------------


def _normalise_finish_reason(value: Any) -> str:
    """Map Vercel AI's finishReason enum onto Shadow's stop_reason space."""
    if value is None:
        return "end_turn"
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return "end_turn"
    lowered = value.lower()
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool-calls": "tool_use",
        "tool_calls": "tool_use",
        "content-filter": "content_filter",
        "content_filter": "content_filter",
        "error": "error",
        "other": "end_turn",
    }
    return mapping.get(lowered, lowered)


def _span_duration_ms(span: dict[str, Any]) -> int:
    """Compute latency in ms from whatever timestamp fields are present."""
    if isinstance(span.get("durationMs"), int | float):
        return int(span["durationMs"])
    start = _parse_ts(span.get("startTime") or span.get("startTimeUnixNano"))
    end = _parse_ts(span.get("endTime") or span.get("endTimeUnixNano"))
    if start is not None and end is not None and end >= start:
        return int(round((end - start) * 1000))
    return 0


def _parse_ts(value: Any) -> float | None:
    """Return a seconds-since-epoch float, or None.

    Accepts ISO-8601 strings, epoch-millis ints, OTLP UnixNano strings/ints.
    """
    if value is None:
        return None
    if isinstance(value, int | float):
        v = float(value)
        if v > 1e17:
            return v / 1e9  # nanos
        if v > 1e11:
            return v / 1e3  # millis
        return v  # seconds
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(value) / 1e9
            except ValueError:
                return None
    return None


def _span_start(span: dict[str, Any]) -> str:
    ts = _parse_ts(span.get("startTime") or span.get("startTimeUnixNano"))
    if ts is not None:
        return _iso_from_epoch(ts)
    return _now_iso()


def _span_end(span: dict[str, Any]) -> str:
    ts = _parse_ts(span.get("endTime") or span.get("endTimeUnixNano"))
    if ts is not None:
        return _iso_from_epoch(ts)
    return _now_iso()


def _iso_from_epoch(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _jsonify(obj: Any) -> Any:
    """Make sure extras are JSON-serialisable so content-addressing stays stable."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)


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


__all__ = ["VERCEL_AI_FORMAT", "vercel_ai_to_agentlog"]
