"""Export a Shadow `.agentlog` as OpenTelemetry OTLP/JSON.

Emits the "otlp-json" format (the JSON encoding of the OTLP protobuf
wire format) that OTLP-compatible collectors accept directly. Attribute
keys follow the OpenTelemetry **GenAI semantic conventions** (SPEC §7):

    gen_ai.system                  (anthropic | openai | ...)
    gen_ai.request.model
    gen_ai.response.model
    gen_ai.request.temperature
    gen_ai.request.top_p
    gen_ai.request.max_tokens
    gen_ai.response.finish_reasons   (list)
    gen_ai.usage.input_tokens
    gen_ai.usage.output_tokens
    gen_ai.usage.thinking_tokens

One span per `chat_request`/`chat_response` pair. `tool_call` /
`tool_result` pairs get their own child spans with `gen_ai.tool.*`
attributes. Non-paired records (orphan requests, errors) become
single-sided spans with the status set accordingly.

Shadow trace IDs (SHA-256 of canonical payload) are 64 hex chars; OTel
spans use 16-hex spanId and 32-hex traceId. We derive them by slicing
the first 32 / 16 hex chars of the Shadow id, which preserves
collision resistance within any realistic trace volume and keeps
records deterministically linked.
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import __version__


def agentlog_to_otel(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a parsed `.agentlog` record list to OTLP/JSON.

    Returns a dict ready to `json.dumps(...)` into a `.otel.json` file
    that OTLP collectors can ingest via the `/v1/traces` endpoint.
    """
    meta = records[0] if records else None
    trace_id = _trace_id_from(meta) if meta else _pad_hex("0" * 32, 32)
    spans: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in records}
    responded_request_ids: set[str] = set()
    tool_result_for_call: dict[str, dict[str, Any]] = {}

    for r in records:
        if r.get("kind") == "chat_response":
            parent_id = r.get("parent")
            if parent_id is not None:
                responded_request_ids.add(parent_id)
        elif r.get("kind") == "tool_result":
            payload = r.get("payload") or {}
            tc_id = payload.get("tool_call_id")
            if isinstance(tc_id, str):
                # Keep the FIRST tool_result for a given id. Subsequent
                # results with the same id (model bug / data corruption)
                # are ignored rather than silently overwriting — the span
                # for the duplicate tool_call will just lack a result,
                # which is honest.
                tool_result_for_call.setdefault(tc_id, r)

    for r in records:
        kind = r.get("kind")
        if kind == "chat_response":
            req = by_id.get(r.get("parent") or "")
            if req is None or req.get("kind") != "chat_request":
                continue
            spans.append(_chat_span(trace_id, req, r))
        elif kind == "chat_request":
            if r["id"] not in responded_request_ids:
                # Request with no matching response (e.g., error followed).
                spans.append(_chat_span(trace_id, r, None))
        elif kind == "tool_call":
            payload = r.get("payload") or {}
            call_id = payload.get("tool_call_id")
            result = tool_result_for_call.get(call_id) if isinstance(call_id, str) else None
            spans.append(_tool_span(trace_id, r, result))
        elif kind == "error":
            # Orphan errors get their own single-sided span for visibility.
            spans.append(_error_span(trace_id, r))

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", "shadow-export"),
                        _attr("shadow.spec.version", "0.1"),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "shadow", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _chat_span(
    trace_id: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    req_payload = request.get("payload") or {}
    resp_payload = (response or {}).get("payload") or {}
    start_ts = request.get("ts", "")
    end_ts = (response or {}).get("ts", start_ts)
    params = req_payload.get("params") or {}
    provider = _infer_system(req_payload.get("model", ""))
    attrs: list[dict[str, Any]] = [
        # v1.37+ semconv: `gen_ai.provider.name` (was `gen_ai.system`).
        # We emit both during the deprecation window so older ingestors
        # (pre-v1.37) still see something familiar, and new ingestors see
        # the current key.
        _attr("gen_ai.provider.name", provider),
        _attr("gen_ai.system", provider),  # deprecated, kept for compat
        _attr("gen_ai.operation.name", "chat"),  # required in v1.37+
        _attr("gen_ai.request.model", req_payload.get("model", "")),
    ]
    trace_meta = request.get("meta") or {}
    if trace_meta.get("trace_id"):
        attrs.append(_attr("gen_ai.conversation.id", trace_meta["trace_id"]))
    if "temperature" in params:
        attrs.append(_attr("gen_ai.request.temperature", float(params["temperature"])))
    if "top_p" in params:
        attrs.append(_attr("gen_ai.request.top_p", float(params["top_p"])))
    if "max_tokens" in params:
        attrs.append(_attr("gen_ai.request.max_tokens", int(params["max_tokens"])))
    if response is not None:
        resp_id = (response or {}).get("id", "")
        if resp_id:
            attrs.append(_attr("gen_ai.response.id", resp_id))
        attrs.append(_attr("gen_ai.response.model", resp_payload.get("model", "")))
        stop = resp_payload.get("stop_reason")
        if stop is not None:
            attrs.append(_attr("gen_ai.response.finish_reasons", [str(stop)]))
        usage = resp_payload.get("usage") or {}
        for key, attr_name in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
            # `thinking_tokens` is a Shadow-ism; OTel has no equivalent yet.
            ("thinking_tokens", "gen_ai.usage.thinking_tokens"),
            ("cached_input_tokens", "gen_ai.usage.cached_input_tokens"),
        ):
            if key in usage:
                attrs.append(_attr(attr_name, int(usage[key])))
        if "latency_ms" in resp_payload:
            attrs.append(_attr("shadow.latency_ms", int(resp_payload["latency_ms"])))
    status = {"code": 1} if response is not None else {"code": 2}  # OK vs ERROR
    return {
        "traceId": trace_id,
        "spanId": _span_id_from(request["id"]),
        "parentSpanId": _span_id_from_optional(request.get("parent")),
        "name": f"gen_ai.chat {req_payload.get('model', 'unknown')}",
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": _rfc3339_to_unix_nano(start_ts),
        "endTimeUnixNano": _rfc3339_to_unix_nano(end_ts),
        "attributes": attrs,
        "status": status,
    }


def _tool_span(
    trace_id: str,
    call: dict[str, Any],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    call_payload = call.get("payload") or {}
    result_payload = (result or {}).get("payload") or {}
    start_ts = call.get("ts", "")
    end_ts = (result or {}).get("ts", start_ts)
    attrs = [
        _attr("gen_ai.tool.name", call_payload.get("tool_name", "")),
        _attr("gen_ai.tool.call_id", call_payload.get("tool_call_id", "")),
    ]
    if result is not None:
        attrs.append(_attr("gen_ai.tool.is_error", bool(result_payload.get("is_error", False))))
        if "latency_ms" in result_payload:
            attrs.append(_attr("shadow.latency_ms", int(result_payload["latency_ms"])))
    status = {"code": 1} if result is not None else {"code": 2}
    if result is not None and result_payload.get("is_error"):
        status = {"code": 2}
    return {
        "traceId": trace_id,
        "spanId": _span_id_from(call["id"]),
        "parentSpanId": _span_id_from_optional(call.get("parent")),
        "name": f"gen_ai.tool.call {call_payload.get('tool_name', 'unknown')}",
        "kind": 3,
        "startTimeUnixNano": _rfc3339_to_unix_nano(start_ts),
        "endTimeUnixNano": _rfc3339_to_unix_nano(end_ts),
        "attributes": attrs,
        "status": status,
    }


def _error_span(trace_id: str, record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload") or {}
    ts = record.get("ts", "")
    attrs = [
        _attr("error.type", payload.get("code", "error")),
        _attr("error.source", payload.get("source", "unknown")),
    ]
    msg = payload.get("message")
    if isinstance(msg, str) and msg:
        attrs.append(_attr("error.message", msg))
    return {
        "traceId": trace_id,
        "spanId": _span_id_from(record["id"]),
        "parentSpanId": _span_id_from_optional(record.get("parent")),
        "name": "shadow.error",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _rfc3339_to_unix_nano(ts),
        "endTimeUnixNano": _rfc3339_to_unix_nano(ts),
        "attributes": attrs,
        "status": {"code": 2},
    }


def _attr(key: str, value: Any) -> dict[str, Any]:
    """Build a single OTLP attribute entry in the JSON protobuf encoding."""
    if isinstance(value, bool):
        val: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        val = {"intValue": str(value)}
    elif isinstance(value, float):
        val = {"doubleValue": value}
    elif isinstance(value, list):
        val = {
            "arrayValue": {
                "values": [_scalar_value(v) for v in value],
            }
        }
    else:
        val = {"stringValue": str(value)}
    return {"key": key, "value": val}


def _scalar_value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _infer_system(model: str) -> str:
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("openai"):
        return "openai"
    return "unknown"


def _trace_id_from(meta: dict[str, Any]) -> str:
    hex_id = _strip_prefix(meta.get("id", ""))
    return _pad_hex(hex_id[:32] if len(hex_id) >= 32 else hex_id, 32)


def _span_id_from(record_id: str) -> str:
    hex_id = _strip_prefix(record_id)
    return _pad_hex(hex_id[:16] if len(hex_id) >= 16 else hex_id, 16)


def _span_id_from_optional(parent_id: str | None) -> str:
    if parent_id is None:
        return ""
    return _span_id_from(parent_id)


def _strip_prefix(s: str) -> str:
    return s[len("sha256:") :] if s.startswith("sha256:") else s


def _pad_hex(s: str, width: int) -> str:
    return s.ljust(width, "0")[:width]


def _rfc3339_to_unix_nano(ts: str) -> str:
    if not ts:
        return "0"
    try:
        cleaned = ts.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(cleaned)
        return str(int(dt.timestamp() * 1_000_000_000))
    except ValueError:
        return "0"


__all__ = ["agentlog_to_otel"]
