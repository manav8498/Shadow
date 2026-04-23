"""OpenTelemetry OTLP/JSON → Shadow `.agentlog` importer.

This is the network-effect converter: ANY OpenTelemetry-instrumented
agent (Python, Go, Rust, Node, Java — thousands of existing
deployments) can export OTLP/JSON and pipe it into Shadow without
rewriting instrumentation.

OTLP shape:

    {
      "resourceSpans": [
        {
          "resource": { "attributes": [...] },
          "scopeSpans": [
            {
              "scope": { "name": "openai", ... },
              "spans": [
                {
                  "traceId": "...",
                  "spanId": "...",
                  "parentSpanId": "...",
                  "name": "gen_ai.chat",
                  "kind": 3,
                  "startTimeUnixNano": "...",
                  "endTimeUnixNano": "...",
                  "attributes": [
                    {"key": "gen_ai.request.model", "value": {"stringValue": "..."}},
                    ...
                  ],
                  "status": {...}
                }
              ]
            }
          ]
        }
      ]
    }

We emit one Shadow `chat_request` + `chat_response` pair per span whose
attributes contain `gen_ai.*` keys. Other spans are dropped.

Attribute mapping:

  gen_ai.request.model          → request.model
  gen_ai.response.model         → response.model (falls back to request)
  gen_ai.request.temperature    → request.params.temperature
  gen_ai.request.top_p          → request.params.top_p
  gen_ai.request.max_tokens     → request.params.max_tokens
  gen_ai.response.finish_reasons → response.stop_reason
  gen_ai.usage.input_tokens     → response.usage.input_tokens
  gen_ai.usage.output_tokens    → response.usage.output_tokens
  gen_ai.usage.thinking_tokens  → response.usage.thinking_tokens
  gen_ai.prompt.N.*             → request.messages[N]   (role, content)
  gen_ai.completion.N.*         → response.content[N]

Shadow's IDs are SHA-256 of the canonical payload. OTel trace_id and
span_id are preserved in `meta.otel_trace_id` / `meta.otel_span_id`
for traceability back to the source.
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

OTEL_FORMAT = "otel"


def otel_to_agentlog(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an OTLP/JSON payload to Shadow records."""
    resource_spans = data.get("resourceSpans")
    if not isinstance(resource_spans, list):
        raise ShadowConfigError(
            "OTLP input missing top-level `resourceSpans` list.\n"
            "hint: pipe output of `otel-cli export` or any collector to a file."
        )

    meta_payload: dict[str, Any] = {
        "sdk": {"name": "shadow", "version": "0.1.0"},
        "imported_from": OTEL_FORMAT,
    }
    meta_id = _core.content_id(meta_payload)
    records: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": meta_payload,
        }
    ]
    last_parent = meta_id

    spans = _flatten_spans(resource_spans)
    chat_spans = [s for s in spans if _is_chat_span(s)]
    for span in chat_spans:
        attrs = _attrs_as_dict(span.get("attributes", []))
        start_ts = _nano_to_iso(span.get("startTimeUnixNano"))
        end_ts = _nano_to_iso(span.get("endTimeUnixNano"))
        req_payload = _span_to_request_payload(span, attrs)
        req_id = _core.content_id(req_payload)
        records.append(
            _envelope(
                "chat_request",
                req_id,
                start_ts,
                last_parent,
                req_payload,
                otel_span_id=span.get("spanId", ""),
                otel_trace_id=span.get("traceId", ""),
            )
        )
        resp_payload = _span_to_response_payload(span, attrs, start_ts, end_ts)
        resp_id = _core.content_id(resp_payload)
        records.append(
            _envelope(
                "chat_response",
                resp_id,
                end_ts,
                req_id,
                resp_payload,
                otel_span_id=span.get("spanId", ""),
                otel_trace_id=span.get("traceId", ""),
            )
        )
        last_parent = resp_id

    return records


def _flatten_spans(resource_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rs in resource_spans:
        for ss in rs.get("scopeSpans", []):
            out.extend(ss.get("spans", []))
    return out


def _is_chat_span(span: dict[str, Any]) -> bool:
    attrs = _attrs_as_dict(span.get("attributes", []))
    # Heuristic: any gen_ai.request.model attribute or name starting
    # with gen_ai.chat.
    return "gen_ai.request.model" in attrs or str(span.get("name", "")).startswith("gen_ai.chat")


def _attrs_as_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in attrs:
        key = a.get("key")
        if not isinstance(key, str):
            continue
        val = a.get("value") or {}
        out[key] = _decode_any_value(val)
    return out


def _decode_any_value(v: dict[str, Any]) -> Any:
    """Decode the OTLP AnyValue oneof into a Python scalar / list."""
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_decode_any_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {
            kv.get("key", ""): _decode_any_value(kv.get("value") or {})
            for kv in v["kvlistValue"].get("values", [])
        }
    return None


def _span_to_request_payload(span: dict[str, Any], attrs: dict[str, Any]) -> dict[str, Any]:
    model = str(attrs.get("gen_ai.request.model", ""))
    messages = _extract_prompt_messages(attrs)
    params: dict[str, Any] = {}
    for src, dst in (
        ("gen_ai.request.temperature", "temperature"),
        ("gen_ai.request.top_p", "top_p"),
        ("gen_ai.request.max_tokens", "max_tokens"),
    ):
        if src in attrs and attrs[src] is not None:
            params[dst] = attrs[src]
    return {
        "model": model,
        "messages": messages,
        "params": params,
    }


def _span_to_response_payload(
    span: dict[str, Any],
    attrs: dict[str, Any],
    start_ts: str,
    end_ts: str,
) -> dict[str, Any]:
    model = str(attrs.get("gen_ai.response.model", attrs.get("gen_ai.request.model", "")))
    content = _extract_completion_content(attrs)
    stop_reason = _extract_stop_reason(attrs, span)
    latency_ms = _duration_ms(start_ts, end_ts)
    usage = {
        "input_tokens": int(attrs.get("gen_ai.usage.input_tokens", 0) or 0),
        "output_tokens": int(attrs.get("gen_ai.usage.output_tokens", 0) or 0),
        "thinking_tokens": int(attrs.get("gen_ai.usage.thinking_tokens", 0) or 0),
    }
    return {
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _extract_prompt_messages(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect `gen_ai.prompt.N.role` / `.content` into a messages list."""
    by_idx: dict[int, dict[str, Any]] = {}
    for key, val in attrs.items():
        if not key.startswith("gen_ai.prompt."):
            continue
        rest = key[len("gen_ai.prompt.") :]
        if "." not in rest:
            continue
        idx_str, field = rest.split(".", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        by_idx.setdefault(idx, {})[field] = val
    messages: list[dict[str, Any]] = []
    for i in sorted(by_idx.keys()):
        m = by_idx[i]
        messages.append({"role": str(m.get("role", "user")), "content": m.get("content", "")})
    return messages


def _extract_completion_content(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect `gen_ai.completion.N.*` into a content parts list."""
    by_idx: dict[int, dict[str, Any]] = {}
    for key, val in attrs.items():
        if not key.startswith("gen_ai.completion."):
            continue
        rest = key[len("gen_ai.completion.") :]
        if "." not in rest:
            continue
        idx_str, field = rest.split(".", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        by_idx.setdefault(idx, {})[field] = val
    parts: list[dict[str, Any]] = []
    for i in sorted(by_idx.keys()):
        p = by_idx[i]
        content = p.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
    return parts


def _extract_stop_reason(attrs: dict[str, Any], span: dict[str, Any]) -> str:
    fr = attrs.get("gen_ai.response.finish_reasons")
    if isinstance(fr, list) and fr:
        return str(fr[0])
    if isinstance(fr, str):
        return fr
    status_code = (span.get("status") or {}).get("code", 1)
    if status_code == 2:
        return "error"
    return "end_turn"


def _nano_to_iso(ns: Any) -> str:
    if ns is None:
        return _now_iso()
    try:
        n = int(ns)
        secs = n // 1_000_000_000
        ms = (n // 1_000_000) % 1000
        dt = datetime.datetime.fromtimestamp(secs, tz=datetime.UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
    except (TypeError, ValueError):
        return _now_iso()


def _duration_ms(start_ts: str, end_ts: str) -> int:
    try:
        start_dt = datetime.datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        end_dt = datetime.datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        return max(0, int((end_dt - start_dt).total_seconds() * 1000))
    except ValueError:
        return 0


def _envelope(
    kind: str,
    record_id: str,
    ts: str,
    parent: str | None,
    payload: dict[str, Any],
    *,
    otel_span_id: str,
    otel_trace_id: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if otel_trace_id:
        meta["otel_trace_id"] = otel_trace_id
    if otel_span_id:
        meta["otel_span_id"] = otel_span_id
    env: dict[str, Any] = {
        "version": "0.1",
        "id": record_id,
        "kind": kind,
        "ts": ts,
        "parent": parent,
        "payload": payload,
    }
    if meta:
        env["meta"] = meta
    return env


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["OTEL_FORMAT", "otel_to_agentlog"]
