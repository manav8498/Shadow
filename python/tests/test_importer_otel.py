"""Tests for the OTLP/JSON → .agentlog importer."""

from __future__ import annotations

from typing import Any

import pytest

from shadow.importers import otel_to_agentlog


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, list):
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}},
        }
    return {"key": key, "value": {"stringValue": str(value)}}


def _otlp(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "test")]},
                "scopeSpans": [{"scope": {"name": "test"}, "spans": spans}],
            }
        ]
    }


def _chat_span(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "traceId": "a" * 32,
        "spanId": "b" * 16,
        "parentSpanId": "",
        "name": "gen_ai.chat gpt-4.1",
        "kind": 3,
        "startTimeUnixNano": "1761134400000000000",  # 2025-10-22T12:00:00Z
        "endTimeUnixNano": "1761134400150000000",
        "attributes": [
            _attr("gen_ai.system", "openai"),
            _attr("gen_ai.request.model", "gpt-4.1"),
            _attr("gen_ai.request.temperature", 0.2),
            _attr("gen_ai.request.max_tokens", 256),
            _attr("gen_ai.response.model", "gpt-4.1"),
            _attr("gen_ai.response.finish_reasons", ["stop"]),
            _attr("gen_ai.usage.input_tokens", 4),
            _attr("gen_ai.usage.output_tokens", 1),
            _attr("gen_ai.prompt.0.role", "user"),
            _attr("gen_ai.prompt.0.content", "hi"),
            _attr("gen_ai.completion.0.role", "assistant"),
            _attr("gen_ai.completion.0.content", "hello"),
        ],
        "status": {"code": 1},
    }
    defaults.update(overrides)
    return defaults


def test_otel_minimal_span_produces_chat_pair() -> None:
    data = _otlp([_chat_span()])
    records = otel_to_agentlog(data)
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4.1"
    assert req["messages"][0] == {"role": "user", "content": "hi"}
    assert req["params"]["temperature"] == 0.2
    assert req["params"]["max_tokens"] == 256
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "hello"
    assert resp["usage"]["input_tokens"] == 4
    assert resp["usage"]["output_tokens"] == 1
    assert resp["latency_ms"] == 150
    assert resp["stop_reason"] == "stop"


def test_otel_preserves_otel_ids_in_meta() -> None:
    data = _otlp([_chat_span()])
    records = otel_to_agentlog(data)
    meta = records[1].get("meta", {})
    assert meta.get("otel_trace_id") == "a" * 32
    assert meta.get("otel_span_id") == "b" * 16


def test_otel_drops_non_genai_spans() -> None:
    non_chat = {
        "traceId": "c" * 32,
        "spanId": "d" * 16,
        "name": "http.request",
        "startTimeUnixNano": "1761134400000000000",
        "endTimeUnixNano": "1761134400001000000",
        "attributes": [_attr("http.method", "GET")],
        "status": {"code": 1},
    }
    data = _otlp([non_chat, _chat_span()])
    records = otel_to_agentlog(data)
    # metadata + one request + one response (the non-chat span is dropped).
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_otel_multiple_chat_spans_chain_parents() -> None:
    s1 = _chat_span(spanId="1" * 16)
    s2 = _chat_span(spanId="2" * 16)
    records = otel_to_agentlog(_otlp([s1, s2]))
    # metadata, req1, resp1, req2, resp2 — resp1 is the parent of req2
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
        "chat_request",
        "chat_response",
    ]
    assert records[3]["parent"] == records[2]["id"]


def test_otel_missing_resource_spans_raises() -> None:
    from shadow.errors import ShadowConfigError

    with pytest.raises(ShadowConfigError, match="resourceSpans"):
        otel_to_agentlog({})


def test_otel_roundtrip_from_our_own_exporter() -> None:
    """Shadow's own OTel export, re-imported, should round-trip cleanly."""
    from shadow.otel import agentlog_to_otel

    baseline = [
        {
            "version": "0.1",
            "id": "sha256:" + "a" * 64,
            "kind": "metadata",
            "ts": "2026-04-21T10:00:00.000Z",
            "parent": None,
            "payload": {"sdk": {"name": "shadow"}},
        },
        {
            "version": "0.1",
            "id": "sha256:" + "b" * 64,
            "kind": "chat_request",
            "ts": "2026-04-21T10:00:01.000Z",
            "parent": "sha256:" + "a" * 64,
            "payload": {
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "hi"}],
                "params": {"temperature": 0.2, "max_tokens": 256},
            },
        },
        {
            "version": "0.1",
            "id": "sha256:" + "c" * 64,
            "kind": "chat_response",
            "ts": "2026-04-21T10:00:01.150Z",
            "parent": "sha256:" + "b" * 64,
            "payload": {
                "model": "gpt-4.1",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "latency_ms": 150,
                "usage": {"input_tokens": 4, "output_tokens": 1, "thinking_tokens": 0},
            },
        },
    ]
    exported = agentlog_to_otel(baseline)
    reimported = otel_to_agentlog(exported)
    # Our exporter doesn't serialize message/content into gen_ai.prompt.N.*
    # attrs yet (that'd round-trip; scoped for a later pass), but model +
    # usage round-trip cleanly.
    resp = next(r for r in reimported if r["kind"] == "chat_response")
    assert resp["payload"]["model"] == "gpt-4.1"
    assert resp["payload"]["usage"]["input_tokens"] == 4
    assert resp["payload"]["usage"]["output_tokens"] == 1
    assert resp["payload"]["latency_ms"] == 150
