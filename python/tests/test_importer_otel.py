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


# ---- v1.37+ structured-messages path --------------------------------------


def _kvlist_attr(key: str, value: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Emit an attribute whose value is a structured JSON (serialised as string).

    The spec allows either kvlist or JSON-string encoding. We test the
    JSON-string form because that's what most instrumenters actually
    emit (OpenLLMetry, AG2's exporter).
    """
    import json as _json

    return {"key": key, "value": {"stringValue": _json.dumps(value)}}


def _structured_v140_span(**overrides: Any) -> dict[str, Any]:
    """A v1.40-shaped chat span: structured messages, provider.name, cache tokens."""
    defaults: dict[str, Any] = {
        "traceId": "f" * 32,
        "spanId": "e" * 16,
        "parentSpanId": "",
        "name": "chat gpt-5",
        "kind": 3,
        "startTimeUnixNano": "1800000000000000000",
        "endTimeUnixNano": "1800000000500000000",
        "attributes": [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.provider.name", "openai"),
            _attr("gen_ai.request.model", "gpt-5"),
            _attr("gen_ai.response.model", "gpt-5-2025-10"),
            _attr("gen_ai.response.id", "resp_abc123"),
            _attr("gen_ai.conversation.id", "conv_xyz"),
            _attr("gen_ai.request.temperature", 0.0),
            _attr("gen_ai.request.top_p", 1.0),
            _attr("gen_ai.request.max_tokens", 500),
            _attr("gen_ai.request.frequency_penalty", 0.1),
            _attr("gen_ai.request.seed", 42),
            _attr("gen_ai.request.stop_sequences", ["\n\n", "END"]),
            _attr("gen_ai.usage.input_tokens", 100),
            _attr("gen_ai.usage.output_tokens", 50),
            _attr("gen_ai.usage.cache_read.input_tokens", 80),
            _attr("gen_ai.usage.cache_creation.input_tokens", 20),
            _attr("gen_ai.usage.reasoning_tokens", 30),
            _attr("gen_ai.response.finish_reasons", ["stop"]),
            _attr("gen_ai.output.type", "text"),
            _attr("server.address", "api.openai.com"),
            _attr("server.port", 443),
            _kvlist_attr(
                "gen_ai.input.messages",
                [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": "What's the weather in SF?"}],
                    }
                ],
            ),
            _kvlist_attr(
                "gen_ai.output.messages",
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": "Let me check."}],
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "name": "get_weather",
                                "arguments": {"city": "SF"},
                            }
                        ],
                    }
                ],
            ),
            _kvlist_attr(
                "gen_ai.system_instructions",
                [{"type": "text", "content": "You are a helpful assistant."}],
            ),
            _kvlist_attr(
                "gen_ai.tool.definitions",
                [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Look up the weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            ),
        ],
        "status": {"code": 1},
    }
    defaults.update(overrides)
    return defaults


def test_v140_structured_input_messages_parsed() -> None:
    data = _otlp([_structured_v140_span()])
    records = otel_to_agentlog(data)
    req = records[1]["payload"]
    # system_instructions is prepended to messages.
    assert req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert req["messages"][1]["role"] == "user"
    # Single-text-part messages collapse back to string content for readability.
    assert req["messages"][1]["content"] == "What's the weather in SF?"


def test_v140_structured_output_with_tool_calls_produces_content_blocks() -> None:
    data = _otlp([_structured_v140_span()])
    records = otel_to_agentlog(data)
    resp = records[2]["payload"]
    blocks = resp["content"]
    # One text block + one tool_use block from the assistant message.
    assert any(b.get("type") == "text" and b["text"] == "Let me check." for b in blocks)
    tool_block = next(b for b in blocks if b.get("type") == "tool_use")
    assert tool_block["id"] == "call_abc"
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"city": "SF"}


def test_v140_extra_request_params_round_trip() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    params = records[1]["payload"]["params"]
    assert params["temperature"] == 0.0
    assert params["top_p"] == 1.0
    assert params["max_tokens"] == 500
    assert params["frequency_penalty"] == 0.1
    assert params["seed"] == 42
    assert params["stop"] == ["\n\n", "END"]


def test_v140_cache_tokens_sum_into_cached_input_tokens() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    usage = records[2]["payload"]["usage"]
    # cache_read (80) + cache_creation (20) = 100
    assert usage["cached_input_tokens"] == 100
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    # reasoning_tokens maps to thinking_tokens
    assert usage["thinking_tokens"] == 30


def test_v140_response_id_and_output_type_surface() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    resp = records[2]["payload"]
    assert resp["response_id"] == "resp_abc123"
    assert resp["output_type"] == "text"


def test_v140_tool_definitions_land_in_request() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    tools = records[1]["payload"].get("tools", [])
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"


def test_v140_conversation_id_round_trips() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    assert records[1]["payload"]["conversation_id"] == "conv_xyz"


def test_v140_provider_name_lands_in_metadata() -> None:
    records = otel_to_agentlog(_otlp([_structured_v140_span()]))
    meta = records[0]["payload"]
    assert meta["provider"] == "openai"
    assert meta["server"]["address"] == "api.openai.com"


def test_v140_error_type_surfaces_on_response() -> None:
    span = _structured_v140_span()
    span["attributes"].append(_attr("error.type", "RateLimitError"))
    records = otel_to_agentlog(_otlp([span]))
    resp = records[2]["payload"]
    assert resp.get("error", {}).get("type") == "RateLimitError"
    assert resp["stop_reason"] == "stop"  # finish_reasons still takes precedence


def test_v140_inference_details_event_carries_messages() -> None:
    """Some instrumenters attach messages to the event, not the span attrs.

    The importer must merge event-bag attributes onto the span's bag.
    """
    import json as _json

    span = _structured_v140_span()
    # Strip the attribute-level input.messages; put it only on the event.
    span["attributes"] = [a for a in span["attributes"] if a["key"] != "gen_ai.input.messages"]
    span["events"] = [
        {
            "name": "gen_ai.client.inference.operation.details",
            "timeUnixNano": span["startTimeUnixNano"],
            "attributes": [
                {
                    "key": "gen_ai.input.messages",
                    "value": {
                        "stringValue": _json.dumps(
                            [
                                {
                                    "role": "user",
                                    "parts": [{"type": "text", "content": "event-carried msg"}],
                                }
                            ]
                        )
                    },
                }
            ],
        }
    ]
    records = otel_to_agentlog(_otlp([span]))
    req = records[1]["payload"]
    assert any(m.get("content") == "event-carried msg" for m in req["messages"])


def test_v140_deprecated_per_message_events_still_parse() -> None:
    """v1.28-v1.36 per-message event model must still round-trip."""
    import json as _json

    span = _chat_span()
    # Strip legacy flat-indexed prompt/completion attrs.
    span["attributes"] = [
        a
        for a in span["attributes"]
        if not a["key"].startswith("gen_ai.prompt.")
        and not a["key"].startswith("gen_ai.completion.")
    ]
    span["events"] = [
        {
            "name": "gen_ai.user.message",
            "attributes": [{"key": "content", "value": {"stringValue": "hi via event"}}],
        },
        {
            "name": "gen_ai.choice",
            "attributes": [
                {
                    "key": "message",
                    "value": {
                        "stringValue": _json.dumps(
                            {
                                "role": "assistant",
                                "parts": [{"type": "text", "content": "hello via event"}],
                            }
                        )
                    },
                }
            ],
        },
    ]
    records = otel_to_agentlog(_otlp([span]))
    req = records[1]["payload"]
    assert any(m.get("content") == "hi via event" for m in req["messages"])
    resp = records[2]["payload"]
    assert any(b.get("text") == "hello via event" for b in resp["content"])


def test_v140_agent_spans_surface_in_metadata() -> None:
    agent_span = {
        "traceId": "0" * 32,
        "spanId": "a" * 16,
        "parentSpanId": "",
        "name": "invoke_agent researcher",
        "kind": 3,
        "startTimeUnixNano": "1800000000000000000",
        "endTimeUnixNano": "1800000000001000000",
        "attributes": [
            _attr("gen_ai.operation.name", "invoke_agent"),
            _attr("gen_ai.provider.name", "anthropic"),
            _attr("gen_ai.agent.id", "agent-1"),
            _attr("gen_ai.agent.name", "researcher"),
            _attr("gen_ai.agent.description", "Finds sources"),
            _attr("gen_ai.agent.version", "1.2.0"),
        ],
        "status": {"code": 1},
    }
    # Needs at least one chat span so import isn't empty; agent span
    # contributes to metadata only.
    records = otel_to_agentlog(_otlp([agent_span, _structured_v140_span()]))
    agents = records[0]["payload"].get("agents", [])
    assert agents and agents[0]["name"] == "researcher"
    assert agents[0]["version"] == "1.2.0"


def test_v140_evaluation_event_carried_to_metadata() -> None:
    span = _structured_v140_span()
    span["events"] = [
        {
            "name": "gen_ai.evaluation.result",
            "attributes": [
                _attr("gen_ai.evaluation.name", "helpfulness"),
                _attr("gen_ai.evaluation.score.value", 0.85),
                _attr("gen_ai.evaluation.score.label", "high"),
            ],
        }
    ]
    records = otel_to_agentlog(_otlp([span]))
    evals = records[0]["payload"].get("evaluations", [])
    assert evals and evals[0]["name"] == "helpfulness"
    assert evals[0]["score"] == 0.85


def test_v140_spans_sorted_by_start_time_for_deterministic_ids() -> None:
    """Two imports of the same OTLP payload must produce identical records."""
    s1 = _structured_v140_span(
        spanId="1" * 16,
        startTimeUnixNano="2000000000000000000",
        endTimeUnixNano="2000000000500000000",
    )
    s2 = _structured_v140_span(
        spanId="2" * 16,
        startTimeUnixNano="1900000000000000000",
        endTimeUnixNano="1900000000500000000",
    )
    # Spans passed in reverse order → importer must sort by startTime.
    forward = otel_to_agentlog(_otlp([s2, s1]))
    reverse = otel_to_agentlog(_otlp([s1, s2]))
    # The chat spans (after metadata) should come out in the same order
    # regardless of input order.
    # The metadata record now also carries `meta` (with trace_id),
    # but only chat/tool records have `otel_span_id`. Filter to
    # records that actually have it.
    assert [
        r["meta"]["otel_span_id"] for r in forward if "meta" in r and "otel_span_id" in r["meta"]
    ] == [r["meta"]["otel_span_id"] for r in reverse if "meta" in r and "otel_span_id" in r["meta"]]
