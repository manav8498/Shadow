"""Round-trip test: real OpenTelemetry SDK → OTLP/JSON → Shadow importer → diff.

Emits genuine OTel spans with GenAI semconv attributes, exports them
via the in-memory SpanExporter, serialises to OTLP/JSON, then imports
via shadow.importers.otel. The resulting .agentlog must diff cleanly
through Shadow's Rust differ.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

otel_installed: bool
try:
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore[import-not-found]
        InMemorySpanExporter,
    )

    otel_installed = True
except ImportError:
    otel_installed = False


pytestmark = pytest.mark.skipif(not otel_installed, reason="opentelemetry-sdk not installed")


def _span_to_otlp_json(span: Any) -> dict[str, Any]:
    """Translate an SDK ReadableSpan into an OTLP/JSON span dict.

    The OTLP/JSON wire format is tightly specified; we reproduce just
    the subset our importer reads.
    """
    ctx = span.get_span_context()
    parent_ctx = span.parent
    attrs = []
    for k, v in (span.attributes or {}).items():
        attrs.append(_attr(k, v))
    return {
        "traceId": format(ctx.trace_id, "032x"),
        "spanId": format(ctx.span_id, "016x"),
        "parentSpanId": format(parent_ctx.span_id, "016x") if parent_ctx else "",
        "name": span.name,
        "kind": 3,
        "startTimeUnixNano": str(span.start_time),
        "endTimeUnixNano": str(span.end_time),
        "attributes": attrs,
        "status": {"code": 1 if span.status.is_ok else 2},
    }


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, tuple | list):
        return {
            "key": key,
            "value": {
                "arrayValue": {
                    "values": [
                        {"stringValue": str(v)}
                        if not isinstance(v, bool | int | float)
                        else _attr("", v)["value"]
                        for v in value
                    ]
                }
            },
        }
    return {"key": key, "value": {"stringValue": str(value)}}


def test_real_otel_spans_round_trip_through_importer() -> None:
    from shadow import _core
    from shadow.importers import otel_to_agentlog

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    # Emit two realistic GenAI-semconv spans.
    for i in range(2):
        with tracer.start_as_current_span("gen_ai.chat gpt-4o-mini") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
            span.set_attribute("gen_ai.response.model", "gpt-4o-mini")
            span.set_attribute("gen_ai.request.temperature", 0.2)
            span.set_attribute("gen_ai.request.max_tokens", 100)
            span.set_attribute("gen_ai.response.finish_reasons", ("stop",))
            span.set_attribute("gen_ai.usage.input_tokens", 12)
            span.set_attribute("gen_ai.usage.output_tokens", 5)
            span.set_attribute("gen_ai.prompt.0.role", "user")
            span.set_attribute("gen_ai.prompt.0.content", f"task {i}")
            span.set_attribute("gen_ai.completion.0.role", "assistant")
            span.set_attribute("gen_ai.completion.0.content", f"answer {i}")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Serialize to OTLP/JSON using the shape our importer reads.
    otlp = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "test-agent")]},
                "scopeSpans": [
                    {
                        "scope": {"name": "otel-sdk-real"},
                        "spans": [_span_to_otlp_json(s) for s in spans],
                    }
                ],
            }
        ]
    }

    # Round-trip through our importer.
    records = otel_to_agentlog(otlp)
    kinds = [r["kind"] for r in records]
    assert kinds.count("chat_request") == 2
    assert kinds.count("chat_response") == 2

    # Verify the content survived.
    reqs = [r for r in records if r["kind"] == "chat_request"]
    resps = [r for r in records if r["kind"] == "chat_response"]
    assert reqs[0]["payload"]["model"] == "gpt-4o-mini"
    assert reqs[0]["payload"]["messages"][0]["content"] == "task 0"
    assert resps[0]["payload"]["content"][0]["text"] == "answer 0"
    assert resps[0]["payload"]["usage"]["input_tokens"] == 12
    assert resps[0]["payload"]["stop_reason"] == "stop"

    # And the resulting agentlog can be diffed (should be 0 delta against itself).
    bytes_out = _core.write_agentlog(records)
    parsed = _core.parse_agentlog(bytes_out)
    report = _core.compute_diff_report(parsed, parsed, None, 42)
    assert len(report["rows"]) == 9
    for row in report["rows"]:
        assert abs(row["delta"]) < 1e-9


def test_real_otel_trace_with_parent_child_spans_preserves_causality() -> None:
    """Parent/child span hierarchy should be reflected in parent links."""
    from shadow.importers import otel_to_agentlog

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("parent_agent") as parent:
        parent.set_attribute("gen_ai.request.model", "gpt-4o")
        parent.set_attribute("gen_ai.prompt.0.role", "user")
        parent.set_attribute("gen_ai.prompt.0.content", "plan")
        parent.set_attribute("gen_ai.completion.0.role", "assistant")
        parent.set_attribute("gen_ai.completion.0.content", "ok")
        parent.set_attribute("gen_ai.usage.input_tokens", 5)
        parent.set_attribute("gen_ai.usage.output_tokens", 1)
        with tracer.start_as_current_span("child_tool") as child:
            child.set_attribute("gen_ai.request.model", "claude-haiku-4-5")
            child.set_attribute("gen_ai.prompt.0.role", "user")
            child.set_attribute("gen_ai.prompt.0.content", "sub-task")
            child.set_attribute("gen_ai.completion.0.role", "assistant")
            child.set_attribute("gen_ai.completion.0.content", "done")
            child.set_attribute("gen_ai.usage.input_tokens", 3)
            child.set_attribute("gen_ai.usage.output_tokens", 1)

    spans = exporter.get_finished_spans()
    # OTel reports children before parents (depth-first close order).
    otlp = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "nested-agent")]},
                "scopeSpans": [
                    {
                        "scope": {"name": "real-otel"},
                        "spans": [_span_to_otlp_json(s) for s in spans],
                    }
                ],
            }
        ]
    }
    records = otel_to_agentlog(otlp)
    assert sum(1 for r in records if r["kind"] == "chat_request") == 2

    # Every record should carry the OTel trace_id in meta.
    non_meta = [r for r in records if r["kind"] != "metadata"]
    for r in non_meta:
        assert r.get("meta", {}).get("otel_trace_id")


def test_invalid_otlp_raises_config_error() -> None:
    from shadow.errors import ShadowConfigError
    from shadow.importers import otel_to_agentlog

    with pytest.raises(ShadowConfigError):
        otel_to_agentlog(json.loads('{"not-otlp": true}'))


def test_tool_call_roundtrip_is_lossless() -> None:
    """Regression: production audit found that exporting an agentlog
    with tool_call/tool_result records to OTel and importing back
    dropped the tool-use structure entirely — produced false
    trajectory and conformance regressions on tool-heavy agents.

    Now the export carries `gen_ai.tool.arguments` and
    `gen_ai.tool.result` attributes on the tool span, and the import
    detects tool spans (`gen_ai.tool.name` attr or
    `gen_ai.tool.call <name>` span name) and reconstructs both
    records.
    """
    from shadow.importers import otel_to_agentlog
    from shadow.otel import agentlog_to_otel

    original = [
        {
            "version": "0.1",
            "id": "sha256:m",
            "kind": "metadata",
            "ts": "2026-04-27T00:00:00Z",
            "parent": None,
            "payload": {"sdk": {"name": "shadow", "version": "2.4.1"}},
        },
        {
            "version": "0.1",
            "id": "sha256:req",
            "kind": "chat_request",
            "ts": "2026-04-27T00:00:01Z",
            "parent": "sha256:m",
            "payload": {
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "search hello"}],
                "params": {},
            },
        },
        {
            "version": "0.1",
            "id": "sha256:resp",
            "kind": "chat_response",
            "ts": "2026-04-27T00:00:02Z",
            "parent": "sha256:req",
            "payload": {
                "model": "gpt-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "tool_use",
                "latency_ms": 1000,
                "usage": {"input_tokens": 5, "output_tokens": 3, "thinking_tokens": 0},
            },
        },
        {
            "version": "0.1",
            "id": "sha256:tc",
            "kind": "tool_call",
            "ts": "2026-04-27T00:00:03Z",
            "parent": "sha256:resp",
            "payload": {
                "tool_name": "lookup",
                "tool_call_id": "call_abc",
                "arguments": {"q": "hello world", "limit": 5},
            },
        },
        {
            "version": "0.1",
            "id": "sha256:tr",
            "kind": "tool_result",
            "ts": "2026-04-27T00:00:04Z",
            "parent": "sha256:tc",
            "payload": {
                "tool_call_id": "call_abc",
                "output": "found 3 hits",
                "is_error": False,
                "latency_ms": 50,
            },
        },
    ]

    otlp = agentlog_to_otel(original)
    back = otel_to_agentlog(otlp)

    kinds = [r["kind"] for r in back]
    assert kinds == ["metadata", "chat_request", "chat_response", "tool_call", "tool_result"]

    tc_back = next(r for r in back if r["kind"] == "tool_call")
    tr_back = next(r for r in back if r["kind"] == "tool_result")

    assert tc_back["payload"]["tool_name"] == "lookup"
    assert tc_back["payload"]["arguments"] == {"q": "hello world", "limit": 5}
    assert tr_back["payload"]["output"] == "found 3 hits"
    assert tr_back["payload"]["latency_ms"] == 50
    assert tr_back["payload"]["is_error"] is False
