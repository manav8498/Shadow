"""Tests for the OpenTelemetry OTLP/JSON exporter."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from shadow import _core
from shadow.cli.app import app
from shadow.otel import agentlog_to_otel
from shadow.sdk import Session


def _write_trace(path: Path) -> None:
    with Session(output_path=path, auto_instrument=False) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "hi"}],
                "params": {"temperature": 0.2, "max_tokens": 128},
            },
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "latency_ms": 150,
                "usage": {"input_tokens": 4, "output_tokens": 1, "thinking_tokens": 0},
            },
        )


def test_exporter_emits_one_span_per_chat_pair(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    spans = otel["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    span = spans[0]
    assert span["name"].startswith("gen_ai.chat")


def test_exporter_uses_genai_semconv_attribute_names(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    attrs = otel["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    keys = {a["key"] for a in attrs}
    # OTel GenAI semconv v1.37+: provider.name + operation.name required;
    # legacy `gen_ai.system` kept during deprecation window.
    assert "gen_ai.provider.name" in keys
    assert "gen_ai.operation.name" in keys
    assert "gen_ai.system" in keys  # compat
    assert "gen_ai.request.model" in keys
    assert "gen_ai.response.model" in keys
    assert "gen_ai.request.temperature" in keys
    assert "gen_ai.request.max_tokens" in keys
    assert "gen_ai.usage.input_tokens" in keys
    assert "gen_ai.usage.output_tokens" in keys
    assert "gen_ai.response.finish_reasons" in keys


def test_exporter_emits_conversation_id_from_trace_id(tmp_path: Path) -> None:
    """v1.37+ gen_ai.conversation.id should be present when trace_id is known."""
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    attrs = otel["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    keys = {a["key"] for a in attrs}
    assert "gen_ai.conversation.id" in keys


def test_exporter_operation_name_is_chat(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    attrs = otel["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    op = next(a for a in attrs if a["key"] == "gen_ai.operation.name")
    assert op["value"]["stringValue"] == "chat"


def test_exporter_infers_anthropic_from_claude_model(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    attrs = otel["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
    system_attr = next(a for a in attrs if a["key"] == "gen_ai.system")
    assert system_attr["value"]["stringValue"] == "anthropic"


def test_exporter_span_and_trace_ids_are_hex(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    records = _core.parse_agentlog(trace.read_bytes())
    otel = agentlog_to_otel(records)
    span = otel["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert len(span["traceId"]) == 32 and all(c in "0123456789abcdef" for c in span["traceId"])
    assert len(span["spanId"]) == 16 and all(c in "0123456789abcdef" for c in span["spanId"])


def test_exporter_cli_writes_valid_json(tmp_path: Path) -> None:
    trace = tmp_path / "t.agentlog"
    _write_trace(trace)
    out = tmp_path / "out.otel.json"
    result = CliRunner().invoke(app, ["export", str(trace), "--output", str(out)])
    assert result.exit_code == 0, result.output
    parsed = json.loads(out.read_text())
    assert "resourceSpans" in parsed


def test_exporter_orphan_request_gets_error_status(tmp_path: Path) -> None:
    # A trace with a request but no response → span status should be ERROR.
    records = [
        {
            "version": "0.1",
            "id": "sha256:" + "a" * 64,
            "kind": "metadata",
            "ts": "2026-04-21T10:00:00.000Z",
            "parent": None,
            "payload": {},
        },
        {
            "version": "0.1",
            "id": "sha256:" + "b" * 64,
            "kind": "chat_request",
            "ts": "2026-04-21T10:00:01.000Z",
            "parent": "sha256:" + "a" * 64,
            "payload": {"model": "m", "messages": [], "params": {}},
        },
    ]
    otel = agentlog_to_otel(records)
    spans = otel["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    assert spans[0]["status"]["code"] == 2  # ERROR
