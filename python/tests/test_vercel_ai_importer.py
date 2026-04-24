"""Tests for `shadow import --format vercel-ai`.

Vercel AI SDK traces ship via OpenTelemetry with `ai.*` attributes
on each generation span. This test suite covers both OTLP and
dashboard export shapes end to end, locking the round-trip into
Shadow's `.agentlog` so the Rust differ can work on imported
traces without per-format wiring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.errors import ShadowConfigError
from shadow.importers.vercel_ai import vercel_ai_to_agentlog

# ---- fixtures -------------------------------------------------------------


def _otlp_span(
    *,
    model: str = "gpt-4o-mini",
    messages: list[dict] | None = None,
    text: str | None = None,
    tool_calls: list[dict] | None = None,
    tools: list[dict] | None = None,
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    temperature: float | None = 0.7,
    finish_reason: str = "stop",
    start: str = "2026-04-24T00:00:00.000Z",
    end: str = "2026-04-24T00:00:00.350Z",
) -> dict:
    attrs: dict = {
        "ai.model.id": model,
        "ai.prompt.messages": json.dumps(messages or [{"role": "user", "content": "hi"}]),
        "ai.usage.promptTokens": prompt_tokens,
        "ai.usage.completionTokens": completion_tokens,
        "ai.response.finishReason": finish_reason,
    }
    if text is not None:
        attrs["ai.response.text"] = text
    if tool_calls is not None:
        attrs["ai.response.toolCalls"] = json.dumps(tool_calls)
    if tools is not None:
        attrs["ai.tools"] = json.dumps(tools)
    if temperature is not None:
        attrs["ai.settings.temperature"] = temperature
    return {
        "name": "ai.generateText",
        "startTime": start,
        "endTime": end,
        "attributes": attrs,
    }


# ---- happy path ----------------------------------------------------------


def test_otlp_span_shape_roundtrips() -> None:
    data = {"spans": [_otlp_span(text="hello!")]}
    records = vercel_ai_to_agentlog(data)
    # metadata + request + response
    assert len(records) == 3
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    resp = records[2]["payload"]
    assert req["model"] == "gpt-4o-mini"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["params"] == {"temperature": 0.7}
    assert resp["content"] == [{"type": "text", "text": "hello!"}]
    assert resp["stop_reason"] == "end_turn"
    assert resp["usage"]["input_tokens"] == 12
    assert resp["usage"]["output_tokens"] == 8
    assert resp["latency_ms"] == 350


def test_events_shape_also_supported() -> None:
    data = {"events": [_otlp_span(text="ok")]}
    records = vercel_ai_to_agentlog(data)
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_plain_list_of_spans() -> None:
    records = vercel_ai_to_agentlog([_otlp_span(text="ok")])
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_tool_call_becomes_anthropic_tool_use_block() -> None:
    data = {
        "spans": [
            _otlp_span(
                text=None,
                tool_calls=[
                    {
                        "toolCallId": "call_1",
                        "toolName": "get_weather",
                        "args": {"city": "SF"},
                    }
                ],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                finish_reason="tool-calls",
            )
        ]
    }
    records = vercel_ai_to_agentlog(data)
    req = records[1]["payload"]
    resp = records[2]["payload"]
    assert req["tools"] == [
        {
            "name": "get_weather",
            "description": "Current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    assert resp["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"city": "SF"},
        }
    ]
    assert resp["stop_reason"] == "tool_use"


def test_system_prompt_prepended() -> None:
    span = _otlp_span(
        messages=[{"role": "user", "content": "hello"}],
        text="hi there",
    )
    span["attributes"]["ai.prompt.system"] = "be terse"
    records = vercel_ai_to_agentlog({"spans": [span]})
    req = records[1]["payload"]
    assert req["messages"][0] == {"role": "system", "content": "be terse"}
    assert req["messages"][1] == {"role": "user", "content": "hello"}


def test_multiple_spans_chain_parents() -> None:
    s1 = "2026-04-24T00:00:00Z"
    e1 = "2026-04-24T00:00:00.100Z"
    s2 = "2026-04-24T00:00:01Z"
    e2 = "2026-04-24T00:00:01.100Z"
    records = vercel_ai_to_agentlog(
        {
            "spans": [
                _otlp_span(text="first", start=s1, end=e1),
                _otlp_span(text="second", start=s2, end=e2),
            ]
        }
    )
    # metadata + 2x (request + response)
    assert len(records) == 5
    # Second chat_request's parent should be the first chat_response's id.
    assert records[3]["parent"] == records[2]["id"]


def test_unknown_ai_keys_stashed_in_extras() -> None:
    span = _otlp_span(text="ok")
    span["attributes"]["ai.experimental.custom"] = "value"
    records = vercel_ai_to_agentlog({"spans": [span]})
    req = records[1]["payload"]
    assert req["source"]["extras"] == {"ai.experimental.custom": "value"}


def test_object_response_serialised_as_text() -> None:
    span = _otlp_span(text=None)
    span["attributes"]["ai.response.object"] = {"answer": 42}
    records = vercel_ai_to_agentlog({"spans": [span]})
    resp = records[2]["payload"]
    assert resp["content"] == [{"type": "text", "text": '{"answer": 42}'}]


def test_non_generation_spans_skipped() -> None:
    # Spans without ai.* input/output keys should not produce pairs.
    records = vercel_ai_to_agentlog(
        {"spans": [{"name": "http.request", "attributes": {"http.method": "GET"}}]}
    )
    assert [r["kind"] for r in records] == ["metadata"]


def test_error_span_surfaces_as_error_stop_reason() -> None:
    span = _otlp_span(text=None)
    span["status"] = {"code": "ERROR", "message": "rate limited"}
    records = vercel_ai_to_agentlog({"spans": [span]})
    resp = records[2]["payload"]
    assert resp["stop_reason"] == "error"
    assert "rate limited" in resp["content"][0]["text"]


# ---- edge cases ----------------------------------------------------------


def test_empty_input_raises() -> None:
    with pytest.raises(ShadowConfigError):
        vercel_ai_to_agentlog({"spans": []})


def test_scalar_input_raises() -> None:
    with pytest.raises(ShadowConfigError):
        vercel_ai_to_agentlog(42)


def test_unrecognised_shape_raises() -> None:
    with pytest.raises(ShadowConfigError):
        vercel_ai_to_agentlog({"not_spans": "x"})


def test_finish_reason_normalisation() -> None:
    span = _otlp_span(text="ok", finish_reason="length")
    records = vercel_ai_to_agentlog({"spans": [span]})
    assert records[2]["payload"]["stop_reason"] == "max_tokens"


# ---- CLI integration -----------------------------------------------------


def test_cli_import_vercel_ai(tmp_path: Path) -> None:
    data = {"spans": [_otlp_span(text="hi")]}
    src = tmp_path / "trace.json"
    src.write_text(json.dumps(data))
    out = tmp_path / "out.agentlog"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "import",
            str(src),
            "--format",
            "vercel-ai",
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    # It's valid JSONL with ≥3 records.
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 3
    kinds = [json.loads(ln)["kind"] for ln in lines]
    assert kinds == ["metadata", "chat_request", "chat_response"]
