"""Tests for `shadow import --format llamaindex`.

LlamaIndex publishes agent traces through its instrumentation event
bus. Each operation surfaces as a paired start/end event sharing an
`id_` correlation key. This suite locks the round-trip into Shadow's
`.agentlog` envelope so the Rust differ can consume LlamaIndex
exports without any LlamaIndex install requirement on the importer
side.
"""

from __future__ import annotations

import pytest

from shadow.errors import ShadowConfigError
from shadow.importers.llamaindex import (
    import_llamaindex_events,
    llamaindex_to_agentlog,
)

# ---- fixtures -------------------------------------------------------------


def _chat_pair(
    *,
    span: str = "span-1",
    model: str = "gpt-4o-mini",
    text: str = "hello!",
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    start_ts: str = "2026-04-24T00:00:00.000Z",
    end_ts: str = "2026-04-24T00:00:00.350Z",
) -> list[dict]:
    """Build a synthetic LLMChatStartEvent + LLMChatEndEvent pair."""
    return [
        {
            "class_name": "LLMChatStartEvent",
            "id_": span,
            "timestamp": start_ts,
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "additional_kwargs": {"temperature": 0.2},
        },
        {
            "class_name": "LLMChatEndEvent",
            "id_": span,
            "timestamp": end_ts,
            "response": {
                "message": {"role": "assistant", "content": text},
                "raw": {
                    "model": model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "finish_reason": "stop",
                },
            },
        },
    ]


def _tool_pair(
    *,
    span: str = "tool-1",
    name: str = "get_weather",
    args: dict | None = None,
    output: str = "sunny, 68F",
    is_error: bool = False,
) -> list[dict]:
    return [
        {
            "class_name": "ToolCallStartEvent",
            "id_": span,
            "timestamp": "2026-04-24T00:00:01.000Z",
            "tool_name": name,
            "tool_kwargs": args if args is not None else {"city": "SF"},
        },
        {
            "class_name": "ToolCallEndEvent",
            "id_": span,
            "timestamp": "2026-04-24T00:00:01.050Z",
            "tool_name": name,
            "output": output,
            "is_error": is_error,
        },
    ]


# ---- happy path ----------------------------------------------------------


def test_chat_pair_produces_request_response_records() -> None:
    records = import_llamaindex_events(_chat_pair())
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
    ]
    meta, req_rec, resp_rec = records
    # Envelope shape: every record carries version, id, kind, ts, parent, payload.
    for rec in records:
        assert set(rec.keys()) == {"version", "id", "kind", "ts", "parent", "payload"}
        assert rec["version"] == "0.1"
        assert isinstance(rec["id"], str) and rec["id"].startswith("sha256:")
    # Parent chain: request parents metadata, response parents request.
    assert req_rec["parent"] == meta["id"]
    assert resp_rec["parent"] == req_rec["id"]
    # Request payload.
    req = req_rec["payload"]
    assert req["model"] == "gpt-4o-mini"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["params"] == {"temperature": 0.2}
    # Response payload.
    resp = resp_rec["payload"]
    assert resp["model"] == "gpt-4o-mini"
    assert resp["content"] == [{"type": "text", "text": "hello!"}]
    assert resp["stop_reason"] == "end_turn"
    assert resp["usage"]["input_tokens"] == 12
    assert resp["usage"]["output_tokens"] == 8
    assert resp["latency_ms"] == 350


def test_tool_pair_produces_tool_call_and_tool_result() -> None:
    events = _chat_pair() + _tool_pair()
    records = import_llamaindex_events(events)
    kinds = [r["kind"] for r in records]
    assert kinds == [
        "metadata",
        "chat_request",
        "chat_response",
        "tool_call",
        "tool_result",
    ]
    tool_call_rec = records[3]
    tool_result_rec = records[4]
    # tool_call payload shape.
    call_payload = tool_call_rec["payload"]
    assert call_payload["tool_name"] == "get_weather"
    assert call_payload["tool_call_id"] == "tool-1"
    assert call_payload["arguments"] == {"city": "SF"}
    # tool_result parents off the tool_call.
    assert tool_result_rec["parent"] == tool_call_rec["id"]
    result_payload = tool_result_rec["payload"]
    assert result_payload["tool_call_id"] == "tool-1"
    assert result_payload["output"] == "sunny, 68F"
    assert result_payload["is_error"] is False


def test_idempotent_content_addressing() -> None:
    """Same events twice → byte-identical record ids."""
    events = _chat_pair() + _tool_pair()
    first = import_llamaindex_events(events)
    second = import_llamaindex_events(events)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert [r["payload"] for r in first] == [r["payload"] for r in second]


def test_round_trip_envelope_fields() -> None:
    """Round-trip: every emitted record has the .agentlog envelope shape."""
    events = _chat_pair() + _tool_pair()
    records = import_llamaindex_events(events)
    parents = {r["id"] for r in records} | {None}
    for rec in records:
        assert rec["version"] == "0.1"
        assert rec["kind"] in {
            "metadata",
            "chat_request",
            "chat_response",
            "tool_call",
            "tool_result",
        }
        assert isinstance(rec["payload"], dict)
        # `parent` is either None (metadata only) or an id that exists
        # earlier in the chain.
        assert rec["parent"] in parents
        # Timestamps are ISO 8601 with trailing Z or +00:00.
        assert isinstance(rec["ts"], str) and rec["ts"]


def test_model_field_present_on_chat_response_when_event_lacks_it() -> None:
    """Spec requires `model` on chat_response even if the event omits it."""
    events = [
        {"class_name": "LLMChatStartEvent", "id_": "s1", "messages": []},
        {
            "class_name": "LLMChatEndEvent",
            "id_": "s1",
            "response": {"message": {"role": "assistant", "content": "ok"}},
        },
    ]
    records = import_llamaindex_events(events)
    resp = records[2]["payload"]
    assert "model" in resp
    assert resp["model"] == ""


def test_wrapped_events_object() -> None:
    """`{events: [...]}` wrapper is accepted and metadata preserved."""
    data = {"events": _chat_pair(), "session_id": "abc-123"}
    records = llamaindex_to_agentlog(data)
    assert records[0]["payload"]["llamaindex_metadata"] == {"session_id": "abc-123"}
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
    ]


def test_agent_run_step_events_are_dropped() -> None:
    """Step boundaries are noise and shouldn't produce records."""
    events = [
        {"class_name": "AgentRunStepStartEvent", "id_": "step1"},
        *_chat_pair(),
        {"class_name": "AgentRunStepEndEvent", "id_": "step1"},
    ]
    records = import_llamaindex_events(events)
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
    ]


def test_unknown_event_count_is_recorded() -> None:
    events = [
        *_chat_pair(),
        {"class_name": "SomeFutureLlamaIndexEvent", "id_": "x"},
    ]
    records = import_llamaindex_events(events)
    assert records[0]["payload"]["unknown_event_count"] == 1


def test_agent_tool_call_event_emits_tool_call_record() -> None:
    """`AgentToolCallEvent` is a single fire-and-forget event."""
    events = [
        {
            "class_name": "AgentToolCallEvent",
            "id_": "atc-1",
            "tool_name": "search",
            "arguments": {"q": "shadow regression testing"},
        },
    ]
    records = import_llamaindex_events(events)
    assert [r["kind"] for r in records] == ["metadata", "tool_call"]
    payload = records[1]["payload"]
    assert payload["tool_name"] == "search"
    assert payload["arguments"] == {"q": "shadow regression testing"}


def test_tool_error_marks_result_as_error() -> None:
    events = _tool_pair(output="boom", is_error=True)
    records = import_llamaindex_events(events)
    assert records[-1]["payload"]["is_error"] is True
    assert records[-1]["payload"]["output"] == "boom"


def test_completion_pair_also_supported() -> None:
    events = [
        {
            "class_name": "LLMCompletionStartEvent",
            "id_": "c1",
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "complete me",
        },
        {
            "class_name": "LLMCompletionEndEvent",
            "id_": "c1",
            "response": {"text": "completion!"},
        },
    ]
    records = import_llamaindex_events(events)
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
    ]
    req = records[1]["payload"]
    resp = records[2]["payload"]
    assert req["messages"] == [{"role": "user", "content": "complete me"}]
    assert resp["content"] == [{"type": "text", "text": "completion!"}]


# ---- edge cases ----------------------------------------------------------


def test_empty_events_raises() -> None:
    with pytest.raises(ShadowConfigError):
        llamaindex_to_agentlog([])


def test_scalar_input_raises() -> None:
    with pytest.raises(ShadowConfigError):
        llamaindex_to_agentlog(42)


def test_fifo_fallback_when_id_missing() -> None:
    """Events without `id_` should still pair up by arrival order."""
    events = [
        {
            "class_name": "LLMChatStartEvent",
            "messages": [{"role": "user", "content": "hi"}],
        },
        {
            "class_name": "LLMChatEndEvent",
            "response": {"message": {"role": "assistant", "content": "hello"}},
        },
    ]
    records = import_llamaindex_events(events)
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
    ]
    assert records[2]["payload"]["content"] == [{"type": "text", "text": "hello"}]
