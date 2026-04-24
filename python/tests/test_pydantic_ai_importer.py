"""Tests for `shadow import --format pydantic-ai`.

PydanticAI's message history is the canonical shape: a list of
`ModelRequest` / `ModelResponse` objects each carrying typed `parts`.
We also accept Logfire span exports that wrap the same list under
`attributes.all_messages_json`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.errors import ShadowConfigError
from shadow.importers.pydantic_ai import pydantic_ai_to_agentlog

# ---- fixtures -------------------------------------------------------------


def _simple_chat_history() -> list[dict]:
    return [
        {
            "kind": "request",
            "parts": [
                {"part_kind": "system-prompt", "content": "you are a helper"},
                {"part_kind": "user-prompt", "content": "hello?"},
            ],
            "model_name": "gpt-4o",
        },
        {
            "kind": "response",
            "parts": [{"part_kind": "text", "content": "hi there"}],
            "model_name": "gpt-4o",
            "usage": {"request_tokens": 10, "response_tokens": 3, "total_tokens": 13},
            "timestamp": "2026-04-24T00:00:00Z",
        },
    ]


def _tool_call_history() -> list[dict]:
    return [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "what's the weather in SF?"}],
            "model_name": "gpt-4o",
            "model_request_parameters": {
                "function_tools": [
                    {
                        "name": "get_weather",
                        "description": "get weather for a city",
                        "parameters_json_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            },
        },
        {
            "kind": "response",
            "parts": [
                {
                    "part_kind": "tool-call",
                    "tool_call_id": "call_1",
                    "tool_name": "get_weather",
                    "args": {"city": "SF"},
                }
            ],
            "model_name": "gpt-4o",
            "usage": {"request_tokens": 20, "response_tokens": 10},
        },
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "tool-return",
                    "tool_call_id": "call_1",
                    "tool_name": "get_weather",
                    "content": "sunny 68F",
                }
            ],
            "model_name": "gpt-4o",
        },
        {
            "kind": "response",
            "parts": [{"part_kind": "text", "content": "It's sunny, 68°F."}],
            "model_name": "gpt-4o",
            "usage": {"request_tokens": 30, "response_tokens": 8},
        },
    ]


# ---- happy path ----------------------------------------------------------


def test_simple_chat_roundtrips() -> None:
    records = pydantic_ai_to_agentlog(_simple_chat_history())
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    resp = records[2]["payload"]
    assert req["messages"] == [
        {"role": "system", "content": "you are a helper"},
        {"role": "user", "content": "hello?"},
    ]
    assert req["model"] == "gpt-4o"
    assert resp["content"] == [{"type": "text", "text": "hi there"}]
    assert resp["stop_reason"] == "end_turn"
    assert resp["usage"]["input_tokens"] == 10
    assert resp["usage"]["output_tokens"] == 3


def test_tool_call_flow_produces_tool_use_and_tool_result() -> None:
    records = pydantic_ai_to_agentlog(_tool_call_history())
    kinds = [r["kind"] for r in records]
    assert kinds == [
        "metadata",
        "chat_request",
        "chat_response",
        "chat_request",
        "chat_response",
    ]
    # Second response's content: model replies after tool return.
    first_resp = records[2]["payload"]
    assert first_resp["content"][0]["type"] == "tool_use"
    assert first_resp["stop_reason"] == "tool_use"
    # Second chat_request should contain the tool_result as a user turn.
    second_req = records[3]["payload"]
    user_turn = second_req["messages"][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"][0]["type"] == "tool_result"
    assert user_turn["content"][0]["tool_use_id"] == "call_1"
    # Tool schemas advertised on the first request propagate.
    assert second_req["tools"][0]["name"] == "get_weather"


def test_wrapped_messages_object() -> None:
    data = {"messages": _simple_chat_history(), "session_id": "abc-123"}
    records = pydantic_ai_to_agentlog(data)
    assert records[0]["payload"]["pydantic_ai_metadata"] == {"session_id": "abc-123"}


def test_logfire_spans_shape() -> None:
    span = {
        "name": "pydantic_ai.agent",
        "attributes": {"all_messages_json": json.dumps(_simple_chat_history())},
    }
    records = pydantic_ai_to_agentlog([span])
    assert [r["kind"] for r in records] == ["metadata", "chat_request", "chat_response"]


def test_alternate_part_kind_casing() -> None:
    # Some PydanticAI versions emit CamelCase `kind` fields.
    history = [
        {
            "kind": "ModelRequest",
            "parts": [{"kind": "UserPromptPart", "content": "hi"}],
        },
        {
            "kind": "ModelResponse",
            "parts": [{"kind": "TextPart", "content": "hello"}],
        },
    ]
    records = pydantic_ai_to_agentlog(history)
    assert records[1]["payload"]["messages"] == [{"role": "user", "content": "hi"}]
    assert records[2]["payload"]["content"] == [{"type": "text", "text": "hello"}]


def test_retry_prompt_folded_into_user_turn() -> None:
    history = [
        {
            "kind": "request",
            "parts": [
                {"part_kind": "user-prompt", "content": "first"},
                {
                    "part_kind": "retry-prompt",
                    "content": "you must use the tool",
                    "tool_name": "get_weather",
                },
            ],
        },
        {
            "kind": "response",
            "parts": [{"part_kind": "text", "content": "ok"}],
        },
    ]
    records = pydantic_ai_to_agentlog(history)
    msgs = records[1]["payload"]["messages"]
    assert any("[retry]" in str(m["content"]) for m in msgs)


def test_params_propagate_from_request() -> None:
    history = [
        {
            "kind": "request",
            "parts": [{"part_kind": "user-prompt", "content": "hi"}],
            "model_request_parameters": {"temperature": 0.2, "max_tokens": 128},
        },
        {
            "kind": "response",
            "parts": [{"part_kind": "text", "content": "ok"}],
        },
    ]
    records = pydantic_ai_to_agentlog(history)
    assert records[1]["payload"]["params"] == {"temperature": 0.2, "max_tokens": 128}


# ---- edge cases ----------------------------------------------------------


def test_empty_history_raises() -> None:
    with pytest.raises(ShadowConfigError):
        pydantic_ai_to_agentlog([])


def test_scalar_input_raises() -> None:
    with pytest.raises(ShadowConfigError):
        pydantic_ai_to_agentlog(42)


def test_dangling_request_without_response_still_emitted() -> None:
    history = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "hi"}]},
    ]
    records = pydantic_ai_to_agentlog(history)
    assert [r["kind"] for r in records] == ["metadata", "chat_request"]


# ---- CLI integration -----------------------------------------------------


def test_cli_import_pydantic_ai(tmp_path: Path) -> None:
    src = tmp_path / "msgs.json"
    src.write_text(json.dumps(_simple_chat_history()))
    out = tmp_path / "out.agentlog"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "import",
            str(src),
            "--format",
            "pydantic-ai",
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    kinds = [json.loads(ln)["kind"] for ln in lines]
    assert kinds == ["metadata", "chat_request", "chat_response"]
