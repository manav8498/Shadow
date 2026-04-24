"""Tests for `shadow import --format a2a`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shadow.errors import ShadowConfigError
from shadow.importers.a2a import a2a_to_agentlog


def _simple_session() -> list[dict]:
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "message": {"parts": [{"text": "what is the weather in SF?"}]},
                "context": {"model": "gpt-4o", "temperature": 0.2, "remote_agent": "weather-bot"},
                "agent_card": {
                    "name": "weather-bot",
                    "version": "1.0",
                    "capabilities": ["get_weather"],
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "output": {"parts": [{"text": "sunny and 68F"}]},
                "usage": {"input_tokens": 20, "output_tokens": 8},
            },
        },
    ]


def test_round_trips_a_simple_session() -> None:
    records = a2a_to_agentlog(_simple_session())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    assert records[1]["payload"]["model"] == "gpt-4o"
    assert records[1]["payload"]["messages"][0]["content"] == "what is the weather in SF?"
    assert records[2]["payload"]["content"] == [{"type": "text", "text": "sunny and 68F"}]
    assert records[2]["payload"]["usage"]["input_tokens"] == 20


def test_agent_card_surfaces_in_metadata() -> None:
    records = a2a_to_agentlog(_simple_session())
    cards = records[0]["payload"].get("agent_cards")
    assert cards and cards[0]["name"] == "weather-bot"


def test_handles_error_response() -> None:
    session = [
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tasks/send",
            "params": {"message": {"parts": [{"text": "do x"}]}, "context": {"model": "m"}},
        },
        {"jsonrpc": "2.0", "id": 5, "error": {"code": 503, "message": "remote agent overloaded"}},
    ]
    records = a2a_to_agentlog(session)
    assert records[-1]["payload"]["stop_reason"] == "error"
    assert "503" in records[-1]["payload"]["content"][0]["text"]


def test_wrapped_object_with_metadata() -> None:
    wrapped = {"messages": _simple_session(), "session_id": "abc"}
    records = a2a_to_agentlog(wrapped)
    assert records[0]["payload"]["a2a_session_metadata"] == {"session_id": "abc"}


def test_empty_list_raises() -> None:
    with pytest.raises(ShadowConfigError):
        a2a_to_agentlog([])


def test_scalar_input_raises() -> None:
    with pytest.raises(ShadowConfigError):
        a2a_to_agentlog(42)


def test_non_tasks_methods_folded_into_events() -> None:
    session = [
        {"jsonrpc": "2.0", "id": 1, "method": "session/init", "params": {}},
        {"jsonrpc": "2.0", "id": 1, "result": {"session_id": "s1"}},
    ]
    records = a2a_to_agentlog(session)
    # no chat pair because method is session/init
    assert [r["kind"] for r in records] == ["metadata"]
    assert records[0]["payload"].get("warnings")


def test_orphan_request_still_emits_chat_request() -> None:
    session = [
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tasks/send",
            "params": {"message": "hello", "context": {"model": "m"}},
        },
    ]
    records = a2a_to_agentlog(session)
    # metadata + chat_request + chat_response (empty text, end_turn)
    kinds = [r["kind"] for r in records]
    assert "chat_request" in kinds
    assert "chat_response" in kinds


def test_cli_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "session.json"
    src.write_text(json.dumps(_simple_session()))
    dst = tmp_path / "out.agentlog"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow.cli.app",
            "import",
            str(src),
            "--format",
            "a2a",
            "--output",
            str(dst),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in dst.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
