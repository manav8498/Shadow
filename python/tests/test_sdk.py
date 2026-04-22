"""Tests for shadow.sdk.Session."""

from __future__ import annotations

from pathlib import Path

from shadow import _core
from shadow.sdk import Session


def test_session_writes_agentlog_with_metadata_root(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out, tags={"env": "test"}, session_tag="unit-test") as session:
        session.record_chat(
            request={"model": "claude-opus-4-7", "messages": [], "params": {}},
            response={
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    assert out.exists()
    records = _core.parse_agentlog(out.read_bytes())
    assert len(records) == 3  # metadata, chat_request, chat_response
    assert records[0]["kind"] == "metadata"
    assert records[0]["parent"] is None
    assert records[1]["kind"] == "chat_request"
    assert records[1]["parent"] == records[0]["id"]
    assert records[2]["kind"] == "chat_response"
    assert records[2]["parent"] == records[1]["id"]
    # session_tag is in envelope meta per the Session contract.
    assert records[0].get("meta", {}).get("session_tag") == "unit-test"


def test_session_redacts_by_default(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out) as session:
        session.record_chat(
            request={
                "model": "x",
                "messages": [{"role": "user", "content": "alice@example.com please help"}],
                "params": {},
            },
            response={
                "model": "x",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    records = _core.parse_agentlog(out.read_bytes())
    req = records[1]
    user_msg = req["payload"]["messages"][0]["content"]
    assert "alice@example.com" not in user_msg
    assert "[REDACTED:email]" in user_msg
    # The redacted stamp was placed on the request record's meta.
    assert req.get("meta", {}).get("redacted") is True


def test_session_tool_call_records_link_to_response(tmp_path: Path) -> None:
    out = tmp_path / "session.agentlog"
    with Session(output_path=out) as session:
        _req_id, resp_id = session.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [],
                "stop_reason": "tool_use",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
        tool_call_id = session.record_tool_call(
            "search_files", "toolu_01", {"query": "*.py"}, parent_id=resp_id
        )
        session.record_tool_result(
            "toolu_01", "hello.py\nworld.py", is_error=False, latency_ms=2, parent_id=tool_call_id
        )
    records = _core.parse_agentlog(out.read_bytes())
    # metadata, chat_request, chat_response, tool_call, tool_result
    assert [r["kind"] for r in records] == [
        "metadata",
        "chat_request",
        "chat_response",
        "tool_call",
        "tool_result",
    ]
