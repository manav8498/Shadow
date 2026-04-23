"""Tests for distributed / multi-agent trace joining."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shadow import _core
from shadow.cli.app import app
from shadow.sdk import Session
from shadow.sdk.tracing import (
    PARENT_SPAN_ENV,
    TRACE_ID_ENV,
    W3C_TRACEPARENT_ENV,
    current_parent_span_id,
    current_trace_id,
    env_for_child,
    new_span_id,
    new_trace_id,
)


def test_session_mints_fresh_trace_id_by_default(tmp_path: Path) -> None:
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        s.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    records = _core.parse_agentlog(out.read_bytes())
    tids = {r["meta"]["trace_id"] for r in records}
    assert len(tids) == 1  # all records share one trace id
    trace_id = tids.pop()
    assert len(trace_id) == 32  # 128-bit hex


def test_session_inherits_trace_id_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_trace = "a" * 32
    parent_span = "b" * 16
    monkeypatch.setenv(TRACE_ID_ENV, parent_trace)
    monkeypatch.setenv(PARENT_SPAN_ENV, parent_span)
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        s.record_chat(
            request={"model": "x", "messages": [], "params": {}},
            response={
                "model": "x",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "latency_ms": 1,
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        )
    records = _core.parse_agentlog(out.read_bytes())
    assert records[0]["meta"]["trace_id"] == parent_trace
    assert records[0]["meta"]["parent_span_id"] == parent_span


def test_session_inherits_trace_id_from_w3c_traceparent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_trace = "c" * 32
    parent_span = "d" * 16
    monkeypatch.setenv(W3C_TRACEPARENT_ENV, f"00-{parent_trace}-{parent_span}-01")
    monkeypatch.delenv(TRACE_ID_ENV, raising=False)
    monkeypatch.delenv(PARENT_SPAN_ENV, raising=False)
    assert current_trace_id() == parent_trace
    assert current_parent_span_id() == parent_span
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        _ = s  # session exists
    records = _core.parse_agentlog(out.read_bytes())
    assert records[0]["meta"]["trace_id"] == parent_trace


def test_env_for_child_sets_all_three_vars() -> None:
    tid = new_trace_id()
    sid = new_span_id()
    env = env_for_child(tid, sid)
    assert env[TRACE_ID_ENV] == tid
    assert env[PARENT_SPAN_ENV] == sid
    assert env[W3C_TRACEPARENT_ENV] == f"00-{tid}-{sid}-01"


def test_session_env_for_child_matches_its_trace(tmp_path: Path) -> None:
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        child_env = s.env_for_child()
    assert child_env[TRACE_ID_ENV] == s.trace_id


def test_cli_join_merges_by_trace_id(tmp_path: Path) -> None:
    """Two sessions sharing a trace_id merge into one ordered trace."""
    # Set up shared trace in environ, then spin up two Sessions that
    # both inherit the same trace_id.
    shared_trace = "e" * 32
    shared_parent = "f" * 16
    os.environ[TRACE_ID_ENV] = shared_trace
    os.environ[PARENT_SPAN_ENV] = shared_parent
    try:
        out_a = tmp_path / "a.agentlog"
        out_b = tmp_path / "b.agentlog"
        with Session(output_path=out_a, auto_instrument=False) as s:
            s.record_chat(
                request={"model": "x", "messages": [], "params": {}},
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "A"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )
        with Session(output_path=out_b, auto_instrument=False) as s:
            s.record_chat(
                request={"model": "x", "messages": [], "params": {}},
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": "B"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )
    finally:
        os.environ.pop(TRACE_ID_ENV, None)
        os.environ.pop(PARENT_SPAN_ENV, None)

    merged = tmp_path / "merged.agentlog"
    result = CliRunner().invoke(app, ["join", str(out_a), str(out_b), "--output", str(merged)])
    assert result.exit_code == 0, result.output
    records = _core.parse_agentlog(merged.read_bytes())
    # All records share the same trace_id.
    tids = {r["meta"]["trace_id"] for r in records}
    assert tids == {shared_trace}
    # And the join preserves arrival order.
    kinds = [r["kind"] for r in records]
    assert kinds.count("metadata") == 2
    assert kinds.count("chat_response") == 2
