"""Tests for ``shadow.adapters.crewai.ShadowCrewAIListener``.

We drive CrewAI's real event bus with synthetic Pydantic event
instances. No real LLM or Agent/Crew objects are constructed — the
listener only reads event fields, so fully-populated synthetic events
cover the contract we ship.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

try:
    from crewai.events import crewai_event_bus
    from crewai.events.types.crew_events import CrewKickoffStartedEvent
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
except ImportError:
    pytest.skip("crewai not installed", allow_module_level=True)

from shadow import _core
from shadow.adapters.crewai import ShadowCrewAIListener
from shadow.sdk import Session


def _emit(event: object) -> None:
    """Emit a CrewAI event and wait for its background handlers.

    CrewAI 1.14's event bus runs handlers in a thread-pool executor and
    returns a ``Future`` from ``emit()``; tests must join on it before
    asserting on downstream state.
    """
    fut = crewai_event_bus.emit("test", event)
    if fut is not None:
        fut.result(timeout=5)


# ---- LLM pair capture ----------------------------------------------------


def test_llm_start_and_complete_produce_one_chat_pair(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        _emit(
            LLMCallStartedEvent(
                model="gpt-4.1",
                call_id="call-1",
                messages=[{"role": "user", "content": "what is 2+2"}],
            )
        )
        _emit(
            LLMCallCompletedEvent(
                model="gpt-4.1",
                call_id="call-1",
                messages=[{"role": "user", "content": "what is 2+2"}],
                response="4",
                call_type="llm_call",
                usage={"prompt_tokens": 8, "completion_tokens": 1},
            )
        )

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert kinds == ["metadata", "chat_request", "chat_response"]
    req = records[1]["payload"]
    assert req["model"] == "gpt-4.1"
    assert req["messages"][0]["content"] == "what is 2+2"
    resp = records[2]["payload"]
    assert resp["content"][0]["text"] == "4"
    assert resp["usage"]["input_tokens"] == 8
    assert resp["usage"]["output_tokens"] == 1


def test_failed_llm_call_produces_error_response(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        _emit(
            LLMCallStartedEvent(
                model="gpt-4.1",
                call_id="call-fail",
                messages=[{"role": "user", "content": "provoke"}],
            )
        )
        _emit(
            LLMCallFailedEvent(
                model="gpt-4.1",
                call_id="call-fail",
                error="upstream 503",
            )
        )

    records = _core.parse_agentlog(out.read_bytes())
    resp = next(r for r in records if r["kind"] == "chat_response")
    assert resp["payload"]["stop_reason"] == "error"
    assert "503" in str(resp["payload"]["error"])


def test_concurrent_call_ids_do_not_cross_contaminate(tmp_path: Path) -> None:
    """Two concurrent LLM calls with distinct call_ids must end up paired correctly."""
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        _emit(
            LLMCallStartedEvent(
                model="a", call_id="alpha", messages=[{"role": "user", "content": "alpha-in"}]
            )
        )
        _emit(
            LLMCallStartedEvent(
                model="b", call_id="beta", messages=[{"role": "user", "content": "beta-in"}]
            )
        )
        # Complete beta before alpha — order-flip shouldn't matter.
        _emit(
            LLMCallCompletedEvent(
                model="b",
                call_id="beta",
                messages=[],
                response="beta-out",
                call_type="llm_call",
                usage={},
            )
        )
        _emit(
            LLMCallCompletedEvent(
                model="a",
                call_id="alpha",
                messages=[],
                response="alpha-out",
                call_type="llm_call",
                usage={},
            )
        )

    records = _core.parse_agentlog(out.read_bytes())
    responses = [r for r in records if r["kind"] == "chat_response"]
    assert len(responses) == 2
    texts = {r["payload"]["content"][0]["text"] for r in responses}
    assert texts == {"alpha-out", "beta-out"}
    for resp in responses:
        resp_idx = records.index(resp)
        req = records[resp_idx - 1]
        assert req["payload"]["model"] == resp["payload"]["model"]


# ---- tool lifecycle ------------------------------------------------------


def test_tool_usage_produces_tool_call_and_result(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        started = ToolUsageStartedEvent(
            agent_key="agent-1",
            tool_name="search_web",
            tool_args={"query": "crewai"},
            tool_class="WebSearch",
            run_attempts=1,
            delegations=0,
        )
        _emit(started)
        now = datetime.datetime.now(datetime.UTC)
        finished = ToolUsageFinishedEvent(
            agent_key="agent-1",
            tool_name="search_web",
            tool_args={"query": "crewai"},
            tool_class="WebSearch",
            run_attempts=1,
            delegations=0,
            started_event_id=str(started.event_id),
            from_cache=False,
            output="ranked results: ...",
            started_at=now,
            finished_at=now,
        )
        _emit(finished)

    records = _core.parse_agentlog(out.read_bytes())
    kinds = [r["kind"] for r in records]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    tc = next(r for r in records if r["kind"] == "tool_call")
    assert tc["payload"]["tool_name"] == "search_web"
    assert tc["payload"]["arguments"] == {"query": "crewai"}
    tr = next(r for r in records if r["kind"] == "tool_result")
    assert "ranked" in str(tr["payload"]["output"])
    assert tr["payload"].get("is_error") in (False, None)


def test_kickoff_events_emit_session_markers(tmp_path: Path) -> None:
    """Each CrewKickoffStartedEvent writes an explicit metadata record
    so Shadow's session detector treats one kickoff as one session
    — even when every LLMCallCompleted ends with ``end_turn``, which
    would otherwise fragment the trace.
    """
    from shadow.hierarchical import _compute_session_of_pair
    from shadow.sdk.session import output_path_from_env  # noqa: F401

    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        # Two kickoffs, each with two LLM pairs. Without the marker,
        # we'd get 4 separate sessions (end_turn after every pair);
        # with the marker we get exactly 2.
        for kickoff in range(2):
            _emit(CrewKickoffStartedEvent(crew_name=f"crew-{kickoff}", inputs={}))
            for pair in range(2):
                call_id = f"k{kickoff}-call{pair}"
                _emit(
                    LLMCallStartedEvent(
                        model="m", call_id=call_id, messages=[{"role": "user", "content": "q"}]
                    )
                )
                _emit(
                    LLMCallCompletedEvent(
                        model="m",
                        call_id=call_id,
                        messages=[],
                        response="a",
                        call_type="llm_call",
                        usage={},
                    )
                )

    records = _core.parse_agentlog(out.read_bytes())
    metadata_count = sum(1 for r in records if r["kind"] == "metadata")
    # Three metadata records total: Session's own init-marker + two
    # kickoff markers (one per CrewKickoffStartedEvent). That's what
    # triggers explicit-marker mode.
    assert metadata_count == 3

    sessions = _compute_session_of_pair(records)
    # 4 chat pairs total, split across 2 populated sessions — two per
    # kickoff. Without the adapter's kickoff markers, the heuristic
    # detector would see 4 separate sessions (one per end_turn pair).
    assert len(sessions) == 4
    assert len(set(sessions)) == 2, f"expected 2 sessions from 2 kickoffs, got {sessions}"
    # First two pairs in the same session, next two in the same
    # session, and those two sessions are different.
    assert sessions[0] == sessions[1]
    assert sessions[2] == sessions[3]
    assert sessions[0] != sessions[2]


def test_quiet_internal_listeners_silences_synthetic_event_noise(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """With quiet_internal_listeners=True, CrewAI's built-in handlers
    that choke on synthetic event sources are detached and no
    ``'str' object has no attribute 'id'`` errors print to stderr.
    """
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s, quiet_internal_listeners=True)
        _emit(CrewKickoffStartedEvent(crew_name="c", inputs={}))
        _emit(
            LLMCallStartedEvent(
                model="m", call_id="c1", messages=[{"role": "user", "content": "q"}]
            )
        )
        _emit(
            LLMCallCompletedEvent(
                model="m",
                call_id="c1",
                messages=[],
                response="a",
                call_type="llm_call",
                usage={},
            )
        )

    err = capfd.readouterr().err
    assert "'str' object has no attribute 'id'" not in err, "noisy internal handler still fired"
    # And we still record the chat pair and kickoff marker.
    records = _core.parse_agentlog(out.read_bytes())
    assert any(r["kind"] == "chat_response" for r in records)
    assert sum(1 for r in records if r["kind"] == "metadata") >= 2


def test_tool_error_event_marks_result_as_error(tmp_path: Path) -> None:
    out = tmp_path / "trace.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        ShadowCrewAIListener(s)
        started = ToolUsageStartedEvent(
            agent_key="agent-1",
            tool_name="risky_op",
            tool_args={},
            tool_class="RiskyTool",
            run_attempts=1,
            delegations=0,
        )
        _emit(started)
        _emit(
            ToolUsageErrorEvent(
                agent_key="agent-1",
                tool_name="risky_op",
                tool_args={},
                tool_class="RiskyTool",
                run_attempts=1,
                delegations=0,
                started_event_id=str(started.event_id),
                error=RuntimeError("boom"),
            )
        )

    records = _core.parse_agentlog(out.read_bytes())
    tr = next(r for r in records if r["kind"] == "tool_result")
    assert tr["payload"]["is_error"] is True
    assert "boom" in str(tr["payload"]["output"])
