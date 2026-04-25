"""Tests for the agent-loop replay engine."""

from __future__ import annotations

import asyncio
from typing import Any

from shadow import _core
from shadow.llm.mock import MockLLM
from shadow.replay_loop import (
    DEFAULT_MAX_TURNS,
    AgentLoopConfig,
    run_agent_loop_replay,
)
from shadow.tools.novel import StubPolicy
from shadow.tools.replay import ReplayToolBackend
from shadow.tools.stub import StubToolBackend


def _build_baseline() -> list[dict[str, Any]]:
    """One session: user asks, agent calls search, gets a result, says 'done'."""
    metadata = {
        "version": "0.1",
        "id": "sha256:meta",
        "kind": "metadata",
        "ts": "2026-04-25T00:00:00.000Z",
        "parent": None,
        "payload": {"sdk": {"name": "test"}},
    }
    req1_payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "params": {},
    }
    req1_id = _core.content_id(req1_payload)
    resp1_payload = {
        "model": "m",
        "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "rust"}}],
        "stop_reason": "tool_use",
        "latency_ms": 10,
        "usage": {"input_tokens": 5, "output_tokens": 3, "thinking_tokens": 0},
    }
    resp1_id = _core.content_id(resp1_payload)
    req2_payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "rust"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "rust is great"},
        ],
        "params": {},
    }
    req2_id = _core.content_id(req2_payload)
    resp2_payload = {
        "model": "m",
        "content": [{"type": "text", "text": "done."}],
        "stop_reason": "end_turn",
        "latency_ms": 8,
        "usage": {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 0},
    }
    resp2_id = _core.content_id(resp2_payload)
    return [
        metadata,
        {
            "version": "0.1",
            "id": req1_id,
            "kind": "chat_request",
            "ts": "...",
            "parent": "sha256:meta",
            "payload": req1_payload,
        },
        {
            "version": "0.1",
            "id": resp1_id,
            "kind": "chat_response",
            "ts": "...",
            "parent": req1_id,
            "payload": resp1_payload,
        },
        {
            "version": "0.1",
            "id": "sha256:tc",
            "kind": "tool_call",
            "ts": "...",
            "parent": resp1_id,
            "payload": {
                "tool_name": "search",
                "tool_call_id": "t1",
                "arguments": {"q": "rust"},
            },
        },
        {
            "version": "0.1",
            "id": "sha256:tr",
            "kind": "tool_result",
            "ts": "...",
            "parent": "sha256:tc",
            "payload": {
                "tool_call_id": "t1",
                "output": "rust is great",
                "is_error": False,
                "latency_ms": 5,
            },
        },
        {
            "version": "0.1",
            "id": req2_id,
            "kind": "chat_request",
            "ts": "...",
            "parent": "sha256:tr",
            "payload": req2_payload,
        },
        {
            "version": "0.1",
            "id": resp2_id,
            "kind": "chat_response",
            "ts": "...",
            "parent": req2_id,
            "payload": resp2_payload,
        },
    ]


def test_engine_drives_loop_to_completion() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline, novel_policy=StubPolicy())
    out, summary = asyncio.run(run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=tools))
    kinds = [r["kind"] for r in out]
    # metadata + (chat_request → chat_response → tool_call → tool_result)
    # + (chat_request → chat_response) + replay-summary metadata.
    assert kinds == [
        "metadata",
        "chat_request",
        "chat_response",
        "tool_call",
        "tool_result",
        "chat_request",
        "chat_response",
        "metadata",  # summary
    ]
    assert summary.sessions_replayed == 1
    assert summary.total_llm_calls == 2
    assert summary.total_tool_calls == 1
    assert summary.total_tool_errors == 0


def test_engine_summary_metadata_lands_at_end() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline)
    out, _ = asyncio.run(run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=tools))
    final = out[-1]
    assert final["kind"] == "metadata"
    assert "replay_summary" in final["payload"]


def test_engine_root_metadata_carries_baseline_of() -> None:
    """Provenance pointer back at the baseline must be present."""
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline)
    out, _ = asyncio.run(run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=tools))
    assert out[0]["kind"] == "metadata"
    assert out[0]["payload"]["baseline_of"] == "sha256:meta"
    assert out[0]["payload"]["replay"]["engine"] == "agent_loop"


def test_engine_caps_at_max_turns_with_truncation_marker() -> None:
    """An infinite-loop agent must be cut off and produce an error
    record with code=loop_max_exceeded."""
    # Build a baseline whose mock LLM always responds with a tool_use,
    # never end_turn — engine will loop forever without max_turns.
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)

    # Patch the mock to ALWAYS return a tool_use response so the loop
    # never terminates organically.
    looping_response = {
        "model": "m",
        "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "rust"}}],
        "stop_reason": "tool_use",
        "latency_ms": 0,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }

    class LoopingLLM:
        @property
        def id(self) -> str:
            return "loop-mock"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            return looping_response

    tools = ReplayToolBackend.from_trace(baseline, novel_policy=StubPolicy())
    out, summary = asyncio.run(
        run_agent_loop_replay(
            baseline,
            llm_backend=LoopingLLM(),  # type: ignore[arg-type]
            tool_backend=tools,
            config=AgentLoopConfig(max_turns=3),
        )
    )
    kinds = [r["kind"] for r in out]
    assert "error" in kinds
    truncate_err = next(r for r in out if r["kind"] == "error")
    assert truncate_err["payload"]["code"] == "loop_max_exceeded"
    assert summary.sessions_truncated == 1


def test_engine_records_tool_errors_in_summary() -> None:
    """A tool that raises must surface as an is_error tool_result and
    bump the summary counter, but the loop continues."""
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)

    class ExplodingTools:
        @property
        def id(self) -> str:
            return "explode"

        async def execute(self, call: Any) -> Any:
            raise RuntimeError("simulated tool failure")

    out, summary = asyncio.run(
        run_agent_loop_replay(
            baseline,
            llm_backend=llm,
            tool_backend=ExplodingTools(),  # type: ignore[arg-type]
        )
    )
    assert summary.total_tool_errors == 1
    # The error result lands as a real tool_result record with is_error=True.
    tr = next(r for r in out if r["kind"] == "tool_result")
    assert tr["payload"]["is_error"] is True


def test_engine_default_max_turns_constant() -> None:
    """DEFAULT_MAX_TURNS must remain stable across releases for users
    who relied on it as a generous-but-bounded ceiling."""
    assert DEFAULT_MAX_TURNS == 32


def test_engine_with_stub_tool_backend_passes_through() -> None:
    """The stub backend always succeeds; the loop terminates when the
    LLM hits end_turn (which MockLLM serves on the second turn)."""
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    out, summary = asyncio.run(
        run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=StubToolBackend())
    )
    assert summary.total_tool_errors == 0
    assert summary.total_tool_calls == 1


def test_engine_records_are_content_addressed() -> None:
    """Every record's id must equal the content_id of its payload —
    Shadow's invariant. The agent loop must respect it."""
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline)
    out, _ = asyncio.run(run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=tools))
    for rec in out:
        assert rec["id"] == _core.content_id(rec["payload"])
