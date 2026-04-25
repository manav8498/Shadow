"""Tests for ``ReplayToolBackend`` and the novel-call policy framework.

Covers:
- exact-match recorded-result lookup
- novel-call paths under each policy (strict / stub / fuzzy / delegate)
- multi-trace merge
- tool calls embedded in chat_response content blocks (auto-instrument shape)
- standalone tool_call records (Session API shape)
- orphan tool_call (no paired tool_result) silently dropped
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from shadow.errors import ShadowBackendError
from shadow.tools.base import ToolCall, ToolResult
from shadow.tools.novel import (
    DelegatePolicy,
    FuzzyMatchPolicy,
    StrictPolicy,
    StubPolicy,
)
from shadow.tools.replay import ReplayToolBackend


def _trace_with_tool(
    tool_name: str = "search",
    args: dict[str, Any] | None = None,
    output: str = "found",
) -> list[dict[str, Any]]:
    """Tiny baseline: one chat_response with a tool_use block, paired
    tool_call record, paired tool_result record."""
    args = args if args is not None else {"q": "rust"}
    return [
        {
            "kind": "metadata",
            "id": "sha256:m",
            "ts": "t",
            "parent": None,
            "payload": {"sdk": {"name": "test"}},
        },
        {
            "kind": "chat_response",
            "id": "sha256:r",
            "ts": "t",
            "parent": "sha256:m",
            "payload": {
                "content": [{"type": "tool_use", "id": "t1", "name": tool_name, "input": args}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
            },
        },
        {
            "kind": "tool_call",
            "id": "sha256:tc",
            "ts": "t",
            "parent": "sha256:r",
            "payload": {
                "tool_name": tool_name,
                "tool_call_id": "t1",
                "arguments": args,
            },
        },
        {
            "kind": "tool_result",
            "id": "sha256:tr",
            "ts": "t",
            "parent": "sha256:tc",
            "payload": {
                "tool_call_id": "t1",
                "output": output,
                "is_error": False,
                "latency_ms": 12,
            },
        },
    ]


# ---- exact-match lookup -------------------------------------------------


def test_replay_backend_returns_recorded_result() -> None:
    backend = ReplayToolBackend.from_trace(_trace_with_tool(output="rust is great"))
    call = ToolCall(id="new-id", name="search", arguments={"q": "rust"})
    result = asyncio.run(backend.execute(call))
    assert result.output == "rust is great"
    assert not result.is_error
    assert result.latency_ms == 12
    # The result mirrors the candidate's call id, not the baseline's.
    assert result.tool_call_id == "new-id"


def test_replay_backend_args_order_independence() -> None:
    """Recording with one key order must match a candidate with the
    other (canonical hashing)."""
    backend = ReplayToolBackend.from_trace(_trace_with_tool(args={"q": "rust", "limit": 10}))
    call = ToolCall(id="x", name="search", arguments={"limit": 10, "q": "rust"})
    result = asyncio.run(backend.execute(call))
    assert result.output == "found"


# ---- novel-call policies ------------------------------------------------


def test_strict_policy_raises() -> None:
    backend = ReplayToolBackend.from_trace(_trace_with_tool(), novel_policy=StrictPolicy())
    novel = ToolCall(id="x", name="other_tool", arguments={})
    with pytest.raises(ShadowBackendError):
        asyncio.run(backend.execute(novel))


def test_stub_policy_returns_placeholder() -> None:
    backend = ReplayToolBackend.from_trace(_trace_with_tool(), novel_policy=StubPolicy())
    novel = ToolCall(id="x", name="other_tool", arguments={"k": "v"})
    result = asyncio.run(backend.execute(novel))
    assert result.tool_call_id == "x"
    assert "novel" in str(result.output)
    assert "other_tool" in str(result.output)


def test_stub_policy_can_mark_results_as_errors() -> None:
    backend = ReplayToolBackend.from_trace(
        _trace_with_tool(), novel_policy=StubPolicy(is_error=True)
    )
    result = asyncio.run(backend.execute(ToolCall("x", "novel", {})))
    assert result.is_error


def test_fuzzy_match_policy_picks_nearest_recorded_call() -> None:
    """When the candidate uses the same tool with overlapping arg
    keys, fuzzy match returns the nearest recorded result."""
    backend = ReplayToolBackend.from_trace(
        _trace_with_tool(args={"q": "rust", "limit": 10}, output="rust source"),
        novel_policy=FuzzyMatchPolicy(),
    )
    # Candidate's args differ in `limit` value but share the key set.
    new_call = ToolCall(id="new-id", name="search", arguments={"q": "python", "limit": 25})
    result = asyncio.run(backend.execute(new_call))
    assert result.output == "rust source"


def test_fuzzy_match_policy_falls_back_to_stub_when_no_same_tool() -> None:
    backend = ReplayToolBackend.from_trace(
        _trace_with_tool(tool_name="search"),
        novel_policy=FuzzyMatchPolicy(),
    )
    novel = ToolCall(id="x", name="completely_different_tool", arguments={"k": 1})
    result = asyncio.run(backend.execute(novel))
    # No recorded same-tool calls → stub fallback fires.
    assert "novel" in str(result.output)


def test_delegate_policy_calls_user_function() -> None:
    seen: list[ToolCall] = []

    async def my_fallback(call: ToolCall) -> ToolResult:
        seen.append(call)
        return ToolResult(call.id, output=f"delegated:{call.name}")

    backend = ReplayToolBackend.from_trace(
        _trace_with_tool(), novel_policy=DelegatePolicy(my_fallback)
    )
    novel = ToolCall(id="x", name="custom", arguments={"k": "v"})
    result = asyncio.run(backend.execute(novel))
    assert result.output == "delegated:custom"
    assert len(seen) == 1
    assert seen[0].name == "custom"


# ---- without novel_policy: must raise ----------------------------------


def test_replay_backend_without_policy_raises_on_novel_call() -> None:
    backend = ReplayToolBackend.from_trace(_trace_with_tool())
    novel = ToolCall(id="x", name="other_tool", arguments={})
    with pytest.raises(ShadowBackendError):
        asyncio.run(backend.execute(novel))


# ---- multi-trace merge --------------------------------------------------


def test_from_traces_merges_results_with_later_winning() -> None:
    t1 = _trace_with_tool(output="first")
    t2 = _trace_with_tool(output="second")
    backend = ReplayToolBackend.from_traces([t1, t2])
    call = ToolCall("x", "search", {"q": "rust"})
    assert asyncio.run(backend.execute(call)).output == "second"


# ---- introspection ------------------------------------------------------


def test_contains_works_on_recorded_call() -> None:
    backend = ReplayToolBackend.from_trace(_trace_with_tool())
    assert ToolCall("any-id", "search", {"q": "rust"}) in backend
    assert ToolCall("any-id", "other", {}) not in backend


def test_orphan_tool_call_without_result_silently_dropped() -> None:
    """A tool_call with no paired tool_result must not appear in the index."""
    trace = _trace_with_tool()
    # Strip the tool_result.
    trace = [r for r in trace if r["kind"] != "tool_result"]
    backend = ReplayToolBackend.from_trace(trace)
    assert len(backend) == 0
