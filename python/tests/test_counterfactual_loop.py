"""Tests for the tool-loop counterfactual primitives."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from shadow import _core
from shadow.counterfactual_loop import (
    branch_at_turn,
    replace_tool_args,
    replace_tool_result,
)
from shadow.errors import ShadowConfigError
from shadow.llm.mock import MockLLM
from shadow.tools.novel import StubPolicy
from shadow.tools.replay import ReplayToolBackend


def _build_baseline() -> list[dict[str, Any]]:
    """Reusable two-turn baseline (search → done)."""
    metadata = {
        "version": "0.1",
        "id": "sha256:meta",
        "kind": "metadata",
        "ts": "...",
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


# ---- replace_tool_result ------------------------------------------------


def test_replace_tool_result_overrides_recorded_result() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    cf = asyncio.run(
        replace_tool_result(
            baseline,
            tool_call_id="t1",
            new_output="<no results found>",
            llm_backend=llm,
        )
    )
    tr = next(r for r in cf.trace if r["kind"] == "tool_result")
    assert tr["payload"]["output"] == "<no results found>"
    assert cf.override["kind"] == "replace_tool_result"
    assert cf.override["tool_name"] == "search"


def test_replace_tool_result_can_mark_as_error() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    cf = asyncio.run(
        replace_tool_result(
            baseline,
            tool_call_id="t1",
            new_output="rate limited",
            new_is_error=True,
            llm_backend=llm,
        )
    )
    tr = next(r for r in cf.trace if r["kind"] == "tool_result")
    assert tr["payload"]["is_error"] is True


def test_replace_tool_result_unknown_id_raises() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    with pytest.raises(ShadowConfigError, match="no tool_call"):
        asyncio.run(
            replace_tool_result(
                baseline,
                tool_call_id="t99",
                new_output="x",
                llm_backend=llm,
            )
        )


# ---- replace_tool_args --------------------------------------------------


def test_replace_tool_args_patch_mode_overrides_call_arguments() -> None:
    """Default mode (no backends) is a deterministic baseline patch:
    the named tool_call's args are replaced; everything else is
    untouched. Useful for sensitivity analysis without needing a
    live backend."""
    baseline = _build_baseline()
    cf = asyncio.run(
        replace_tool_args(
            baseline,
            tool_call_id="t1",
            new_arguments={"q": "python"},
        )
    )
    tc = next(r for r in cf.trace if r["kind"] == "tool_call")
    assert tc["payload"]["arguments"] == {"q": "python"}
    # And the embedded tool_use block on the chat_response is patched too.
    resp_with_tool = next(
        r
        for r in cf.trace
        if r["kind"] == "chat_response"
        and any(b.get("type") == "tool_use" for b in r["payload"].get("content", []))
    )
    tu = next(b for b in resp_with_tool["payload"]["content"] if b["type"] == "tool_use")
    assert tu["input"] == {"q": "python"}
    assert cf.override["kind"] == "replace_tool_args"
    assert cf.override["mode"] == "patch"


def test_replace_tool_args_redispatch_mode_runs_tool_backend_on_new_args() -> None:
    """With a tool_backend supplied, the patched call is dispatched
    so the paired tool_result reflects what the new args actually
    return."""
    baseline = _build_baseline()

    class Echoer:
        @property
        def id(self) -> str:
            return "echoer"

        async def execute(self, call: Any) -> Any:
            from shadow.tools.base import ToolResult

            return ToolResult(
                tool_call_id=call.id,
                output=f"echoed:{call.arguments}",
                is_error=False,
                latency_ms=0,
            )

    cf = asyncio.run(
        replace_tool_args(
            baseline,
            tool_call_id="t1",
            new_arguments={"q": "python"},
            tool_backend=Echoer(),  # type: ignore[arg-type]
        )
    )
    tr = next(r for r in cf.trace if r["kind"] == "tool_result")
    assert "echoed:" in str(tr["payload"]["output"])
    assert "python" in str(tr["payload"]["output"])
    assert cf.override["mode"] == "redispatch"


def test_replace_tool_args_unknown_id_raises() -> None:
    baseline = _build_baseline()
    with pytest.raises(ShadowConfigError):
        asyncio.run(
            replace_tool_args(
                baseline,
                tool_call_id="missing",
                new_arguments={},
            )
        )


# ---- branch_at_turn -----------------------------------------------------


def test_branch_at_turn_zero_is_full_replay() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline, novel_policy=StubPolicy())
    cf = asyncio.run(branch_at_turn(baseline, turn=0, llm_backend=llm, tool_backend=tools))
    # turn=0 means a full replay; both LLM calls fire.
    assert cf.summary.total_llm_calls == 2
    assert cf.override["turn"] == 0


def test_branch_at_turn_negative_raises() -> None:
    baseline = _build_baseline()
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline)
    with pytest.raises(ShadowConfigError, match=">= 0"):
        asyncio.run(branch_at_turn(baseline, turn=-1, llm_backend=llm, tool_backend=tools))
