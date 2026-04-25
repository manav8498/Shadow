"""Worked example: sandboxed agent-loop replay.

This script builds a tiny baseline trace by hand (so it's runnable
with no API keys and no network), then runs three different
sandboxed-replay flavours against it:

1. Plain agent-loop replay — recorded results served back.
2. Replace-tool-result counterfactual — what would the agent do if
   the search had returned "no results"?
3. Sandboxed real-tool replay — the same loop but driven against a
   real (in-process) Python tool function whose side effects are
   blocked by the sandbox.

Run:

    python examples/sandboxed-replay/run.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from shadow import _core
from shadow.counterfactual_loop import replace_tool_result
from shadow.llm.mock import MockLLM
from shadow.replay_loop import run_agent_loop_replay
from shadow.tools.novel import StubPolicy
from shadow.tools.replay import ReplayToolBackend
from shadow.tools.sandbox import SandboxedToolBackend


def build_baseline() -> list[dict[str, Any]]:
    """Two-turn baseline: agent calls search, gets a result, then says 'done.'"""
    metadata = {
        "version": "0.1",
        "id": "sha256:meta",
        "kind": "metadata",
        "ts": "2026-04-25T00:00:00.000Z",
        "parent": None,
        "payload": {"sdk": {"name": "example"}},
    }
    req1 = {
        "model": "m",
        "messages": [{"role": "user", "content": "find me a python tutorial"}],
        "params": {},
    }
    req1_id = _core.content_id(req1)
    resp1 = {
        "model": "m",
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "search",
                "input": {"q": "python tutorial"},
            }
        ],
        "stop_reason": "tool_use",
        "latency_ms": 10,
        "usage": {"input_tokens": 5, "output_tokens": 3, "thinking_tokens": 0},
    }
    resp1_id = _core.content_id(resp1)
    req2 = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "find me a python tutorial"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "python tutorial"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "Top result: realpython.com/intro",
            },
        ],
        "params": {},
    }
    req2_id = _core.content_id(req2)
    resp2 = {
        "model": "m",
        "content": [{"type": "text", "text": "Try realpython.com/intro."}],
        "stop_reason": "end_turn",
        "latency_ms": 8,
        "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
    }
    resp2_id = _core.content_id(resp2)
    return [
        metadata,
        {
            "version": "0.1",
            "id": req1_id,
            "kind": "chat_request",
            "ts": "...",
            "parent": "sha256:meta",
            "payload": req1,
        },
        {
            "version": "0.1",
            "id": resp1_id,
            "kind": "chat_response",
            "ts": "...",
            "parent": req1_id,
            "payload": resp1,
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
                "arguments": {"q": "python tutorial"},
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
                "output": "Top result: realpython.com/intro",
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
            "payload": req2,
        },
        {
            "version": "0.1",
            "id": resp2_id,
            "kind": "chat_response",
            "ts": "...",
            "parent": req2_id,
            "payload": resp2,
        },
    ]


async def demo_plain_replay(baseline: list[dict[str, Any]]) -> None:
    print("\n--- 1. plain agent-loop replay ---")
    llm = MockLLM.from_trace(baseline)
    tools = ReplayToolBackend.from_trace(baseline, novel_policy=StubPolicy())
    out, summary = await run_agent_loop_replay(
        baseline, llm_backend=llm, tool_backend=tools
    )
    print(f"  produced {len(out)} records, {summary.total_llm_calls} LLM calls,")
    print(f"  {summary.total_tool_calls} tool calls, "
          f"{summary.total_tool_errors} tool errors")


async def demo_counterfactual(baseline: list[dict[str, Any]]) -> None:
    print("\n--- 2. counterfactual: what if search returned no results? ---")
    cf = await replace_tool_result(
        baseline,
        tool_call_id="t1",
        new_output="<no results found>",
    )
    new_result = next(r for r in cf.trace if r["kind"] == "tool_result")
    print(f"  patched tool_result.output = {new_result['payload']['output']!r}")
    print(f"  override metadata = {cf.override}")


async def demo_sandboxed(baseline: list[dict[str, Any]]) -> None:
    print("\n--- 3. sandboxed real tool function ---")

    async def search(args: dict) -> str:
        # In production this would hit a real KB API. The sandbox
        # blocks network/subprocess/fs writes during execution, so a
        # naive socket.connect inside this function would raise a
        # SandboxViolation rather than reach the network.
        return f"[sandboxed] handled query: {args['q']}"

    sandbox = SandboxedToolBackend(tool_registry={"search": search})
    llm = MockLLM.from_trace(baseline)
    out, summary = await run_agent_loop_replay(
        baseline, llm_backend=llm, tool_backend=sandbox
    )
    sandboxed_result = next(r for r in out if r["kind"] == "tool_result")
    print(f"  sandboxed tool_result = {sandboxed_result['payload']['output']!r}")
    print(f"  total LLM calls = {summary.total_llm_calls}, "
          f"tool calls = {summary.total_tool_calls}")


async def main() -> None:
    baseline = build_baseline()
    print(f"baseline has {len(baseline)} records")
    await demo_plain_replay(baseline)
    await demo_counterfactual(baseline)
    await demo_sandboxed(baseline)


if __name__ == "__main__":
    asyncio.run(main())
