"""Real-LLM end-to-end test for the agent-loop replay engine.

Gated behind ``SHADOW_RUN_NETWORK_TESTS=1``. CI never sets this; the
test only runs when an engineer explicitly opts in with their own
``OPENAI_API_KEY``.

The test builds a tiny baseline by hand, instantiates a live
``OpenAILLM`` backend, drives the agent-loop engine forward against
a real GPT-4o-mini call, and asserts the resulting trace has the
shape Shadow promises (metadata + chat pair + summary), every record
is content-addressed, and the response contains real text.

Cost: a single chat-completion call against gpt-4o-mini, well under
$0.001.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from shadow import _core
from shadow.replay_loop import run_agent_loop_replay
from shadow.tools.novel import StubPolicy
from shadow.tools.replay import ReplayToolBackend

pytestmark = pytest.mark.network


def _build_minimal_baseline() -> list[dict[str, Any]]:
    """One-turn baseline with a user message; no tool calls."""
    metadata = {
        "version": "0.1",
        "id": "sha256:meta",
        "kind": "metadata",
        "ts": "2026-04-25T00:00:00.000Z",
        "parent": None,
        "payload": {"sdk": {"name": "test"}},
    }
    req = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly the word 'pong' and nothing else.",
            }
        ],
        "params": {"temperature": 0.0, "max_tokens": 5},
    }
    req_id = _core.content_id(req)
    resp_payload: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "content": [{"type": "text", "text": "pong"}],
        "stop_reason": "end_turn",
        "latency_ms": 0,
        "usage": {"input_tokens": 12, "output_tokens": 1, "thinking_tokens": 0},
    }
    resp_id = _core.content_id(resp_payload)
    return [
        metadata,
        {
            "version": "0.1",
            "id": req_id,
            "kind": "chat_request",
            "ts": "...",
            "parent": "sha256:meta",
            "payload": req,
        },
        {
            "version": "0.1",
            "id": resp_id,
            "kind": "chat_response",
            "ts": "...",
            "parent": req_id,
            "payload": resp_payload,
        },
    ]


@pytest.mark.skipif(
    os.environ.get("SHADOW_RUN_NETWORK_TESTS") != "1",
    reason="SHADOW_RUN_NETWORK_TESTS != 1 — skipping live LLM integration test",
)
def test_agent_loop_replay_against_real_openai_returns_a_valid_trace() -> None:
    """Drive the agent-loop engine against a real OpenAI GPT-4o-mini
    call and assert the output trace has Shadow's expected shape and
    invariants. Costs a fraction of a cent per run.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from shadow.llm.openai_backend import OpenAILLM

    baseline = _build_minimal_baseline()
    llm = OpenAILLM()
    tools = ReplayToolBackend.from_trace(baseline, novel_policy=StubPolicy())

    out, summary = asyncio.run(run_agent_loop_replay(baseline, llm_backend=llm, tool_backend=tools))

    # Engine must produce a structurally valid trace.
    kinds = [r["kind"] for r in out]
    assert kinds[0] == "metadata"  # root
    assert kinds[-1] == "metadata"  # summary
    assert "chat_request" in kinds
    assert "chat_response" in kinds

    # Every record's id is the content_id of its payload.
    for rec in out:
        assert rec["id"] == _core.content_id(
            rec["payload"]
        ), f"record {rec['kind']} fails content-addressing invariant"

    # Real LLM call: response carries non-empty text content.
    response = next(r for r in out if r["kind"] == "chat_response")
    text_blocks = [b for b in response["payload"]["content"] if b.get("type") == "text"]
    assert text_blocks, "expected at least one text block in the live response"
    response_text = text_blocks[0]["text"].strip().lower()
    assert response_text, "expected non-empty response text from live API"

    # Summary captures the call.
    assert summary.sessions_replayed == 1
    assert summary.total_llm_calls >= 1
