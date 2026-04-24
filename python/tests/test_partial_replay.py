"""Tests for partial replay — the 2nd replay-as-science slice.

Partial replay locks the baseline prefix verbatim, then switches to
live replay at a branch point. These tests lock the semantics so the
rest of the diff machinery treats the result the same as any other
candidate trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from shadow.errors import ShadowParseError
from shadow.replay import run_partial_replay

# ---- test helpers ---------------------------------------------------------


class _ScriptedBackend:
    """Minimal LlmBackend stub returning a fixed response sequence."""

    id = "scripted-test"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        if not self._responses:
            return {
                "model": "scripted",
                "content": [{"type": "text", "text": "default"}],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            }
        return self._responses.pop(0)


def _baseline_trace(turns: int = 4) -> list[dict[str, Any]]:
    """Build a deterministic baseline with `turns` request/response pairs."""
    recs: list[dict[str, Any]] = []
    meta_payload = {"sdk": {"name": "test", "version": "0"}}
    recs.append(
        {
            "version": "0.1",
            "id": "meta0",
            "kind": "metadata",
            "ts": "2026-04-24T00:00:00Z",
            "parent": None,
            "payload": meta_payload,
        }
    )
    last = "meta0"
    for i in range(turns):
        req_id = f"req{i}"
        resp_id = f"resp{i}"
        recs.append(
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": f"2026-04-24T00:00:0{i}Z",
                "parent": last,
                "payload": {
                    "model": "m",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "params": {},
                },
            }
        )
        recs.append(
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": f"2026-04-24T00:00:0{i}.500Z",
                "parent": req_id,
                "payload": {
                    "model": "m",
                    "content": [{"type": "text", "text": f"baseline-answer-{i}"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 10,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            }
        )
        last = resp_id
    return recs


def _live_response(i: int) -> dict[str, Any]:
    return {
        "model": "m",
        "content": [{"type": "text", "text": f"live-answer-{i}"}],
        "stop_reason": "end_turn",
        "latency_ms": 20,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }


# ---- semantics ------------------------------------------------------------


def test_branch_at_zero_is_fully_live() -> None:
    baseline = _baseline_trace(3)
    backend = _ScriptedBackend([_live_response(i) for i in range(3)])
    out = asyncio.run(run_partial_replay(baseline, branch_at=0, backend=backend))
    # 3 turns → 3 live requests.
    assert len(backend.requests) == 3
    texts = [r["payload"]["content"][0]["text"] for r in out if r.get("kind") == "chat_response"]
    assert texts == ["live-answer-0", "live-answer-1", "live-answer-2"]


def test_branch_at_midpoint_preserves_prefix_and_replays_suffix() -> None:
    baseline = _baseline_trace(4)
    backend = _ScriptedBackend([_live_response(2), _live_response(3)])
    out = asyncio.run(run_partial_replay(baseline, branch_at=2, backend=backend))
    # Prefix turns 0-1 should be verbatim baseline answers.
    # Suffix turns 2-3 should be live answers.
    texts = [r["payload"]["content"][0]["text"] for r in out if r.get("kind") == "chat_response"]
    assert texts == [
        "baseline-answer-0",
        "baseline-answer-1",
        "live-answer-2",
        "live-answer-3",
    ]
    # Exactly 2 live requests — prefix did not hit the backend.
    assert len(backend.requests) == 2


def test_branch_at_end_produces_full_baseline_copy() -> None:
    baseline = _baseline_trace(3)
    backend = _ScriptedBackend([])
    out = asyncio.run(run_partial_replay(baseline, branch_at=3, backend=backend))
    texts = [r["payload"]["content"][0]["text"] for r in out if r.get("kind") == "chat_response"]
    assert texts == ["baseline-answer-0", "baseline-answer-1", "baseline-answer-2"]
    assert len(backend.requests) == 0


def test_branch_at_beyond_length_clamps_gracefully() -> None:
    baseline = _baseline_trace(2)
    backend = _ScriptedBackend([])
    out = asyncio.run(run_partial_replay(baseline, branch_at=99, backend=backend))
    assert [r for r in out if r.get("kind") == "chat_response"]
    # No backend calls (clamped to len).
    assert len(backend.requests) == 0


def test_metadata_root_is_tagged_with_branch_point() -> None:
    baseline = _baseline_trace(2)
    backend = _ScriptedBackend([_live_response(1)])
    out = asyncio.run(run_partial_replay(baseline, branch_at=1, backend=backend))
    meta = out[0]
    assert meta["kind"] == "metadata"
    assert meta["payload"]["baseline_of"] == "meta0"
    assert meta["payload"]["partial_replay"] == {"branch_at": 1}


def test_replay_summary_records_prefix_count() -> None:
    baseline = _baseline_trace(4)
    backend = _ScriptedBackend([_live_response(i) for i in range(2, 4)])
    out = asyncio.run(run_partial_replay(baseline, branch_at=2, backend=backend))
    summary = next(r for r in out if r.get("kind") == "replay_summary")
    assert summary["payload"]["branch_at"] == 2
    assert summary["payload"]["prefix_turn_count"] == 2
    # output_count includes prefix responses + live responses.
    assert summary["payload"]["output_count"] == 4


def test_parents_form_consistent_dag() -> None:
    baseline = _baseline_trace(3)
    backend = _ScriptedBackend([_live_response(1), _live_response(2)])
    out = asyncio.run(run_partial_replay(baseline, branch_at=1, backend=backend))
    # Every record's parent should point at a record earlier in the list.
    seen_ids = set()
    for rec in out:
        parent = rec.get("parent")
        if parent is not None:
            assert parent in seen_ids, f"dangling parent for {rec.get('kind')}"
        seen_ids.add(rec["id"])


def test_negative_branch_at_rejected() -> None:
    with pytest.raises(ShadowParseError):
        asyncio.run(
            run_partial_replay(_baseline_trace(2), branch_at=-1, backend=_ScriptedBackend([]))
        )


def test_empty_trace_rejected() -> None:
    with pytest.raises(ShadowParseError):
        asyncio.run(run_partial_replay([], branch_at=0, backend=_ScriptedBackend([])))


def test_backend_errors_captured_as_error_records() -> None:
    from shadow.errors import ShadowBackendError

    class _FailingBackend:
        id = "failing"

        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise ShadowBackendError("rate limit")

    out = asyncio.run(run_partial_replay(_baseline_trace(2), 0, _FailingBackend()))
    err_records = [r for r in out if r.get("kind") == "error"]
    assert len(err_records) == 2
    assert err_records[0]["payload"]["message"] == "rate limit"
