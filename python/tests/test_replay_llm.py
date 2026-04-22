"""Coverage-boosting tests for shadow.replay + shadow.llm.mock/positional."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from shadow import _core
from shadow.errors import ShadowBackendError, ShadowParseError
from shadow.llm import MockLLM, PositionalMockLLM
from shadow.replay import run_replay
from shadow.sdk import Session


def _write_trace(path: Path, latencies: list[int], texts: list[str]) -> None:
    with Session(output_path=path, tags={"env": "test"}) as s:
        for lat, text in zip(latencies, texts, strict=True):
            s.record_chat(
                request={
                    "model": "x",
                    "messages": [{"role": "user", "content": text}],
                    "params": {},
                },
                response={
                    "model": "x",
                    "content": [{"type": "text", "text": f"echo: {text}"}],
                    "stop_reason": "end_turn",
                    "latency_ms": lat,
                    "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
                },
            )


def test_mock_llm_from_path_and_len(tmp_path: Path) -> None:
    path = tmp_path / "t.agentlog"
    _write_trace(path, [10, 20], ["q1", "q2"])
    mock = MockLLM.from_path(path)
    assert len(mock) == 2
    assert mock.id == "mock"


def test_mock_llm_from_traces_merges(tmp_path: Path) -> None:
    p1 = tmp_path / "t1.agentlog"
    p2 = tmp_path / "t2.agentlog"
    _write_trace(p1, [10], ["a"])
    _write_trace(p2, [20], ["b"])
    t1 = _core.parse_agentlog(p1.read_bytes())
    t2 = _core.parse_agentlog(p2.read_bytes())
    mock = MockLLM.from_traces([t1, t2])
    assert len(mock) == 2


def test_positional_mock_llm_cycles_through_responses(tmp_path: Path) -> None:
    path = tmp_path / "t.agentlog"
    _write_trace(path, [10, 20, 30], ["a", "b", "c"])
    mock = PositionalMockLLM.from_path(path)
    assert len(mock) == 3
    assert mock.id == "positional-mock"

    async def drive() -> None:
        r1 = await mock.complete({"anything": 1})
        r2 = await mock.complete({"different": 2})
        r3 = await mock.complete({"totally_different": 3})
        assert r1["latency_ms"] == 10
        assert r2["latency_ms"] == 20
        assert r3["latency_ms"] == 30

    asyncio.run(drive())


def test_positional_mock_raises_when_exhausted() -> None:
    mock = PositionalMockLLM(
        [
            {
                "model": "x",
                "content": [],
                "latency_ms": 1,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            }
        ]
    )

    async def drive() -> None:
        await mock.complete({})
        with pytest.raises(ShadowBackendError, match="exhausted"):
            await mock.complete({})

    asyncio.run(drive())


def test_run_replay_happy_path(tmp_path: Path) -> None:
    path = tmp_path / "t.agentlog"
    _write_trace(path, [10, 20], ["a", "b"])
    baseline = _core.parse_agentlog(path.read_bytes())
    mock = MockLLM.from_trace(baseline)

    candidate = asyncio.run(run_replay(baseline, mock))
    kinds = [r["kind"] for r in candidate]
    assert kinds[0] == "metadata"
    assert kinds[-1] == "replay_summary"
    # New metadata root records baseline_of.
    assert candidate[0]["payload"]["baseline_of"] == baseline[0]["id"]


def test_run_replay_empty_baseline_errors() -> None:
    mock = MockLLM({})
    with pytest.raises(ShadowParseError, match="empty"):
        asyncio.run(run_replay([], mock))


def test_run_replay_non_metadata_root_errors() -> None:
    mock = MockLLM({})
    bad = [
        {
            "version": "0.1",
            "id": "sha256:" + "a" * 64,
            "kind": "chat_request",
            "ts": "2026-04-21T10:00:00Z",
            "parent": None,
            "payload": {},
        }
    ]
    with pytest.raises(ShadowParseError, match="root is"):
        asyncio.run(run_replay(bad, mock))


def test_run_replay_missing_response_produces_error_record(tmp_path: Path) -> None:
    # Baseline has 2 requests, mock only knows about 1.
    path = tmp_path / "t.agentlog"
    _write_trace(path, [10, 20], ["a", "b"])
    baseline = _core.parse_agentlog(path.read_bytes())

    # Drop the second response from the mock's map so req #2 fails.
    trimmed = MockLLM.from_trace(baseline[:-1])  # strips last response

    candidate = asyncio.run(run_replay(baseline, trimmed))
    errors = [r for r in candidate if r["kind"] == "error"]
    assert len(errors) == 1
    summary = candidate[-1]
    assert summary["payload"]["error_count"] == 1
