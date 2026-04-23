"""Tests for the SanityJudge and the judge-axis aggregation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from shadow.judge import SanityJudge, aggregate_scores


class _FakeBackend:
    """Backend that returns a pre-programmed JSON verdict per call."""

    def __init__(self, verdicts: list[dict[str, Any]]) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return "fake-judge"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        payload = self._verdicts[self._i % len(self._verdicts)]
        self._i += 1
        return {
            "model": "fake",
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": "end_turn",
            "latency_ms": 1,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        }


def _resp(text: str) -> dict[str, Any]:
    return {
        "model": "x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "latency_ms": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }


def test_sanity_judge_maps_better_equal_worse_to_scores() -> None:
    backend = _FakeBackend(
        [
            {"verdict": "better", "confidence": 0.9, "reason": "same info, shorter"},
            {"verdict": "equal", "confidence": 0.8, "reason": "equivalent"},
            {"verdict": "worse", "confidence": 0.95, "reason": "missed details"},
        ]
    )
    j = SanityJudge(backend)
    v1 = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    v2 = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    v3 = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v1["score"] == 1.0
    assert v2["score"] == 1.0
    assert v3["score"] == 0.0


def test_sanity_judge_handles_malformed_output_as_neutral() -> None:
    class _GarbageBackend:
        id = "garbage"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            return _resp("not JSON at all")

    j = SanityJudge(_GarbageBackend())  # type: ignore[arg-type]
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "error"
    assert v["score"] == 0.5


def test_sanity_judge_strips_markdown_fences() -> None:
    fenced = '```json\n{"verdict":"better","confidence":0.9,"reason":"ok"}\n```'

    class _FencedBackend:
        id = "fenced"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            return _resp(fenced)

    j = SanityJudge(_FencedBackend())  # type: ignore[arg-type]
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "better"
    assert v["score"] == 1.0


def test_sanity_judge_falls_back_to_neutral_when_backend_raises() -> None:
    class _BrokenBackend:
        id = "broken"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

    j = SanityJudge(_BrokenBackend())  # type: ignore[arg-type]
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "error"
    assert v["score"] == 0.5


def test_sanity_judge_includes_task_context_when_provided() -> None:
    backend = _FakeBackend([{"verdict": "equal", "confidence": 1.0, "reason": "ok"}])
    j = SanityJudge(backend)
    request_ctx = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "what is the capital of France?"},
        ]
    }
    asyncio.run(j.score_pair(_resp("Paris"), _resp("Paris."), request_ctx))
    sent = backend.calls[0]["messages"][0]["content"]
    assert "TASK:" in sent
    assert "capital of France" in sent


def test_aggregate_scores_flags_perfect_run_as_none() -> None:
    row = aggregate_scores([1.0] * 10, seed=1)
    assert row["axis"] == "judge"
    assert row["candidate_median"] == 1.0
    assert row["delta"] == 0.0
    assert row["severity"] == "none"


def test_aggregate_scores_flags_half_regression_as_nonzero() -> None:
    row = aggregate_scores([0.0] * 10, seed=1)
    assert row["candidate_median"] == 0.0
    assert row["delta"] == pytest.approx(-1.0)
    # delta is the whole scale → severity escalates once CI stabilises
    assert row["severity"] in {"moderate", "severe", "minor"}
    assert row["n"] == 10


def test_aggregate_scores_low_power_flag_on_small_n() -> None:
    row = aggregate_scores([1.0, 0.5, 1.0], seed=1)
    assert "low_power" in row["flags"]


def test_aggregate_scores_empty_input_returns_empty_row() -> None:
    row = aggregate_scores([])
    assert row["n"] == 0
    assert row["severity"] == "none"
