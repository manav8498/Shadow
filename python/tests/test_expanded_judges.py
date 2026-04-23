"""Tests for CorrectnessJudge, PairwiseJudge, FormatJudge."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from shadow.judge import CorrectnessJudge, FormatJudge, PairwiseJudge


class _ProgrammedBackend:
    """Returns pre-programmed JSON verdicts in sequence."""

    def __init__(self, verdicts: list[dict[str, Any]]) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return "programmed"

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


# ---------------------------------------------------------------------------
# CorrectnessJudge
# ---------------------------------------------------------------------------


def test_correctness_judge_matches_expected() -> None:
    backend = _ProgrammedBackend(
        [{"verdict": "match", "confidence": 0.9, "reason": "same content"}]
    )
    rubric = {"what is 2+2?": "4"}
    judge = CorrectnessJudge(backend, rubric=rubric)
    v = asyncio.run(
        judge.score_pair(
            _resp("Four"),
            _resp("4"),
            request_context={"messages": [{"role": "user", "content": "what is 2+2?"}]},
        )
    )
    assert v["verdict"] == "match"
    assert v["score"] == 1.0


def test_correctness_judge_partial_maps_to_half_score() -> None:
    backend = _ProgrammedBackend(
        [{"verdict": "partial", "confidence": 0.7, "reason": "mostly right"}]
    )
    judge = CorrectnessJudge(backend, rubric={"q": "answer"})
    v = asyncio.run(
        judge.score_pair(
            _resp("baseline"),
            _resp("close"),
            request_context={"messages": [{"role": "user", "content": "q"}]},
        )
    )
    assert v["verdict"] == "partial"
    assert v["score"] == 0.5


def test_correctness_judge_without_rubric_entry_returns_neutral() -> None:
    judge = CorrectnessJudge(_ProgrammedBackend([]), rubric={"other": "x"})
    v = asyncio.run(
        judge.score_pair(
            _resp("b"),
            _resp("c"),
            request_context={"messages": [{"role": "user", "content": "unknown"}]},
        )
    )
    assert v["verdict"] == "error"
    assert v["score"] == 0.5


# ---------------------------------------------------------------------------
# PairwiseJudge — position-bias agreement
# ---------------------------------------------------------------------------


def test_pairwise_judge_both_orderings_agree_candidate_wins() -> None:
    # First call: A=candidate, B=baseline → judge says a_better
    # Second call: A=baseline, B=candidate → judge says b_better
    # Both say candidate wins → agreement → confident.
    backend = _ProgrammedBackend(
        [
            {"verdict": "a_better", "reason": "A is clearer"},
            {"verdict": "b_better", "reason": "B is clearer"},
        ]
    )
    judge = PairwiseJudge(backend)
    v = asyncio.run(judge.score_pair(_resp("baseline"), _resp("candidate")))
    assert v["verdict"] == "candidate_better"
    assert v["score"] == 1.0
    assert v["confidence"] >= 0.8


def test_pairwise_judge_disagreement_returns_uncertain() -> None:
    # First call: a_better → candidate_better.
    # Second call: a_better → baseline_better (position bias!).
    # Disagree → uncertain.
    backend = _ProgrammedBackend(
        [
            {"verdict": "a_better", "reason": "A"},
            {"verdict": "a_better", "reason": "A again"},
        ]
    )
    judge = PairwiseJudge(backend)
    v = asyncio.run(judge.score_pair(_resp("baseline"), _resp("candidate")))
    assert v["verdict"] == "uncertain"
    assert v["score"] == 0.5


def test_pairwise_judge_consistent_tie() -> None:
    backend = _ProgrammedBackend(
        [
            {"verdict": "tie", "reason": "equal"},
            {"verdict": "tie", "reason": "equal"},
        ]
    )
    judge = PairwiseJudge(backend)
    v = asyncio.run(judge.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "tie"
    assert v["score"] == 0.5


def test_pairwise_judge_calls_backend_exactly_twice() -> None:
    backend = _ProgrammedBackend(
        [
            {"verdict": "tie", "reason": "equal"},
            {"verdict": "tie", "reason": "equal"},
        ]
    )
    judge = PairwiseJudge(backend)
    asyncio.run(judge.score_pair(_resp("a"), _resp("b")))
    assert len(backend.calls) == 2


# ---------------------------------------------------------------------------
# FormatJudge — schema conformance without LLM
# ---------------------------------------------------------------------------


def test_format_judge_valid_json_no_schema_scores_1() -> None:
    judge = FormatJudge()
    v = asyncio.run(judge.score_pair(_resp("x"), _resp('{"ok": true}')))
    assert v["score"] == 1.0
    assert v["verdict"] == "json_ok"


def test_format_judge_not_json_scores_0() -> None:
    judge = FormatJudge()
    v = asyncio.run(judge.score_pair(_resp("x"), _resp("plain prose here")))
    assert v["score"] == 0.0
    assert v["verdict"] == "not_json"


def test_format_judge_schema_pass_scores_1() -> None:
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string", "enum": ["ok", "err"]}},
    }
    judge = FormatJudge(schema=schema)
    v = asyncio.run(judge.score_pair(_resp("x"), _resp('{"status": "ok"}')))
    assert v["score"] == 1.0
    assert v["verdict"] == "schema_ok"


def test_format_judge_schema_fail_scores_partial() -> None:
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string", "enum": ["ok", "err"]}},
    }
    judge = FormatJudge(schema=schema)
    v = asyncio.run(judge.score_pair(_resp("x"), _resp('{"status": "bogus"}')))
    assert v["score"] == 0.3
    assert v["verdict"] == "schema_fail"


def test_format_judge_strips_markdown_fences() -> None:
    judge = FormatJudge()
    fenced = '```json\n{"a": 1}\n```'
    v = asyncio.run(judge.score_pair(_resp("x"), _resp(fenced)))
    assert v["score"] == 1.0


def test_format_judge_nested_schema_validation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "integer"}},
                },
            }
        },
    }
    judge = FormatJudge(schema=schema)
    good = asyncio.run(judge.score_pair(_resp("x"), _resp('{"items":[{"id":1},{"id":2}]}')))
    assert good["score"] == 1.0
    bad = asyncio.run(judge.score_pair(_resp("x"), _resp('{"items":[{"id":"not-int"}]}')))
    assert bad["score"] == 0.3
