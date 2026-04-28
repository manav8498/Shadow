"""Tests for LlmJudge + the five new rubric-driven judges.

Uses a deterministic `_FakeBackend` that returns a pre-programmed JSON
verdict per call, so every test is reproducible without any API key.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from shadow.judge import (
    FactualityJudge,
    LlmJudge,
    ProcedureAdherenceJudge,
    RefusalAppropriateJudge,
    SchemaConformanceJudge,
    ToneJudge,
)


class _FakeBackend:
    """Backend that cycles through a list of canned JSON verdicts."""

    def __init__(self, verdicts: list[dict[str, Any]]) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return "fake"

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


def _request(task: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": task}]}


# ---- LlmJudge generic behaviour -------------------------------------------


def test_llm_judge_rejects_unknown_placeholders() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        LlmJudge(
            backend=_FakeBackend([]),
            rubric="Evaluate {oops} — that's not a real placeholder.",
        )


def test_llm_judge_accepts_any_subset_of_allowed_placeholders() -> None:
    # No placeholders at all — valid.
    LlmJudge(backend=_FakeBackend([]), rubric="Just respond.")
    # Only {candidate} — valid.
    LlmJudge(backend=_FakeBackend([]), rubric="Rate: {candidate}")
    # All three — valid.
    LlmJudge(
        backend=_FakeBackend([]),
        rubric="{task}\n{baseline}\n{candidate}",
    )


def test_llm_judge_maps_custom_verdict_via_score_map() -> None:
    backend = _FakeBackend([{"verdict": "great", "confidence": 0.9, "reason": "nice"}])
    j = LlmJudge(
        backend=backend,
        rubric="{candidate}",
        score_map={"great": 1.0, "poor": 0.2, "ugly": 0.0},
    )
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "great"
    assert v["score"] == 1.0
    assert v["confidence"] == pytest.approx(0.9)


def test_llm_judge_errors_on_verdict_not_in_map() -> None:
    backend = _FakeBackend([{"verdict": "excellent", "confidence": 0.9, "reason": "ok"}])
    j = LlmJudge(
        backend=backend,
        rubric="{candidate}",
        score_map={"pass": 1.0, "fail": 0.0},
    )
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "error"
    assert v["score"] == 0.5  # neutral fallback


def test_llm_judge_extracts_json_when_model_adds_prose() -> None:
    # The model prepends explanation prose — should still parse.
    noisy = (
        'Let me think about this...\n\n{"verdict": "pass", "confidence": 0.8, '
        '"reason": "matches"}\n\nThat is my answer.'
    )

    class _ProseBackend:
        id = "prose"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            return _resp(noisy)

    j = LlmJudge(
        backend=_ProseBackend(),  # type: ignore[arg-type]
        rubric="{candidate}",
        score_map={"pass": 1.0, "fail": 0.0},
    )
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "pass"
    assert v["score"] == 1.0


def test_llm_judge_handles_backend_exception_as_error() -> None:
    class _BrokenBackend:
        id = "broken"

        async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("connection refused")

    j = LlmJudge(
        backend=_BrokenBackend(),  # type: ignore[arg-type]
        rubric="{candidate}",
        score_map={"pass": 1.0, "fail": 0.0},
    )
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["verdict"] == "error"
    assert "connection refused" in v["reason"]
    assert v["score"] == 0.5


def test_llm_judge_renders_placeholders_into_prompt() -> None:
    backend = _FakeBackend([{"verdict": "pass", "confidence": 1.0, "reason": "ok"}])
    j = LlmJudge(
        backend=backend,
        rubric="TASK={task} BASELINE={baseline} CANDIDATE={candidate}",
        score_map={"pass": 1.0, "fail": 0.0},
    )
    asyncio.run(j.score_pair(_resp("B-TEXT"), _resp("C-TEXT"), _request("T-TEXT")))
    sent = backend.calls[0]["messages"][0]["content"]
    assert "TASK=T-TEXT" in sent
    assert "BASELINE=B-TEXT" in sent
    assert "CANDIDATE=C-TEXT" in sent


def test_llm_judge_handles_literal_json_braces_in_rubric() -> None:
    """Regression test: rubric with literal JSON example must not crash.

    Previously the implementation called rubric.format(...), which threw
    KeyError on every JSON key in the example. External eval flagged
    this as a major footgun; rubric authors had no way to include JSON
    examples without escaping every brace.
    """
    rubric = """Rate the candidate. Reply ONLY with JSON:
        {"verdict": "pass" | "fail", "confidence": 0.0-1.0, "reason": "..."}
        Example: {"verdict": "pass", "confidence": 0.9, "reason": "good"}

        TASK: {task}
        BASELINE: {baseline}
        CANDIDATE: {candidate}
        """
    backend = _FakeBackend([{"verdict": "pass", "confidence": 1.0, "reason": "ok"}])
    j = LlmJudge(
        backend=backend,
        rubric=rubric,
        score_map={"pass": 1.0, "fail": 0.0},
    )
    # Must not raise KeyError or anything else.
    v = asyncio.run(j.score_pair(_resp("B-TEXT"), _resp("C-TEXT"), _request("T-TEXT")))
    assert v["score"] == 1.0
    sent = backend.calls[0]["messages"][0]["content"]
    # Placeholders substituted.
    assert "TASK: T-TEXT" in sent
    assert "BASELINE: B-TEXT" in sent
    assert "CANDIDATE: C-TEXT" in sent
    # Literal JSON braces preserved in the prompt.
    assert '{"verdict": "pass" | "fail"' in sent
    assert '{"verdict": "pass", "confidence": 0.9' in sent


def test_llm_judge_clamps_out_of_range_confidence() -> None:
    backend = _FakeBackend([{"verdict": "pass", "confidence": 2.5, "reason": "overshoot"}])
    j = LlmJudge(
        backend=backend,
        rubric="{candidate}",
        score_map={"pass": 1.0, "fail": 0.0},
    )
    v = asyncio.run(j.score_pair(_resp("a"), _resp("b")))
    assert v["confidence"] == 1.0  # clamped


# ---- ProcedureAdherenceJudge ---------------------------------------------


def test_procedure_judge_scores_followed_as_1() -> None:
    backend = _FakeBackend(
        [{"verdict": "followed", "confidence": 0.92, "reason": "all steps followed"}]
    )
    j = ProcedureAdherenceJudge(
        backend,
        required_procedure=[
            "call backup_database before run_migration",
            "pause_replication before restore",
        ],
    )
    v = asyncio.run(j.score_pair(_resp("ok"), _resp("did it"), _request("migrate prod")))
    assert v["score"] == 1.0
    sent = backend.calls[0]["messages"][0]["content"]
    assert "backup_database" in sent
    assert "pause_replication" in sent


def test_procedure_judge_scores_violated_as_0() -> None:
    backend = _FakeBackend(
        [{"verdict": "violated", "confidence": 0.95, "reason": "skipped backup"}]
    )
    j = ProcedureAdherenceJudge(
        backend, required_procedure=["call backup_database before run_migration"]
    )
    v = asyncio.run(j.score_pair(_resp(""), _resp(""), _request("")))
    assert v["score"] == 0.0
    assert "skipped backup" in v["reason"]


def test_procedure_judge_rejects_empty_procedure() -> None:
    with pytest.raises(ValueError):
        ProcedureAdherenceJudge(_FakeBackend([]), required_procedure=[])


# ---- SchemaConformanceJudge ----------------------------------------------


def test_schema_judge_passes_conforming_output() -> None:
    backend = _FakeBackend(
        [{"verdict": "conforms", "confidence": 0.9, "reason": "shape + meaning ok"}]
    )
    j = SchemaConformanceJudge(
        backend,
        expected_schema={
            "order_id": "string (Acme order ID)",
            "action": "one of refund, exchange",
        },
    )
    v = asyncio.run(j.score_pair(_resp(""), _resp("{...}"), _request("refund")))
    assert v["score"] == 1.0


def test_schema_judge_flags_semantic_violation() -> None:
    backend = _FakeBackend(
        [{"verdict": "violates", "confidence": 0.9, "reason": "amount_usd is 'many'"}]
    )
    j = SchemaConformanceJudge(backend, expected_schema={"amount_usd": "positive number"})
    v = asyncio.run(j.score_pair(_resp(""), _resp(""), _request("")))
    assert v["score"] == 0.0


def test_schema_judge_rejects_empty_schema() -> None:
    with pytest.raises(ValueError):
        SchemaConformanceJudge(_FakeBackend([]), expected_schema={})


# ---- FactualityJudge ------------------------------------------------------


def test_factuality_judge_passes_consistent_response() -> None:
    backend = _FakeBackend(
        [{"verdict": "consistent", "confidence": 0.9, "reason": "no claims conflict"}]
    )
    j = FactualityJudge(backend, known_facts=["Acme offers refunds within 30 days."])
    v = asyncio.run(j.score_pair(_resp(""), _resp("ok"), _request("")))
    assert v["score"] == 1.0


def test_factuality_judge_flags_contradiction() -> None:
    backend = _FakeBackend(
        [
            {
                "verdict": "contradicts",
                "confidence": 0.95,
                "reason": "claims 90-day refund window",
            }
        ]
    )
    j = FactualityJudge(backend, known_facts=["Acme offers refunds within 30 days."])
    v = asyncio.run(j.score_pair(_resp(""), _resp("90 days"), _request("")))
    assert v["score"] == 0.0
    assert "90-day" in v["reason"]


def test_factuality_judge_rejects_empty_facts() -> None:
    with pytest.raises(ValueError):
        FactualityJudge(_FakeBackend([]), known_facts=[])


# ---- RefusalAppropriateJudge ----------------------------------------------


def test_refusal_judge_approves_appropriate_compliance() -> None:
    backend = _FakeBackend(
        [{"verdict": "appropriate", "confidence": 0.9, "reason": "benign request answered"}]
    )
    j = RefusalAppropriateJudge(backend, policy="Refuse medical advice. Answer everything else.")
    v = asyncio.run(j.score_pair(_resp(""), _resp(""), _request("what time is it")))
    assert v["score"] == 1.0


def test_refusal_judge_flags_inappropriate_over_refusal() -> None:
    backend = _FakeBackend(
        [{"verdict": "inappropriate", "confidence": 0.9, "reason": "over-refused benign task"}]
    )
    j = RefusalAppropriateJudge(backend, policy="Refuse medical advice. Answer everything else.")
    v = asyncio.run(j.score_pair(_resp(""), _resp(""), _request("")))
    assert v["score"] == 0.0
    assert "over-refused" in v["reason"]


def test_refusal_judge_rejects_empty_policy() -> None:
    with pytest.raises(ValueError):
        RefusalAppropriateJudge(_FakeBackend([]), policy="   ")


# ---- ToneJudge ------------------------------------------------------------


def test_tone_judge_matches_target_tone() -> None:
    backend = _FakeBackend(
        [{"verdict": "matches", "confidence": 0.88, "reason": "concise and professional"}]
    )
    j = ToneJudge(backend, target_tone="concise, professional, ≤ 3 sentences")
    v = asyncio.run(j.score_pair(_resp(""), _resp("Done."), _request("")))
    assert v["score"] == 1.0


def test_tone_judge_flags_tone_deviation() -> None:
    backend = _FakeBackend(
        [{"verdict": "deviates", "confidence": 0.9, "reason": "response was 6 sentences"}]
    )
    j = ToneJudge(backend, target_tone="≤ 3 sentences")
    v = asyncio.run(j.score_pair(_resp(""), _resp("a" * 500), _request("")))
    assert v["score"] == 0.0


def test_tone_judge_rejects_empty_target() -> None:
    with pytest.raises(ValueError):
        ToneJudge(_FakeBackend([]), target_tone="")
