"""Tests for the v2.9 LLM-backed novel-pattern diagnosis fallback.

The LLM fallback closes the "rule-based ceiling" gap on the
deterministic recommendations engine: when no hardcoded pattern
matches but a severe regression is present, an LLM is asked to
propose a novel root cause.

These offline tests inject mock LLM callers so no API calls happen.
The corresponding live test lives in
``test_recommendations_llm_live.py`` and gates on
``SHADOW_RUN_NETWORK_TESTS=1`` + ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shadow.diff_py.recommendations import (
    LLMDiagnosedRecommendation,
    enrich_with_llm,
)


def _report(
    *,
    severities: dict[str, str] | None = None,
    deltas: dict[str, float] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal DiffReport-shaped dict for tests."""
    severities = severities or {}
    deltas = deltas or {}
    rows = []
    for axis in (
        "semantic",
        "trajectory",
        "safety",
        "verbosity",
        "latency",
        "cost",
        "reasoning",
        "judge",
        "conformance",
    ):
        rows.append(
            {
                "axis": axis,
                "severity": severities.get(axis, "none"),
                "delta": deltas.get(axis, 0.0),
                "ci95_low": deltas.get(axis, 0.0) - 0.05,
                "ci95_high": deltas.get(axis, 0.0) + 0.05,
            }
        )
    return {
        "rows": rows,
        "recommendations": list(recommendations or []),
    }


def _good_diagnosis() -> str:
    """A well-formed JSON diagnosis the LLM might return."""
    return json.dumps(
        {
            "severity": "warning",
            "message": "Possible vocabulary drift in instruction following.",
            "rationale": (
                "The verbosity axis shifted moderately on traces with neither "
                "tool-sequence nor refusal changes — pattern not matched by the "
                "10 hardcoded signatures. Consider: did a recent change to the "
                "user-prompt template alter how the model interprets task "
                "instructions?"
            ),
            "axis": "verbosity",
            "confidence": 0.55,
        }
    )


class TestLLMFallbackGating:
    def test_returns_unchanged_when_root_cause_already_present(self) -> None:
        report = _report(
            severities={"latency": "severe"},
            recommendations=[
                {
                    "severity": "error",
                    "action": "root_cause",
                    "turn": 0,
                    "message": "Looks like a model change.",
                    "rationale": "...",
                    "axis": "cost",
                    "confidence": 0.85,
                }
            ],
        )
        # Caller would never run because root_cause is present.
        called = []

        def fake_caller(_summary: str) -> str:
            called.append(1)
            return _good_diagnosis()

        out = enrich_with_llm(report, llm_caller=fake_caller)
        assert called == [], "LLM must NOT be called when root_cause exists"
        assert out is report or out == report  # unchanged

    def test_returns_unchanged_when_no_severe_axis(self) -> None:
        report = _report(
            severities={"latency": "moderate"},  # not severe
            recommendations=[],
        )
        called = []

        def fake_caller(_summary: str) -> str:
            called.append(1)
            return _good_diagnosis()

        out = enrich_with_llm(report, llm_caller=fake_caller)
        assert called == []
        assert out["recommendations"] == []

    def test_no_api_key_and_no_caller_returns_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        report = _report(severities={"semantic": "severe"})
        out = enrich_with_llm(report)  # no llm_caller arg
        assert out["recommendations"] == []


class TestLLMFallbackHappyPath:
    def test_severe_axis_with_no_root_cause_invokes_llm(self) -> None:
        report = _report(
            severities={"verbosity": "severe"},
            deltas={"verbosity": 0.6},
        )
        out = enrich_with_llm(report, llm_caller=lambda _s: _good_diagnosis())
        recs = out["recommendations"]
        assert len(recs) == 1
        assert recs[0]["source"] == "llm"
        assert recs[0]["action"] == "root_cause"
        assert recs[0]["confidence"] <= 0.65, (
            f"LLM confidence must be capped at 0.65 even if the LLM claimed "
            f"more; got {recs[0]['confidence']}"
        )

    def test_existing_per_divergence_recommendations_preserved(self) -> None:
        report = _report(
            severities={"verbosity": "severe"},
            recommendations=[
                {
                    "severity": "warning",
                    "action": "review",
                    "turn": 3,
                    "message": "Review the verbosity shift at turn 3.",
                    "rationale": "...",
                    "axis": "verbosity",
                    "confidence": 0.5,
                }
            ],
        )
        out = enrich_with_llm(report, llm_caller=lambda _s: _good_diagnosis())
        assert len(out["recommendations"]) == 2
        # Existing review recommendation preserved.
        assert out["recommendations"][0]["action"] == "review"
        # LLM diagnosis appended.
        assert out["recommendations"][1]["source"] == "llm"

    def test_llm_diagnosis_shape_matches_recommendation_dict(self) -> None:
        report = _report(severities={"semantic": "severe"})
        out = enrich_with_llm(report, llm_caller=lambda _s: _good_diagnosis())
        rec = out["recommendations"][0]
        for field in ("severity", "action", "turn", "message", "rationale", "axis", "confidence"):
            assert field in rec


class TestLLMFallbackValidation:
    def test_malformed_json_dropped(self) -> None:
        report = _report(severities={"semantic": "severe"})
        out = enrich_with_llm(report, llm_caller=lambda _s: "not-valid-json-at-all")
        assert out["recommendations"] == []

    def test_missing_required_field_dropped(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            # Missing "rationale".
            return json.dumps(
                {
                    "severity": "warning",
                    "message": "something happened",
                    "axis": "semantic",
                    "confidence": 0.5,
                }
            )

        out = enrich_with_llm(report, llm_caller=caller)
        assert out["recommendations"] == []

    def test_invalid_severity_enum_dropped(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            return json.dumps(
                {
                    "severity": "catastrophic",  # not in enum
                    "message": "very bad indeed",
                    "rationale": "more than twenty characters here for sure.",
                    "axis": "semantic",
                    "confidence": 0.5,
                }
            )

        out = enrich_with_llm(report, llm_caller=caller)
        assert out["recommendations"] == []

    def test_invalid_axis_dropped(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            return json.dumps(
                {
                    "severity": "warning",
                    "message": "vague but plausible",
                    "rationale": "this rationale is at least twenty chars long.",
                    "axis": "fictional_axis",
                    "confidence": 0.5,
                }
            )

        out = enrich_with_llm(report, llm_caller=caller)
        assert out["recommendations"] == []

    def test_message_too_short_dropped(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            return json.dumps(
                {
                    "severity": "warning",
                    "message": "tiny",  # < 8 chars
                    "rationale": "this rationale is at least twenty chars long.",
                    "axis": "semantic",
                    "confidence": 0.5,
                }
            )

        out = enrich_with_llm(report, llm_caller=caller)
        assert out["recommendations"] == []

    def test_caller_raises_returns_unchanged(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            raise RuntimeError("network down")

        out = enrich_with_llm(report, llm_caller=caller)
        assert out["recommendations"] == []


class TestConfidenceCap:
    def test_high_llm_confidence_capped_to_065(self) -> None:
        report = _report(severities={"semantic": "severe"})

        def caller(_s: str) -> str:
            return json.dumps(
                {
                    "severity": "error",
                    "message": "I am extremely sure about this diagnosis.",
                    "rationale": "rationale of suitable length to pass validation.",
                    "axis": "semantic",
                    "confidence": 0.99,
                }
            )

        out = enrich_with_llm(report, llm_caller=caller)
        rec = out["recommendations"][0]
        assert rec["confidence"] == pytest.approx(0.65), (
            "LLM-diagnosed confidence must be capped at 0.65 to honestly "
            f"signal it's beyond curated patterns; got {rec['confidence']}"
        )


class TestDataclassRoundTrip:
    def test_llm_recommendation_to_dict_preserves_source_marker(self) -> None:
        rec = LLMDiagnosedRecommendation(
            severity="warning",
            action="root_cause",
            turn=0,
            message="example",
            rationale="rationale of suitable length to pass validation.",
            axis="semantic",
            confidence=0.4,
        )
        d = rec.to_dict()
        assert d["source"] == "llm"
        assert d["action"] == "root_cause"
