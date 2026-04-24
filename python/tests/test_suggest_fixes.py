"""Tests for LLM-assisted prescriptive fix generation."""

from __future__ import annotations

import json
from typing import Any

from shadow.suggest_fixes import (
    FLAG_CONFIDENCE,
    MAX_EVIDENCE_CHARS,
    SuggestedFix,
    render_terminal,
    suggest_fixes,
)

# ---- test backend ---------------------------------------------------------


class _ScriptedJsonBackend:
    """Backend returning a pre-baked JSON response."""

    id = "scripted-json"

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.last_request: dict[str, Any] | None = None

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_request = payload
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return {
            "model": "test-model-1",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 1,
            "usage": {"input_tokens": 123, "output_tokens": 456, "thinking_tokens": 0},
        }


# ---- fixtures -------------------------------------------------------------


def _report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pair_count": 3,
        "rows": [
            {
                "axis": "trajectory",
                "severity": "severe",
                "baseline_median": 0.0,
                "candidate_median": 1.0,
                "delta": 1.0,
            },
            {
                "axis": "latency",
                "severity": "moderate",
                "baseline_median": 100.0,
                "candidate_median": 400.0,
                "delta": 300.0,
            },
        ],
        "first_divergence": {"turn_index": 0, "kind": "structural_drift"},
        "recommendations": [
            {
                "id": "rec-traj-0",
                "title": "Review tool-schema change at turn 0",
                "severity": "error",
                "axis": "trajectory",
                "action": "Diff the tool schemas; the call shape changed.",
                "rationale": "trajectory severity=severe + structural_drift at turn 0",
                "turn_index": 0,
            },
            {
                "id": "rec-lat-0",
                "title": "Review latency regression",
                "severity": "warning",
                "axis": "latency",
                "action": "Check the new model's p95.",
                "rationale": "latency severity=moderate, +300ms",
                "turn_index": 1,
            },
        ],
    }
    base.update(overrides)
    return base


def _records(text: str = "ok", turns: int = 3) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = [{"kind": "metadata", "id": "meta", "parent": None, "payload": {}}]
    for i in range(turns):
        recs.append(
            {
                "kind": "chat_request",
                "id": f"req{i}",
                "parent": recs[-1]["id"],
                "payload": {
                    "model": "m",
                    "messages": [{"role": "user", "content": f"q{i}"}],
                    "params": {},
                },
            }
        )
        recs.append(
            {
                "kind": "chat_response",
                "id": f"resp{i}",
                "parent": f"req{i}",
                "payload": {
                    "model": "m",
                    "content": [{"type": "text", "text": f"{text}-{i}"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 10,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "thinking_tokens": 0,
                    },
                },
            }
        )
    return recs


def _valid_llm_response() -> dict[str, Any]:
    return {
        "suggestions": [
            {
                "anchor": "rec-traj-0",
                "proposal": "Rename `customer_id` back to `cid` in search_files tool schema",
                "snippet": (
                    "tools:\n  - name: search_files\n"
                    "    input_schema:\n      properties:\n"
                    "        cid: {type: string}"
                ),
                "confidence": 0.82,
                "rationale": "evidence.turns[0].candidate_request.tools shows the rename",
            },
            {
                "anchor": "rec-lat-0",
                "proposal": "Revert model swap in config.yaml to claude-haiku-4-5",
                "snippet": None,
                "confidence": 0.65,
                "rationale": "top_axes.latency delta +300ms after model switch",
            },
        ]
    }


# ---- happy path -----------------------------------------------------------


def test_suggest_fixes_binds_anchors_and_sorts() -> None:
    backend = _ScriptedJsonBackend(_valid_llm_response())
    result = suggest_fixes(_report(), _records(turns=3), _records(text="changed", turns=3), backend)
    assert len(result.suggestions) == 2
    # error severity (trajectory) should come first.
    assert result.suggestions[0].anchor == "rec-traj-0"
    assert result.suggestions[0].severity == "error"
    assert result.suggestions[0].axis == "trajectory"
    # tokens propagate.
    assert result.prompt_tokens == 123
    assert result.completion_tokens == 456
    # anchors_considered matches input.
    assert result.anchors_considered == 2


def test_suggest_fixes_preserves_snippet() -> None:
    backend = _ScriptedJsonBackend(_valid_llm_response())
    result = suggest_fixes(_report(), _records(), _records(), backend)
    traj = result.suggestions[0]
    assert traj.snippet and "search_files" in traj.snippet


def test_suggest_fixes_rejects_ungrounded_anchor() -> None:
    # LLM invented a recommendation id that wasn't in the report.
    bad = {
        "suggestions": [
            {
                "anchor": "hallucinated",
                "proposal": "rewrite the whole agent",
                "snippet": None,
                "confidence": 0.99,
                "rationale": "",
            }
        ]
    }
    backend = _ScriptedJsonBackend(bad)
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert result.suggestions == []


def test_suggest_fixes_tolerates_markdown_fence() -> None:
    # Some models wrap JSON in ```json ``` despite system-prompt asks.
    fenced = "```json\n" + json.dumps(_valid_llm_response()) + "\n```"
    backend = _ScriptedJsonBackend(fenced)
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert len(result.suggestions) == 2


def test_suggest_fixes_tolerates_trailing_chatter() -> None:
    tail = json.dumps(_valid_llm_response()) + "\n\nLet me know if you'd like more."
    backend = _ScriptedJsonBackend(tail)
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert len(result.suggestions) == 2


def test_suggest_fixes_handles_bad_json() -> None:
    backend = _ScriptedJsonBackend("I think you should just start over.")
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert result.suggestions == []
    # anchors still considered — we still called the LLM.
    assert result.anchors_considered == 2


def test_suggest_fixes_no_recommendations_short_circuits() -> None:
    report = _report()
    report["recommendations"] = []
    backend = _ScriptedJsonBackend(_valid_llm_response())
    result = suggest_fixes(report, _records(), _records(), backend)
    assert result.suggestions == []
    # Backend must NOT have been called.
    assert backend.last_request is None


def test_suggest_fixes_clamps_confidence() -> None:
    # Model reports confidence out of range.
    bad_conf = {
        "suggestions": [
            {
                "anchor": "rec-traj-0",
                "proposal": "do something",
                "snippet": None,
                "confidence": 1.5,
                "rationale": "",
            },
            {
                "anchor": "rec-lat-0",
                "proposal": "do something else",
                "snippet": None,
                "confidence": -0.2,
                "rationale": "",
            },
        ]
    }
    backend = _ScriptedJsonBackend(bad_conf)
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert all(0.0 <= s.confidence <= 1.0 for s in result.suggestions)


def test_suggest_fixes_drops_empty_proposals() -> None:
    thin = {
        "suggestions": [
            {
                "anchor": "rec-traj-0",
                "proposal": "",
                "snippet": None,
                "confidence": 0.8,
                "rationale": "",
            }
        ]
    }
    backend = _ScriptedJsonBackend(thin)
    result = suggest_fixes(_report(), _records(), _records(), backend)
    assert result.suggestions == []


def test_large_payloads_get_truncated() -> None:
    big_text = "x" * (MAX_EVIDENCE_CHARS * 2)
    big_records = _records(text=big_text)
    backend = _ScriptedJsonBackend(_valid_llm_response())
    result = suggest_fixes(_report(), big_records, big_records, backend)
    assert result.suggestions  # still works
    # Prompt text must contain the truncation marker.
    prompt_text = backend.last_request["messages"][1]["content"]
    assert "_truncated" in prompt_text


# ---- rendering ------------------------------------------------------------


def test_render_terminal_marks_speculative_below_threshold() -> None:
    result = suggest_fixes(
        _report(),
        _records(),
        _records(),
        _ScriptedJsonBackend(
            {
                "suggestions": [
                    {
                        "anchor": "rec-traj-0",
                        "proposal": "do something",
                        "snippet": None,
                        "confidence": 0.1,
                        "rationale": "",
                    }
                ]
            }
        ),
    )
    rendered = render_terminal(result)
    assert "speculative" in rendered
    assert FLAG_CONFIDENCE == 0.3  # sanity — constant matches test expectation


def test_render_terminal_reports_empty_when_no_suggestions() -> None:
    report = _report()
    report["recommendations"] = []
    backend = _ScriptedJsonBackend("{}")
    result = suggest_fixes(report, _records(), _records(), backend)
    assert "no concrete suggestions" in render_terminal(result)


def test_to_dict_shape() -> None:
    fix = SuggestedFix(
        anchor="a",
        severity="error",
        axis="trajectory",
        proposal="p",
        snippet=None,
        confidence=0.5,
        rationale="r",
    )
    assert fix.to_dict() == {
        "anchor": "a",
        "severity": "error",
        "axis": "trajectory",
        "proposal": "p",
        "snippet": None,
        "confidence": 0.5,
        "rationale": "r",
    }
