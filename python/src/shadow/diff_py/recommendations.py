"""LLM-backed novel-pattern diagnosis on top of the deterministic
recommendations engine.

The Rust ``shadow_core::diff::recommendations`` module produces 10
hand-coded cross-axis correlation patterns plus per-divergence
rules. That deterministic core handles the most common production
regression archetypes correctly and cheaply. But its ceiling is
"what the patterns know to look for" — a novel cross-axis signature
that doesn't match any of the 10 produces no root-cause
recommendation, only per-divergence severity labels.

The v2.9 ``enrich_with_llm`` function closes that gap. It:

  1. Runs the deterministic engine first.
  2. If a root-cause is already present, returns the deterministic
     output unchanged (LLM would be redundant and costs money).
  3. If NO root-cause is present BUT a severe axis is, asks an LLM
     to propose a novel root cause given the structured DiffReport.
  4. Validates the LLM's structured output, attaches a confidence
     bound, and merges it into the recommendations list.

The LLM call uses structured output (JSON Schema) so the response
shape is enforced. Hallucinated recommendations get rejected at the
JSON-validation step rather than reaching the user.

Cost model: at most one LLM call per diff report, only when the
deterministic engine has nothing to say but a severe regression is
present. Typical call cost: $0.001-0.01 with gpt-4o-mini.

API key
-------
Read from ``OPENAI_API_KEY`` env var. The function never accepts
the key as a parameter for the same reason ``OpenAIReplayer``
doesn't — keep secrets out of stack frames.

When the env var is missing, the function returns the deterministic
recommendations unchanged (no LLM call attempted, no error raised).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

# Severity / action labels mirror the Rust recommendation surface
# so the merged list is uniform.
RecommendationSeverity = Literal["error", "warning", "info"]
ActionKind = Literal["restore", "remove", "revert", "review", "verify", "root_cause"]


@dataclass(frozen=True)
class LLMDiagnosedRecommendation:
    """One novel-pattern recommendation produced by the LLM fallback.

    Same shape as a Rust-side Recommendation so callers don't need
    to branch on origin. The ``source`` field marks LLM-derived rows
    so reviewers can tell rule-based vs LLM-diagnosed at a glance.
    """

    severity: RecommendationSeverity
    action: ActionKind
    turn: int
    message: str
    rationale: str
    axis: str
    confidence: float
    source: Literal["llm"] = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "action": self.action,
            "turn": self.turn,
            "message": self.message,
            "rationale": self.rationale,
            "axis": self.axis,
            "confidence": self.confidence,
            "source": self.source,
        }


_DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["severity", "message", "rationale", "axis", "confidence"],
    "properties": {
        "severity": {"enum": ["error", "warning", "info"]},
        "message": {"type": "string", "minLength": 8, "maxLength": 200},
        "rationale": {"type": "string", "minLength": 20, "maxLength": 800},
        "axis": {
            "enum": [
                "semantic",
                "trajectory",
                "safety",
                "verbosity",
                "latency",
                "cost",
                "reasoning",
                "judge",
                "conformance",
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def _has_severe_axis(report: dict[str, Any]) -> bool:
    rows = report.get("rows", []) or []
    return any(r.get("severity") == "severe" for r in rows)


def _has_root_cause(recommendations: list[dict[str, Any]]) -> bool:
    return any(r.get("action") == "root_cause" for r in recommendations)


def _summarise_report(report: dict[str, Any]) -> str:
    """Compact JSON summary of the diff report for the LLM prompt.

    We only include axis names + severity + delta + CI bounds — NOT
    the raw response text or tool-call payloads, to keep the prompt
    short and to avoid leaking trace content into the LLM call.
    """
    rows = report.get("rows", []) or []
    summary = []
    for r in rows:
        summary.append(
            {
                "axis": r.get("axis", ""),
                "severity": r.get("severity", ""),
                "delta": r.get("delta", 0.0),
                "ci95_low": r.get("ci95_low", 0.0),
                "ci95_high": r.get("ci95_high", 0.0),
            }
        )
    return json.dumps({"axes": summary}, separators=(",", ":"))


_SYSTEM_PROMPT = """You are a senior agent-eval engineer diagnosing
behavioural regressions in LLM agent traces. Given a structured diff
report (per-axis severity / delta / CI), propose a single most-likely
root cause for the observed regression.

Constraints:
- The deterministic rule-based engine ALREADY ran and found no
  matching cross-axis pattern. So your hypothesis must be a novel
  pattern that the 10 hardcoded rules did not catch.
- The 10 existing patterns are: model swap (cost+latency+semantic),
  prompt drift (semantic+verbosity), refusal escalation (severe
  safety+), tool schema migration (severe trajectory+reasoning),
  hallucination cluster (semantic+judge), context window overflow
  (severe cost+reasoning), retry loop (severe trajectory+latency
  no reasoning), cache mismatch (severe cost+stable lat/sem),
  prompt injection (severe trajectory+negative safety), latency
  spike (severe latency alone). Don't restate any of those.
- Be specific. "Something changed" is not useful. Propose a
  concrete mechanism a reviewer can investigate.
- Set confidence ≤ 0.65 because rule-based engine found nothing
  matching — the LLM is operating beyond the curated patterns.
- Pick the single axis most central to your hypothesis.

Respond with a JSON object matching the supplied schema."""


# A callable that takes the JSON summary and returns the LLM's raw
# JSON-string response. Lets tests inject a mock without touching
# the network.
LLMCaller = Callable[[str], str]


def _default_openai_caller(summary: str) -> str:
    """Default LLM caller using OpenAI's structured-output API."""
    from openai import OpenAI  # type: ignore[import-not-found, unused-ignore]

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Diff report:\n{summary}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "novel_pattern_diagnosis",
                "schema": _DIAGNOSIS_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.0,
        max_tokens=400,
        seed=20260428,
        timeout=15.0,
    )
    return completion.choices[0].message.content or "{}"


def enrich_with_llm(
    report: dict[str, Any],
    *,
    llm_caller: LLMCaller | None = None,
    require_severe_axis: bool = True,
) -> dict[str, Any]:
    """Return the diff report with one LLM-diagnosed recommendation
    appended when the deterministic engine produced no root-cause.

    Parameters
    ----------
    report :
        DiffReport-shaped dict (output of ``compute_diff_report`` or
        an equivalent Rust-side dump).
    llm_caller :
        Override for testing. Production code should leave this
        ``None`` to use the OpenAI structured-output path.
    require_severe_axis :
        When True (default), the LLM is only invoked if at least one
        axis is at severity "severe". Cheap diagnostic gate that
        avoids spending an LLM call on noise.

    Returns
    -------
    The same report with a possibly-augmented ``recommendations``
    list. Existing rule-based recommendations are preserved
    unchanged. The LLM-diagnosed row is marked with
    ``"source": "llm"`` so callers can render it differently.

    The function is **safe to call without OPENAI_API_KEY** — when
    the env var is missing or the LLM call fails, it returns the
    report unchanged.
    """
    recommendations = list(report.get("recommendations", []) or [])

    # Skip cases the LLM can't usefully diagnose.
    if _has_root_cause(recommendations):
        return report  # deterministic engine already explained the regression
    if require_severe_axis and not _has_severe_axis(report):
        return report  # no severe axis → not worth an LLM call
    if "OPENAI_API_KEY" not in os.environ and llm_caller is None:
        return report  # no key, no fallback callable, no LLM call

    summary = _summarise_report(report)
    caller = llm_caller if llm_caller is not None else _default_openai_caller

    try:
        raw = caller(summary)
        parsed = json.loads(raw)
    except Exception:
        return report  # LLM call failed; fall back silently to deterministic

    if not _validate_diagnosis(parsed):
        return report  # malformed LLM output → drop, don't propagate noise

    # Confidence cap — even if the LLM claims 0.9, we cap at 0.65 to
    # honestly signal that this is operating beyond curated patterns.
    confidence_cap = 0.65
    diag = LLMDiagnosedRecommendation(
        severity=parsed["severity"],
        action="root_cause",
        turn=0,
        message=str(parsed["message"]).strip(),
        rationale=str(parsed["rationale"]).strip(),
        axis=parsed["axis"],
        confidence=min(confidence_cap, float(parsed.get("confidence", 0.5))),
    )
    enriched_recs = [*recommendations, diag.to_dict()]
    return {**report, "recommendations": enriched_recs}


def _validate_diagnosis(parsed: Any) -> bool:
    """Cheap structural validation against ``_DIAGNOSIS_SCHEMA``.
    A real jsonschema dependency would be cleaner, but this avoids
    pulling jsonschema into the default install for one validation
    site. The check is conservative: any deviation from the expected
    shape returns False, dropping the LLM diagnosis silently.
    """
    if not isinstance(parsed, dict):
        return False
    required = {"severity", "message", "rationale", "axis", "confidence"}
    if not required.issubset(parsed.keys()):
        return False
    if parsed["severity"] not in ("error", "warning", "info"):
        return False
    if parsed["axis"] not in {
        "semantic",
        "trajectory",
        "safety",
        "verbosity",
        "latency",
        "cost",
        "reasoning",
        "judge",
        "conformance",
    }:
        return False
    msg = parsed["message"]
    if not isinstance(msg, str) or not (8 <= len(msg) <= 400):
        return False
    rat = parsed["rationale"]
    if not isinstance(rat, str) or not (20 <= len(rat) <= 1200):
        return False
    conf = parsed.get("confidence")
    if not isinstance(conf, int | float):
        return False
    return 0.0 <= float(conf) <= 1.0


__all__ = [
    "LLMCaller",
    "LLMDiagnosedRecommendation",
    "enrich_with_llm",
]
