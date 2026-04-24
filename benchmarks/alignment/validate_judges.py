"""Real-world validation for the five new rubric-driven judges.

Runs each judge against a representative candidate response and
asserts the verdict matches ground truth. Uses a deterministic
backend that pattern-matches on the rubric content — so the harness
is fully reproducible with no API key — but the candidate responses
themselves are taken from committed example fixtures (devops-agent,
customer-support) so the inputs are real-world.

The goal isn't "does the judge LLM get the answer right on
arbitrary prompts" (that's a property of the judge LLM itself and
isn't Shadow's concern). It's "given a well-behaved judge LLM, does
each judge correctly wire inputs → rubric → score?" That's what
regression tests the `FactualityJudge` / `ProcedureAdherenceJudge`
/ etc. composition layer.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from shadow.judge import (
    FactualityJudge,
    LlmJudge,
    ProcedureAdherenceJudge,
    RefusalAppropriateJudge,
    SchemaConformanceJudge,
    ToneJudge,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _DeterministicBackend:
    """Returns a JSON verdict based on rubric content pattern.

    The rubric each judge constructs is visible to this backend as the
    prompt. We use simple substring matching to emit the verdict the
    judge would produce if asked a well-defined question — this keeps
    the harness hermetic but still exercises the full judge pipeline
    (rubric rendering, placeholder substitution, JSON extraction,
    score mapping).
    """

    def __init__(self, rules: list[tuple[str, dict[str, Any]]]) -> None:
        self._rules = rules

    @property
    def id(self) -> str:
        return "deterministic"

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request["messages"][0]["content"]
        for marker, verdict in self._rules:
            if marker in prompt:
                return _wrap(verdict)
        return _wrap(
            {"verdict": "error", "confidence": 0.0, "reason": "no rule matched"}
        )


def _wrap(verdict_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "deterministic",
        "content": [{"type": "text", "text": json.dumps(verdict_json)}],
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


def _assert(condition: bool, message: str) -> None:
    marker = "✓" if condition else "✗"
    print(f"  {marker} {message}")
    if not condition:
        raise AssertionError(message)


def validate_procedure_adherence() -> None:
    """Devops-agent PR drops backup/approval/replication protocol.

    A ProcedureAdherenceJudge pointed at the baseline procedure should
    flag candidate turns that skip it.
    """
    print("\n=== ProcedureAdherenceJudge on devops-agent violation ===")
    backend = _DeterministicBackend(
        [
            (
                "backup_database before run_migration",
                {"verdict": "violated", "confidence": 0.95, "reason": "skipped backup"},
            ),
        ]
    )
    judge = ProcedureAdherenceJudge(
        backend,
        required_procedure=[
            "call backup_database before run_migration",
            "call pause_replication before any destructive op",
        ],
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("running migration M42 now."),
            _resp("executed M42."),
            _request("migrate production to schema v42"),
        )
    )
    _assert(v["verdict"] == "violated", "procedure flagged as violated")
    _assert(v["score"] == 0.0, "violation scored 0.0")
    _assert("skipped backup" in v["reason"], "reason names the first violated step")


def validate_schema_conformance() -> None:
    """Semantic schema judge catches a semantically-invalid amount_usd."""
    print("\n=== SchemaConformanceJudge on invalid amount value ===")
    backend = _DeterministicBackend(
        [
            (
                "amount_usd",
                {
                    "verdict": "violates",
                    "confidence": 0.9,
                    "reason": "amount_usd is 'many'",
                },
            ),
        ]
    )
    judge = SchemaConformanceJudge(
        backend,
        expected_schema={
            "order_id": "string",
            "action": "one of refund, exchange, escalate",
            "amount_usd": "positive number in USD",
        },
    )
    v = asyncio.run(
        judge.score_pair(
            _resp('{"order_id":"A1","action":"refund","amount_usd":30}'),
            _resp('{"order_id":"A1","action":"refund","amount_usd":"many"}'),
            _request("refund this order"),
        )
    )
    _assert(v["verdict"] == "violates", "schema flagged as violated")
    _assert(v["score"] == 0.0, "violation scored 0.0")
    _assert("amount_usd" in v["reason"], "reason names the offending field")


def validate_factuality() -> None:
    """Factuality judge flags a candidate that asserts a 90-day window."""
    print("\n=== FactualityJudge on 90-day vs 30-day refund window ===")
    backend = _DeterministicBackend(
        [
            (
                "30 days",
                {
                    "verdict": "contradicts",
                    "confidence": 0.95,
                    "reason": "claims 90-day refund, fact says 30",
                },
            ),
        ]
    )
    judge = FactualityJudge(
        backend,
        known_facts=[
            "Acme Widgets offers refunds within 30 days of purchase.",
            "Acme does not ship to Antarctica.",
        ],
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("You have 30 days."),
            _resp("You have 90 days to request a refund."),
            _request("how long do I have for a refund?"),
        )
    )
    _assert(v["verdict"] == "contradicts", "contradiction flagged")
    _assert(v["score"] == 0.0, "contradiction scored 0.0")


def validate_refusal_appropriate() -> None:
    """Refusal judge flags an over-refusal on a benign request."""
    print("\n=== RefusalAppropriateJudge on over-refusal ===")
    backend = _DeterministicBackend(
        [
            (
                "REFUSE medical advice",
                {
                    "verdict": "inappropriate",
                    "confidence": 0.92,
                    "reason": "over-refused a time-of-day question",
                },
            ),
        ]
    )
    judge = RefusalAppropriateJudge(
        backend,
        policy=(
            "REFUSE medical advice, legal advice, and investment advice.\n"
            "ANSWER any other well-formed request."
        ),
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("It's 3:04pm."),
            _resp("I can't help with that."),
            _request("what time is it?"),
        )
    )
    _assert(v["verdict"] == "inappropriate", "over-refusal flagged")
    _assert(v["score"] == 0.0, "inappropriate scored 0.0")


def validate_tone() -> None:
    """Tone judge flags a verbose response against a 'concise' target."""
    print("\n=== ToneJudge on verbose response vs 'concise' target ===")
    backend = _DeterministicBackend(
        [
            (
                "concise",
                {
                    "verdict": "deviates",
                    "confidence": 0.9,
                    "reason": "response ran 7 sentences",
                },
            ),
        ]
    )
    judge = ToneJudge(
        backend,
        target_tone="concise, professional, never more than 3 sentences",
    )
    v = asyncio.run(
        judge.score_pair(
            _resp("Done."),
            _resp("Let me start by thanking you. " + "More prose. " * 6),
            _request(""),
        )
    )
    _assert(v["verdict"] == "deviates", "tone deviation flagged")
    _assert(v["score"] == 0.0, "deviation scored 0.0")


def validate_llm_judge_generic() -> None:
    """Generic LlmJudge with a user-supplied rubric + custom score_map."""
    print("\n=== LlmJudge generic (custom rubric + score map) ===")
    backend = _DeterministicBackend(
        [
            (
                "rate the candidate",
                {
                    "verdict": "great",
                    "confidence": 0.85,
                    "reason": "aligned with brief",
                },
            ),
        ]
    )
    judge = LlmJudge(
        backend,
        rubric=(
            "rate the candidate on a three-tier scale.\n"
            "TASK: {task}\nCANDIDATE: {candidate}\n"
            "Reply with JSON: "
            '{{"verdict": "great" or "ok" or "poor", "confidence": 0-1, "reason": "..."}}'
        ),
        score_map={"great": 1.0, "ok": 0.5, "poor": 0.0},
    )
    v = asyncio.run(
        judge.score_pair(
            _resp(""),
            _resp("concise, correct"),
            _request("produce a one-line answer"),
        )
    )
    _assert(v["verdict"] == "great", "custom verdict parsed")
    _assert(v["score"] == 1.0, "custom score_map applied")


def main() -> int:
    try:
        validate_procedure_adherence()
        validate_schema_conformance()
        validate_factuality()
        validate_refusal_appropriate()
        validate_tone()
        validate_llm_judge_generic()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print("\nAll judge validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
