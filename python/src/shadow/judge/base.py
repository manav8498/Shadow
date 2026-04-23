"""Judge Protocol + shared verdict shape."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class JudgeVerdict(TypedDict):
    """Structured output of a single judge call."""

    verdict: str  # "better" | "equal" | "worse" | "error"
    confidence: float  # in [0.0, 1.0]
    reason: str
    score: float  # in [0.0, 1.0]; 1.0 = candidate ≥ baseline


class Judge(Protocol):
    """Score a single (baseline_response, candidate_response) pair."""

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict: ...


__all__ = ["Judge", "JudgeVerdict"]
