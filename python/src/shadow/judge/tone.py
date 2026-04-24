"""ToneJudge — does the candidate maintain the required persona/tone?

Real regression source: a prompt rewrite changes the registered
persona ("concise customer-support agent" → "helpful assistant"), and
the agent starts producing paragraphs where it used to produce
bullet points. Accuracy dashboards miss this entirely.

Usage:

    judge = ToneJudge(
        backend=OpenAILLM(),
        target_tone="concise, professional, never more than 3 sentences",
    )

Output is binary (matches / doesn't match). For continuous tone
scoring, pair this with `SanityJudge` in an aggregate.
"""

from __future__ import annotations

from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.judge.llm import LlmJudge
from shadow.llm.base import LlmBackend

_RUBRIC_TEMPLATE = """You are judging whether the CANDIDATE response matches the
TARGET TONE below.

TARGET TONE:
__TONE__

TASK:
{task}

CANDIDATE RESPONSE:
{candidate}

Reply with ONLY this JSON object (no preamble, no code fence):
  {{"verdict": "matches" or "deviates",
    "confidence": 0-1,
    "reason": "ONE short sentence naming the specific tone deviation"}}
"""


class ToneJudge:
    """Flag responses that deviate from a required tone/persona."""

    def __init__(
        self,
        backend: LlmBackend,
        target_tone: str,
        model: str | None = None,
    ) -> None:
        if not target_tone.strip():
            raise ValueError("target_tone must be a non-empty string")
        rubric = _RUBRIC_TEMPLATE.replace("__TONE__", target_tone.strip())
        self._judge = LlmJudge(
            backend=backend,
            rubric=rubric,
            score_map={"matches": 1.0, "deviates": 0.0},
            model=model,
        )

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        return await self._judge.score_pair(baseline_response, candidate_response, request_context)


__all__ = ["ToneJudge"]
