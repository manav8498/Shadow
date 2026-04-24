"""FactualityJudge — does the candidate contradict known facts?

Regression pattern: a prompt change or model swap makes the agent
hallucinate about things it had right before. This judge takes a set
of known-true statements and asks the judge LLM "does the candidate
response contradict any of these?"

Intentionally asymmetric: the score penalises contradictions but
doesn't reward verbosity. Silence (not mentioning a fact) is scored
neutral; assertion-that-contradicts is scored worst.

Usage:

    judge = FactualityJudge(
        backend=AnthropicLLM(),
        known_facts=[
            "Acme Widgets offers refunds within 30 days of purchase.",
            "Acme does not ship to Antarctica.",
            "The minimum order amount is $25 USD.",
        ],
    )

Notes on rigour: this is a regression-eval helper, not a factuality
benchmark. The judge LLM itself can be wrong; PairwiseJudge's
position-bias mitigation does not apply here because the comparison
is fact-set vs single-response. For high-stakes factuality
evaluation, pair this with an external knowledge-grounded checker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.judge.llm import LlmJudge
from shadow.llm.base import LlmBackend

_RUBRIC_TEMPLATE = """You are checking whether the CANDIDATE response contradicts any
of the KNOWN FACTS below.

"Contradicts" means the candidate explicitly asserts something that
directly conflicts with a known fact. Silence on a topic is NOT a
contradiction.

KNOWN FACTS:
__FACTS__

TASK:
{task}

CANDIDATE RESPONSE:
{candidate}

Reply with ONLY this JSON object (no preamble, no code fence):
  {{"verdict": "consistent" or "contradicts",
    "confidence": 0-1,
    "reason": "ONE short sentence; if 'contradicts', quote the offending claim"}}
"""


class FactualityJudge:
    """Flag candidates whose claims contradict a known-fact set."""

    def __init__(
        self,
        backend: LlmBackend,
        known_facts: Sequence[str],
        model: str | None = None,
    ) -> None:
        if not known_facts:
            raise ValueError("known_facts must contain at least one fact")
        facts_block = "\n".join(f"  - {f}" for f in known_facts)
        rubric = _RUBRIC_TEMPLATE.replace("__FACTS__", facts_block)
        self._judge = LlmJudge(
            backend=backend,
            rubric=rubric,
            score_map={"consistent": 1.0, "contradicts": 0.0},
            model=model,
        )

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        return await self._judge.score_pair(baseline_response, candidate_response, request_context)


__all__ = ["FactualityJudge"]
