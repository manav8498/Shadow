"""PairwiseJudge — position-bias-resistant A/B judge.

The SanityJudge asks "is the candidate at least as good?" with a fixed
baseline→candidate ordering. LLM judges are known to have **position
bias**: they favour the first or last option presented regardless of
content.

PairwiseJudge runs each comparison **twice** with the order flipped,
and agrees only when both calls land on the same verdict. If they
disagree, the pair is scored 0.5 (uncertain). This doubles the cost
but removes position bias as a confounder — necessary for any serious
eval setup.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.llm.base import LlmBackend

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class PairwiseJudge:
    """Position-bias-resistant pairwise preference judge."""

    def __init__(
        self,
        backend: LlmBackend,
        model: str | None = None,
    ) -> None:
        self._backend = backend
        self._model = model

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        task = _task_text(request_context)
        b_text = _response_text(baseline_response)
        c_text = _response_text(candidate_response)

        # Ask twice — once candidate=A baseline=B, once flipped.
        verdict_ab = await self._ask(task, a=c_text, b=b_text)
        verdict_ba = await self._ask(task, a=b_text, b=c_text)

        # Normalize: what did the judge prefer, from candidate's POV?
        candidate_verdict_ab = verdict_ab  # raw: A wins / B wins / tie
        if verdict_ab == "a_better":
            candidate_first = "candidate_better"
        elif verdict_ab == "b_better":
            candidate_first = "baseline_better"
        else:
            candidate_first = "tie"
        if verdict_ba == "a_better":
            candidate_second = "baseline_better"
        elif verdict_ba == "b_better":
            candidate_second = "candidate_better"
        else:
            candidate_second = "tie"

        # Agreement → confident. Disagreement → uncertain.
        if candidate_first == candidate_second:
            v = candidate_first
            score = {"candidate_better": 1.0, "tie": 0.5, "baseline_better": 0.0}[v]
            return {
                "verdict": v,
                "confidence": 0.9,
                "reason": f"both orderings agreed ({candidate_verdict_ab})",
                "score": score,
            }
        return {
            "verdict": "uncertain",
            "confidence": 0.3,
            "reason": f"position bias: A/B={candidate_first} B/A={candidate_second}",
            "score": 0.5,
        }

    async def _ask(self, task: str, *, a: str, b: str) -> str:
        prompt = _PROMPT.format(task=task or "(not provided)", a=a, b=b)
        try:
            resp = await self._backend.complete(
                {
                    "model": self._model or "",
                    "messages": [{"role": "user", "content": prompt}],
                    "params": {"temperature": 0.0, "max_tokens": 100},
                }
            )
        except Exception:
            return "tie"
        return _parse_ab_verdict(_response_text(resp))


_PROMPT = """You are an impartial judge. Given a task and two candidate
responses, decide which is better, or if they are tied.

TASK:
{task}

RESPONSE A:
{a}

RESPONSE B:
{b}

Return ONLY a JSON object (no preamble, no code fence):
  verdict: "a_better" | "b_better" | "tie"
  reason: ONE short sentence"""


def _parse_ab_verdict(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return "tie"
    raw = str(data.get("verdict", "tie")).lower().strip()
    if raw in ("a_better", "b_better", "tie"):
        return raw
    return "tie"


def _task_text(request_context: dict[str, Any] | None) -> str:
    if not request_context:
        return ""
    for m in reversed(request_context.get("messages", [])):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
    return ""


def _response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content") or []
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts).strip()


__all__ = ["PairwiseJudge"]
