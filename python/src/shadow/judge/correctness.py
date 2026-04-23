"""CorrectnessJudge — matches candidate against a reference answer.

Given a `{request → expected}` rubric, this judge asks the backing LLM
whether the candidate response matches the expected answer's meaning
(not its exact wording). Use when you know what the right answer is —
regression eval, SWE-bench-style setups, documented FAQs.

API:

    rubric = {
        "what is the capital of France?": "Paris",
        "what is 2 + 2?": "4",
        ...
    }
    judge = CorrectnessJudge(backend=OpenAILLM(), rubric=rubric, key_fn="last_user")

`key_fn` selects how to look up the rubric key from a request. Default
`"last_user"` uses the text of the last user message; pass a callable
for custom matching.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.llm.base import LlmBackend

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class CorrectnessJudge:
    """Scores candidate vs a reference-answer rubric."""

    def __init__(
        self,
        backend: LlmBackend,
        rubric: dict[str, str],
        model: str | None = None,
        key_fn: str | Callable[[dict[str, Any] | None], str] = "last_user",
    ) -> None:
        self._backend = backend
        self._rubric = rubric
        self._model = model
        self._key_fn: Callable[[dict[str, Any] | None], str]
        if callable(key_fn):
            self._key_fn = key_fn
        elif key_fn == "last_user":
            self._key_fn = _last_user_key
        else:
            raise ValueError(f"unknown key_fn: {key_fn}")

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        key = self._key_fn(request_context)
        expected = self._rubric.get(key)
        if expected is None:
            return {
                "verdict": "error",
                "confidence": 0.0,
                "reason": "no rubric entry for this request",
                "score": 0.5,
            }
        candidate_text = _response_text(candidate_response)
        prompt = _PROMPT.format(expected=expected, candidate=candidate_text)
        try:
            resp = await self._backend.complete(
                {
                    "model": self._model or "",
                    "messages": [{"role": "user", "content": prompt}],
                    "params": {"temperature": 0.0, "max_tokens": 200},
                }
            )
        except Exception as e:
            return {
                "verdict": "error",
                "confidence": 0.0,
                "reason": f"judge call failed: {e}",
                "score": 0.5,
            }
        return _parse_verdict(_response_text(resp))


_PROMPT = """Compare a CANDIDATE answer against an EXPECTED answer.
Judge ONLY whether the candidate conveys the same factual content as
the expected answer. Differences in wording, phrasing, or formatting
do NOT count as wrong — only factual mismatch does.

EXPECTED:
{expected}

CANDIDATE:
{candidate}

Return ONLY a JSON object (no preamble, no code fence):
  verdict: "match" | "partial" | "mismatch"
  confidence: number in [0, 1]
  reason: ONE short sentence"""


_VERDICT_SCORE = {"match": 1.0, "partial": 0.5, "mismatch": 0.0, "error": 0.5}


def _parse_verdict(text: str) -> JudgeVerdict:
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reason": "judge output was not JSON",
            "score": 0.5,
        }
    raw = str(data.get("verdict", "error")).lower().strip()
    verdict = raw if raw in _VERDICT_SCORE else "error"
    confidence = data.get("confidence", 0.5)
    try:
        conf_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf_f = 0.5
    return {
        "verdict": verdict,
        "confidence": conf_f,
        "reason": str(data.get("reason", ""))[:240],
        "score": _VERDICT_SCORE[verdict],
    }


def _last_user_key(request_context: dict[str, Any] | None) -> str:
    if not request_context:
        return ""
    for m in reversed(request_context.get("messages", [])):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                return "\n".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
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


__all__ = ["CorrectnessJudge"]
