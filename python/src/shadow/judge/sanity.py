"""SanityJudge — the default domain-agnostic judge.

Asks the backing LLM "is the candidate response at least as good as the
baseline response for this task?" and maps the answer to a score in
`[0, 1]`. This is deliberately coarse: it's a regression floor, not a
rubric. Teams should prefer a domain-specific Judge for production use.

Prompt is short, JSON-structured, temperature 0. If the judge returns
non-parseable output, the verdict is `error` and the score is 0.5 (a
neutral value so a broken judge doesn't drag the axis toward either
extreme).
"""

from __future__ import annotations

import json
import re
from typing import Any

from shadow.judge.base import JudgeVerdict
from shadow.llm.base import LlmBackend

_SCORE_BY_VERDICT = {"better": 1.0, "equal": 1.0, "worse": 0.0, "error": 0.5}

_JUDGE_PROMPT = """You are reviewing whether a CANDIDATE response is at least as good as a
BASELINE response for the same task.

{task_block}BASELINE RESPONSE:
{baseline}

CANDIDATE RESPONSE:
{candidate}

Return ONLY a JSON object with these fields (no preamble, no code fence):
  verdict: "better" | "equal" | "worse"
  confidence: number in [0, 1]
  reason: ONE short sentence

"better" and "equal" both mean the candidate is acceptable. "worse" means it regressed."""


class SanityJudge:
    """Coarse regression-floor judge that populates Shadow's axis 8.

    Parameters
    ----------
    backend:
        Any `LlmBackend` (MockLLM, AnthropicLLM, OpenAILLM). The judge
        will hit this backend once per response pair.
    model:
        Optional model name to use for the judge call. If `None`, the
        backend's default is used.
    """

    def __init__(self, backend: LlmBackend, model: str | None = None) -> None:
        self._backend = backend
        self._model = model

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        prompt = _JUDGE_PROMPT.format(
            task_block=_format_task_block(request_context),
            baseline=_response_text(baseline_response),
            candidate=_response_text(candidate_response),
        )
        request: dict[str, Any] = {
            "model": self._model or "",
            "messages": [{"role": "user", "content": prompt}],
            "params": {"temperature": 0.0, "max_tokens": 256},
        }
        try:
            response = await self._backend.complete(request)
        except Exception as e:
            return {
                "verdict": "error",
                "confidence": 0.0,
                "reason": f"judge call failed: {e}",
                "score": 0.5,
            }
        return _parse_verdict(_response_text(response))


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("content") or []
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            t = part.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts).strip()


def _format_task_block(request_context: dict[str, Any] | None) -> str:
    if not request_context:
        return ""
    messages = request_context.get("messages") or []
    user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return ""
    last = user_msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        task = content
    elif isinstance(content, list):
        task = "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    else:
        return ""
    task = task.strip()
    if not task:
        return ""
    return f"TASK:\n{task}\n\n"


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


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
    verdict_raw = str(data.get("verdict", "error")).lower().strip()
    verdict = verdict_raw if verdict_raw in _SCORE_BY_VERDICT else "error"
    confidence = data.get("confidence", 0.5)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.5
    reason = str(data.get("reason", ""))[:240]
    return {
        "verdict": verdict,
        "confidence": confidence_f,
        "reason": reason,
        "score": _SCORE_BY_VERDICT[verdict],
    }


__all__ = ["SanityJudge"]
