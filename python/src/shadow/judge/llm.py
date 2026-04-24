"""LlmJudge — configurable LLM-as-judge adapter.

The generic building block for user-supplied rubrics. Callers supply
a rubric string with placeholders and get a `Judge`-compatible
instance back — no need to re-implement the backend call, JSON
parsing, or error fallbacks.

Design goals:

1. **Predictable placeholders.** The rubric may reference any of
   `{task}`, `{baseline}`, `{candidate}` — nothing else — so rubrics
   are portable across teams.
2. **Explicit scoring contract.** The model is told exactly what JSON
   shape to emit. The verdict field is constrained to strings that
   map to scores via `score_map`. Out-of-set verdicts fall into the
   `error` bucket with a neutral 0.5 score, never a panic.
3. **Low variance.** Defaults: `temperature=0.0`, `max_tokens=512`.
   The rubric author can override per-call parameters but these
   defaults match how SanityJudge behaves and match published
   best practice for LLM-as-judge setups (Zheng et al. 2024,
   "Judging LLM-as-a-Judge").
4. **No hidden network calls.** All I/O goes through the injected
   `LlmBackend`. A `MockLLM` + canned responses reproduces every
   judgment deterministically for tests.

Example:

    rubric = '''Rate whether the candidate response correctly followed
    the safety procedure for the given task. Reply with JSON:
      {"verdict": "pass"|"fail", "confidence": 0-1, "reason": "..."}

    TASK: {task}
    CANDIDATE: {candidate}
    '''
    judge = LlmJudge(
        backend=OpenAILLM(),
        rubric=rubric,
        score_map={"pass": 1.0, "fail": 0.0},
        model="gpt-4o-mini",
    )
"""

from __future__ import annotations

from collections.abc import Mapping
from string import Formatter
from typing import Any

from shadow.judge._common import (
    clamp,
    error_verdict,
    extract_json,
    response_text,
    task_text,
)
from shadow.judge.base import JudgeVerdict
from shadow.llm.base import LlmBackend

# Placeholders the rubric is allowed to reference. Any other
# `{name}` in the rubric raises ValueError at construction time so
# failures surface early (not at every judge call).
_ALLOWED_PLACEHOLDERS = frozenset({"task", "baseline", "candidate"})


def _rubric_placeholders(rubric: str) -> set[str]:
    """Return the set of `{name}` placeholders used in `rubric`."""
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(rubric):
        if field_name:
            # Strip any attribute-access tail (`x.y` → `x`).
            root = field_name.split(".", 1)[0].split("[", 1)[0]
            names.add(root)
    return names


class LlmJudge:
    """User-configurable LLM-as-judge.

    Parameters
    ----------
    backend:
        `LlmBackend` the judge will call (e.g. `AnthropicLLM()`,
        `OpenAILLM()`, or a `MockLLM` for tests).
    rubric:
        Rubric prompt. May reference `{task}`, `{baseline}`, and
        `{candidate}` placeholders; must instruct the model to emit a
        JSON object with at least a `verdict` field (optionally
        `confidence` and `reason`).
    score_map:
        Maps the verdict string the model emits to a score in
        `[0, 1]`. Keys are lower-cased before lookup. A verdict not
        in the map is treated as an error (score 0.5).
    model:
        Optional model name passed to the backend.
    temperature, max_tokens:
        Per-call completion parameters. Defaults chosen for
        reproducibility (temp 0, max 512 tokens).
    """

    def __init__(
        self,
        backend: LlmBackend,
        rubric: str,
        score_map: Mapping[str, float] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        placeholders = _rubric_placeholders(rubric)
        unknown = placeholders - _ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"rubric uses unknown placeholder(s): {sorted(unknown)}. "
                f"Allowed: {sorted(_ALLOWED_PLACEHOLDERS)}."
            )
        self._backend = backend
        self._rubric = rubric
        self._score_map: dict[str, float] = {
            k.lower(): float(v) for k, v in (score_map or {"pass": 1.0, "fail": 0.0}).items()
        }
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        prompt = self._render(
            task=task_text(request_context),
            baseline=response_text(baseline_response),
            candidate=response_text(candidate_response),
        )
        request: dict[str, Any] = {
            "model": self._model or "",
            "messages": [{"role": "user", "content": prompt}],
            "params": {
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
        }
        try:
            response = await self._backend.complete(request)
        except Exception as e:
            return error_verdict(f"judge call failed: {e}")
        return self._parse(response_text(response))

    # ---- internals ---------------------------------------------------------

    def _render(self, *, task: str, baseline: str, candidate: str) -> str:
        return self._rubric.format(task=task, baseline=baseline, candidate=candidate)

    def _parse(self, text: str) -> JudgeVerdict:
        data = extract_json(text)
        if data is None:
            return error_verdict("judge output contained no JSON object")
        verdict_raw = str(data.get("verdict", "")).lower().strip()
        if verdict_raw not in self._score_map:
            return error_verdict(
                f"verdict {verdict_raw!r} not in score_map " f"{sorted(self._score_map)}"
            )
        return {
            "verdict": verdict_raw,
            "confidence": clamp(data.get("confidence", 0.5)),
            "reason": str(data.get("reason", ""))[:240],
            "score": clamp(self._score_map[verdict_raw]),
        }


__all__ = ["LlmJudge"]
