"""Shared helpers for Judge implementations.

Centralises the parsing/formatting code that was previously duplicated
across `SanityJudge`, `PairwiseJudge`, `CorrectnessJudge`, and
`FormatJudge`. Every judge builds its prompt, calls the backend, and
parses a JSON verdict — the bits that differ are the rubric text and
the mapping from LLM output to a 0-1 score.

Any breaking change to judge-response parsing should be made here, not
by re-implementing the same regex in each subclass.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shadow.judge.base import JudgeVerdict

# Strip GFM code fences the model sometimes wraps its JSON in.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def response_text(response: dict[str, Any]) -> str:
    """Extract the human-readable text from a `chat_response` payload.

    Handles both Anthropic-style content arrays and OpenAI-style flat
    strings. Returns the empty string for non-text payloads.
    """
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


def task_text(request_context: dict[str, Any] | None) -> str:
    """Best-effort 'what was the user asking?' extractor."""
    if not request_context:
        return ""
    messages = request_context.get("messages") or []
    user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return ""
    last = user_msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return ""


def format_task_block(request_context: dict[str, Any] | None, header: str = "TASK") -> str:
    """Wrap `task_text` in a labelled block, or return empty string."""
    task = task_text(request_context)
    return f"{header}:\n{task}\n\n" if task else ""


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object out of a judge response.

    Tolerates:
    - leading/trailing prose
    - code-fenced blocks (```json ... ```)
    - trailing commas (best-effort)

    Returns `None` if no parseable object is found.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    # Fast path: whole response is a JSON object.
    if cleaned.startswith("{"):
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    # Fallback: find the first `{...}` span via brace matching.
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    candidate = cleaned[start:end]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def clamp(value: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    """Coerce an arbitrary value to a float in `[lo, hi]`."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(lo, min(hi, f))


def error_verdict(reason: str, score: float = 0.5) -> JudgeVerdict:
    """Build a uniform error verdict (used for parse/call failures)."""
    return {
        "verdict": "error",
        "confidence": 0.0,
        "reason": reason[:240],
        "score": score,
    }


__all__ = [
    "clamp",
    "error_verdict",
    "extract_json",
    "format_task_block",
    "response_text",
    "task_text",
]
