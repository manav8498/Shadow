"""FormatJudge — schema-conformance without calling an LLM.

Given a JSON schema (or a "must parse as JSON" flag), FormatJudge
scores whether each candidate response conforms. No LLM calls, no API
cost — purely mechanical. Complements the Rust-side `conformance`
axis (which compares baseline-vs-candidate parse rates); `FormatJudge`
instead asserts against an absolute schema.

Scoring:
  - Candidate response parses as JSON AND matches schema → 1.0
  - Parses as JSON but schema validation fails → 0.3
  - Does not parse as JSON → 0.0
"""

from __future__ import annotations

import json
from typing import Any

from shadow.judge.base import JudgeVerdict


class FormatJudge:
    """Score candidate responses by JSON-schema conformance."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self._schema = schema

    async def score_pair(
        self,
        baseline_response: dict[str, Any],
        candidate_response: dict[str, Any],
        request_context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        text = _response_text(candidate_response).strip()
        text = _strip_fences(text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return {
                "verdict": "not_json",
                "confidence": 1.0,
                "reason": f"JSON parse failed: {e}",
                "score": 0.0,
            }
        if self._schema is None:
            return {
                "verdict": "json_ok",
                "confidence": 1.0,
                "reason": "parses as JSON (no schema)",
                "score": 1.0,
            }
        errors = _validate(parsed, self._schema)
        if not errors:
            return {
                "verdict": "schema_ok",
                "confidence": 1.0,
                "reason": "parses and matches schema",
                "score": 1.0,
            }
        return {
            "verdict": "schema_fail",
            "confidence": 1.0,
            "reason": "; ".join(errors)[:240],
            "score": 0.3,
        }


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            # Strip leading ```(?:json) and trailing ```
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            lines = lines[1:]
            return "\n".join(lines).strip()
    return text


def _validate(value: Any, schema: dict[str, Any]) -> list[str]:
    """Tiny JSON-schema subset validator — enough to guard typical
    structured outputs. Supports:
      - type: object/array/string/number/integer/boolean/null
      - properties (for objects)
      - required (for objects)
      - items (for arrays)
      - enum

    Returns a list of human-readable errors; empty = valid.
    """
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _type_match(value, expected_type):
        errors.append(f"expected {expected_type}, got {type(value).__name__}")
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"value {value!r} not in enum {schema['enum']}")
    if expected_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(value, dict):
            return errors
        for k in required:
            if k not in value:
                errors.append(f"missing required key '{k}'")
        for k, sub in props.items():
            if k in value:
                errors.extend(f"{k}.{e}" for e in _validate(value[k], sub))
    elif expected_type == "array":
        items_schema = schema.get("items")
        if isinstance(value, list) and isinstance(items_schema, dict):
            for i, item in enumerate(value):
                errors.extend(f"[{i}].{e}" for e in _validate(item, items_schema))
    return errors


def _type_match(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


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


__all__ = ["FormatJudge"]
