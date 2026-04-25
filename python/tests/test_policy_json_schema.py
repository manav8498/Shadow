"""Tests for the ``must_match_json_schema`` policy rule kind."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shadow.hierarchical import (
    PolicyRule,
    _check_must_match_json_schema,
    load_policy,
    policy_diff,
)


def _resp(text: str) -> dict[str, Any]:
    return {
        "kind": "chat_response",
        "id": "sha256:r",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 0,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    }


SCHEMA = {
    "type": "object",
    "required": ["decision", "amount"],
    "properties": {
        "decision": {"type": "string", "enum": ["refund", "decline", "escalate"]},
        "amount": {"type": "number", "minimum": 0},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


def test_load_policy_accepts_must_match_json_schema_kind() -> None:
    rules = load_policy(
        {
            "rules": [
                {
                    "kind": "must_match_json_schema",
                    "params": {"schema": SCHEMA},
                    "severity": "error",
                }
            ]
        }
    )
    assert len(rules) == 1
    assert rules[0].kind == "must_match_json_schema"


def test_valid_json_response_passes_schema() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": SCHEMA},
        severity="error",
    )
    response = _resp(json.dumps({"decision": "refund", "amount": 49.99}))
    violations = _check_must_match_json_schema(rule, [response])
    assert violations == []


def test_invalid_json_response_surfaces_violation() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": SCHEMA},
        severity="error",
    )
    response = _resp("Sure, I'll refund that — $49.99 to your card.")
    violations = _check_must_match_json_schema(rule, [response])
    assert len(violations) == 1
    assert "not valid JSON" in violations[0].detail


def test_schema_mismatch_reports_path_and_message() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": SCHEMA},
        severity="error",
    )
    # Wrong type for `amount` and missing `decision`.
    response = _resp(json.dumps({"amount": "forty-nine"}))
    violations = _check_must_match_json_schema(rule, [response])
    assert len(violations) == 1
    assert "json schema mismatch" in violations[0].detail


def test_schema_path_param_loads_external_file(tmp_path: Path) -> None:
    schema_file = tmp_path / "refund_decision.schema.json"
    schema_file.write_text(json.dumps(SCHEMA))
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema_path": str(schema_file)},
        severity="error",
    )
    response = _resp(json.dumps({"decision": "decline", "amount": 0}))
    violations = _check_must_match_json_schema(rule, [response])
    assert violations == []


def test_both_schema_and_schema_path_is_a_violation() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": SCHEMA, "schema_path": "/nonexistent"},
        severity="error",
    )
    violations = _check_must_match_json_schema(
        rule, [_resp(json.dumps({"decision": "refund", "amount": 1}))]
    )
    assert len(violations) == 1
    assert "exactly one" in violations[0].detail


def test_neither_schema_nor_schema_path_is_a_violation() -> None:
    rule = PolicyRule(id="r1", kind="must_match_json_schema", params={}, severity="error")
    violations = _check_must_match_json_schema(rule, [_resp("{}")])
    assert len(violations) == 1


def test_schema_path_not_found_reports_clearly(tmp_path: Path) -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema_path": str(tmp_path / "missing.schema.json")},
        severity="error",
    )
    violations = _check_must_match_json_schema(rule, [_resp("{}")])
    assert len(violations) == 1
    assert "schema_path not found" in violations[0].detail


def test_invalid_schema_itself_short_circuits_to_one_violation() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": {"type": "not-a-real-type"}},
        severity="error",
    )
    violations = _check_must_match_json_schema(
        rule, [_resp(json.dumps({"x": 1})), _resp(json.dumps({"y": 2}))]
    )
    assert len(violations) == 1
    assert "schema itself is invalid" in violations[0].detail


def test_empty_response_text_is_a_violation() -> None:
    rule = PolicyRule(
        id="r1",
        kind="must_match_json_schema",
        params={"schema": SCHEMA},
        severity="error",
    )
    # tool_use only, no text block.
    response: dict[str, Any] = {
        "kind": "chat_response",
        "id": "sha256:r",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {
            "model": "m",
            "content": [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {}}],
            "stop_reason": "tool_use",
            "latency_ms": 0,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    }
    violations = _check_must_match_json_schema(rule, [response])
    assert len(violations) == 1
    assert "no text content" in violations[0].detail


def test_nan_and_infinity_rejected_as_non_rfc_8259_json() -> None:
    """Python's json.loads accepts NaN / Infinity / -Infinity as a
    CPython extension. They are NOT valid JSON per RFC 8259 and
    downstream consumers (browsers, other-language parsers) will
    choke on them, so the schema rule must reject them as invalid
    JSON regardless of whether the schema would have allowed the
    underlying number type."""
    rule = PolicyRule(
        id="r",
        kind="must_match_json_schema",
        params={
            "schema": {
                "type": "object",
                "required": ["amount"],
                "properties": {"amount": {"type": "number"}},
            }
        },
        severity="error",
    )
    for raw in ('{"amount": NaN}', '{"amount": Infinity}', '{"amount": -Infinity}'):
        out = _check_must_match_json_schema(rule, [_resp(raw)])
        assert len(out) == 1, f"{raw!r} should be a violation"
        assert (
            "non-standard" in out[0].detail or "NaN" in out[0].detail or "Infinity" in out[0].detail
        )


def test_policy_diff_threads_must_match_json_schema_through() -> None:
    """The full policy_diff path must accept must_match_json_schema rules
    and report regressions when the candidate adds violations the
    baseline didn't have."""
    rules = load_policy(
        [
            {
                "id": "schema-check",
                "kind": "must_match_json_schema",
                "params": {"schema": SCHEMA},
                "severity": "error",
            }
        ]
    )
    metadata: dict[str, Any] = {
        "kind": "metadata",
        "id": "sha256:m",
        "ts": "t",
        "parent": None,
        "payload": {},
    }
    request: dict[str, Any] = {
        "kind": "chat_request",
        "id": "sha256:q",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {"model": "m", "messages": [], "params": {}},
    }
    baseline = [
        metadata,
        request,
        _resp(json.dumps({"decision": "refund", "amount": 1})),
    ]
    candidate = [
        metadata,
        request,
        _resp("Sorry, can't help with that."),  # invalid JSON
    ]
    diff = policy_diff(baseline, candidate, rules)
    assert len(diff.regressions) == 1
    assert diff.regressions[0].kind == "must_match_json_schema"
