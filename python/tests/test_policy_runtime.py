"""Tests for ``shadow.policy_runtime`` — runtime policy enforcement
on top of ``shadow.sdk.Session``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow.errors import ShadowConfigError
from shadow.hierarchical import PolicyRule, load_policy
from shadow.policy_runtime import (
    EnforcedSession,
    PolicyEnforcer,
    PolicyViolationError,
    default_replacement_response,
)


def _request(**kw: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": "m", "messages": [], "params": {}}
    payload.update(kw)
    return payload


def _response(text: str = "hi") -> dict[str, Any]:
    return {
        "model": "m",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "latency_ms": 10,
        "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
    }


def _rules_no_call_tool_x() -> list[PolicyRule]:
    return load_policy(
        [{"id": "no-x", "kind": "no_call", "params": {"tool": "x"}, "severity": "error"}]
    )


def _rules_must_be_grounded() -> list[PolicyRule]:
    return load_policy(
        [
            {
                "id": "grounded",
                "kind": "must_be_grounded",
                "params": {
                    "retrieval_path": "request.metadata.retrieved_chunks",
                    "min_unigram_precision": 0.5,
                },
                "severity": "error",
            }
        ]
    )


# ---- Verdict and default replacement -----------------------------------


def test_default_replacement_preserves_structural_fields() -> None:
    original = {
        "model": "claude-opus",
        "content": [{"type": "text", "text": "the original answer"}],
        "stop_reason": "end_turn",
        "latency_ms": 123,
        "usage": {"input_tokens": 10, "output_tokens": 20, "thinking_tokens": 5},
    }
    from shadow.hierarchical import PolicyViolation

    v = [
        PolicyViolation(
            rule_id="no-x", kind="no_call", severity="error", pair_index=0, detail="x called"
        )
    ]
    out = default_replacement_response(v, original)
    assert out["stop_reason"] == "policy_blocked"
    assert "policy_runtime" in out["content"][0]["text"]
    # Structural fields preserved so downstream renderers don't break.
    assert out["model"] == "claude-opus"
    assert out["latency_ms"] == 123
    assert out["usage"]["input_tokens"] == 10


def test_default_replacement_handles_missing_original() -> None:
    """When called with an empty / non-dict 'original', the builder
    must still produce a valid chat_response payload."""
    from shadow.hierarchical import PolicyViolation

    v = [PolicyViolation("r", "no_call", "error", 0, "boom")]
    out = default_replacement_response(v, {})
    assert "model" in out
    assert "usage" in out
    assert "latency_ms" in out


# ---- PolicyEnforcer ----------------------------------------------------


def test_enforcer_returns_allow_when_no_violations() -> None:
    enforcer = PolicyEnforcer(_rules_no_call_tool_x())
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
    ]
    v = enforcer.evaluate(records)
    assert v.allow is True
    assert v.violations == []
    assert v.replacement is None


def test_enforcer_only_reports_new_violations_across_calls() -> None:
    """The enforcer is incremental: after reporting a violation once,
    subsequent evaluate() calls on the same trace must NOT re-report
    it. Otherwise every turn would re-fire whole-trace rules."""
    rules = load_policy(
        [{"id": "max-2", "kind": "max_turns", "params": {"n": 2}, "severity": "error"}]
    )
    enforcer = PolicyEnforcer(rules, on_violation="warn")
    records: list[dict[str, Any]] = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
    ]
    # Add three turn pairs — turn 3 trips max_turns.
    for i in range(3):
        records.append(
            {
                "kind": "chat_request",
                "id": f"sha256:q{i}",
                "ts": "t",
                "parent": "sha256:m",
                "payload": _request(),
            }
        )
        records.append(
            {
                "kind": "chat_response",
                "id": f"sha256:r{i}",
                "ts": "t",
                "parent": f"sha256:q{i}",
                "payload": _response(),
            }
        )
    v1 = enforcer.evaluate(records)
    v2 = enforcer.evaluate(records)
    assert len(v1.violations) >= 1
    assert v2.violations == [], "second evaluate() on unchanged trace must report nothing new"


def test_enforcer_replace_mode_builds_a_replacement_response() -> None:
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="replace")
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
        {
            "kind": "chat_request",
            "id": "sha256:q",
            "ts": "t",
            "parent": "sha256:m",
            "payload": _request(metadata={"retrieved_chunks": ["pythons are large snakes"]}),
        },
        {
            "kind": "chat_response",
            "id": "sha256:r",
            "ts": "t",
            "parent": "sha256:q",
            "payload": _response("totally unrelated answer about quantum physics"),
        },
    ]
    v = enforcer.evaluate(records)
    assert v.allow is False
    assert v.replacement is not None
    assert v.replacement["stop_reason"] == "policy_blocked"
    assert "policy_runtime" in v.replacement["content"][0]["text"]


def test_enforcer_warn_mode_does_not_build_replacement() -> None:
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="warn")
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
        {
            "kind": "chat_request",
            "id": "sha256:q",
            "ts": "t",
            "parent": "sha256:m",
            "payload": _request(metadata={"retrieved_chunks": ["pythons are large snakes"]}),
        },
        {
            "kind": "chat_response",
            "id": "sha256:r",
            "ts": "t",
            "parent": "sha256:q",
            "payload": _response("unrelated"),
        },
    ]
    v = enforcer.evaluate(records)
    assert v.allow is False
    assert v.replacement is None


def test_enforcer_from_policy_file_loads_yaml(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        "rules:\n"
        "  - id: no-x\n"
        "    kind: no_call\n"
        "    params: {tool: x}\n"
        "    severity: error\n"
    )
    enforcer = PolicyEnforcer.from_policy_file(p)
    assert enforcer.on_violation == "replace"


def test_enforcer_custom_replacement_builder_is_called() -> None:
    captured: dict[str, Any] = {}

    def my_builder(violations: list[Any], original: dict[str, Any]) -> dict[str, Any]:
        captured["count"] = len(violations)
        return {**original, "content": [{"type": "text", "text": "custom"}]}

    enforcer = PolicyEnforcer(
        _rules_must_be_grounded(),
        on_violation="replace",
        replacement_builder=my_builder,
    )
    records = [
        {"kind": "metadata", "id": "sha256:m", "ts": "t", "parent": None, "payload": {}},
        {
            "kind": "chat_request",
            "id": "sha256:q",
            "ts": "t",
            "parent": "sha256:m",
            "payload": _request(metadata={"retrieved_chunks": ["alpha bravo"]}),
        },
        {
            "kind": "chat_response",
            "id": "sha256:r",
            "ts": "t",
            "parent": "sha256:q",
            "payload": _response("delta echo"),
        },
    ]
    v = enforcer.evaluate(records)
    assert v.replacement is not None
    assert v.replacement["content"][0]["text"] == "custom"
    assert captured["count"] >= 1


# ---- EnforcedSession ---------------------------------------------------


def test_enforced_session_replace_mode_swaps_offending_response(tmp_path: Path) -> None:
    output = tmp_path / "run.agentlog"
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="replace")
    with EnforcedSession(enforcer=enforcer, output_path=output, tags={"env": "test"}) as s:
        s.record_chat(
            request=_request(metadata={"retrieved_chunks": ["pythons are large snakes"]}),
            response=_response("the moon is made of green cheese"),
        )
    from shadow import _core

    records = _core.parse_agentlog(output.read_bytes())
    response_record = next(r for r in records if r["kind"] == "chat_response")
    assert response_record["payload"]["stop_reason"] == "policy_blocked"
    assert "policy_runtime" in response_record["payload"]["content"][0]["text"]


def test_enforced_session_passes_through_when_grounded(tmp_path: Path) -> None:
    output = tmp_path / "run.agentlog"
    enforcer = PolicyEnforcer(_rules_must_be_grounded())
    with EnforcedSession(enforcer=enforcer, output_path=output, tags={"env": "test"}) as s:
        s.record_chat(
            request=_request(metadata={"retrieved_chunks": ["the refund window is thirty days"]}),
            response=_response("Your refund window is thirty days from purchase."),
        )
    from shadow import _core

    records = _core.parse_agentlog(output.read_bytes())
    response_record = next(r for r in records if r["kind"] == "chat_response")
    assert response_record["payload"]["stop_reason"] == "end_turn"
    assert "policy_blocked" not in str(response_record["payload"])


def test_enforced_session_raise_mode_throws_on_violation(tmp_path: Path) -> None:
    output = tmp_path / "run.agentlog"
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="raise")
    with (
        pytest.raises(PolicyViolationError),
        EnforcedSession(enforcer=enforcer, output_path=output, tags={"env": "test"}) as s,
    ):
        s.record_chat(
            request=_request(metadata={"retrieved_chunks": ["alpha bravo charlie"]}),
            response=_response("delta echo foxtrot"),
        )


def test_enforced_session_raise_mode_pops_offending_records(tmp_path: Path) -> None:
    """Raising must not leave the violating chat pair in the in-memory
    record list (so the trace flushed on __exit__ is structurally
    valid up to the previous turn)."""
    output = tmp_path / "run.agentlog"
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="raise")
    sess = EnforcedSession(enforcer=enforcer, output_path=output, tags={"env": "test"})
    sess.__enter__()
    try:
        # First, a clean turn that records normally.
        sess.record_chat(
            request=_request(metadata={"retrieved_chunks": ["the refund window is thirty days"]}),
            response=_response("Your refund window is thirty days."),
        )
        n_before = len(sess._records)
        with pytest.raises(PolicyViolationError):
            sess.record_chat(
                request=_request(metadata={"retrieved_chunks": ["alpha bravo"]}),
                response=_response("delta echo"),
            )
        # The violating chat pair must have been rolled back.
        assert len(sess._records) == n_before
    finally:
        sess.__exit__(None, None, None)


def test_enforced_session_warn_mode_records_unchanged(tmp_path: Path) -> None:
    output = tmp_path / "run.agentlog"
    enforcer = PolicyEnforcer(_rules_must_be_grounded(), on_violation="warn")
    with EnforcedSession(enforcer=enforcer, output_path=output, tags={"env": "test"}) as s:
        s.record_chat(
            request=_request(metadata={"retrieved_chunks": ["pythons are large snakes"]}),
            response=_response("the moon is made of green cheese"),
        )
    from shadow import _core

    records = _core.parse_agentlog(output.read_bytes())
    response_record = next(r for r in records if r["kind"] == "chat_response")
    # Warn mode does NOT modify the response.
    assert "green cheese" in response_record["payload"]["content"][0]["text"]
    assert response_record["payload"]["stop_reason"] == "end_turn"


# ---- error paths -------------------------------------------------------


def test_enforcer_from_policy_file_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("rules:\n  - this is not a mapping\n")
    with pytest.raises(ShadowConfigError):
        PolicyEnforcer.from_policy_file(p)


def test_policy_violation_error_carries_violations() -> None:
    from shadow.hierarchical import PolicyViolation

    v = [
        PolicyViolation("r1", "no_call", "error", 0, "x"),
        PolicyViolation("r2", "must_be_grounded", "error", 1, "y"),
    ]
    err = PolicyViolationError(v)
    assert err.violations == v
    assert "r1" in str(err)
    assert "r2" in str(err)
