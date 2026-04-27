"""Tests for v2.1 pre-tool-call (pre-dispatch) policy enforcement.

The post-response :class:`EnforcedSession` checks ``record_chat`` after
the model returned. This module's :func:`wrap_tools` and
:class:`GuardedTool` move the check BEFORE the tool function runs, so
tool-sequence rules like ``no_call``, ``must_call_before``, and
``must_call_once`` block dangerous calls at the dispatch site instead
of after the tool's side effects already happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow.hierarchical import load_policy
from shadow.policy_runtime import (
    EnforcedSession,
    PolicyEnforcer,
    PolicyViolationError,
    wrap_tools,
)

# ---- shared fixtures ---------------------------------------------------


def _meta_record() -> dict[str, Any]:
    return {
        "version": "0.1",
        "kind": "metadata",
        "id": "sha256:m",
        "ts": "t",
        "parent": None,
        "payload": {"sdk": {"name": "test"}},
    }


def _no_call_rules(tool: str = "delete_user") -> list[Any]:
    return load_policy(
        [{"id": "no-x", "kind": "no_call", "params": {"tool": tool}, "severity": "error"}]
    )


def _must_call_before_rules(first: str, then: str) -> list[Any]:
    return load_policy(
        [
            {
                "id": "ordering",
                "kind": "must_call_before",
                "params": {"first": first, "then": then},
                "severity": "error",
            }
        ]
    )


# ---- probe (non-mutating evaluate) -------------------------------------


def test_probe_does_not_mutate_known_set() -> None:
    """probe() asks "what would happen" without remembering.
    A subsequent evaluate() with the same trace must still surface the
    violation."""
    rules = _no_call_rules("send_email")
    enforcer = PolicyEnforcer(rules, on_violation="warn")
    records = [_meta_record()]
    # Append a tool_call that violates no_call.
    bad_call = {
        "version": "0.1",
        "kind": "tool_call",
        "id": "sha256:tc",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {
            "tool_name": "send_email",
            "tool_call_id": "t1",
            "arguments": {"to": "alice@example.com"},
        },
    }
    pre = enforcer.probe([*records, bad_call])
    post = enforcer.evaluate([*records, bad_call])
    assert pre.allow is False
    assert post.allow is False  # evaluate would have skipped if probe leaked state
    assert len(post.violations) == 1


# ---- GuardedTool basic shape -------------------------------------------


def test_guarded_tool_calls_underlying_fn_when_allowed(tmp_path: Path) -> None:
    rules = _no_call_rules("send_email")
    enforcer = PolicyEnforcer(rules, on_violation="raise")

    def lookup_order(order_id: str) -> str:
        return f"ORDER:{order_id}"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "ok.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"lookup_order": lookup_order})
        result = guarded["lookup_order"](order_id="ORD-1")
    assert result == "ORDER:ORD-1"


def test_guarded_tool_blocks_no_call_violation_in_replace_mode(tmp_path: Path) -> None:
    rules = _no_call_rules("delete_user")
    enforcer = PolicyEnforcer(rules, on_violation="replace")

    delete_was_called = []

    def delete_user(user_id: str) -> str:
        delete_was_called.append(user_id)
        return f"deleted {user_id}"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "blocked.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"delete_user": delete_user})
        result = guarded["delete_user"](user_id="u-42")

    # The real function was NOT called.
    assert delete_was_called == []
    # Returned the placeholder.
    assert "blocked" in str(result).lower()


def test_guarded_tool_raises_in_raise_mode(tmp_path: Path) -> None:
    rules = _no_call_rules("execute_sql")
    enforcer = PolicyEnforcer(rules, on_violation="raise")

    def execute_sql(query: str) -> str:
        return "executed"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "raise.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"execute_sql": execute_sql})
        with pytest.raises(PolicyViolationError, match="execute_sql"):
            guarded["execute_sql"](query="DROP TABLE users")


def test_guarded_tool_calls_anyway_in_warn_mode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rules = _no_call_rules("issue_refund")
    enforcer = PolicyEnforcer(rules, on_violation="warn")

    called = []

    def issue_refund(amount: float) -> str:
        called.append(amount)
        return f"refunded {amount}"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "warn.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"issue_refund": issue_refund})
        with caplog.at_level("WARNING", logger="shadow.policy_runtime"):
            result = guarded["issue_refund"](amount=100.0)

    assert called == [100.0]  # warn mode = call anyway
    assert result == "refunded 100.0"
    assert any("blocked but warn" in r.message for r in caplog.records)


# ---- must_call_before pre-dispatch -------------------------------------


def test_guarded_tool_blocks_must_call_before_violation(tmp_path: Path) -> None:
    """Calling `process_refund` before `confirm_with_user` should be
    blocked at the dispatch site — the canonical pattern for dangerous
    operations."""
    rules = _must_call_before_rules(first="confirm_with_user", then="process_refund")
    enforcer = PolicyEnforcer(rules, on_violation="raise")

    process_was_called = []

    def confirm_with_user(amount: float) -> str:
        return "confirmed"

    def process_refund(amount: float) -> str:
        process_was_called.append(amount)
        return "processed"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "ordering.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools(
            {"confirm_with_user": confirm_with_user, "process_refund": process_refund}
        )
        # Skipping the confirm call — process_refund should raise.
        with pytest.raises(PolicyViolationError, match="process_refund"):
            guarded["process_refund"](amount=500)
    assert process_was_called == []


def test_guarded_tool_allows_must_call_before_when_ordered_correctly(
    tmp_path: Path,
) -> None:
    """Same rule, but called in the right order — both should run.

    Note: must_call_before only fires when `then` is invoked, and only
    if `first` hasn't been called yet. We confirm by recording a real
    `tool_call` for confirm_with_user (via Session.record_tool_call)
    BEFORE invoking the guarded process_refund."""
    rules = _must_call_before_rules(first="confirm_with_user", then="process_refund")
    enforcer = PolicyEnforcer(rules, on_violation="raise")

    process_was_called = []

    def confirm_with_user(amount: float) -> str:
        return "confirmed"

    def process_refund(amount: float) -> str:
        process_was_called.append(amount)
        return "processed"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "ok-ordering.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        # Manually record the confirm_with_user tool_call so the rule's
        # `first` precondition is satisfied. (In production an LLM
        # would have caused this via record_chat / a real tool dispatch;
        # here we record it explicitly to keep the test pure.)
        s.record_tool_call(
            tool_name="confirm_with_user",
            tool_call_id="t1",
            arguments={"amount": 500},
        )
        guarded = s.wrap_tools(
            {"confirm_with_user": confirm_with_user, "process_refund": process_refund}
        )
        result = guarded["process_refund"](amount=500)
    assert result == "processed"
    assert process_was_called == [500]


# ---- standalone wrap_tools (no EnforcedSession) ------------------------


def test_wrap_tools_with_records_provider_callable() -> None:
    """When the caller already has its own session-management layer
    (LangChain / framework adapter), wrap_tools accepts an explicit
    records_provider callable instead of an EnforcedSession."""
    rules = _no_call_rules("delete_user")
    enforcer = PolicyEnforcer(rules, on_violation="raise")

    records: list[dict[str, Any]] = [_meta_record()]

    def delete_user(user_id: str) -> str:
        return "deleted"

    guarded = wrap_tools(
        {"delete_user": delete_user},
        enforcer,
        records_provider=lambda: records,
    )
    with pytest.raises(PolicyViolationError):
        guarded["delete_user"](user_id="x")


def test_wrap_tools_requires_session_or_records_provider() -> None:
    rules = _no_call_rules("x")
    enforcer = PolicyEnforcer(rules)
    with pytest.raises(ValueError, match="records_provider"):
        wrap_tools({"x": lambda: None}, enforcer)


# ---- custom blocked_replacement ----------------------------------------


def test_guarded_tool_custom_blocked_replacement(tmp_path: Path) -> None:
    rules = _no_call_rules("send_email")
    enforcer = PolicyEnforcer(rules, on_violation="replace")

    def send_email(to: str, body: str) -> dict[str, Any]:
        return {"sent": True, "to": to}

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "custom-rep.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = wrap_tools(
            {"send_email": send_email},
            enforcer,
            session=s,
            blocked_replacement={"sent": False, "reason": "policy_blocked"},
        )
        result = guarded["send_email"](to="bob@x.com", body="hi")
    assert result == {"sent": False, "reason": "policy_blocked"}


# ---- behavioural: probe stays non-mutating across many calls ----------


def test_repeated_blocked_calls_do_not_pollute_enforcer_state(tmp_path: Path) -> None:
    """Probe is non-mutating, so the same tool can be probed-blocked
    many times without leaking state into a future evaluate() call.
    Confirms the v2.0.1 dedup-fix interacts cleanly with probe."""
    rules = _no_call_rules("delete_user")
    enforcer = PolicyEnforcer(rules, on_violation="replace")

    def delete_user(uid: str) -> str:
        return "deleted"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "repeat.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"delete_user": delete_user})
        # Five blocked attempts.
        for _ in range(5):
            r = guarded["delete_user"](uid="u")
            assert "blocked" in str(r).lower()
        # Enforcer's _known should still be empty — probe didn't
        # mutate it. So an evaluate() now should still surface the
        # violation if a real call had been recorded.
        assert len(enforcer._known) == 0


def test_wrap_tools_auto_records_successful_calls(tmp_path: Path) -> None:
    """Default auto_record=True makes a successful tool call visible
    to subsequent sequential-rule checks. Without this the user had to
    call s.record_tool_call() manually, and a `must_call_before`
    policy would silently fail to enforce ordering even when the
    caller did the right thing in the right order."""
    policy_yaml = (
        "version: '1'\n"
        "rules:\n"
        "  - id: confirm-before-refund\n"
        "    kind: must_call_before\n"
        "    severity: error\n"
        "    params: {first: confirm_amount, then: issue_refund}\n"
    )
    pol_path = tmp_path / "policy.yaml"
    pol_path.write_text(policy_yaml)
    enforcer = PolicyEnforcer.from_policy_file(pol_path)

    confirm_calls: list[Any] = []
    refund_calls: list[Any] = []

    def confirm_amount(amount: int) -> str:
        confirm_calls.append(amount)
        return "confirmed"

    def issue_refund(amount: int) -> str:
        refund_calls.append(amount)
        return f"refunded ${amount}"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "trace.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools({"confirm_amount": confirm_amount, "issue_refund": issue_refund})
        # Calling confirm first then refund must succeed — auto-record
        # makes the confirm call visible to the must_call_before rule
        # so refund passes.
        confirm_result = guarded["confirm_amount"](42)
        refund_result = guarded["issue_refund"](42)

    assert confirm_result == "confirmed"
    assert refund_result == "refunded $42"
    assert confirm_calls == [42]
    assert refund_calls == [42]


def test_wrap_tools_auto_record_off_preserves_old_behaviour(tmp_path: Path) -> None:
    """auto_record=False keeps the historical 'caller records
    explicitly' behaviour: tool_call records aren't appended, so a
    must_call_before rule that depends on them will not see the
    confirm call and will block the refund."""
    policy_yaml = (
        "version: '1'\n"
        "rules:\n"
        "  - id: confirm-before-refund\n"
        "    kind: must_call_before\n"
        "    severity: error\n"
        "    params: {first: confirm_amount, then: issue_refund}\n"
    )
    pol_path = tmp_path / "policy.yaml"
    pol_path.write_text(policy_yaml)
    enforcer = PolicyEnforcer(
        list(_load_yaml_rules(pol_path)),
        on_violation="replace",
    )

    def confirm_amount(amount: int) -> str:
        return "confirmed"

    def issue_refund(amount: int) -> str:
        return f"refunded ${amount}"

    with EnforcedSession(
        enforcer=enforcer,
        output_path=tmp_path / "trace.agentlog",
        tags={"env": "t"},
        auto_instrument=False,
    ) as s:
        guarded = s.wrap_tools(
            {"confirm_amount": confirm_amount, "issue_refund": issue_refund},
            auto_record=False,
        )
        guarded["confirm_amount"](42)
        # auto_record=False means confirm call is not recorded; the
        # must_call_before rule fires and refund is blocked.
        result = guarded["issue_refund"](42)
        assert "blocked" in str(result).lower()


def _load_yaml_rules(path: Path):
    import yaml

    from shadow.hierarchical import load_policy

    return load_policy(yaml.safe_load(path.read_text()))
