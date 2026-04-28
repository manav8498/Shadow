"""Tests for shadow.ltl — formula AST, model checker, and compiler."""

from __future__ import annotations

import pytest

from shadow.ltl import check_trace
from shadow.ltl.checker import TraceState, check, trace_from_records
from shadow.ltl.compiler import parse_ltl, rule_to_ltl
from shadow.ltl.formula import (
    FALSE,
    TRUE,
    And,
    Atom,
    Finally,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    Until,
    g,
    neg,
)

# ---- helpers ----------------------------------------------------------------


def _states(*tool_call_lists: list[str], stop_reasons: list[str] | None = None) -> list[TraceState]:
    """Build a list of TraceState from tool-call lists."""
    stops = stop_reasons or ["end_turn"] * len(tool_call_lists)
    return [
        TraceState(pair_index=i, tool_calls=tc, stop_reason=stops[i])
        for i, tc in enumerate(tool_call_lists)
    ]


# ---- formula AST ------------------------------------------------------------


class TestFormulaStr:
    def test_atom_str(self):
        assert str(Atom("tool_call:search")) == "tool_call:search"

    def test_globally_str(self):
        assert "G" in str(Globally(Atom("true")))

    def test_finally_str(self):
        assert "F" in str(Finally(Atom("true")))

    def test_not_str(self):
        assert "¬" in str(Not(Atom("false")))

    def test_and_str(self):
        assert "∧" in str(And(Atom("a"), Atom("b")))

    def test_implies_str(self):
        assert "→" in str(Implies(Atom("a"), Atom("b")))


# ---- LTL checker ------------------------------------------------------------


class TestAtom:
    def test_true_holds_anywhere(self):
        states = _states([])
        assert check(TRUE, states, 0)
        assert check(TRUE, states, 1)  # past end is True for Atom("true")

    def test_false_never_holds(self):
        states = _states([])
        assert not check(FALSE, states, 0)

    def test_tool_call_predicate(self):
        states = _states(["search"], [])
        assert check(Atom("tool_call:search"), states, 0)
        assert not check(Atom("tool_call:search"), states, 1)

    def test_stop_reason_predicate(self):
        states = _states([], stop_reasons=["end_turn"])
        assert check(Atom("stop_reason:end_turn"), states, 0)
        assert not check(Atom("stop_reason:tool_use"), states, 0)

    def test_text_contains_predicate(self):
        state = TraceState(
            pair_index=0,
            tool_calls=[],
            stop_reason="end_turn",
            text_content="hello world",
        )
        assert check(Atom("text_contains:hello"), [state], 0)
        assert not check(Atom("text_contains:goodbye"), [state], 0)


class TestNot:
    def test_negates_true(self):
        assert not check(Not(TRUE), _states([]), 0)

    def test_negates_false(self):
        assert check(Not(FALSE), _states([]), 0)


class TestAnd:
    def test_both_true(self):
        states = _states(["search"], stop_reasons=["end_turn"])
        phi = And(Atom("tool_call:search"), Atom("stop_reason:end_turn"))
        assert check(phi, states, 0)

    def test_one_false(self):
        states = _states(["search"], stop_reasons=["end_turn"])
        phi = And(Atom("tool_call:search"), Atom("tool_call:delete"))
        assert not check(phi, states, 0)


class TestOr:
    def test_either_true(self):
        states = _states(["search"])
        phi = Or(Atom("tool_call:search"), Atom("tool_call:delete"))
        assert check(phi, states, 0)

    def test_both_false(self):
        states = _states([])
        phi = Or(Atom("tool_call:search"), Atom("tool_call:delete"))
        assert not check(phi, states, 0)


class TestGlobally:
    def test_globally_on_empty_trace(self):
        # G φ on empty trace is vacuously true.
        assert check(Globally(FALSE), [], 0)

    def test_globally_holds_when_all_states_satisfy(self):
        states = _states([], [], [])  # 3 states, all stop_reason="end_turn"
        phi = Globally(Atom("stop_reason:end_turn"))
        assert check(phi, states, 0)

    def test_globally_fails_when_one_state_violates(self):
        states = _states([], [], stop_reasons=["end_turn", "tool_use"])
        phi = Globally(Atom("stop_reason:end_turn"))
        assert not check(phi, states, 0)

    def test_globally_holds_after_violation_is_past(self):
        # At position 1 onwards all states satisfy.
        states = _states([], [], stop_reasons=["tool_use", "end_turn"])
        phi = Globally(Atom("stop_reason:end_turn"))
        assert not check(phi, states, 0)  # fails at 0 (tool_use)
        assert check(phi, states, 1)  # holds from 1 onwards


class TestFinally:
    def test_finally_on_empty_trace_is_false(self):
        assert not check(Finally(TRUE), [], 0)

    def test_finally_holds_when_witnessed(self):
        states = _states([], ["search"])
        phi = Finally(Atom("tool_call:search"))
        assert check(phi, states, 0)
        assert check(phi, states, 1)
        assert not check(phi, states, 2)  # past end

    def test_finally_fails_when_never_witnessed(self):
        states = _states([], [])
        phi = Finally(Atom("tool_call:search"))
        assert not check(phi, states, 0)


class TestNext:
    def test_next_holds_on_next_state(self):
        states = _states([], ["search"])
        phi = Next(Atom("tool_call:search"))
        assert check(phi, states, 0)

    def test_next_false_at_last_state(self):
        states = _states(["search"])
        phi = Next(Atom("tool_call:search"))
        assert not check(phi, states, 0)  # no next state

    def test_next_false_past_end(self):
        states = _states([])
        assert not check(Next(TRUE), states, 1)


class TestUntil:
    def test_until_satisfied(self):
        # φ U ψ: φ holds until ψ.
        states = _states([], [], ["search"])  # 3 states
        # "not search" U "search" — first two states have no search,
        # third does.
        phi = Until(Not(Atom("tool_call:search")), Atom("tool_call:search"))
        assert check(phi, states, 0)

    def test_until_violated_when_psi_never_holds(self):
        states = _states([], [])
        phi = Until(TRUE, Atom("tool_call:search"))
        assert not check(phi, states, 0)

    def test_until_trivially_satisfied_when_psi_holds_immediately(self):
        states = _states(["search"])
        phi = Until(FALSE, Atom("tool_call:search"))
        assert check(phi, states, 0)


class TestImplies:
    def test_implies_vacuously_true_when_antecedent_false(self):
        states = _states([])
        phi = Implies(Atom("tool_call:search"), FALSE)
        assert check(phi, states, 0)

    def test_implies_requires_consequent_when_antecedent_true(self):
        states = _states(["search"])
        phi = Implies(Atom("tool_call:search"), Atom("tool_call:delete"))
        assert not check(phi, states, 0)
        phi_ok = Implies(Atom("tool_call:search"), Atom("tool_call:search"))
        assert check(phi_ok, states, 0)


# ---- compiler / rule_to_ltl -------------------------------------------------


class TestRuleToLtl:
    def test_no_call(self):
        formula = rule_to_ltl("no_call", {"tool": "delete_all"})
        assert formula is not None
        # Should be G(¬tool_call:delete_all)
        states_clean = _states([], [])
        states_with_call = _states(["delete_all"])
        assert check(formula, states_clean, 0)
        assert not check(formula, states_with_call, 0)

    def test_must_call_before(self):
        formula = rule_to_ltl("must_call_before", {"first": "auth", "then": "payment"})
        assert formula is not None
        # auth before payment — should pass.
        states_ok = _states(["auth"], ["payment"])
        assert check(formula, states_ok, 0)
        # payment without prior auth — should fail.
        states_bad = _states(["payment"])
        assert not check(formula, states_bad, 0)

    def test_must_call_once(self):
        formula = rule_to_ltl("must_call_once", {"tool": "log_action"})
        assert formula is not None
        # Called exactly once: OK.
        states_once = _states([], ["log_action"], [])
        assert check(formula, states_once, 0)
        # Never called: fails F(call).
        states_never = _states([], [])
        assert not check(formula, states_never, 0)
        # Called twice: fails G(call → X(G(¬call))).
        states_twice = _states(["log_action"], ["log_action"])
        assert not check(formula, states_twice, 0)

    def test_required_stop_reason(self):
        formula = rule_to_ltl("required_stop_reason", {"allowed": ["end_turn"]})
        assert formula is not None
        states_ok = _states([], stop_reasons=["end_turn"])
        states_bad = _states([], stop_reasons=["content_filter"])
        assert check(formula, states_ok, 0)
        assert not check(formula, states_bad, 0)

    def test_forbidden_text(self):
        formula = rule_to_ltl("forbidden_text", {"text": "confidential"})
        assert formula is not None
        state_clean = TraceState(0, text_content="all good")
        state_bad = TraceState(0, text_content="this is confidential data")
        assert check(formula, [state_clean], 0)
        assert not check(formula, [state_bad], 0)

    def test_must_include_text(self):
        formula = rule_to_ltl("must_include_text", {"text": "approved"})
        assert formula is not None
        state_with = TraceState(0, text_content="your request is approved")
        state_without = TraceState(0, text_content="pending review")
        assert check(formula, [state_with], 0)
        assert not check(formula, [state_without], 0)

    def test_max_turns_returns_none(self):
        # max_turns has no LTL encoding.
        assert rule_to_ltl("max_turns", {"limit": 5}) is None

    def test_missing_params_returns_none(self):
        assert rule_to_ltl("no_call", {"tool": 123}) is None  # non-str tool
        assert rule_to_ltl("must_call_before", {"first": "a"}) is None  # missing then


# ---- parse_ltl --------------------------------------------------------------


class TestParseLtl:
    def test_atom(self):
        f = parse_ltl("tool_call:search")
        assert isinstance(f, Atom)
        assert f.pred == "tool_call:search"

    def test_true_false(self):
        assert isinstance(parse_ltl("true"), Atom)
        assert parse_ltl("true").pred == "true"
        assert isinstance(parse_ltl("false"), Atom)

    def test_globally(self):
        f = parse_ltl("G tool_call:search")
        assert isinstance(f, Globally)
        assert isinstance(f.child, Atom)

    def test_finally(self):
        f = parse_ltl("F stop_reason:end_turn")
        assert isinstance(f, Finally)

    def test_next(self):
        f = parse_ltl("X tool_call:search")
        assert isinstance(f, Next)

    def test_not(self):
        f = parse_ltl("! tool_call:search")
        assert isinstance(f, Not)

    def test_and(self):
        f = parse_ltl("tool_call:a & tool_call:b")
        assert isinstance(f, And)

    def test_or(self):
        f = parse_ltl("tool_call:a | tool_call:b")
        assert isinstance(f, Or)

    def test_implies(self):
        f = parse_ltl("tool_call:a -> tool_call:b")
        assert isinstance(f, Implies)

    def test_until(self):
        f = parse_ltl("tool_call:a U tool_call:b")
        assert isinstance(f, Until)

    def test_grouping(self):
        f = parse_ltl("G (tool_call:a | tool_call:b)")
        assert isinstance(f, Globally)
        assert isinstance(f.child, Or)

    def test_nested(self):
        f = parse_ltl("G ! tool_call:delete")
        assert isinstance(f, Globally)
        assert isinstance(f.child, Not)

    def test_complex_expression(self):
        f = parse_ltl("G (tool_call:auth -> F tool_call:payment)")
        assert isinstance(f, Globally)

    def test_right_associative_implies(self):
        # a -> b -> c should parse as a -> (b -> c)
        f = parse_ltl("true -> true -> false")
        assert isinstance(f, Implies)
        assert isinstance(f.right, Implies)

    def test_ltl_formula_kind(self):
        formula = rule_to_ltl("ltl_formula", {"formula": "G ! tool_call:delete"})
        assert formula is not None
        states_clean = _states([])
        states_delete = _states(["delete"])
        assert check(formula, states_clean, 0)
        assert not check(formula, states_delete, 0)

    def test_parse_error_on_junk(self):
        with pytest.raises(ValueError):
            parse_ltl("G")  # operator without argument

    def test_unexpected_token(self):
        with pytest.raises(ValueError):
            parse_ltl("true false")  # two tokens without operator


# ---- trace_from_records -----------------------------------------------------


class TestTraceFromRecords:
    def test_extracts_tool_calls(self):
        records = [
            {
                "kind": "chat_response",
                "id": "r1",
                "payload": {
                    "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "name": "search", "input": {}}],
                },
            }
        ]
        states = trace_from_records(records)
        assert len(states) == 1
        assert "search" in states[0].tool_calls
        assert states[0].stop_reason == "tool_use"

    def test_skips_non_response_records(self):
        end_payload = {"stop_reason": "end_turn", "content": []}
        records = [
            {"kind": "metadata", "id": "m1", "payload": {}},
            {"kind": "chat_request", "id": "q1", "payload": {}},
            {"kind": "chat_response", "id": "r1", "payload": end_payload},
        ]
        states = trace_from_records(records)
        assert len(states) == 1

    def test_pair_index_increments(self):
        end_payload = {"stop_reason": "end_turn", "content": []}
        records = [
            {"kind": "chat_response", "id": f"r{i}", "payload": end_payload} for i in range(3)
        ]
        states = trace_from_records(records)
        assert [s.pair_index for s in states] == [0, 1, 2]


# ---- check_trace (integration) ----------------------------------------------


class TestCheckTrace:
    def _records(self, tool_calls_per_turn: list[list[str]]) -> list[dict]:
        records: list[dict] = []
        for i, tc in enumerate(tool_calls_per_turn):
            content = [{"type": "tool_use", "name": n, "input": {}} for n in tc]
            content.append({"type": "text", "text": "done"})
            records.append(
                {
                    "kind": "chat_response",
                    "id": f"r{i}",
                    "payload": {"stop_reason": "end_turn", "content": content},
                }
            )
        return records

    def test_no_violations_returns_empty(self):
        formula = g(neg(Atom("tool_call:delete")))
        records = self._records([["search"], ["search"]])
        assert check_trace(formula, records) == []

    def test_violation_returns_pair_index(self):
        formula = g(neg(Atom("tool_call:delete")))
        records = self._records([["search"], ["delete"]])
        violations = check_trace(formula, records)
        assert 1 in violations

    def test_empty_records_no_violations(self):
        formula = g(neg(Atom("tool_call:delete")))
        assert check_trace(formula, []) == []


# ---- ltl_formula policy kind via hierarchical.py ---------------------------


class TestLtlPolicyKind:
    def _records(self, tool_calls_per_turn: list[list[str]]) -> list[dict]:
        records: list[dict] = [{"kind": "metadata", "id": "m0", "payload": {}}]
        for i, tc in enumerate(tool_calls_per_turn):
            content = [{"type": "tool_use", "name": n, "input": {}} for n in tc]
            content.append({"type": "text", "text": "ok"})
            records.append(
                {
                    "kind": "chat_response",
                    "id": f"r{i}",
                    "payload": {"stop_reason": "end_turn", "content": content},
                }
            )
        return records

    def test_ltl_formula_no_violation(self):
        from shadow.hierarchical import check_policy, load_policy

        rules = load_policy(
            [
                {
                    "id": "no-delete",
                    "kind": "ltl_formula",
                    "params": {"formula": "G !tool_call:delete"},
                    "severity": "error",
                }
            ]
        )
        records = self._records([["search"], ["fetch"]])
        violations = check_policy(records, rules)
        assert violations == []

    def test_ltl_formula_detects_violation(self):
        from shadow.hierarchical import check_policy, load_policy

        rules = load_policy(
            [
                {
                    "id": "no-delete",
                    "kind": "ltl_formula",
                    "params": {"formula": "G !tool_call:delete"},
                    "severity": "error",
                }
            ]
        )
        records = self._records([["search"], ["delete"]])
        violations = check_policy(records, rules)
        assert any(v.rule_id == "no-delete" for v in violations)

    def test_ltl_formula_missing_param_whole_trace_violation(self):
        from shadow.hierarchical import check_policy, load_policy

        rules = load_policy(
            [
                {
                    "id": "bad",
                    "kind": "ltl_formula",
                    "params": {},  # no formula
                    "severity": "warning",
                }
            ]
        )
        records = self._records([[]])
        violations = check_policy(records, rules)
        assert len(violations) == 1
        assert "formula" in violations[0].detail

    def test_ltl_formula_parse_error_returns_violation(self):
        from shadow.hierarchical import check_policy, load_policy

        rules = load_policy(
            [
                {
                    "id": "bad-ltl",
                    "kind": "ltl_formula",
                    "params": {"formula": "G"},  # incomplete formula
                    "severity": "error",
                }
            ]
        )
        records = self._records([[]])
        violations = check_policy(records, rules)
        assert len(violations) == 1
        assert "parse" in violations[0].detail.lower() or "error" in violations[0].detail.lower()
