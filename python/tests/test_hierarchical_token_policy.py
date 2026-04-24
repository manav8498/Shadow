"""Tests for the token-level + policy-level layers of hierarchical diff."""

from __future__ import annotations

from typing import Any

import pytest

from shadow.errors import ShadowConfigError
from shadow.hierarchical import (
    check_policy,
    load_policy,
    policy_diff,
    render_policy_diff,
    render_token_diff,
    token_diff,
)

# ---- fixtures -------------------------------------------------------------


def _response(
    input_tokens: int = 100,
    output_tokens: int = 50,
    thinking_tokens: int = 0,
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "kind": "chat_response",
        "payload": {
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
            },
            "content": content or [{"type": "text", "text": "ok"}],
            "stop_reason": stop_reason,
        },
    }


def _tool_use(name: str, args: dict | None = None) -> dict:
    return {
        "type": "tool_use",
        "id": f"call_{name}",
        "name": name,
        "input": args or {},
    }


# ---- token diff -----------------------------------------------------------


def test_token_diff_identical_returns_zero_shift() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(100, 50) for _ in range(5)]
    out = token_diff(base, cand)
    assert out.pair_count == 5
    assert out.normalised_shift["input_tokens"] == 0.0
    assert out.normalised_shift["output_tokens"] == 0.0
    # All per-pair deltas are zero.
    for pair in out.worst_pairs:
        assert sum(abs(v) for v in pair.delta.values()) == 0


def test_token_diff_catches_output_blow_up() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(100, 200) for _ in range(5)]  # 4x output tokens
    out = token_diff(base, cand)
    assert out.normalised_shift["output_tokens"] == pytest.approx(3.0)
    assert out.worst_pairs[0].delta["output_tokens"] == 150


def test_token_diff_worst_pairs_ordered() -> None:
    base = [_response(100, 50) for _ in range(3)]
    cand = [
        _response(100, 50),  # no change
        _response(500, 50),  # +400 input
        _response(100, 200),  # +150 output
    ]
    out = token_diff(base, cand)
    # Pair #1 (+400) should come before pair #2 (+150).
    assert out.worst_pairs[0].pair_index == 1
    assert out.worst_pairs[1].pair_index == 2


def test_token_diff_handles_empty_traces() -> None:
    out = token_diff([], [])
    assert out.pair_count == 0
    assert all(v == 0.0 for v in out.normalised_shift.values())


def test_token_diff_unequal_lengths_uses_min() -> None:
    base = [_response() for _ in range(3)]
    cand = [_response() for _ in range(5)]
    out = token_diff(base, cand)
    assert out.pair_count == 3


def test_token_diff_handles_baseline_zero_median() -> None:
    base = [_response(thinking_tokens=0) for _ in range(3)]
    cand = [_response(thinking_tokens=1000) for _ in range(3)]
    out = token_diff(base, cand)
    assert out.normalised_shift["thinking_tokens"] == float("inf")


def test_render_token_diff_has_expected_fields() -> None:
    base = [_response(100, 50) for _ in range(5)]
    cand = [_response(150, 50) for _ in range(5)]
    out = token_diff(base, cand)
    rendered = render_token_diff(out)
    assert "Token-level diff" in rendered
    assert "input_tokens" in rendered
    assert "+50.00%" in rendered


def test_token_diff_percentiles_correct() -> None:
    base = [_response(output_tokens=v) for v in [10, 20, 30, 40, 50]]
    cand = [_response(output_tokens=v) for v in [10, 20, 30, 40, 50]]
    out = token_diff(base, cand)
    assert out.baseline["output_tokens"].median == pytest.approx(30)
    # p95 across [10,20,30,40,50] linear-interp = 48
    assert out.baseline["output_tokens"].p95 == pytest.approx(48)
    assert out.baseline["output_tokens"].maximum == 50
    assert out.baseline["output_tokens"].total == 150


# ---- policy diff ----------------------------------------------------------


def test_load_policy_accepts_wrapped_and_bare() -> None:
    wrapped = {"rules": [{"id": "r1", "kind": "no_call", "params": {"tool": "bad"}}]}
    rules = load_policy(wrapped)
    assert rules[0].id == "r1"
    assert rules[0].kind == "no_call"
    bare = [{"id": "r1", "kind": "no_call", "params": {"tool": "bad"}}]
    rules2 = load_policy(bare)
    assert rules2[0].id == "r1"


def test_load_policy_rejects_unknown_kind() -> None:
    with pytest.raises(ShadowConfigError):
        load_policy([{"id": "r", "kind": "make-coffee", "params": {}}])


def test_load_policy_rejects_wrong_shape() -> None:
    with pytest.raises(ShadowConfigError):
        load_policy(42)


def test_must_call_before_passes_when_ordered_correctly() -> None:
    rules = load_policy(
        [
            {
                "id": "backup-before-migrate",
                "kind": "must_call_before",
                "params": {"first": "backup_database", "then": "run_migration"},
            }
        ]
    )
    records = [
        _response(content=[_tool_use("backup_database")], stop_reason="tool_use"),
        _response(content=[_tool_use("run_migration")], stop_reason="tool_use"),
    ]
    assert check_policy(records, rules) == []


def test_must_call_before_flags_wrong_order() -> None:
    rules = load_policy(
        [
            {
                "id": "backup-before-migrate",
                "kind": "must_call_before",
                "params": {"first": "backup_database", "then": "run_migration"},
            }
        ]
    )
    records = [
        _response(content=[_tool_use("run_migration")], stop_reason="tool_use"),
        _response(content=[_tool_use("backup_database")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert violations[0].rule_id == "backup-before-migrate"
    assert violations[0].pair_index == 0  # the run_migration turn


def test_no_call_flags_each_invocation() -> None:
    rules = load_policy([{"id": "no-rm-rf", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    records = [
        _response(content=[_tool_use("rm_rf")], stop_reason="tool_use"),
        _response(content=[_tool_use("rm_rf")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert [v.pair_index for v in violations] == [0, 1]


def test_must_call_once_catches_duplicates() -> None:
    rules = load_policy(
        [{"id": "one-login", "kind": "must_call_once", "params": {"tool": "login"}}]
    )
    records = [
        _response(content=[_tool_use("login")], stop_reason="tool_use"),
        _response(content=[_tool_use("login")], stop_reason="tool_use"),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert "2 times" in violations[0].detail


def test_max_turns_limit() -> None:
    rules = load_policy([{"id": "short", "kind": "max_turns", "params": {"limit": 2}}])
    records = [_response() for _ in range(3)]
    violations = check_policy(records, rules)
    assert len(violations) == 1


def test_required_stop_reason() -> None:
    rules = load_policy(
        [
            {
                "id": "finish-clean",
                "kind": "required_stop_reason",
                "params": {"allowed": ["end_turn"]},
            }
        ]
    )
    # final stop_reason = content_filter → violation
    bad = [_response(stop_reason="end_turn"), _response(stop_reason="content_filter")]
    assert check_policy(bad, rules)
    # final = end_turn → pass
    good = [_response(stop_reason="end_turn"), _response(stop_reason="end_turn")]
    assert check_policy(good, rules) == []


def test_max_total_tokens() -> None:
    rules = load_policy([{"id": "budget", "kind": "max_total_tokens", "params": {"limit": 500}}])
    records = [
        _response(input_tokens=200, output_tokens=200),
        _response(input_tokens=200, output_tokens=50),
    ]
    violations = check_policy(records, rules)
    assert len(violations) == 1
    assert "650" in violations[0].detail


def test_must_include_text_and_forbidden_text() -> None:
    must_rules = load_policy(
        [{"id": "cite", "kind": "must_include_text", "params": {"text": "SOURCE:"}}]
    )
    forbid_rules = load_policy(
        [{"id": "no-pii", "kind": "forbidden_text", "params": {"text": "SSN:"}}]
    )
    records_bad = [
        _response(content=[{"type": "text", "text": "answer without source"}]),
    ]
    records_good = [
        _response(content=[{"type": "text", "text": "answer with citation. SOURCE: wiki"}]),
    ]
    records_pii = [
        _response(content=[{"type": "text", "text": "leaked SSN: 123-45-6789"}]),
    ]
    assert check_policy(records_bad, must_rules)
    assert check_policy(records_good, must_rules) == []
    assert check_policy(records_pii, forbid_rules)


def test_policy_diff_classifies_regressions_and_fixes() -> None:
    rules = load_policy([{"id": "no-rm", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    # baseline: clean. candidate: calls rm_rf → regression.
    base = [_response()]
    cand = [_response(content=[_tool_use("rm_rf")], stop_reason="tool_use")]
    diff = policy_diff(base, cand, rules)
    assert len(diff.regressions) == 1
    assert len(diff.fixes) == 0
    # Flip it: baseline has violation, candidate clean → fix.
    diff2 = policy_diff(cand, base, rules)
    assert len(diff2.fixes) == 1
    assert len(diff2.regressions) == 0


def test_render_policy_diff_has_labels() -> None:
    rules = load_policy([{"id": "no-rm", "kind": "no_call", "params": {"tool": "rm_rf"}}])
    base = [_response()]
    cand = [_response(content=[_tool_use("rm_rf")], stop_reason="tool_use")]
    diff = policy_diff(base, cand, rules)
    rendered = render_policy_diff(diff)
    assert "regressions" in rendered
    assert "no-rm" in rendered


# ---- session-scoped rules ------------------------------------------------
#
# Multi-session traces (many tickets concatenated — the common production
# shape) require per-session evaluation. The tests below are written against
# a realistic multi-ticket shape: each ticket starts with a chat_request
# whose messages[-1].role == "user", followed by possibly multiple
# (tool_use → tool_result) response turns, then ends with a terminal
# chat_response. Session boundaries are inferred purely from record
# topology — no manual markers needed.


def _request(user_text: str, follow_up_tool_results: list[str] | None = None) -> dict[str, Any]:
    """Build a chat_request. If follow_up_tool_results is non-empty this
    request is a mid-session turn (last message role = tool)."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": user_text})
    for tr in follow_up_tool_results or []:
        msgs.append({"role": "assistant", "content": "thinking"})
        msgs.append({"role": "tool", "content": tr})
    return {"kind": "chat_request", "payload": {"model": "m", "messages": msgs}}


def _session_start_req(user_text: str) -> dict[str, Any]:
    """A pristine session-starting request (last message = user)."""
    return _request(user_text, follow_up_tool_results=None)


def _mid_session_req(user_text: str, tool_result: str) -> dict[str, Any]:
    """A mid-session continuation request (last message = tool result)."""
    return _request(user_text, follow_up_tool_results=[tool_result])


def _ticket(user_text: str, response_contents: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build one full user-initiated ticket: the first request carries the
    user message; each response may be followed by a tool-result continuation
    before the next response. Turn count = len(response_contents).
    """
    out: list[dict[str, Any]] = [_session_start_req(user_text)]
    for i, content in enumerate(response_contents):
        stop = "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
        out.append(
            {
                "kind": "chat_response",
                "payload": {
                    "content": content,
                    "stop_reason": stop,
                    "usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
                },
            }
        )
        if i < len(response_contents) - 1:
            out.append(_mid_session_req(user_text, "tool-result-text"))
    return out


def test_session_scope_must_call_before_catches_every_violating_session() -> None:
    """The canonical adversarial case from the real-world test.

    Ten support tickets. Four correctly call `search_kb` then `escalate`.
    Six skip the KB and escalate directly. Trace-scoped `must_call_before`
    is satisfied by ticket #1 and reports zero violations — that's the bug.
    Session-scoped reports exactly six, one per offending session.
    """
    correct = _ticket(
        "kb-able",
        [
            [_tool_use("search_kb")],
            [_tool_use("escalate")],
        ],
    )
    skip_kb = _ticket("urgent", [[_tool_use("escalate")]])
    trace: list[dict[str, Any]] = []
    # 4 correct tickets, then 6 KB-skippers — matches the real candidate run.
    for _ in range(4):
        trace.extend(correct)
    for _ in range(6):
        trace.extend(skip_kb)

    rule_trace = load_policy(
        [
            {
                "id": "kb-before-escalate",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                "scope": "trace",
            }
        ]
    )
    rule_session = load_policy(
        [
            {
                "id": "kb-before-escalate",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                "scope": "session",
            }
        ]
    )

    assert check_policy(trace, rule_trace) == []  # the old behavior (the bug)
    vs = check_policy(trace, rule_session)
    assert len(vs) == 6
    assert all(v.rule_id == "kb-before-escalate" for v in vs)


def test_session_scope_max_turns_counts_per_session() -> None:
    """max_turns with session scope is a per-ticket turn cap.

    A 5-turn ticket (four mid-session tool-use turns + one terminal
    text reply — the realistic shape of a long agentic loop) bundled
    with three 1-turn tickets should violate a cap of 3 only for the
    5-turn ticket. Intermediate turns must carry ``stop="tool_use"``
    so session-boundary detection recognises them as continuations.
    """
    big_ticket = _ticket(
        "huge",
        [
            [_tool_use("step1")],
            [_tool_use("step2")],
            [_tool_use("step3")],
            [_tool_use("step4")],
            [{"type": "text", "text": "done"}],
        ],
    )
    small = _ticket("s", [[{"type": "text", "text": "t"}]])
    trace = big_ticket + small + small + small
    rules = load_policy(
        [
            {
                "id": "turn-cap",
                "kind": "max_turns",
                "params": {"limit": 3},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1
    assert "5 turns" in vs[0].detail


def test_session_scope_must_call_once_is_per_session() -> None:
    """must_call_once with session scope = called exactly once per session."""
    ok = _ticket("t", [[_tool_use("login")], [{"type": "text", "text": "done"}]])
    dup = _ticket("t", [[_tool_use("login")], [_tool_use("login")]])
    trace = ok + dup + ok  # only middle ticket violates
    rules = load_policy(
        [
            {
                "id": "one-login",
                "kind": "must_call_once",
                "params": {"tool": "login"},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1
    assert "2 times" in vs[0].detail


def test_session_scope_required_stop_reason_checks_each_session_last() -> None:
    """required_stop_reason under session scope must check the final
    response of *each* session, not only the final response of the trace."""
    good = _ticket("a", [[{"type": "text", "text": "hi"}]])  # end_turn
    bad_ticket = [
        _session_start_req("b"),
        {
            "kind": "chat_response",
            "payload": {
                "content": [{"type": "text", "text": "oops"}],
                "stop_reason": "content_filter",
                "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
            },
        },
    ]
    trace = good + bad_ticket + good  # middle ticket ends with refusal
    rules = load_policy(
        [
            {
                "id": "clean-stop",
                "kind": "required_stop_reason",
                "params": {"allowed": ["end_turn"]},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1
    assert "content_filter" in vs[0].detail


def test_session_scope_max_total_tokens_is_per_session_budget() -> None:
    """Session scope on max_total_tokens = per-session budget (fair
    comparison of one heavy ticket against many light ones)."""
    light = [
        _session_start_req("l"),
        {
            "kind": "chat_response",
            "payload": {
                "content": [{"type": "text", "text": "a"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 50, "output_tokens": 50, "thinking_tokens": 0},
            },
        },
    ]
    heavy = [
        _session_start_req("h"),
        {
            "kind": "chat_response",
            "payload": {
                "content": [{"type": "text", "text": "a"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5000, "output_tokens": 5000, "thinking_tokens": 0},
            },
        },
    ]
    trace = light + heavy + light
    rules = load_policy(
        [
            {
                "id": "budget",
                "kind": "max_total_tokens",
                "params": {"limit": 500},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1
    assert "10000" in vs[0].detail


def test_session_scope_preserves_absolute_pair_index_in_violations() -> None:
    """Violations emitted from session scope should still carry the
    trace-absolute pair_index, so reviewers can locate the failing turn."""
    trace = _ticket(
        "t1",
        [
            [_tool_use("escalate")],  # pair_index 0
        ],
    ) + _ticket(
        "t2",
        [
            [_tool_use("escalate")],  # pair_index 1
        ],
    )
    rules = load_policy(
        [
            {
                "id": "kb-before-escalate",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert sorted(v.pair_index for v in vs if v.pair_index is not None) == [0, 1]


def test_session_scope_rejects_invalid_value() -> None:
    with pytest.raises(ShadowConfigError):
        load_policy(
            [
                {
                    "id": "x",
                    "kind": "no_call",
                    "params": {"tool": "rm_rf"},
                    "scope": "weekly",
                }
            ]
        )


def test_trace_scope_default_preserves_legacy_behavior() -> None:
    """Rules without explicit scope behave exactly as before (trace-wide).

    This is the contract existing users depend on; a session-start
    detector must not change the answer for rules that never asked for
    session scope.
    """
    correct = _ticket(
        "c",
        [[_tool_use("search_kb")], [_tool_use("escalate")]],
    )
    skip_kb = _ticket("u", [[_tool_use("escalate")]])
    trace = correct + skip_kb  # session-scope would flag the 2nd ticket
    rules = load_policy(
        [
            {
                "id": "kb-before-escalate",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                # no scope — defaults to "trace"
            }
        ]
    )
    assert check_policy(trace, rules) == []


def test_session_scope_recovers_boundaries_from_terminal_stop_reason() -> None:
    """Even when message history is abbreviated or corrupted — so the
    role-based session-start detector fails — a prior response with a
    non-``tool_use`` stop_reason still marks a session boundary.

    This models imports from foreign tracers that don't preserve the
    full message history and trace-harness bugs where ``messages`` got
    mutated post-recording. Without this fallback, session scope would
    silently degrade to trace scope — the exact bug that hid the
    original 6 violations.
    """

    def _abbreviated_req() -> dict[str, Any]:
        # Last role is "tool" (mid-session marker) — role-based detector
        # won't flag a session start. The stop_reason of the preceding
        # response is the only signal left.
        return {
            "kind": "chat_request",
            "payload": {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "ambiguous"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "t"},
                ],
            },
        }

    def _session_block(tools: list[str]) -> list[dict[str, Any]]:
        """One mid-session-shaped ticket: all requests look like
        continuations, each response calls the listed tool and ends
        with end_turn on the final response."""
        out: list[dict[str, Any]] = []
        for i, t in enumerate(tools):
            out.append(_abbreviated_req())
            stop = "end_turn" if i == len(tools) - 1 else "tool_use"
            out.append(
                {
                    "kind": "chat_response",
                    "payload": {
                        "content": [_tool_use(t)]
                        if t != "text"
                        else [{"type": "text", "text": "ok"}],
                        "stop_reason": stop,
                        "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
                    },
                }
            )
        return out

    # Three tickets: correct, skip-kb (violation), correct.
    trace = (
        _session_block(["search_kb", "escalate", "text"])
        + _session_block(["escalate", "text"])
        + _session_block(["search_kb", "text"])
    )

    rules = load_policy(
        [
            {
                "id": "kb-before-escalate",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1, (
        f"expected 1 violation (ticket 2), got {len(vs)}: "
        f"{[(v.rule_id, v.pair_index, v.detail) for v in vs]}"
    )


def test_explicit_metadata_markers_override_stop_reason_heuristic() -> None:
    """Adapters that emit Session.record_metadata per logical session
    must have their markers treated as authoritative. Without this,
    a trace where every chat pair ends with ``end_turn`` (typical of
    CrewAI's per-LLMCall events) would fragment into one session per
    pair and misfire ``must_call_once`` rules.
    """
    # Two kickoffs, each with two LLM pairs that end with end_turn.
    # Pair 1 of kickoff 1 calls `foo`, pair 2 doesn't. Kickoff 2 has
    # no `foo` calls at all. Under session-scoped ``must_call_once`` on
    # foo: kickoff 1 has exactly one foo call (pass), kickoff 2 has zero
    # (fail). The stop_reason heuristic alone would fragment this into
    # 4 sessions and misreport.

    def _meta(payload: dict) -> dict:
        return {"kind": "metadata", "payload": payload}

    def _req() -> dict:
        return {
            "kind": "chat_request",
            "payload": {
                "model": "m",
                "messages": [{"role": "system", "content": "s"}],
            },
        }

    def _resp(tools: list[str]) -> dict:
        content = [_tool_use(t) for t in tools] or [{"type": "text", "text": "ok"}]
        return {
            "kind": "chat_response",
            "payload": {
                "content": content,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
            },
        }

    trace = [
        _meta({"source": "adapter", "kickoff": "k1"}),
        _req(),
        _resp(["foo"]),
        _req(),
        _resp([]),  # no foo in kickoff 1's second pair
        _meta({"source": "adapter", "kickoff": "k2"}),
        _req(),
        _resp([]),
        _req(),
        _resp([]),  # kickoff 2 has no foo at all → violation
    ]

    rules = load_policy(
        [
            {
                "id": "one-foo-per-kickoff",
                "kind": "must_call_once",
                "params": {"tool": "foo"},
                "scope": "session",
            }
        ]
    )
    vs = check_policy(trace, rules)
    assert len(vs) == 1, f"expected exactly one violation (kickoff 2), got {len(vs)}"
    assert "0 times" in vs[0].detail


def test_explicit_markers_ignored_when_only_one_metadata_record() -> None:
    """A trace with a single metadata record falls back to the
    heuristic detector (one metadata = still a standard SDK trace)."""
    trace = [
        {"kind": "metadata", "payload": {"source": "sdk"}},
        _response(content=[_tool_use("foo")], stop_reason="tool_use"),
    ]
    # Should behave like no explicit markers — single session, foo
    # called once, rule passes.
    rules = load_policy(
        [
            {
                "id": "one-foo",
                "kind": "must_call_once",
                "params": {"tool": "foo"},
                "scope": "session",
            }
        ]
    )
    assert check_policy(trace, rules) == []


def test_session_scope_falls_back_to_single_session_when_no_requests() -> None:
    """Fixture-style traces (just responses, no requests) have no session
    markers; scope=session should degrade to single-session (trace-like)
    behavior rather than crash."""
    records = [
        _response(content=[_tool_use("escalate")], stop_reason="tool_use"),
        _response(content=[_tool_use("search_kb")], stop_reason="tool_use"),
    ]
    rules = load_policy(
        [
            {
                "id": "kb-first",
                "kind": "must_call_before",
                "params": {"first": "search_kb", "then": "escalate"},
                "scope": "session",
            }
        ]
    )
    # same output as trace-scope on this shape (one violation at pair 0).
    vs = check_policy(records, rules)
    assert len(vs) == 1
