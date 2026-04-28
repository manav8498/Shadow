"""Tests for shadow.policy_suggest — mine traces for must_call_before rules."""

from __future__ import annotations

from typing import Any

from shadow.policy_suggest import PolicySuggestion, suggest_policies


def _record(
    kind: str,
    payload: dict[str, Any] | None,
    *,
    idx: int,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if scenario_id:
        meta["scenario_id"] = scenario_id
    return {
        "version": "0.1",
        "id": f"sha256:{idx:064x}",
        "kind": kind,
        "ts": "2026-04-28T00:00:00.000Z",
        "parent": "sha256:" + "0" * 64,
        "meta": meta,
        "payload": payload,
    }


def _response_with_tools(
    tools: list[str], *, idx: int, scenario_id: str | None = None
) -> dict[str, Any]:
    return _record(
        "chat_response",
        {
            "model": "test",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": t, "input": {}}
                for i, t in enumerate(tools)
            ],
            "stop_reason": "tool_use",
            "latency_ms": 100,
            "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
        },
        idx=idx,
        scenario_id=scenario_id,
    )


def _scenario(
    tool_sequences_per_turn: list[list[str]], *, base_idx: int, scenario_id: str
) -> list[dict[str, Any]]:
    return [
        _response_with_tools(tools, idx=base_idx + i, scenario_id=scenario_id)
        for i, tools in enumerate(tool_sequences_per_turn)
    ]


# ---------------------------------------------------------------------------
# Basic suggestion shape
# ---------------------------------------------------------------------------


class TestSuggestionShape:
    def test_returns_list_of_policysuggestion(self):
        suggestions = suggest_policies([])
        assert suggestions == []

    def test_no_tool_calls_no_suggestions(self):
        # Three scenarios with no tools at all — nothing to suggest.
        records = [
            _record(
                "chat_response",
                {
                    "model": "x",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                    "latency_ms": 1,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
                },
                idx=i,
                scenario_id=f"sc{i}",
            )
            for i in range(3)
        ]
        suggestions = suggest_policies(records)
        assert suggestions == []


# ---------------------------------------------------------------------------
# Mining must_call_before patterns
# ---------------------------------------------------------------------------


class TestMustCallBeforeMining:
    def test_consistent_ordering_across_scenarios(self):
        """In 5 scenarios A always precedes B → suggest A-before-B with confidence 1.0."""
        records: list[dict[str, Any]] = []
        for i in range(5):
            records.extend(
                _scenario(
                    [["verify_user"], ["check_balance"], ["transfer_funds"]],
                    base_idx=i * 100,
                    scenario_id=f"sc-{i}",
                )
            )
        suggestions = suggest_policies(records)
        # Multiple pairs satisfy the rule (verify-before-check, verify-before-transfer,
        # check-before-transfer); all should be suggested.
        rule_ids = {s.rule_id for s in suggestions}
        assert "verify_user-before-check_balance" in rule_ids
        assert "verify_user-before-transfer_funds" in rule_ids
        assert "check_balance-before-transfer_funds" in rule_ids
        # All should have confidence 1.0
        for s in suggestions:
            assert s.confidence == 1.0

    def test_inconsistent_ordering_drops_below_min_consistency(self):
        """If A precedes B in only 2/4 scenarios, default min_consistency=1.0
        rejects the suggestion."""
        records = [
            *_scenario([["A"], ["B"]], base_idx=0, scenario_id="s1"),
            *_scenario([["A"], ["B"]], base_idx=10, scenario_id="s2"),
            *_scenario([["B"], ["A"]], base_idx=20, scenario_id="s3"),
            *_scenario([["B"], ["A"]], base_idx=30, scenario_id="s4"),
        ]
        suggestions = suggest_policies(records)
        # Neither direction is consistent → no suggestions.
        rule_ids = {s.rule_id for s in suggestions}
        assert "A-before-B" not in rule_ids
        assert "B-before-A" not in rule_ids

    def test_min_support_filters_pairs_with_few_observations(self):
        """A pair appearing in only 2 scenarios is below min_support=3 → no suggestion."""
        records = [
            *_scenario([["A"], ["B"]], base_idx=0, scenario_id="s1"),
            *_scenario([["A"], ["B"]], base_idx=10, scenario_id="s2"),
        ]
        suggestions = suggest_policies(records)  # default min_support=3
        assert all(s.rule_id != "A-before-B" for s in suggestions)

    def test_relax_min_support_yields_more(self):
        records = [
            *_scenario([["A"], ["B"]], base_idx=0, scenario_id="s1"),
            *_scenario([["A"], ["B"]], base_idx=10, scenario_id="s2"),
        ]
        suggestions = suggest_policies(records, min_support=2)
        rule_ids = {s.rule_id for s in suggestions}
        assert "A-before-B" in rule_ids

    def test_relax_consistency_yields_more(self):
        """3 of 4 scenarios have A before B → confidence 0.75 → only suggested
        when min_consistency drops below 0.75."""
        records = [
            *_scenario([["A"], ["B"]], base_idx=0, scenario_id="s1"),
            *_scenario([["A"], ["B"]], base_idx=10, scenario_id="s2"),
            *_scenario([["A"], ["B"]], base_idx=20, scenario_id="s3"),
            *_scenario([["B"], ["A"]], base_idx=30, scenario_id="s4"),
        ]
        # Default min_consistency=1.0 → no suggestion.
        assert all(s.rule_id != "A-before-B" for s in suggest_policies(records))
        # Relax to 0.7 → one suggestion at 0.75 confidence.
        relaxed = suggest_policies(records, min_consistency=0.7)
        a_before_b = [s for s in relaxed if s.rule_id == "A-before-B"]
        assert len(a_before_b) == 1
        assert a_before_b[0].confidence == 0.75

    def test_no_scenario_ids_treats_whole_trace_as_one(self):
        """When records have no scenario_id, the whole trace is one
        scenario. With only one scenario we can't satisfy min_support=3."""
        records = _scenario(
            [["A"], ["B"], ["C"]],
            base_idx=0,
            scenario_id=None,  # type: ignore[arg-type]
        )
        # The helper still adds scenario_id=None; remove the meta for this test.
        for r in records:
            r["meta"] = {}
        suggestions = suggest_policies(records)
        assert suggestions == []  # only 1 scenario, n_both < 3


# ---------------------------------------------------------------------------
# Output sorting and rationale
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_sorted_by_support_then_confidence(self):
        # 5 scenarios where A precedes B; 3 scenarios where C precedes D.
        records: list[dict[str, Any]] = []
        for i in range(5):
            records.extend(_scenario([["A"], ["B"]], base_idx=i * 100, scenario_id=f"ab-{i}"))
        for i in range(3):
            records.extend(_scenario([["C"], ["D"]], base_idx=600 + i * 100, scenario_id=f"cd-{i}"))
        suggestions = suggest_policies(records)
        # First entry should be the higher-support pair.
        assert suggestions[0].rule_id == "A-before-B"

    def test_rationale_includes_counts_and_percentage(self):
        records = []
        for i in range(5):
            records.extend(_scenario([["A"], ["B"]], base_idx=i * 100, scenario_id=f"s-{i}"))
        suggestions = suggest_policies(records)
        rationales = [s.rationale for s in suggestions if s.rule_id == "A-before-B"]
        assert rationales
        text = rationales[0]
        assert "5/5" in text
        assert "100%" in text

    def test_params_match_must_call_before_kind(self):
        records = []
        for i in range(5):
            records.extend(_scenario([["A"], ["B"]], base_idx=i * 100, scenario_id=f"s-{i}"))
        suggestions = suggest_policies(records)
        ab = next((s for s in suggestions if s.rule_id == "A-before-B"), None)
        assert ab is not None
        assert ab.kind == "must_call_before"
        assert ab.params == {"first": "A", "then": "B"}

    def test_policysuggestion_is_immutable(self):
        s = PolicySuggestion(
            rule_id="x-before-y",
            kind="must_call_before",
            params={"first": "x", "then": "y"},
            confidence=1.0,
            n_both=3,
            n_ordered=3,
            rationale="ok",
        )
        try:
            s.confidence = 0.5  # type: ignore[misc]
        except (AttributeError, TypeError):
            return  # frozen dataclass — expected
        raise AssertionError("PolicySuggestion should be frozen")
