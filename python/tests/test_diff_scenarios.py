"""Tests for shadow.diff_py.scenarios — scenario-aware multi-case diff.

Regression target: the external real-world stress evaluation reported
that multi-scenario regression suites with heterogeneous tool sequences
get reduced to spurious "dropped turns" messages, because the Rust
alignment layer aligns the entire trace as one continuous flow without
knowing where one scenario ends and the next begins.

This test suite verifies that ``compute_multi_scenario_report``:

  - partitions correctly by ``meta.scenario_id``
  - runs the Rust diff once per shared scenario
  - reports baseline-only / candidate-only scenarios separately
  - is backward-compatible for single-scenario traces (no
    scenario_ids → one ``__default__`` bucket)
  - never produces "dropped turns" messages purely because two
    different scenarios happen to have different tool sequences
"""

from __future__ import annotations

from typing import Any

from shadow.diff_py import (
    DEFAULT_SCENARIO_ID,
    MultiScenarioReport,
    compute_multi_scenario_report,
    partition_by_scenario,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_NULL_ID = "sha256:" + "0" * 64
_TS = "2026-04-28T00:00:00.000Z"


def _record(
    kind: str,
    payload: dict[str, Any] | None,
    *,
    idx: int,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if scenario_id is not None:
        meta["scenario_id"] = scenario_id
    return {
        "version": "0.1",
        "id": f"sha256:{idx:064x}",
        "kind": kind,
        "ts": _TS,
        "parent": _NULL_ID,
        "meta": meta,
        "payload": payload,
    }


def _chat_response(text: str, *, idx: int, scenario_id: str | None = None) -> dict[str, Any]:
    return _record(
        "chat_response",
        {
            "model": "test-model",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 100,
            "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
        },
        idx=idx,
        scenario_id=scenario_id,
    )


def _chat_request(prompt: str, *, idx: int, scenario_id: str | None = None) -> dict[str, Any]:
    return _record(
        "chat_request",
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
            "params": {"temperature": 0.0, "max_tokens": 64},
            "tools": [],
        },
        idx=idx,
        scenario_id=scenario_id,
    )


def _pair(
    prompt: str,
    response: str,
    *,
    base_idx: int,
    scenario_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _chat_request(prompt, idx=base_idx, scenario_id=scenario_id),
        _chat_response(response, idx=base_idx + 1, scenario_id=scenario_id),
    ]


# ---------------------------------------------------------------------------
# partition_by_scenario
# ---------------------------------------------------------------------------


class TestPartitionByScenario:
    def test_no_scenario_id_falls_into_default_bucket(self):
        records = _pair("hi", "hello", base_idx=1)
        buckets = partition_by_scenario(records)
        assert list(buckets.keys()) == [DEFAULT_SCENARIO_ID]
        assert len(buckets[DEFAULT_SCENARIO_ID]) == 2

    def test_partitions_by_scenario_id(self):
        records = (
            _pair("a", "A", base_idx=1, scenario_id="sc1")
            + _pair("b", "B", base_idx=10, scenario_id="sc2")
            + _pair("c", "C", base_idx=20, scenario_id="sc1")
        )
        buckets = partition_by_scenario(records)
        assert set(buckets.keys()) == {"sc1", "sc2"}
        assert len(buckets["sc1"]) == 4  # 2 pairs interleaved
        assert len(buckets["sc2"]) == 2

    def test_preserves_record_order_within_bucket(self):
        records = [
            _chat_response("first", idx=1, scenario_id="x"),
            _chat_response("middle", idx=2, scenario_id="y"),
            _chat_response("last", idx=3, scenario_id="x"),
        ]
        buckets = partition_by_scenario(records)
        texts = [r["payload"]["content"][0]["text"] for r in buckets["x"]]
        assert texts == ["first", "last"]

    def test_mixed_scenario_and_default(self):
        records = [
            _chat_response("untagged", idx=1),  # no scenario_id
            _chat_response("tagged", idx=2, scenario_id="sc1"),
        ]
        buckets = partition_by_scenario(records)
        assert DEFAULT_SCENARIO_ID in buckets
        assert "sc1" in buckets
        assert len(buckets[DEFAULT_SCENARIO_ID]) == 1
        assert len(buckets["sc1"]) == 1

    def test_empty_meta_falls_into_default_bucket(self):
        rec = _chat_response("x", idx=1)
        rec["meta"] = {}  # explicit empty
        buckets = partition_by_scenario([rec])
        assert DEFAULT_SCENARIO_ID in buckets

    def test_non_string_scenario_id_falls_into_default(self):
        rec = _chat_response("x", idx=1)
        rec["meta"] = {"scenario_id": 42}  # not a string
        buckets = partition_by_scenario([rec])
        assert DEFAULT_SCENARIO_ID in buckets


# ---------------------------------------------------------------------------
# compute_multi_scenario_report — backward compat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_single_scenario_report_has_one_section(self):
        baseline = _pair("hi", "hello", base_idx=1)
        candidate = _pair("hi", "hello world", base_idx=10)
        report = compute_multi_scenario_report(baseline, candidate)
        assert isinstance(report, MultiScenarioReport)
        assert len(report.scenarios) == 1
        assert report.scenarios[0].scenario_id == DEFAULT_SCENARIO_ID
        assert report.is_single_scenario is True

    def test_single_scenario_diff_matches_legacy_core_call(self):
        from shadow import _core

        baseline = _pair("hi", "hello", base_idx=1)
        candidate = _pair("hi", "hi there", base_idx=10)
        legacy = _core.compute_diff_report(baseline, candidate, None, None)
        report = compute_multi_scenario_report(baseline, candidate)
        # The single-scenario diff should be identical to the direct
        # core call: same rows, same recommendations, same pair count.
        sc_diff = report.scenarios[0].diff
        assert sc_diff["rows"] == legacy["rows"]
        assert sc_diff["pair_count"] == legacy["pair_count"]
        assert sc_diff["recommendations"] == legacy["recommendations"]


# ---------------------------------------------------------------------------
# compute_multi_scenario_report — multi-scenario behavior
# ---------------------------------------------------------------------------


class TestMultiScenario:
    def test_three_scenarios_produce_three_sections(self):
        baseline = (
            _pair("p1", "r1", base_idx=1, scenario_id="sc1")
            + _pair("p2", "r2", base_idx=10, scenario_id="sc2")
            + _pair("p3", "r3", base_idx=20, scenario_id="sc3")
        )
        candidate = (
            _pair("p1", "r1-modified", base_idx=100, scenario_id="sc1")
            + _pair("p2", "r2-modified", base_idx=110, scenario_id="sc2")
            + _pair("p3", "r3-modified", base_idx=120, scenario_id="sc3")
        )
        report = compute_multi_scenario_report(baseline, candidate)
        assert len(report.scenarios) == 3
        ids = [s.scenario_id for s in report.scenarios]
        assert ids == ["sc1", "sc2", "sc3"]  # baseline order preserved

    def test_each_scenario_section_has_independent_diff(self):
        baseline = _pair("p1", "r1", base_idx=1, scenario_id="sc1") + _pair(
            "p2", "r2", base_idx=10, scenario_id="sc2"
        )
        # Only sc1 has a regression.
        candidate = _pair("p1", "r1-MODIFIED", base_idx=100, scenario_id="sc1") + _pair(
            "p2", "r2", base_idx=110, scenario_id="sc2"
        )
        report = compute_multi_scenario_report(baseline, candidate)
        assert len(report.scenarios) == 2
        sc1, sc2 = report.scenarios
        assert sc1.scenario_id == "sc1"
        assert sc2.scenario_id == "sc2"
        # Each scenario has its own independent diff structure: rows
        # (per-axis stats), pair_count, recommendations. The structure
        # is independent — regression in sc1 doesn't bleed into sc2.
        for sc in (sc1, sc2):
            assert isinstance(sc.diff, dict)
            assert "rows" in sc.diff
            assert "pair_count" in sc.diff
            assert isinstance(sc.diff["rows"], list)

    def test_baseline_only_scenarios_reported(self):
        baseline = _pair("p1", "r1", base_idx=1, scenario_id="sc1") + _pair(
            "p2", "r2", base_idx=10, scenario_id="sc2"
        )
        # Candidate is missing sc2 entirely (regressed test case).
        candidate = _pair("p1", "r1", base_idx=100, scenario_id="sc1")
        report = compute_multi_scenario_report(baseline, candidate)
        assert report.baseline_only_scenarios == ["sc2"]
        assert report.candidate_only_scenarios == []
        assert len(report.scenarios) == 1
        assert report.scenarios[0].scenario_id == "sc1"

    def test_candidate_only_scenarios_reported(self):
        baseline = _pair("p1", "r1", base_idx=1, scenario_id="sc1")
        candidate = _pair("p1", "r1", base_idx=100, scenario_id="sc1") + _pair(
            "p2", "r2", base_idx=110, scenario_id="sc-new"
        )
        report = compute_multi_scenario_report(baseline, candidate)
        assert report.baseline_only_scenarios == []
        assert report.candidate_only_scenarios == ["sc-new"]
        assert len(report.scenarios) == 1

    def test_record_counts_per_scenario(self):
        baseline = _pair("p1", "r1", base_idx=1, scenario_id="sc1") + _pair(
            "p2", "r2", base_idx=10, scenario_id="sc2"
        )
        candidate = _pair("p1", "r1", base_idx=100, scenario_id="sc1") + _pair(
            "p2", "r2-MOD", base_idx=110, scenario_id="sc2"
        )
        report = compute_multi_scenario_report(baseline, candidate)
        sc1 = report.scenarios[0]
        assert sc1.n_baseline_records == 2
        assert sc1.n_candidate_records == 2
        assert report.total_baseline_records == 4
        assert report.total_candidate_records == 4

    def test_to_dict_round_trip(self):
        baseline = _pair("p1", "r1", base_idx=1, scenario_id="sc1")
        candidate = _pair("p1", "r1-MOD", base_idx=100, scenario_id="sc1")
        report = compute_multi_scenario_report(baseline, candidate)
        d = report.to_dict()
        assert "scenarios" in d
        assert "baseline_only_scenarios" in d
        assert "candidate_only_scenarios" in d
        assert len(d["scenarios"]) == 1
        assert d["scenarios"][0]["scenario_id"] == "sc1"


# ---------------------------------------------------------------------------
# Regression for the specific "dropped turns" misclassification
# ---------------------------------------------------------------------------


class TestDroppedTurnsRegression:
    def test_heterogeneous_scenarios_no_spurious_dropped_turns(self):
        """Two scenarios with different tool sequences. Without
        scenario_id partitioning, the alignment layer would emit
        'dropped turns' messages. With partitioning, each scenario is
        diffed independently and reports its own (correct) findings."""
        # Scenario 1: simple text response.
        # Scenario 2: response with one tool call.
        scenario1_baseline = _pair("query A", "answer A", base_idx=1, scenario_id="text-only")
        scenario2_baseline = [
            _chat_request("query B", idx=10, scenario_id="with-tool"),
            _record(
                "chat_response",
                {
                    "model": "test-model",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                    "latency_ms": 100,
                    "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
                },
                idx=11,
                scenario_id="with-tool",
            ),
        ]
        baseline = scenario1_baseline + scenario2_baseline

        # Candidate has both scenarios but each is slightly different.
        scenario1_candidate = _pair(
            "query A", "answer A modified", base_idx=100, scenario_id="text-only"
        )
        scenario2_candidate = [
            _chat_request("query B", idx=110, scenario_id="with-tool"),
            _record(
                "chat_response",
                {
                    "model": "test-model",
                    "content": [
                        {"type": "tool_use", "id": "t2", "name": "lookup", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                    "latency_ms": 200,  # latency drift
                    "usage": {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0},
                },
                idx=111,
                scenario_id="with-tool",
            ),
        ]
        candidate = scenario1_candidate + scenario2_candidate

        report = compute_multi_scenario_report(baseline, candidate)
        assert len(report.scenarios) == 2

        # In each scenario's recommendations / first_divergence, "dropped"
        # should NOT appear merely because the scenarios have different
        # tool sequences — that classification is reserved for actual
        # drops within the same scenario.
        for sc in report.scenarios:
            recs = sc.diff.get("recommendations", []) or []
            divergences = sc.diff.get("first_divergences", []) or []
            for r in recs:
                msg = (r.get("message") or "").lower()
                rationale = (r.get("rationale") or "").lower()
                assert "dropped tool" not in msg, (
                    f"scenario {sc.scenario_id!r}: spurious 'dropped tool' "
                    f"in recommendation message: {msg}"
                )
                assert "dropped tool" not in rationale, (
                    f"scenario {sc.scenario_id!r}: spurious 'dropped tool' "
                    f"in rationale: {rationale}"
                )
            for dv in divergences:
                exp = (dv.get("explanation") or "").lower()
                # Within a scenario, "dropped tool" is allowed only when
                # the scenario actually drops a tool. Our test scenarios
                # don't drop any tools, just slightly mutate them.
                assert "dropped tool" not in exp, (
                    f"scenario {sc.scenario_id!r}: spurious 'dropped tool' "
                    f"in divergence explanation: {exp}"
                )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_inputs(self):
        report = compute_multi_scenario_report([], [])
        assert report.scenarios == []
        assert report.baseline_only_scenarios == []
        assert report.candidate_only_scenarios == []

    def test_baseline_empty_candidate_has_scenarios(self):
        candidate = _pair("p", "r", base_idx=1, scenario_id="sc1")
        report = compute_multi_scenario_report([], candidate)
        assert report.candidate_only_scenarios == ["sc1"]
        assert report.scenarios == []

    def test_disjoint_scenario_sets(self):
        baseline = _pair("p", "r", base_idx=1, scenario_id="sc-a")
        candidate = _pair("p", "r", base_idx=10, scenario_id="sc-b")
        report = compute_multi_scenario_report(baseline, candidate)
        assert report.scenarios == []  # no shared scenarios
        assert report.baseline_only_scenarios == ["sc-a"]
        assert report.candidate_only_scenarios == ["sc-b"]
