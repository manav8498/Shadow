"""Tests for `shadow.diagnose_pr.report`.

`build_report` is the v0.1 skeleton — it consumes loader output and
delta list, produces a DiagnosePrReport with trivial verdict logic
(real classification arrives in Week 2). `to_json` serialises it
with stable key order so PR-comment diffs are minimal."""

from __future__ import annotations

import json
from pathlib import Path

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.deltas import extract_deltas
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.report import build_report, to_json


def _t(idx: int) -> LoadedTrace:
    return LoadedTrace(
        path=Path(f"/tmp/{idx}.agentlog"),
        trace_id=f"sha256:{idx:064d}",
        records=[
            {"id": f"sha256:{idx:064d}", "kind": "metadata", "payload": {}},
        ],
    )


def test_zero_traces_is_ship_verdict() -> None:
    r = build_report(traces=[], deltas=[], affected_trace_ids=set())
    assert r.verdict == "ship"
    assert r.total_traces == 0
    assert r.affected_traces == 0
    assert r.blast_radius == 0.0
    assert r.dominant_cause is None
    assert r.flags == []


def test_traces_with_zero_affected_is_still_ship() -> None:
    r = build_report(
        traces=[_t(1), _t(2), _t(3)],
        deltas=[],
        affected_trace_ids=set(),
    )
    assert r.verdict == "ship"
    assert r.total_traces == 3
    assert r.affected_traces == 0
    assert r.blast_radius == 0.0


def test_affected_traces_with_no_severe_or_dangerous_is_hold() -> None:
    """Week 2: classify_verdict returns `hold` when affected > 0
    and no severe axis / no dangerous violation. Week 3 will
    distinguish `probe` from `hold` by causal CI excluding zero."""
    traces = [_t(1), _t(2), _t(3)]
    r = build_report(traces=traces, deltas=[], affected_trace_ids={traces[0].trace_id})
    assert r.verdict == "hold"
    assert r.affected_traces == 1
    assert r.blast_radius > 0


def test_low_power_flag_when_n_below_30() -> None:
    r = build_report(traces=[_t(1)], deltas=[], affected_trace_ids=set())
    assert "low_power" in r.flags


def test_no_low_power_flag_when_n_at_least_30() -> None:
    traces = [_t(i) for i in range(30)]
    r = build_report(traces=traces, deltas=[], affected_trace_ids=set())
    assert "low_power" not in r.flags


def test_to_json_includes_schema_version_and_keys_sorted() -> None:
    r = build_report(traces=[_t(1)], deltas=[], affected_trace_ids=set())
    blob = to_json(r)
    parsed = json.loads(blob)
    assert parsed["schema_version"] == SCHEMA_VERSION
    # Stable key order (alphabetical) so PR-side diffs are minimal.
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_to_json_round_trips_dataclass_field_set() -> None:
    deltas = extract_deltas({"model": "a"}, {"model": "b"})
    r = build_report(
        traces=[_t(1), _t(2)],
        deltas=deltas,
        affected_trace_ids={f"sha256:{1:064d}"},
    )
    parsed = json.loads(to_json(r))
    expected_keys = {
        "schema_version",
        "verdict",
        "total_traces",
        "affected_traces",
        "blast_radius",
        "dominant_cause",
        "top_causes",
        "trace_diagnoses",
        "affected_trace_ids",
        "new_policy_violations",
        "worst_policy_rule",
        "suggested_fix",
        "flags",
        # v3.2.2 added two top-level booleans mirroring entries in
        # `flags`. They're derivable from the dataclass but surfaced
        # at the top level so CI pipelines / dashboards don't have to
        # string-match the per-row flag list.
        "is_synthetic",
        "low_statistical_power",
    }
    assert set(parsed.keys()) == expected_keys
    assert parsed["affected_trace_ids"] == [f"sha256:{1:064d}"]
