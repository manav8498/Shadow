"""Scenario-aware diff: partition records by ``meta.scenario_id``,
run the Rust diff per scenario, return a structured per-scenario report.

Why
---
The Rust core's alignment layer aligns baseline and candidate as a
single contiguous turn sequence. A multi-scenario regression suite
concatenates N distinct test cases into one trace. When scenarios
differ in tool counts or stop reasons, the alignment layer reports
""dropped turns" spuriously — what actually happened is that scenario
3's tools are not the same as scenario 4's, not that the candidate
dropped turns.

External evaluation flagged this as the highest-priority defect in the
v2.5 release: "Top-divergence on heterogeneous multi-tool traces
explains as 'dropped turns' instead of per-scenario diffs."

How
---
Each record carries an open-ended ``meta`` dict in the agentlog format
(SPEC §3.1). Authors of multi-scenario suites set
``meta.scenario_id`` on every record belonging to a given test case.
This module partitions baseline and candidate records by that key,
runs ``shadow._core.compute_diff_report`` once per scenario, and
returns a :class:`MultiScenarioReport` that aggregates the per-scenario
results.

Records without a ``scenario_id`` fall into a synthetic
``__default__`` bucket. So a single-scenario trace returns a
one-element report and the API is backward-compatible.

Usage
-----
    from shadow.diff_py import compute_multi_scenario_report

    report = compute_multi_scenario_report(
        baseline_records=load("baseline.agentlog"),
        candidate_records=load("candidate.agentlog"),
    )
    for sc in report.scenarios:
        print(sc.scenario_id, sc.diff["summary"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shadow import _core

DEFAULT_SCENARIO_ID = "__default__"
"""Bucket used for records that do not carry a ``meta.scenario_id``."""


def _scenario_id_of(record: dict[str, Any]) -> str:
    """Return the scenario_id for a record, or DEFAULT_SCENARIO_ID."""
    meta = record.get("meta") or {}
    if isinstance(meta, dict):
        sid = meta.get("scenario_id")
        if isinstance(sid, str) and sid:
            return sid
    return DEFAULT_SCENARIO_ID


def partition_by_scenario(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group records by their ``meta.scenario_id``.

    Preserves order: records appear in each bucket in the same relative
    order as in the input. Records without a scenario_id are assigned
    to :data:`DEFAULT_SCENARIO_ID`.

    Returns
    -------
    Mapping ``scenario_id -> [record, ...]``. Insertion order matches the
    order in which each scenario_id first appears in the input.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        sid = _scenario_id_of(rec)
        buckets.setdefault(sid, []).append(rec)
    return buckets


@dataclass
class ScenarioDiff:
    """Diff result for a single scenario_id."""

    scenario_id: str
    """The scenario_id this diff covers (DEFAULT_SCENARIO_ID for un-tagged
    records)."""
    diff: dict[str, Any]
    """The DiffReport dict produced by ``shadow._core.compute_diff_report``."""
    n_baseline_records: int
    n_candidate_records: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "diff": self.diff,
            "n_baseline_records": self.n_baseline_records,
            "n_candidate_records": self.n_candidate_records,
        }


@dataclass
class MultiScenarioReport:
    """Per-scenario diffs for a multi-scenario regression suite."""

    scenarios: list[ScenarioDiff] = field(default_factory=list)
    """Per-scenario diff results, in the order scenarios first appeared
    in the baseline trace."""
    baseline_only_scenarios: list[str] = field(default_factory=list)
    """Scenario IDs present in baseline but missing from candidate."""
    candidate_only_scenarios: list[str] = field(default_factory=list)
    """Scenario IDs present in candidate but missing from baseline."""

    @property
    def is_single_scenario(self) -> bool:
        """True if the input had exactly one scenario (or no scenario_ids
        at all). Backward-compatibility flag for callers that want to
        decide whether to render the legacy single-scenario format or
        the new multi-section format."""
        return len(self.scenarios) <= 1 and not self.baseline_only_scenarios

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "baseline_only_scenarios": self.baseline_only_scenarios,
            "candidate_only_scenarios": self.candidate_only_scenarios,
        }

    @property
    def total_baseline_records(self) -> int:
        return sum(s.n_baseline_records for s in self.scenarios)

    @property
    def total_candidate_records(self) -> int:
        return sum(s.n_candidate_records for s in self.scenarios)


def compute_multi_scenario_report(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> MultiScenarioReport:
    """Run a per-scenario diff over a multi-scenario regression suite.

    Partitions both sides by ``meta.scenario_id``. For each scenario
    present in *both* sides, runs the standard Rust diff. Reports
    scenario IDs that appear in only one side separately so the caller
    can flag a missing test case (a real regression) rather than burying
    it as a "dropped turns" message.

    Parameters
    ----------
    baseline_records : agentlog records (dicts) for the baseline run(s).
    candidate_records : agentlog records for the candidate run(s).
    pricing : optional model -> (price_per_input_tok, price_per_output_tok).
    seed : optional RNG seed for reproducible bootstrap CIs.

    Returns
    -------
    MultiScenarioReport with one ScenarioDiff per shared scenario_id.

    Backward compatibility: if neither side carries any scenario_id,
    both partition into a single :data:`DEFAULT_SCENARIO_ID` bucket and
    the returned report has exactly one entry, equivalent to calling
    ``shadow._core.compute_diff_report`` directly.
    """
    bp = partition_by_scenario(baseline_records)
    cp = partition_by_scenario(candidate_records)

    baseline_keys = set(bp.keys())
    candidate_keys = set(cp.keys())
    shared = baseline_keys & candidate_keys

    # Preserve baseline insertion order for shared scenarios; the
    # baseline is the authoritative ordering source.
    ordered_shared = [k for k in bp if k in shared]

    scenarios: list[ScenarioDiff] = []
    for sid in ordered_shared:
        b = bp[sid]
        c = cp[sid]
        diff = _core.compute_diff_report(b, c, pricing, seed)
        scenarios.append(
            ScenarioDiff(
                scenario_id=sid,
                diff=diff,
                n_baseline_records=len(b),
                n_candidate_records=len(c),
            )
        )

    baseline_only = sorted(baseline_keys - candidate_keys)
    candidate_only = sorted(candidate_keys - baseline_keys)

    return MultiScenarioReport(
        scenarios=scenarios,
        baseline_only_scenarios=baseline_only,
        candidate_only_scenarios=candidate_only,
    )
