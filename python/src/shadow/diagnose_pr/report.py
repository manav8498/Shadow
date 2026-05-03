"""DiagnosePrReport assembly + JSON serialisation.

Week 1 skeleton: trivial ship-or-probe verdict from a flat
`affected_trace_ids` set.

Week 2 (this version): real verdict from
`shadow.diagnose_pr.risk.classify_verdict`, taking a list of
`TraceDiagnosis` directly so the per-trace state (worst_axis,
first_divergence, policy_violations) flows through to the JSON.

The Week 1 entry point — `build_report(... affected_trace_ids=)` —
is preserved so existing callers (and tests) keep working. The new
Week 2 entry point is `build_report(... diagnoses=)`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict

from shadow.diagnose_pr import SCHEMA_VERSION
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.models import (
    ConfigDelta,
    DiagnosePrReport,
    TraceDiagnosis,
)
from shadow.diagnose_pr.risk import classify_verdict

_LOW_POWER_THRESHOLD = 30


def build_report(
    *,
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    diagnoses: list[TraceDiagnosis] | None = None,
    affected_trace_ids: set[str] | None = None,
    new_policy_violations: int = 0,
    worst_policy_rule: str | None = None,
    has_dangerous_violation: bool = False,
    has_severe_axis: bool = False,
) -> DiagnosePrReport:
    """Assemble a DiagnosePrReport.

    Two call modes:

      * Week 2 path: pass `diagnoses=[TraceDiagnosis, ...]` already
        carrying per-trace `affected`, `worst_axis`, etc. The
        verdict is computed from those + the policy/axis flags.

      * Week 1 path: pass `affected_trace_ids={...}`. We build
        skeleton TraceDiagnosis entries (no per-trace detail) so
        the JSON shape is uniform.

    Exactly one of `diagnoses` / `affected_trace_ids` must be
    supplied; passing both raises a ValueError.
    """
    del deltas  # v0.1: not yet ranked into top_causes (Week 3)

    if diagnoses is not None and affected_trace_ids is not None:
        raise ValueError("pass exactly one of `diagnoses` or `affected_trace_ids`")

    if diagnoses is None:
        ids = affected_trace_ids or set()
        diagnoses = [
            TraceDiagnosis(
                trace_id=t.trace_id,
                affected=t.trace_id in ids,
                risk=0.0,
                worst_axis=None,
                first_divergence=None,
                policy_violations=[],
            )
            for t in traces
        ]

    total = len(traces)
    affected_trace_set = {d.trace_id for d in diagnoses if d.affected}
    affected = len(affected_trace_set)
    blast_radius = (affected / total) if total > 0 else 0.0

    verdict = classify_verdict(
        affected=affected,
        total=total,
        has_dangerous_violation=has_dangerous_violation,
        has_severe_axis=has_severe_axis,
    )

    flags: list[str] = []
    if 0 < total < _LOW_POWER_THRESHOLD:
        flags.append("low_power")

    return DiagnosePrReport(
        schema_version=SCHEMA_VERSION,
        verdict=verdict,
        total_traces=total,
        affected_traces=affected,
        blast_radius=blast_radius,
        dominant_cause=None,  # Week 3
        top_causes=[],  # Week 3
        trace_diagnoses=diagnoses,
        affected_trace_ids=sorted(affected_trace_set),
        new_policy_violations=new_policy_violations,
        worst_policy_rule=worst_policy_rule,
        suggested_fix=None,  # Week 3
        flags=flags,
    )


def to_json(report: DiagnosePrReport, *, indent: int = 2) -> str:
    """Serialise a report with sorted keys (so PR-side diffs are
    minimal) and the requested indent.
    """
    return json.dumps(asdict(report), sort_keys=True, indent=indent, ensure_ascii=False)


def report_from_traces_and_deltas(
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    *,
    affected: Iterable[str] = (),
) -> DiagnosePrReport:
    """Convenience wrapper used by the Week 1 skeleton CLI. Accepts
    an iterable of trace ids for the affected set."""
    return build_report(
        traces=traces,
        deltas=deltas,
        affected_trace_ids=set(affected),
    )


__all__ = [
    "build_report",
    "report_from_traces_and_deltas",
    "to_json",
]
