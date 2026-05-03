"""DiagnosePrReport assembly + JSON serialisation.

`build_report` is the v0.1 skeleton: it consumes loader output
(traces) plus a delta list and an "affected" trace-id set, and
produces a DiagnosePrReport. The verdict logic here is intentionally
trivial:

  * 0 affected            -> ship
  * any affected, no CI   -> probe   (Week 3 will promote to hold
                                      once causal CI excludes zero)

Real verdict logic — `hold` from CI excluding zero, `stop` from
dangerous-tool policy violations — arrives in Week 2 (`risk.py`)
without changing the v0.1 schema.
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
    Verdict,
)

_LOW_POWER_THRESHOLD = 30


def build_report(
    *,
    traces: list[LoadedTrace],
    deltas: list[ConfigDelta],
    affected_trace_ids: set[str],
    new_policy_violations: int = 0,
    worst_policy_rule: str | None = None,
) -> DiagnosePrReport:
    """Assemble a DiagnosePrReport from skeleton inputs.

    `deltas` is currently used for record-keeping (top_causes is
    empty until Week 3). Keeping the parameter in the signature
    means the CLI can wire it up now and Week 3 only changes the
    body of this function.
    """
    del deltas  # v0.1 skeleton: not yet ranked into top_causes

    total = len(traces)
    affected = len(affected_trace_ids)
    blast_radius = (affected / total) if total > 0 else 0.0
    verdict: Verdict = "ship" if affected == 0 else "probe"

    diagnoses = [
        TraceDiagnosis(
            trace_id=t.trace_id,
            affected=t.trace_id in affected_trace_ids,
            risk=0.0,
            worst_axis=None,
            first_divergence=None,
            policy_violations=[],
        )
        for t in traces
    ]

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
        affected_trace_ids=sorted(affected_trace_ids),
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
    """Convenience wrapper used by the CLI command. Accepts an
    iterable of trace ids for the affected set."""
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
