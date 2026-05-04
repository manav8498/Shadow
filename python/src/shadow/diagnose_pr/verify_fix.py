"""`shadow verify-fix` — close the diagnose -> fix -> verify loop.

Given a diagnose-pr report.json + baseline/fixed configs + the
original baseline traces + the candidate-fixed traces, verify:

  * affected traces from the diagnose report no longer diverge
    against the fixed candidate (regression reversed)
  * a sample of "safe" (un-affected) traces still pass against
    the fixed config (no new regression introduced)

The pass criteria are the spec §7 thresholds:

  affected_reversed_rate >= 0.90
  new_policy_violations  == 0
  safe_trace_regression_rate <= 0.02

This module is import-safe (no I/O at import time). The CLI
command in `cli/app.py` does the I/O wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shadow.diagnose_pr.diffing import diff_pair, is_affected
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.policy import evaluate_policy


@dataclass(frozen=True)
class VerifyFixReport:
    """Outcome of `shadow verify-fix`. JSON-serialisable via
    `dataclasses.asdict`."""

    schema_version: str
    passed: bool
    affected_total: int
    affected_reversed: int
    affected_reversed_rate: float
    safe_total: int
    safe_regressed: int
    safe_regression_rate: float
    new_policy_violations: int
    threshold_reversed_rate: float
    threshold_safe_regression_rate: float
    fail_reasons: list[str] = field(default_factory=list)


VERIFY_SCHEMA_VERSION = "verify-fix/v0.1"

_DEFAULT_AFFECTED_THRESHOLD = 0.90
_DEFAULT_SAFE_REGRESSION_CEILING = 0.02


def verify_fix(
    *,
    diagnose_report: dict[str, Any],
    baseline_traces: list[LoadedTrace],
    fixed_traces: list[LoadedTrace],
    policy_path: Path | None = None,
    affected_reversed_threshold: float = _DEFAULT_AFFECTED_THRESHOLD,
    safe_regression_ceiling: float = _DEFAULT_SAFE_REGRESSION_CEILING,
) -> VerifyFixReport:
    """Validate that a fix reverses the regression named in
    `diagnose_report` without introducing new ones.

    `baseline_traces` are the original (compliant) traces from
    the diagnose-pr run. `fixed_traces` are the candidate-with-
    patch traces — paired by filename to baseline. Affected vs
    safe partition comes from `diagnose_report.affected_trace_ids`.
    """
    affected_set = set(diagnose_report.get("affected_trace_ids") or [])

    # Pair fixed traces to baseline by filename.
    fixed_by_name = {t.path.name: t for t in fixed_traces}

    affected_total = 0
    affected_reversed = 0
    safe_total = 0
    safe_regressed = 0
    total_new_violations = 0

    for base in baseline_traces:
        fixed = fixed_by_name.get(base.path.name)
        if fixed is None:
            # No paired fixed trace; skip — verify-fix can only
            # check what the user re-replayed.
            continue
        is_in_affected_set = base.trace_id in affected_set

        diff_report = diff_pair(base.records, fixed.records)
        regressed = is_affected(diff_report)

        if is_in_affected_set:
            affected_total += 1
            if not regressed:
                affected_reversed += 1
        else:
            safe_total += 1
            if regressed:
                safe_regressed += 1

        if policy_path is not None:
            pol = evaluate_policy(policy_path, base.records, fixed.records)
            total_new_violations += pol.new_violations

    reversed_rate = (affected_reversed / affected_total) if affected_total > 0 else 1.0
    safe_regression_rate = (safe_regressed / safe_total) if safe_total > 0 else 0.0

    fail_reasons: list[str] = []
    if affected_total > 0 and reversed_rate < affected_reversed_threshold:
        fail_reasons.append(
            f"only {affected_reversed}/{affected_total} affected traces reversed "
            f"({reversed_rate:.2%}); threshold is {affected_reversed_threshold:.0%}"
        )
    if safe_total > 0 and safe_regression_rate > safe_regression_ceiling:
        fail_reasons.append(
            f"{safe_regressed}/{safe_total} previously-safe traces regressed "
            f"({safe_regression_rate:.2%}); ceiling is {safe_regression_ceiling:.0%}"
        )
    if total_new_violations > 0:
        fail_reasons.append(
            f"fixed candidate introduces {total_new_violations} new policy violations"
        )

    return VerifyFixReport(
        schema_version=VERIFY_SCHEMA_VERSION,
        passed=not fail_reasons,
        affected_total=affected_total,
        affected_reversed=affected_reversed,
        affected_reversed_rate=reversed_rate,
        safe_total=safe_total,
        safe_regressed=safe_regressed,
        safe_regression_rate=safe_regression_rate,
        new_policy_violations=total_new_violations,
        threshold_reversed_rate=affected_reversed_threshold,
        threshold_safe_regression_rate=safe_regression_ceiling,
        fail_reasons=fail_reasons,
    )


__all__ = ["VERIFY_SCHEMA_VERSION", "VerifyFixReport", "verify_fix"]
