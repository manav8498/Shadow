"""Per-trace 9-axis diff for `shadow diagnose-pr`.

Wraps `shadow._core.compute_diff_report` (the Rust 9-axis differ)
for one-pair use, plus two classifiers — `is_affected` and
`worst_axis_for` — that read the report's `severity` and
`first_divergence` fields.

The classifier thresholds live here, not in the Rust differ, so
they're easy to tune without recompiling.
"""

from __future__ import annotations

from typing import Any

from shadow import _core

_AFFECTED_SEVERITIES = {"moderate", "severe"}
_SEVERITY_ORDER = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def diff_pair(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute a 9-axis diff report for one (baseline, candidate)
    record-list pair. Thin wrapper over the Rust differ so the
    diagnose-pr surface doesn't depend on `_core` directly."""
    return _core.compute_diff_report(baseline, candidate, pricing, seed)


def is_affected(report: dict[str, Any]) -> bool:
    """Classify whether a candidate trace meaningfully changed
    behavior relative to its baseline.

    Affected if either:
      * any axis has severity moderate/severe, OR
      * the differ flagged a `first_divergence` (a structural pin
        that something interesting happened, even if the per-axis
        severities are still minor).
    """
    rows = report.get("rows") or []
    for row in rows:
        if row.get("severity") in _AFFECTED_SEVERITIES:
            return True
    fd = report.get("first_divergence")
    return bool(fd and isinstance(fd, dict))


def worst_axis_for(report: dict[str, Any]) -> str | None:
    """Return the axis name with the highest severity in this report,
    or None if all axes are at severity `none`. Ties are broken by
    insertion order (the differ already orders axes deterministically)."""
    rows = report.get("rows") or []
    best_rank = 0
    best_axis: str | None = None
    for row in rows:
        sev = row.get("severity", "none")
        rank = _SEVERITY_ORDER.get(sev, 0)
        if rank > best_rank:
            best_rank = rank
            best_axis = row.get("axis")
    return best_axis


__all__ = ["diff_pair", "is_affected", "worst_axis_for"]
