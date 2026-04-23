"""Aggregate per-pair judge scores into a DiffReport axis row.

The Rust core leaves axis 8 (`judge`) empty because there's no default
Judge trait implementation on the Rust side. The Python CLI, after
running a Judge over each response pair, calls [`aggregate_scores`] to
produce an `AxisStat`-shaped dict and splices it into the
`DiffReport.rows` list in place of the empty placeholder.

Mirrors the Rust-side semantics exactly:
- `baseline_median` is `1.0` (the "perfect" reference).
- `candidate_median` is the median of the judge scores.
- `delta = candidate_median - baseline_median` (always ≤ 0.0).
- Bootstrap 95% CI over candidate_scores, 1000 iterations.
- Severity via the same CI-aware rule used by the Rust axes.
- `flags` via the same low_power / ci_crosses_zero rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_BOOTSTRAP_ITERATIONS = 1000
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


def aggregate_scores(scores: list[float], seed: int = 42) -> dict[str, Any]:
    """Build a judge axis row from per-pair scores in `[0, 1]`."""
    if not scores:
        return _empty_row()
    arr = np.asarray(scores, dtype=float)
    baseline_median = 1.0
    candidate_median = float(np.median(arr))
    delta = candidate_median - baseline_median
    rng = np.random.default_rng(seed)
    resampled_deltas = np.empty(_BOOTSTRAP_ITERATIONS, dtype=float)
    for i in range(_BOOTSTRAP_ITERATIONS):
        sample = rng.choice(arr, size=arr.size, replace=True)
        resampled_deltas[i] = float(np.median(sample)) - baseline_median
    ci_low = float(np.percentile(resampled_deltas, _CI_LOW_PCT))
    ci_high = float(np.percentile(resampled_deltas, _CI_HIGH_PCT))
    return {
        "axis": "judge",
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "delta": delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "severity": _severity(delta, baseline_median, ci_low, ci_high),
        "n": len(scores),
        "flags": _flags(ci_low, ci_high, len(scores)),
    }


def _empty_row() -> dict[str, Any]:
    return {
        "axis": "judge",
        "baseline_median": 0.0,
        "candidate_median": 0.0,
        "delta": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "severity": "none",
        "n": 0,
        "flags": [],
    }


def _severity(delta: float, baseline_median: float, ci_low: float, ci_high: float) -> str:
    if abs(delta) < 1e-9:
        return "none"
    ci_crosses_zero = ci_low <= 0.0 <= ci_high and not (ci_low == 0.0 and ci_high == 0.0)
    if ci_crosses_zero and abs(delta) < max(abs(baseline_median) * 0.05, 1e-9):
        return "none"
    if abs(baseline_median) < 1e-9:
        base = "none" if abs(delta) < 1e-9 else "minor"
    else:
        rel = abs(delta / baseline_median)
        if rel < 0.10:
            base = "minor"
        elif rel < 0.30:
            base = "moderate"
        else:
            base = "severe"
    if ci_crosses_zero and base in ("moderate", "severe"):
        return "minor"
    return base


def _flags(ci_low: float, ci_high: float, n: int) -> list[str]:
    flags: list[str] = []
    if 0 < n < 5:
        flags.append("low_power")
    if ci_low <= 0.0 <= ci_high and not (ci_low == 0.0 and ci_high == 0.0):
        flags.append("ci_crosses_zero")
    return flags


__all__ = ["aggregate_scores"]
