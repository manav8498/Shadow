"""Conformal prediction coverage bounds for Agent Behavior Certificates.

Two entry points are provided.  Use the first whenever real per-run
scores are available — only it carries the distribution-free guarantee.

``conformal_calibrate(per_axis_scores, ...)`` — **real split conformal**
    Takes a dict of per-axis lists of nonconformity scores
    (e.g. ``|baseline_run_i[axis] − μ_baseline[axis]|`` for each
    held-out baseline run) and computes the conformal quantile

        q̂ = scores[ ⌈(n+1)(1−α)⌉ ]   (sorted ascending; α = 1 − coverage)

    The guarantee is exact and distribution-free under exchangeability:

        P( score_{n+1} ≤ q̂ ) ≥ 1 − α

    (Vovk et al. 2005, Angelopoulos & Bates 2022).  This is the
    canonical entry point and the one quoted in agent certificates.

``build_conformal_coverage(axis_rows, ...)`` — **parametric fallback**
    Used when only summary statistics (``delta``, ``ci95_low``,
    ``ci95_high``, ``n``) are available — typically because the diff
    report has been aggregated and the per-run scores were discarded.
    The function reconstructs a synthetic Gaussian calibration set
    from the moments and computes a parametric quantile.  This is
    **not** distribution-free; it is valid only to the extent that the
    score distribution is well-approximated by a normal with the
    inferred mean and variance.  ``ConformalCoverageReport.is_distribution_free``
    is set to ``False`` in this case so downstream code can flag it.

Either entry point returns a :class:`ConformalCoverageReport` with
per-axis :class:`AxisCoverage` rows sorted by ``q_hat`` descending.

References
----------
Vovk, Gammerman & Shafer (2005). Algorithmic Learning in a Random World.
Angelopoulos & Bates (2022). A Gentle Introduction to Conformal Prediction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AxisCoverage:
    """Conformal coverage bound for one behavioral axis."""

    axis: str
    """Axis name, e.g. ``"semantic"``."""
    n_calibration: int
    """Number of calibration runs (i.e. paired observations in the diff)."""
    target_coverage: float
    """Desired coverage level (1−α), e.g. 0.90."""
    q_hat: float
    """Calibration quantile: worst-case |delta| covered at target level."""
    achieved_coverage: float
    """Empirical coverage fraction on the calibration set (≥ target_coverage
    by construction when n_calibration is sufficient)."""
    pac_delta: float
    """PAC confidence: P(achieved_coverage < target_coverage) ≤ pac_delta.
    Derived from the binomial CDF on the coverage indicator."""
    marginal_claim: str
    """Human-readable claim string."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConformalCoverageReport:
    """Per-axis conformal coverage bounds + joint summary."""

    target_coverage: float
    """Requested coverage level (1−α), e.g. 0.90."""
    confidence: float
    """PAC confidence level, e.g. 0.95."""
    n_calibration: int
    """Calibration set size (number of paired runs)."""
    axes: list[AxisCoverage] = field(default_factory=list)
    """Per-axis bounds, sorted by q_hat descending."""
    worst_axis: str = ""
    """Axis with the largest q_hat (the binding constraint)."""
    sufficient_n: bool = True
    """True when n_calibration ≥ n_min(target_coverage, confidence)."""
    n_min: int = 0
    """Minimum n for which the PAC guarantee holds at this coverage/confidence."""
    is_distribution_free: bool = True
    """True iff the bounds were computed from real per-run scores
    (split-conformal); False if reconstructed from summary statistics
    via a parametric Gaussian approximation (see module docstring)."""
    note: str = ""
    """Freeform human-readable summary."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def conformal_calibrate(
    per_axis_scores: dict[str, list[float]],
    target_coverage: float = 0.90,
    confidence: float = 0.95,
) -> ConformalCoverageReport:
    """Distribution-free split-conformal coverage from real per-run scores.

    Parameters
    ----------
    per_axis_scores
        Mapping ``axis_name → list[float]`` of nonconformity scores.
        Each list element is one calibration observation (e.g. the
        absolute deviation of a single baseline run from the baseline
        mean on that axis).  The length is the calibration set size n.
    target_coverage
        Desired marginal coverage 1 − α.  q̂ is the sample
        ⌈(n+1)(1−α)⌉/n quantile of the scores.
    confidence
        PAC confidence level used to compute ``n_min`` and
        ``pac_delta`` (the probability that empirical coverage on
        future runs falls below the target).

    Returns
    -------
    ConformalCoverageReport with ``is_distribution_free=True``.

    Raises
    ------
    ValueError if either ``target_coverage`` or ``confidence`` is
    outside (0, 1), or if any axis has zero scores.
    """
    if not (0 < target_coverage < 1):
        raise ValueError(f"target_coverage must be in (0,1); got {target_coverage}")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0,1); got {confidence}")

    alpha = 1.0 - target_coverage
    n_min = _min_n_for_pac(target_coverage, confidence)

    axis_coverages: list[AxisCoverage] = []
    for axis_name, scores in per_axis_scores.items():
        if not scores:
            continue
        # Coerce to non-negative floats — nonconformity scores are
        # signed-distance-style by convention.
        s = [abs(float(x)) for x in scores]
        coverage = _axis_coverage(
            axis=axis_name,
            scores=s,
            target_coverage=target_coverage,
            confidence=confidence,
            alpha=alpha,
        )
        axis_coverages.append(coverage)

    axis_coverages.sort(key=lambda c: -c.q_hat)
    worst_axis = axis_coverages[0].axis if axis_coverages else ""
    n_calibration = max((c.n_calibration for c in axis_coverages), default=0)
    sufficient = n_calibration >= n_min
    note = _build_note(
        target_coverage, confidence, n_calibration, n_min, worst_axis, sufficient
    )

    return ConformalCoverageReport(
        target_coverage=target_coverage,
        confidence=confidence,
        n_calibration=n_calibration,
        axes=axis_coverages,
        worst_axis=worst_axis,
        sufficient_n=sufficient,
        n_min=n_min,
        is_distribution_free=True,
        note=note,
    )


def build_conformal_coverage(
    axis_rows: list[dict[str, Any]],
    target_coverage: float = 0.90,
    confidence: float = 0.95,
) -> ConformalCoverageReport:
    """Parametric coverage estimate from summary statistics (NOT distribution-free).

    Use this only when the per-run scores are unavailable (e.g. when
    consuming an aggregated diff report).  The function reconstructs a
    synthetic Gaussian calibration set with mean ``|delta|`` and standard
    deviation derived from the CI half-width, then applies the same
    quantile rule as :func:`conformal_calibrate`.

    The returned report has ``is_distribution_free=False``.  Callers
    that require the formal distribution-free guarantee should switch
    to :func:`conformal_calibrate` with real per-run scores.
    """
    if not (0 < target_coverage < 1):
        raise ValueError(f"target_coverage must be in (0,1); got {target_coverage}")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0,1); got {confidence}")

    alpha = 1.0 - target_coverage
    n_min = _min_n_for_pac(target_coverage, confidence)

    axis_coverages: list[AxisCoverage] = []
    # Per-axis seed offset so synthetic calibration sets across axes are
    # not perfectly correlated (otherwise all axes would share the same
    # standardised quantile shape).
    for seed_offset, row in enumerate(axis_rows):
        axis_name = str(row.get("axis") or "")
        n = int(row.get("n") or 0)

        if n == 0:
            continue

        scores = _approximate_calibration_set(row, n, seed_offset)
        coverage = _axis_coverage(
            axis=axis_name,
            scores=scores,
            target_coverage=target_coverage,
            confidence=confidence,
            alpha=alpha,
        )
        axis_coverages.append(coverage)

    axis_coverages.sort(key=lambda c: -c.q_hat)
    worst_axis = axis_coverages[0].axis if axis_coverages else ""
    n_calibration = max((c.n_calibration for c in axis_coverages), default=0)
    sufficient = n_calibration >= n_min
    note = _build_note(
        target_coverage, confidence, n_calibration, n_min, worst_axis, sufficient
    )
    if axis_coverages:
        note = note + (
            "  (parametric estimate from summary statistics — "
            "not a distribution-free guarantee; pass per-run scores to "
            "shadow.conformal.conformal_calibrate for the formal bound)"
        )

    return ConformalCoverageReport(
        target_coverage=target_coverage,
        confidence=confidence,
        n_calibration=n_calibration,
        axes=axis_coverages,
        worst_axis=worst_axis,
        sufficient_n=sufficient,
        n_min=n_min,
        is_distribution_free=False,
        note=note,
    )


def _approximate_calibration_set(
    row: dict[str, Any], n: int, seed_offset: int = 0
) -> list[float]:
    """Synthesize a Gaussian calibration set from summary statistics.

    Used by the parametric ``build_conformal_coverage`` fallback only.
    Mean = |delta|, std derived from ``ci95_high − ci95_low``.  Each
    axis row gets a distinct rng seed so different axes do not share a
    common standardised quantile shape.
    """
    abs_delta = abs(float(row.get("delta") or 0.0))

    if n == 1:
        return [abs_delta]

    ci_low = float(row.get("ci95_low") or 0.0)
    ci_high = float(row.get("ci95_high") or 0.0)
    ci_width = abs(ci_high - ci_low)
    se = ci_width / (2 * 1.96) if ci_width > 1e-12 else abs_delta * 0.1
    sigma = se * math.sqrt(n)

    import numpy as np

    rng = np.random.default_rng(seed=42 + seed_offset)
    raw = rng.normal(loc=abs_delta, scale=max(sigma, 1e-9), size=n)
    return [max(0.0, float(s)) for s in raw]


def _axis_coverage(
    axis: str,
    scores: list[float],
    target_coverage: float,
    confidence: float,
    alpha: float,
) -> AxisCoverage:
    """Compute conformal bounds for one axis given calibration scores."""
    n = len(scores)
    # Conformal quantile: q̂ = score_(⌈(n+1)(1−α)⌉) on the sorted list.
    idx = math.ceil((n + 1) * (1.0 - alpha)) / n
    idx_clamped = min(1.0, idx)
    q_hat = float(_quantile(scores, idx_clamped))

    achieved = float(sum(1 for s in scores if s <= q_hat)) / n
    pac_delta = _pac_coverage_delta(n, target_coverage)

    claim = (
        f"With ≥{confidence:.0%} confidence, {axis} |delta| ≤ {q_hat:.4f} "
        f"on ≥{target_coverage:.0%} of future runs  (n={n})"
    )

    return AxisCoverage(
        axis=axis,
        n_calibration=n,
        target_coverage=target_coverage,
        q_hat=q_hat,
        achieved_coverage=achieved,
        pac_delta=pac_delta,
        marginal_claim=claim,
    )


def _quantile(scores: list[float], p: float) -> float:
    """Linear-interpolation quantile, matching numpy.percentile default."""
    if not scores:
        return 0.0
    sorted_s = sorted(scores)
    n = len(sorted_s)
    if n == 1:
        return sorted_s[0]
    idx = (n - 1) * p
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac


def _pac_coverage_delta(n: int, target_coverage: float) -> float:
    """P(empirical coverage < target) under the binomial model."""
    try:
        from scipy.stats import binom  # type: ignore[import-untyped]

        k = int(math.floor(n * target_coverage))
        pac_delta = float(binom.cdf(k, n, target_coverage))
        return min(1.0, max(0.0, pac_delta))
    except ImportError:
        # Normal-approximation fallback. Z = 0 at the threshold.
        if n < 2:
            return 1.0
        return 0.5


def _min_n_for_pac(coverage: float, confidence: float) -> int:
    """Minimum calibration set size for a non-vacuous PAC guarantee.

    Derived from ``P(score_{n+1} ≤ q̂_(⌈(n+1)(1-α)⌉)) ≥ 1 - α``: we
    require n large enough that the empirical (1-α) quantile is below
    the failure probability ``1 − confidence``:

        n ≥ ⌈log(1−confidence) / log(coverage)⌉

    For coverage=0.90, confidence=0.95: n_min ≈ 29.
    """
    if coverage >= 1.0 or confidence >= 1.0:
        return 1
    raw = math.log(1.0 - confidence) / math.log(coverage)
    return max(1, math.ceil(raw))


def _build_note(
    target_coverage: float,
    confidence: float,
    n: int,
    n_min: int,
    worst_axis: str,
    sufficient: bool,
) -> str:
    if not sufficient:
        return (
            f"n={n} calibration runs is below the minimum n_min={n_min} needed "
            f"for the {confidence:.0%} PAC guarantee at {target_coverage:.0%} "
            f"coverage. Run more replays and re-certify to tighten the bounds."
        )
    if not worst_axis:
        return "No axis observations — conformal bounds are vacuous."
    return (
        f"Conformal coverage at {target_coverage:.0%} level "
        f"({confidence:.0%} PAC confidence). "
        f"Binding axis: {worst_axis}. "
        f"Calibrated on n={n} runs."
    )


__all__ = [
    "AxisCoverage",
    "ConformalCoverageReport",
    "build_conformal_coverage",
    "conformal_calibrate",
]
