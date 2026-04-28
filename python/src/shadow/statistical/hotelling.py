"""Two-sample Hotelling T² test for behavioral drift detection.

Hotelling T² is the multivariate generalisation of the two-sample
t-test.  Given fingerprint matrices X1 (n1 × D) and X2 (n2 × D), it
tests:

    H0: μ1 = μ2   (no behavioral shift)
    H1: μ1 ≠ μ2   (behavioral shift detected)

Under multivariate normality with equal covariances, the F transform

    F = ((n1 + n2 − D − 1) / ((n1 + n2 − 2) D)) · T²

is exactly F-distributed with (D, n1+n2−D−1) degrees of freedom under
H0, giving an exact p-value via :func:`scipy.stats.f.sf`.

**Small-sample / high-dimension robustness.**  When ``n1 + n2 − 2`` is
not much larger than ``D`` the pooled covariance is poorly conditioned.
This module applies the Oracle Approximating Shrinkage (OAS) estimator
of Chen, Wiesel, Eldar & Hero (2010) — a closed-form analytic shrinkage
toward a scaled identity that requires no cross-validation:

    Σ̂_OAS = (1 − ρ̂) S + ρ̂ (tr(S)/D) I

where ρ̂ is computed in closed form from the trace and trace-of-square
of S.  After shrinkage the F-distribution null is no longer exact —
the test becomes asymptotically valid but has finite-sample bias.  For
problems where exact size control matters at small n, callers can
request a permutation p-value via the ``permutations`` argument: the
labels are shuffled B times to build a Monte-Carlo null and the
returned ``p_value`` is the empirical fraction of permuted T² values
that exceed the observed T².

References
----------
Hotelling, H. (1931). The generalization of Student's ratio.
Chen, Wiesel, Eldar & Hero (2010). Shrinkage Algorithms for MMSE
    Covariance Estimation. IEEE Trans. Signal Processing.
Anderson, T. W. (2003). An Introduction to Multivariate Statistical
    Analysis (3rd ed.) — permutation Hotelling T² (§5.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist  # type: ignore[import-untyped]


@dataclass
class HotellingResult:
    """Result of a two-sample Hotelling T² test."""

    t2: float
    """The T² statistic."""
    f_stat: float
    """F approximation: (n1+n2-D-1) / ((n1+n2-2)*D) * T²."""
    p_value: float
    """Two-tailed p-value under F(D, n1+n2-D-1)."""
    df1: int
    """Numerator degrees of freedom (== D)."""
    df2: int
    """Denominator degrees of freedom (== n1+n2-D-1)."""
    n1: int
    """Baseline sample size."""
    n2: int
    """Candidate sample size."""
    d: int
    """Fingerprint dimension."""
    reject_null: bool
    """True when p_value < alpha (behavioral drift detected)."""
    shrinkage: float
    """Ledoit-Wolf shrinkage coefficient applied to S_pooled (0 = none)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "t2": self.t2,
            "f_stat": self.f_stat,
            "p_value": self.p_value,
            "df1": self.df1,
            "df2": self.df2,
            "n1": self.n1,
            "n2": self.n2,
            "d": self.d,
            "reject_null": self.reject_null,
            "shrinkage": self.shrinkage,
        }


def hotelling_t2(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    alpha: float = 0.05,
    permutations: int = 0,
    rng: np.random.Generator | None = None,
) -> HotellingResult:
    """Two-sample Hotelling T² test.

    Parameters
    ----------
    x1 : (n1, D) array — baseline fingerprint matrix.
    x2 : (n2, D) array — candidate fingerprint matrix.
    alpha : significance level (default 0.05).
    permutations
        If 0 (default), the p-value is computed from the F-approximation.
        If > 0, run a permutation test with that many label shuffles
        and return the Monte-Carlo p-value
        ``(1 + #{T²_perm ≥ T²_obs}) / (1 + permutations)``.  Use this
        when the F-approximation is unreliable — small samples, when
        OAS shrinkage was applied, or when the data clearly violates
        multivariate normality.
    rng
        Optional random generator for the permutation path; passing one
        makes the result reproducible.

    Returns
    -------
    HotellingResult with T², F statistic, p-value, and rejection
    decision at the given alpha level.

    Raises
    ------
    ValueError if either matrix has fewer than 2 observations, or if
    the two matrices have different D dimensions, or if ``permutations``
    is negative.
    """
    if permutations < 0:
        raise ValueError(f"permutations must be >= 0; got {permutations}")

    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    n1, d1 = x1.shape
    n2, d2 = x2.shape

    if n1 < 2:
        raise ValueError(f"x1 must have at least 2 rows; got {n1}")
    if n2 < 2:
        raise ValueError(f"x2 must have at least 2 rows; got {n2}")
    if d1 != d2:
        raise ValueError(f"x1 and x2 must have the same number of columns; got {d1} vs {d2}")
    d = d1

    t2_obs, shrinkage = _t2_statistic(x1, x2)

    df2 = n1 + n2 - d - 1

    if permutations > 0:
        # Permutation Monte-Carlo p-value. Robust under non-normality
        # and shrinkage — the only assumption is exchangeability of the
        # combined sample under H0.
        gen = rng if rng is not None else np.random.default_rng()
        combined = np.vstack([x1, x2])
        n_total = combined.shape[0]
        count = 0
        for _ in range(permutations):
            perm = gen.permutation(n_total)
            xa = combined[perm[:n1]]
            xb = combined[perm[n1:]]
            t2_perm, _ = _t2_statistic(xa, xb)
            if t2_perm >= t2_obs:
                count += 1
        p_value = (count + 1) / (permutations + 1)
        f_stat = (
            float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2_obs)
            if df2 > 0
            else math.nan
        )
        return HotellingResult(
            t2=t2_obs,
            f_stat=f_stat,
            p_value=p_value,
            df1=d,
            df2=max(df2, 0),
            n1=n1,
            n2=n2,
            d=d,
            reject_null=p_value < alpha,
            shrinkage=shrinkage,
        )

    if df2 <= 0:
        # More dimensions than observations; F-approximation is undefined.
        # Caller should pass permutations > 0 for a valid test in this regime.
        return HotellingResult(
            t2=t2_obs,
            f_stat=math.nan,
            p_value=1.0,
            df1=d,
            df2=max(df2, 0),
            n1=n1,
            n2=n2,
            d=d,
            reject_null=False,
            shrinkage=shrinkage,
        )

    f_stat = float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2_obs)
    p_value = float(f_dist.sf(f_stat, dfn=d, dfd=df2))

    return HotellingResult(
        t2=t2_obs,
        f_stat=f_stat,
        p_value=p_value,
        df1=d,
        df2=df2,
        n1=n1,
        n2=n2,
        d=d,
        reject_null=p_value < alpha,
        shrinkage=shrinkage,
    )


def _t2_statistic(
    x1: NDArray[np.float64], x2: NDArray[np.float64]
) -> tuple[float, float]:
    """Compute the T² statistic and the OAS shrinkage coefficient used."""
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    mu1 = x1.mean(axis=0)
    mu2 = x2.mean(axis=0)
    diff = mu1 - mu2

    s1 = _sample_cov(x1)
    s2 = _sample_cov(x2)
    s_pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    s_reg, shrinkage = _oas_shrink(s_pooled.astype(np.float64), n1 + n2 - 2)

    try:
        s_inv = np.linalg.inv(s_reg)
    except np.linalg.LinAlgError:
        s_inv = np.linalg.pinv(s_reg)

    t2 = float(n1 * n2 / (n1 + n2) * diff @ s_inv @ diff)
    return t2, shrinkage


def _sample_cov(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unbiased sample covariance (divides by n-1)."""
    n = x.shape[0]
    centered = x - x.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw: NDArray[np.float64] = (centered.T @ centered) / max(n - 1, 1)
    cov = np.where(np.isfinite(raw), raw, 0.0)
    return cov.astype(np.float64)


def _oas_shrink(
    s: NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], float]:
    """Oracle Approximating Shrinkage (OAS, Chen et al. 2010) toward μ I.

    Shrinks S toward (trace(S)/D) · I via the closed-form coefficient

        ρ̂ = ((1 − 2/D) tr(S²) + tr(S)²) /
             ((n + 1 − 2/D) (tr(S²) − tr(S)²/D))

    requiring no cross-validation.  Returns (Σ̂_OAS, ρ̂) with
    ρ̂ ∈ [0, 1].  ρ̂ → 0 when the sample covariance is well-conditioned
    (n ≫ D); ρ̂ → 1 when full shrinkage to a scaled identity is needed.
    """
    d = s.shape[0]
    trace_s = float(np.trace(s))
    trace_s2 = float(np.trace(s @ s))
    mu = trace_s / d

    # OAS shrinkage coefficient (closed form, no CV required).
    # rho* = ((1-2/D)*trace(S²) + trace(S)²) / ((n+1-2/D)*(trace(S²) - trace(S)²/D))
    numerator = (1.0 - 2.0 / d) * trace_s2 + trace_s**2
    denominator = (n + 1.0 - 2.0 / d) * (trace_s2 - trace_s**2 / d)

    if abs(denominator) < 1e-12:
        # S is (near-)isotropic — no shrinkage needed.
        return s, 0.0

    rho = min(1.0, max(0.0, numerator / denominator))
    target = mu * np.eye(d, dtype=np.float64)
    s_reg: NDArray[np.float64] = (1.0 - rho) * s + rho * target
    return s_reg, rho


__all__ = ["HotellingResult", "hotelling_t2"]
