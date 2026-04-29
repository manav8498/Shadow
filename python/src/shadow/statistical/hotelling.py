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

**Exact permutation for tiny n.**  When ``permutations == -1`` AND the
total number of label permutations C(n1+n2, n1) ≤ EXACT_PERM_LIMIT,
the implementation enumerates the full set of permutations and reports
the exact (not Monte-Carlo) p-value.  Use this when n1 + n2 ≤ ~12 and
exact size control matters more than speed.

**Power and decision classification.**  Every result carries:

  * ``power``: post-hoc power approximation under the non-central F
    null at the observed effect size.  Computed via the closed-form
    ncF tail probability (Anderson 2003 Eq. 5.3.10) when df2 > 0;
    set to NaN when the F approximation is undefined (df2 ≤ 0).
  * ``decision``: one of ``"reject"``, ``"fail_to_reject"``, or
    ``"underpowered"``.  The ``underpowered`` state fires when the
    sample is too small to draw a conclusion at the requested
    α-level (df2 ≤ 0 with no permutation request, OR power < 0.5
    on a non-rejecting result).  This prevents the
    "we silently said no regression but the sample size was 5" failure
    mode that the v2.6 review flagged.

References
----------
Hotelling, H. (1931). The generalization of Student's ratio.
Chen, Wiesel, Eldar & Hero (2010). Shrinkage Algorithms for MMSE
    Covariance Estimation. IEEE Trans. Signal Processing.
Anderson, T. W. (2003). An Introduction to Multivariate Statistical
    Analysis (3rd ed.) — permutation Hotelling T² (§5.3).
Phipson, B. & Smyth, G. K. (2010). "Permutation P-values Should Never
    Be Zero." *Statistical Applications in Genetics and Molecular
    Biology* 9(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_dist  # type: ignore[import-untyped, unused-ignore]
from scipy.stats import ncf as ncf_dist  # type: ignore[import-untyped, unused-ignore]

# Total label-permutation enumeration cap. C(12, 6) = 924; C(14, 7) =
# 3432; C(16, 8) = 12870. We cap at ~13_000 enumerations so the exact
# path stays fast (well under 1 s in pure Python). Above this the
# Monte-Carlo path is the right answer.
EXACT_PERM_LIMIT: int = 13_000

# Power threshold under which a non-rejecting result is reported as
# "underpowered" rather than fail_to_reject. Industry convention:
# 0.8 is the conventional adequacy threshold (Cohen 1988); we use 0.5
# as the "this answer might be wrong" threshold rather than the more
# stringent "this answer is reliable" threshold so the underpowered
# state isn't triggered too aggressively on borderline cases.
UNDERPOWERED_BELOW: float = 0.5

Decision = Literal["reject", "fail_to_reject", "underpowered"]


@dataclass
class HotellingResult:
    """Result of a two-sample Hotelling T² test."""

    t2: float
    """The T² statistic."""
    f_stat: float
    """F approximation: (n1+n2-D-1) / ((n1+n2-2)*D) * T²."""
    p_value: float
    """Two-tailed p-value under F(D, n1+n2-D-1) or the permutation null."""
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
    """True when p_value < alpha AND the test was adequately powered.

    A non-rejecting outcome on an underpowered sample sets
    ``reject_null = False`` AND ``decision = "underpowered"`` — callers
    should branch on ``decision``, not just ``reject_null``, when the
    "no regression detected" path matters.
    """
    shrinkage: float
    """Oracle Approximating Shrinkage coefficient (0 = none, 1 = full)."""
    power: float = field(default=float("nan"))
    """Post-hoc power under the non-central F at the observed effect.

    NaN when df2 ≤ 0 (the F approximation is undefined). Callers can
    use the ``decision`` field to branch instead.
    """
    decision: Decision = field(default="fail_to_reject")
    """Three-state classification:

      * ``reject`` — H0 is rejected at the requested α (drift detected).
      * ``fail_to_reject`` — H0 not rejected AND the test was adequately
        powered (≥ 0.5) — i.e. "we looked carefully and saw nothing."
      * ``underpowered`` — sample too small to conclude. Either df2 ≤ 0
        (the F approximation is undefined) or the post-hoc power on
        the observed effect is < 0.5. Reported instead of
        ``fail_to_reject`` so callers don't mistake low power for "all
        clear."
    """
    used_exact_permutation: bool = field(default=False)
    """True iff the exact permutation enumeration path was used."""

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
            "power": self.power,
            "decision": self.decision,
            "used_exact_permutation": self.used_exact_permutation,
        }


def hotelling_t2(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    alpha: float = 0.05,
    permutations: int = 0,
    rng: np.random.Generator | None = None,
) -> HotellingResult:
    """Two-sample Hotelling T² test with power-aware decision classification.

    Parameters
    ----------
    x1 : (n1, D) array — baseline fingerprint matrix.
    x2 : (n2, D) array — candidate fingerprint matrix.
    alpha : significance level (default 0.05).
    permutations
        Three modes:

        * ``0`` (default) — F-approximation p-value via
          :func:`scipy.stats.f.sf`.
        * ``> 0`` — Monte-Carlo permutation test with that many label
          shuffles. Phipson-Smyth (2010) corrected p-value.
        * ``-1`` — **exact permutation** when the total number of
          label permutations C(n1+n2, n1) ≤ ``EXACT_PERM_LIMIT``;
          otherwise falls through to the F-approximation. Use this
          when n1+n2 is small (≤ ~12) and exact size control matters.
    rng
        Optional random generator for the Monte-Carlo permutation path.

    Returns
    -------
    HotellingResult with T², F statistic, p-value, **post-hoc power**,
    and a three-state ``decision`` field that distinguishes
    ``reject`` / ``fail_to_reject`` / ``underpowered`` so the caller
    cannot mistake low statistical power for "no regression detected."

    Raises
    ------
    ValueError if either matrix has fewer than 2 observations, or if
    the two matrices have different D dimensions, or if ``permutations``
    is < -1.
    """
    if permutations < -1:
        raise ValueError(f"permutations must be >= -1; got {permutations}")

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

    # Exact-permutation path: enumerate all C(n1+n2, n1) permutations
    # when feasible and return the exact (not Monte-Carlo) p-value.
    if permutations == -1:
        n_total = n1 + n2
        n_combos = math.comb(n_total, n1)
        if n_combos <= EXACT_PERM_LIMIT:
            combined = np.vstack([x1, x2])
            ge_count = 0
            for combo in combinations(range(n_total), n1):
                mask = np.zeros(n_total, dtype=bool)
                mask[list(combo)] = True
                xa = combined[mask]
                xb = combined[~mask]
                t2_perm, _ = _t2_statistic(xa, xb)
                if t2_perm >= t2_obs:
                    ge_count += 1
            p_value = ge_count / n_combos
            f_stat = (
                float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2_obs) if df2 > 0 else math.nan
            )
            power = _post_hoc_power(t2_obs, n1, n2, d, alpha, df2)
            decision = _classify_decision(p_value, alpha, power, df2)
            reject = decision == "reject"
            return HotellingResult(
                t2=t2_obs,
                f_stat=f_stat,
                p_value=p_value,
                df1=d,
                df2=max(df2, 0),
                n1=n1,
                n2=n2,
                d=d,
                reject_null=reject,
                shrinkage=shrinkage,
                power=power,
                decision=decision,
                used_exact_permutation=True,
            )
        # n too large to enumerate exhaustively — fall through to the
        # F-approximation (the caller asked for exact, but we cannot
        # safely deliver it; the F path is the conservative choice).

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
        # Phipson-Smyth (2010) correction: p̂ = (b + 1) / (B + 1) where b is
        # the count of permutations with statistic ≥ observed and B is the
        # total number of permutations sampled. The "+1" accounts for the
        # observed labelling itself being one of the permutations under H0,
        # ensuring the p-value is never exactly zero (which would imply the
        # observed cannot have arisen by chance, an over-statement) and is
        # an unbiased estimator of the true permutation p-value.
        # Reference: Phipson & Smyth (2010), "Permutation P-values Should
        # Never Be Zero". Statistical Applications in Genetics and
        # Molecular Biology 9(1).
        p_value = (count + 1) / (permutations + 1)
        f_stat = float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2_obs) if df2 > 0 else math.nan
        power = _post_hoc_power(t2_obs, n1, n2, d, alpha, df2)
        decision = _classify_decision(p_value, alpha, power, df2)
        reject = decision == "reject"
        return HotellingResult(
            t2=t2_obs,
            f_stat=f_stat,
            p_value=p_value,
            df1=d,
            df2=max(df2, 0),
            n1=n1,
            n2=n2,
            d=d,
            reject_null=reject,
            shrinkage=shrinkage,
            power=power,
            decision=decision,
            used_exact_permutation=False,
        )

    if df2 <= 0:
        # More dimensions than observations; F-approximation is undefined.
        # Reported as ``underpowered`` so callers don't mistake the
        # neutral p=1.0 for "all clear."
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
            power=float("nan"),
            decision="underpowered",
            used_exact_permutation=False,
        )

    f_stat = float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2_obs)
    p_value = float(f_dist.sf(f_stat, dfn=d, dfd=df2))
    power = _post_hoc_power(t2_obs, n1, n2, d, alpha, df2)
    decision = _classify_decision(p_value, alpha, power, df2)
    reject = decision == "reject"

    return HotellingResult(
        t2=t2_obs,
        f_stat=f_stat,
        p_value=p_value,
        df1=d,
        df2=df2,
        n1=n1,
        n2=n2,
        d=d,
        reject_null=reject,
        shrinkage=shrinkage,
        power=power,
        decision=decision,
        used_exact_permutation=False,
    )


def _post_hoc_power(
    t2_obs: float,
    n1: int,
    n2: int,
    d: int,
    alpha: float,
    df2: int,
) -> float:
    """Post-hoc power approximation under the non-central F.

    Procedure (Anderson 2003 §5.3.10):

      1. Critical value: ``F_crit = F_{1-α; d, df2}``.
      2. Non-centrality parameter: ``λ = T²_obs * (n1 * n2) / (n1 + n2)``
         under the alternative — this is the form for the
         multivariate ANOVA non-central F.
      3. Power: ``1 - F_ncF(F_crit; d, df2, λ)``.

    Returns NaN when df2 ≤ 0 (the F approximation is undefined). The
    caller branches on the ``decision`` field rather than the raw
    power value in that case.
    """
    if df2 <= 0:
        return float("nan")
    try:
        f_crit = float(f_dist.ppf(1.0 - alpha, dfn=d, dfd=df2))
    except Exception:
        return float("nan")
    if not math.isfinite(f_crit):
        return float("nan")
    nc = float(t2_obs * (n1 * n2) / (n1 + n2))
    if nc <= 0.0:
        return float(alpha)  # null effect → power equals nominal size
    try:
        power = 1.0 - float(ncf_dist.cdf(f_crit, dfn=d, dfd=df2, nc=nc))
    except Exception:
        return float("nan")
    if not math.isfinite(power):
        return float("nan")
    return max(0.0, min(1.0, power))


def _classify_decision(p_value: float, alpha: float, power: float, df2: int) -> Decision:
    """Map (p-value, power, df2) to one of three decision states.

    * df2 ≤ 0 → ``underpowered`` (F approximation undefined; the
      caller passed a fingerprint with more dimensions than the sample
      can support).
    * p < α → ``reject`` (drift detected; power is informational only).
    * p ≥ α AND power ≥ UNDERPOWERED_BELOW → ``fail_to_reject`` (we
      looked carefully and saw nothing).
    * p ≥ α AND power < UNDERPOWERED_BELOW → ``underpowered`` (sample
      too small to draw a confident conclusion at this effect size).
    """
    if df2 <= 0:
        return "underpowered"
    if p_value < alpha:
        return "reject"
    if not math.isfinite(power) or power < UNDERPOWERED_BELOW:
        return "underpowered"
    return "fail_to_reject"


def decision_label(result: HotellingResult) -> str:
    """Human-readable one-line label for a HotellingResult.

    Designed for terminal / PR-comment renderers. The label includes
    the decision state, the p-value (or "p=NaN" when undefined), and
    the post-hoc power so the reader can tell the difference between
    "we ran the test with adequate power" and "the test was undefined."
    """
    if result.decision == "reject":
        return f"behavioral drift detected (p={result.p_value:.4f}, " f"power={result.power:.2f})"
    if result.decision == "fail_to_reject":
        return f"no detectable drift (p={result.p_value:.4f}, " f"power={result.power:.2f})"
    # underpowered
    if math.isfinite(result.power):
        return (
            f"underpowered: cannot conclude "
            f"(p={result.p_value:.4f}, power={result.power:.2f}, "
            f"need larger sample)"
        )
    return (
        f"underpowered: F-approximation undefined "
        f"(df2={result.df2}, n1={result.n1}, n2={result.n2}, D={result.d}; "
        f"need n1+n2 > D+1 OR pass permutations=-1 for exact test)"
    )


def _t2_statistic(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> tuple[float, float]:
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


def _oas_shrink(s: NDArray[np.float64], n: int) -> tuple[NDArray[np.float64], float]:
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


__all__ = [
    "EXACT_PERM_LIMIT",
    "UNDERPOWERED_BELOW",
    "Decision",
    "HotellingResult",
    "decision_label",
    "hotelling_t2",
]
