"""Cross-validation: shadow.statistical primitives vs reference impls.

Verifies that ``shadow.statistical.hotelling.hotelling_t2`` agrees
with a hand-derived reference implementation (the Hotelling T²
formula straight from Hotelling 1931 / Anderson 2003) within
floating-point tolerance.

We do not depend on pingouin / scipy.stats for the multivariate test
because:
- pingouin does not ship a Hotelling T² for two independent samples
  in older versions, only the one-sample / paired variant.
- scipy.stats has no built-in two-sample T².

So we verify against a textbook implementation that does not use
shrinkage (the test inputs have n ≫ D so OAS shrinkage ≈ 0 and the
two should agree). For shrinkage-affected regimes we cannot easily
cross-validate; that path is covered by the permutation p-value
sanity tests in test_statistical.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadow.statistical.hotelling import hotelling_t2


def _reference_hotelling_t2_with_oas(x1: np.ndarray, x2: np.ndarray) -> tuple[float, float, int]:
    """Reference implementation: textbook Hotelling T² + OAS shrinkage.

    Implements the same algorithm shadow.statistical.hotelling does,
    but using a separate, hand-derived code path against
    Anderson 2003 (T² formula) and Chen, Wiesel, Eldar & Hero 2010
    (OAS coefficient). If our implementation drifts (e.g. due to a
    refactor introducing a bug), this reference catches it.

    Returns (T², F-statistic, df2).
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]
    mu1 = x1.mean(axis=0)
    mu2 = x2.mean(axis=0)
    diff = mu1 - mu2

    # Pooled covariance.
    centered1 = x1 - mu1
    centered2 = x2 - mu2
    s1 = (centered1.T @ centered1) / (n1 - 1)
    s2 = (centered2.T @ centered2) / (n2 - 1)
    s_pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)

    # OAS shrinkage (Chen et al. 2010).
    n = n1 + n2 - 2
    trace_s = float(np.trace(s_pooled))
    trace_s2 = float(np.trace(s_pooled @ s_pooled))
    mu = trace_s / d
    numerator = (1.0 - 2.0 / d) * trace_s2 + trace_s**2
    denominator = (n + 1.0 - 2.0 / d) * (trace_s2 - trace_s**2 / d)
    if abs(denominator) < 1e-12:
        s_reg = s_pooled
    else:
        rho = min(1.0, max(0.0, numerator / denominator))
        target = mu * np.eye(d)
        s_reg = (1.0 - rho) * s_pooled + rho * target

    s_inv = np.linalg.inv(s_reg)
    t2 = float(n1 * n2 / (n1 + n2) * diff @ s_inv @ diff)
    df2 = n1 + n2 - d - 1
    f_stat = float((n1 + n2 - d - 1) / ((n1 + n2 - 2) * d) * t2)
    return t2, f_stat, df2


def _correlated_design(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Generate samples from a non-isotropic Gaussian (correlated dimensions).

    OAS shrinkage rounds toward a scaled identity. On data that's
    already isotropic (uncorrelated, equal variance), OAS detects no
    structure to preserve and shrinks fully (ρ̂ → 1) — which makes the
    statistic disagree with the no-shrinkage reference. To validate
    against the reference, the data must have non-trivial off-diagonal
    structure that OAS will preserve (ρ̂ ≈ 0).
    """
    # Random covariance with correlated dimensions.
    a = rng.standard_normal((d, d))
    cov = a @ a.T + np.eye(d) * 0.1  # PSD by construction
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, d))
    return z @ chol.T


@pytest.mark.parametrize("seed", range(20))
def test_hotelling_t2_matches_oas_reference(seed: int) -> None:
    """Our T² statistic must agree with a hand-derived reference that
    implements the same algorithm (Hotelling + OAS shrinkage) via a
    separate code path, within 1e-6 relative error.

    Detects: refactor bugs that change the numerical answer; subtle
    issues in shrinkage coefficient computation; matrix-inverse
    routing differences.
    """
    rng = np.random.default_rng(seed)
    n_per = 200
    d = 4
    x1 = _correlated_design(rng, n_per, d)
    x2 = _correlated_design(rng, n_per, d) + 0.3
    ours = hotelling_t2(x1, x2)
    ref_t2, ref_f, ref_df2 = _reference_hotelling_t2_with_oas(x1, x2)
    assert ours.df2 == ref_df2
    rel_t2 = abs(ours.t2 - ref_t2) / max(abs(ref_t2), 1.0)
    assert rel_t2 < 1e-6, (
        f"seed={seed}: T² disagrees with OAS reference. "
        f"ours={ours.t2:.6f}, reference={ref_t2:.6f}, rel_err={rel_t2:.2e}, "
        f"shrinkage={ours.shrinkage:.4f}"
    )
    rel_f = abs(ours.f_stat - ref_f) / max(abs(ref_f), 1.0)
    assert rel_f < 1e-6


def test_hotelling_p_value_matches_scipy_f_dist() -> None:
    """The p-value should match a hand-call to scipy.stats.f.sf with
    matching dfn and dfd."""
    from scipy.stats import f as f_dist  # type: ignore[import-untyped]

    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((50, 3))
    x2 = rng.standard_normal((50, 3)) + 0.5
    ours = hotelling_t2(x1, x2)
    expected = float(f_dist.sf(ours.f_stat, dfn=ours.df1, dfd=ours.df2))
    assert ours.p_value == pytest.approx(expected, abs=1e-10)
