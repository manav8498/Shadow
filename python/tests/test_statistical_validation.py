"""Statistical-property validation for shadow.statistical and shadow.conformal.

Unit tests confirm the code paths execute correctly.  These simulations
confirm the *statistical claims* hold under repeated sampling:

- Hotelling T² controls Type-I error at the nominal α.
- Hotelling T² has non-trivial power against a known shift.
- Wald SPRT respects its (α, β) bounds across many runs.
- mSPRT controls Type-I error simultaneously across all sample sizes.
- Conformal q̂ achieves the target marginal coverage on held-out runs.

These are the tests that distinguish a library that *implements* a
statistical procedure from one that *delivers* the guarantees.

Marked as ``@pytest.mark.slow`` because each runs ~hundreds-thousands
of synthetic trials.  Skip with ``pytest -m "not slow"`` for fast CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadow.conformal import conformal_calibrate
from shadow.statistical.hotelling import hotelling_t2
from shadow.statistical.sprt import MSPRTDetector, SPRTDetector

pytestmark = pytest.mark.slow


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI on a proportion k/n.

    Returns (lower, upper). Used by validation tests to assert that
    the empirical rejection rate is consistent with the nominal α at
    the simulation's sample size — a tighter and more correct check
    than ad-hoc ``α + 0.05`` slack.

    Wilson CI is preferred over Wald (normal approximation) because
    it remains valid near 0 and 1, and is exact-in-distribution for
    the binomial proportion. See Brown, Cai & DasGupta (2001),
    "Interval Estimation for a Binomial Proportion".
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _alpha_in_wilson_ci(rejections: int, n_trials: int, alpha: float, z: float = 1.96) -> bool:
    """True iff α is inside the 95% Wilson CI on the empirical rate."""
    lo, hi = _wilson_ci(rejections, n_trials, z=z)
    return lo <= alpha <= hi


# ---------------------------------------------------------------------------
# Hotelling T² Type-I rate
# ---------------------------------------------------------------------------


class TestHotellingTypeI:
    """Under H0 (μ1 = μ2), the F-test should reject at rate ≈ α."""

    def test_type_i_rate_at_alpha_5pct(self):
        rng = np.random.default_rng(0)
        n_trials = 500
        n_per_group = 30
        d = 4
        alpha = 0.05
        rejections = 0
        for _ in range(n_trials):
            x1 = rng.standard_normal((n_per_group, d))
            x2 = rng.standard_normal((n_per_group, d))
            res = hotelling_t2(x1, x2, alpha=alpha)
            if res.reject_null:
                rejections += 1
        rate = rejections / n_trials
        lo, hi = _wilson_ci(rejections, n_trials)
        assert _alpha_in_wilson_ci(rejections, n_trials, alpha), (
            f"Hotelling Type-I rate {rate:.4f} (Wilson CI [{lo:.4f}, {hi:.4f}]) "
            f"does not include nominal α={alpha:.4f}. Rejecting too often "
            f"or too rarely under H0."
        )

    def test_permutation_p_value_uniform_under_null(self):
        """Under H0, permutation p-values should be ~uniform on [0, 1]."""
        rng = np.random.default_rng(1)
        n_trials = 200
        p_values = []
        for _ in range(n_trials):
            x1 = rng.standard_normal((10, 3))
            x2 = rng.standard_normal((10, 3))
            res = hotelling_t2(x1, x2, alpha=0.05, permutations=200, rng=rng)
            p_values.append(res.p_value)
        # Empirical rejection at α=0.05 should be ~5%.
        rate = sum(1 for p in p_values if p < 0.05) / n_trials
        assert rate < 0.12, f"permutation Type-I rate {rate:.3f} too high"


# ---------------------------------------------------------------------------
# Hotelling T² power
# ---------------------------------------------------------------------------


class TestHotellingPower:
    """Against a known shift, Hotelling should reject at rate >> α."""

    def test_power_against_moderate_shift(self):
        rng = np.random.default_rng(2)
        n_trials = 200
        n_per_group = 50
        d = 4
        # Mean shift of 0.5σ on every dimension.
        shift = 0.5
        rejections = 0
        for _ in range(n_trials):
            x1 = rng.standard_normal((n_per_group, d))
            x2 = rng.standard_normal((n_per_group, d)) + shift
            res = hotelling_t2(x1, x2, alpha=0.05)
            if res.reject_null:
                rejections += 1
        power = rejections / n_trials
        assert power > 0.80, f"Hotelling power {power:.3f} too low against 0.5σ shift on D=4 axes"


# ---------------------------------------------------------------------------
# Wald SPRT operating characteristic
# ---------------------------------------------------------------------------


class TestSPRTOperatingCharacteristic:
    """Wald SPRT (α, β) bounds hold asymptotically with large warmup.

    With plug-in μ̂, σ̂ from finite warmup, the bounds are asymptotic.
    These tests use warmup ≥ 200 so σ̂ is within ~5% of σ_true and
    the empirical Type-I rate sits near the nominal α.
    """

    def test_type_i_rate_under_h0(self):
        """Feeding observations from the null distribution → reject at ≈ α."""
        rng = np.random.default_rng(3)
        n_trials = 300
        alpha = 0.05
        beta = 0.20
        warmup = 200
        false_positives = 0
        for _ in range(n_trials):
            det = SPRTDetector(alpha=alpha, beta=beta, effect_size=0.5, warmup=warmup)
            for _ in range(warmup):
                det.update(float(rng.normal(0, 1)))
            for _ in range(500):
                state = det.update(float(rng.normal(0, 1)))
                if state.decision != "continue":
                    break
            if det.decision == "h1":
                false_positives += 1
        rate = false_positives / n_trials
        # Asymptotic Wald guarantee with large warmup. Empirical Type-I
        # should be near α; allow modest slack for plug-in σ̂ noise and
        # Monte-Carlo error.
        assert (
            rate < alpha + 0.05
        ), f"SPRT Type-I rate {rate:.3f} above α + slack with warmup={warmup}"

    def test_power_at_specified_alternative(self):
        """When the true effect equals the specified δ, reject at ≥ 1-β."""
        rng = np.random.default_rng(4)
        n_trials = 200
        alpha = 0.05
        beta = 0.20
        effect_size = 0.5
        warmup = 100
        true_positives = 0
        for _ in range(n_trials):
            det = SPRTDetector(alpha=alpha, beta=beta, effect_size=effect_size, warmup=warmup)
            for _ in range(warmup):
                det.update(float(rng.normal(0, 1)))
            for _ in range(1000):
                state = det.update(float(rng.normal(effect_size, 1)))
                if state.decision != "continue":
                    break
            if det.decision == "h1":
                true_positives += 1
        power = true_positives / n_trials
        assert power > 1 - beta - 0.10, f"SPRT power {power:.3f} below 1-β={1 - beta:.2f}"


# ---------------------------------------------------------------------------
# mSPRT — always-valid Type-I control
# ---------------------------------------------------------------------------


class TestMSPRTAlwaysValid:
    """mSPRT controls Type-I rate uniformly across all peek-points."""

    def test_type_i_rate_always_valid(self):
        """Robbins (1970): P(reject under H0 at any n) ≤ α when σ is known.

        With plug-in σ̂, the bound is asymptotic; we use warmup=300 so
        σ̂ is within ~4% of σ_true and the empirical rate sits near α.
        """
        rng = np.random.default_rng(5)
        n_trials = 300
        alpha = 0.05
        warmup = 300
        false_positives = 0
        for _ in range(n_trials):
            det = MSPRTDetector(alpha=alpha, tau=1.0, warmup=warmup)
            for _ in range(warmup):
                det.update(float(rng.normal(0, 1)))
            # "Peek" at every step — always-valid Type-I bound says
            # continuous monitoring does not inflate the false-positive
            # rate above α.
            for _ in range(500):
                state = det.update(float(rng.normal(0, 1)))
                if state.decision == "h1":
                    break
            if det.decision == "h1":
                false_positives += 1
        rate = false_positives / n_trials
        assert (
            rate < alpha + 0.05
        ), f"mSPRT Type-I rate {rate:.3f} above α + slack with warmup={warmup}"

    def test_detects_real_drift(self):
        """mSPRT must detect a moderate true effect."""
        rng = np.random.default_rng(6)
        n_trials = 100
        warmup = 100
        det_rejected = 0
        for _ in range(n_trials):
            det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=warmup)
            for _ in range(warmup):
                det.update(float(rng.normal(0, 1)))
            for _ in range(500):
                state = det.update(float(rng.normal(1.0, 1)))
                if state.decision == "h1":
                    break
            if det.decision == "h1":
                det_rejected += 1
        rate = det_rejected / n_trials
        assert rate > 0.85, f"mSPRT power {rate:.3f} too low against 1σ shift"


# ---------------------------------------------------------------------------
# Conformal coverage validation
# ---------------------------------------------------------------------------


class TestConformalCoverageValidation:
    """The q̂ from n calibration scores should cover ≥ target on held-out."""

    def test_marginal_coverage_at_90pct(self):
        rng = np.random.default_rng(7)
        target = 0.90
        n_trials = 100
        n_cal = 200
        n_test = 500
        # Track per-trial empirical coverage.
        coverages = []
        for _ in range(n_trials):
            cal_scores = list(np.abs(rng.standard_normal(n_cal)))
            report = conformal_calibrate({"x": cal_scores}, target_coverage=target, confidence=0.95)
            q_hat = report.axes[0].q_hat
            test_scores = np.abs(rng.standard_normal(n_test))
            cov = float(np.mean(test_scores <= q_hat))
            coverages.append(cov)
        mean_cov = sum(coverages) / len(coverages)
        # Marginal coverage averaged over trials should be ≥ target.
        # Conformal guarantees E[coverage] ≥ target − 1/(n_cal+1).
        assert (
            mean_cov >= target - 0.02
        ), f"Mean held-out coverage {mean_cov:.3f} below target {target}"
        # At most a small fraction of trials should fall below target − slack.
        below = sum(1 for c in coverages if c < target - 0.05)
        assert below / n_trials < 0.20, f"{below}/{n_trials} trials had coverage < target − 5%"

    def test_coverage_robust_to_distribution_shape(self):
        """Conformal is distribution-free: should also cover under heavy tails."""
        rng = np.random.default_rng(8)
        n_cal = 300
        n_test = 1000
        # Calibration and test from a t-distribution (heavy tails).
        cal_scores = list(np.abs(rng.standard_t(df=3, size=n_cal)))
        report = conformal_calibrate({"x": cal_scores}, target_coverage=0.90)
        q_hat = report.axes[0].q_hat
        test_scores = np.abs(rng.standard_t(df=3, size=n_test))
        coverage = float(np.mean(test_scores <= q_hat))
        # Distribution-free guarantee: coverage ≥ 0.90 − 1/(n+1) ≈ 0.897.
        assert coverage >= 0.85, f"Coverage on heavy-tailed test set {coverage:.3f} too low"

    def test_pac_delta_is_valid_probability(self):
        """pac_delta is P(empirical coverage < target) under Binomial(n, target).

        At n=200 and target=0.90 this is ≈ 0.5 by binomial symmetry around
        the mean, NOT < 1−confidence — the confidence parameter governs
        n_min, not pac_delta directly.  The right invariant is just that
        pac_delta is a valid probability and the report is distribution-free.
        """
        rng = np.random.default_rng(9)
        cal_scores = list(np.abs(rng.standard_normal(200)))
        report = conformal_calibrate({"x": cal_scores}, target_coverage=0.90, confidence=0.95)
        ax = report.axes[0]
        assert 0.0 <= ax.pac_delta <= 1.0
        assert report.is_distribution_free is True
        # n=200 ≫ n_min(0.90, 0.95)=29, so the report flags as sufficient.
        assert report.sufficient_n is True


# ---------------------------------------------------------------------------
# Cross-validation: real conformal vs parametric fallback
# ---------------------------------------------------------------------------


class TestRealVsParametricConformal:
    """The real conformal q̂ should be within reason of the parametric q̂
    when the underlying distribution actually IS Gaussian."""

    def test_agreement_under_gaussian_truth(self):
        from shadow.conformal import build_conformal_coverage

        rng = np.random.default_rng(10)
        n = 100
        scores = list(np.abs(rng.standard_normal(n)))
        # Real conformal from the actual scores.
        real = conformal_calibrate({"x": scores}, target_coverage=0.90)
        real_q = real.axes[0].q_hat
        # Parametric from summary stats.
        delta = float(np.mean(scores))
        ci_low = delta - 1.96 * float(np.std(scores)) / math.sqrt(n)
        ci_high = delta + 1.96 * float(np.std(scores)) / math.sqrt(n)
        parametric = build_conformal_coverage(
            [{"axis": "x", "delta": delta, "n": n, "ci95_low": ci_low, "ci95_high": ci_high}],
            target_coverage=0.90,
        )
        para_q = parametric.axes[0].q_hat
        # Both should be in the same order of magnitude under Gaussian truth.
        assert 0.5 * real_q < para_q < 5.0 * real_q
