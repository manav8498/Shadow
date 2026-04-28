"""Sequential testing for behavioral drift: Wald SPRT and mixture SPRT.

Two complementary detectors are provided:

``SPRTDetector`` — classic Wald SPRT (Wald 1945)
    Tests H0: x ~ N(μ0, σ²) vs H1: x ~ N(μ0 + δ, σ²) with a *fixed*
    point alternative δ = effect_size × σ.  Stops as soon as the
    log-likelihood ratio crosses Wald's boundaries
    ``log A = log(β/(1-α))`` or ``log B = log((1-β)/α)``.  Decisions
    are **absorbing**: once a boundary is crossed, ``update`` returns
    the same decision without further accumulation.

    Strength: the decision is fastest when the true effect equals the
    pre-specified δ (E[N | H1] roughly half the equivalent fixed-N test).
    Weakness: if the true effect is smaller than δ the test loses power;
    the α/β bounds are only valid for the *specified* alternative.

``MSPRTDetector`` — mixture SPRT / always-valid inference
    Tests H0: μ = μ0 vs H1: μ ≠ μ0 by mixing over a Gaussian prior
    on the alternative effect: δ ~ N(0, τ² σ²).  The resulting test
    statistic is a non-negative martingale under H0, so

        P( sup_{n ≥ 1} Λ_n ≥ 1/α ) ≤ α    (Robbins 1970)

    holds **simultaneously over all n** — the test is always valid.
    This removes the need to commit to a single alternative and gives
    Type-I control under any true effect size, making it the modern
    choice for production A/B testing (Johari, Pekelis, Walsh 2017).

Both detectors estimate the null parameters (μ0, σ) online from the
first ``warmup`` observations, so they are distribution-free in
practice provided the per-turn scores are roughly i.i.d. during warmup.

**Plug-in caveat.** Wald's (α, β) bounds and the always-valid mSPRT
bound are exact only when μ0 and σ are *known*.  With plug-in μ̂, σ̂
estimated from a finite warmup, the bounds are asymptotic — they hold
in the limit of large warmup but admit O(1/√warmup) finite-sample
slack.  In practice this means: with warmup ≥ 100 and post-warmup
streams ≤ a few hundred observations, the empirical Type-I rate is
within ~2× the nominal α; with warmup = 20 it can be substantially
inflated.  Use a large warmup (≥ 100) when accurate Type-I control
matters; switch to a t-mixture variant (Lai & Xing 2010) when σ is
truly unknown and warmup must be small.

References
----------
Wald, A. (1945) "Sequential Tests of Statistical Hypotheses."
Robbins, H. (1970) "Statistical methods related to the law of the
    iterated logarithm." Ann. Math. Statist. 41(5).
Johari, Pekelis & Walsh (2017) "Always Valid Inference: Bringing
    Sequential Analysis to A/B Testing." KDD 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class SPRTState:
    """Snapshot of detector state after one update."""

    log_lr: float
    """Cumulative log-likelihood ratio."""
    n_observations: int
    """Total observations processed (including warmup)."""
    decision: str
    """``"h0"`` (no drift), ``"h1"`` (drift detected), or ``"continue"``."""
    in_warmup: bool
    """True while the detector is still estimating null parameters."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_lr": self.log_lr,
            "n_observations": self.n_observations,
            "decision": self.decision,
            "in_warmup": self.in_warmup,
        }


class SPRTDetector:
    """Sequential Probability Ratio Test for one behavioral score stream.

    Parameters
    ----------
    alpha : float
        Type-I error bound (false positive rate). Default 0.05.
    beta : float
        Type-II error bound (false negative rate). Default 0.20.
    effect_size : float
        Minimum effect size (in standard deviations of the null) we
        want to detect with power 1-beta. Default 0.5 (medium effect,
        Cohen's convention).
    warmup : int
        Number of observations to use for estimating null mean and
        variance before SPRT accumulation starts. Must be ≥ 2.
        Default 5.

    Raises
    ------
    ValueError
        If alpha or beta are outside (0, 0.5), or warmup < 2.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.20,
        effect_size: float = 0.5,
        warmup: int = 5,
    ) -> None:
        if not (0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5); got {alpha}")
        if not (0 < beta < 0.5):
            raise ValueError(f"beta must be in (0, 0.5); got {beta}")
        if warmup < 2:
            raise ValueError(f"warmup must be >= 2; got {warmup}")

        self.alpha = alpha
        self.beta = beta
        self.effect_size = effect_size
        self.warmup = warmup

        # Wald boundaries (in log-space for numerical stability).
        self._log_a = math.log(beta / (1.0 - alpha))  # accept H0 ≤ this
        self._log_b = math.log((1.0 - beta) / alpha)  # accept H1 ≥ this

        self._reset_state()

    def _reset_state(self) -> None:
        self._warmup_buf: list[float] = []
        self._log_lr: float = 0.0
        self._n: int = 0
        self._mu0: float = 0.0
        self._sigma: float = 1.0
        self._delta: float = 1.0 * self.effect_size  # δ = effect_size × σ
        self._decision: str = "continue"

    def update(self, score: float) -> SPRTState:
        """Process one observation. Returns the current decision.

        ``score`` should be a real-valued behavioral signal where larger
        values indicate greater divergence (e.g. 1 - cosine_similarity,
        Hotelling T² per turn, or any axis |delta|).

        Decisions are absorbing: once the log-LR has crossed a Wald
        boundary, subsequent calls return the same decision without
        further accumulation.  Call :meth:`reset` to start a fresh
        sequential test.
        """
        self._n += 1

        # Absorbing barrier: a Wald SPRT terminates at the first crossing.
        # Continuing to accumulate after a decision invalidates the
        # (α, β) error bounds, since post-decision steps would inflate
        # the test's actual Type-I rate via optional stopping.
        if self._decision != "continue":
            return SPRTState(
                log_lr=self._log_lr,
                n_observations=self._n,
                decision=self._decision,
                in_warmup=False,
            )

        if len(self._warmup_buf) < self.warmup:
            self._warmup_buf.append(float(score))
            if len(self._warmup_buf) == self.warmup:
                self._calibrate()
            return SPRTState(
                log_lr=0.0,
                n_observations=self._n,
                decision="continue",
                in_warmup=True,
            )

        # SPRT update: Gaussian log-likelihood ratio increment.
        # log LR_k = δ/σ² * (x_k - μ0) - δ²/(2σ²)
        sigma2 = max(self._sigma**2, 1e-12)
        x_shift = (self._delta / sigma2) * (float(score) - self._mu0)
        increment = x_shift - self._delta**2 / (2.0 * sigma2)
        self._log_lr += increment

        if self._log_lr >= self._log_b:
            self._decision = "h1"
        elif self._log_lr <= self._log_a:
            self._decision = "h0"
        # else: remain "continue"

        return SPRTState(
            log_lr=self._log_lr,
            n_observations=self._n,
            decision=self._decision,
            in_warmup=False,
        )

    def _calibrate(self) -> None:
        """Estimate null mean and std from warmup observations."""
        buf = self._warmup_buf
        n = len(buf)
        self._mu0 = sum(buf) / n
        variance = sum((x - self._mu0) ** 2 for x in buf) / max(n - 1, 1)
        self._sigma = max(math.sqrt(variance), 1e-6)
        self._delta = self.effect_size * self._sigma

    def reset(self) -> None:
        """Reset detector state (preserves hyperparameters)."""
        self._reset_state()

    @property
    def decision(self) -> str:
        return self._decision

    @property
    def n_observations(self) -> int:
        return self._n

    @property
    def log_lr(self) -> float:
        return self._log_lr

    @property
    def boundaries(self) -> tuple[float, float]:
        """Return (log_A, log_B) boundary thresholds."""
        return (self._log_a, self._log_b)

    def __repr__(self) -> str:
        return (
            f"SPRTDetector(alpha={self.alpha}, beta={self.beta}, "
            f"effect_size={self.effect_size}, n={self._n}, "
            f"decision={self._decision!r})"
        )


class MSPRTDetector:
    """Mixture SPRT for always-valid sequential inference.

    Tests H0: μ = μ0 vs H1: μ ≠ μ0 for streaming Gaussian observations
    without specifying a point alternative.  Mixes the simple-vs-simple
    likelihood ratio over a Gaussian prior δ ~ N(0, τ² σ²) on the
    alternative effect.

    For observations x_1, ..., x_n with sample mean x̄_n and known σ²,
    the mixture statistic admits a closed form (Robbins 1970):

        Λ_n = sqrt(σ² / (σ² + n τ² σ²)) ·
              exp( n² (x̄_n − μ0)² τ² / (2 (σ² + n τ² σ²)) )

    Under H0, ``Λ_n`` is a non-negative martingale, so by Ville's
    inequality

        P( sup_{n ≥ 1} Λ_n ≥ 1/α ) ≤ α

    holds simultaneously over **all** sample sizes — there is no
    multiple-testing penalty for peeking, which is why mSPRT is the
    standard choice for production A/B testing.

    Parameters
    ----------
    alpha : float
        Type-I error bound. P(false positive) ≤ alpha across all n.
    tau : float
        Prior std on δ in units of σ.  Larger τ → faster detection of
        large effects, slower for small effects.  τ=1 is a sensible
        default that gives Bayesian "moderate" prior weight.
    warmup : int
        Number of observations used to estimate (μ0, σ²) before the
        mixture statistic starts accumulating.  Must be ≥ 2.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        tau: float = 1.0,
        warmup: int = 5,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0; got {tau}")
        if warmup < 2:
            raise ValueError(f"warmup must be >= 2; got {warmup}")

        self.alpha = alpha
        self.tau = tau
        self.warmup = warmup
        self._log_threshold = math.log(1.0 / alpha)

        self._reset_state()

    def _reset_state(self) -> None:
        self._warmup_buf: list[float] = []
        self._post_warmup_n: int = 0
        self._post_warmup_sum: float = 0.0
        self._n: int = 0
        self._mu0: float = 0.0
        self._sigma2: float = 1.0
        self._log_lambda: float = 0.0
        self._decision: str = "continue"

    def update(self, score: float) -> SPRTState:
        """Process one observation. Returns the current state.

        Once the test rejects H0 the decision is absorbing — subsequent
        calls return the same decision and ``log_lambda`` without further
        updates, preserving the always-valid Type-I bound.
        """
        self._n += 1

        if self._decision != "continue":
            return SPRTState(
                log_lr=self._log_lambda,
                n_observations=self._n,
                decision=self._decision,
                in_warmup=False,
            )

        if len(self._warmup_buf) < self.warmup:
            self._warmup_buf.append(float(score))
            if len(self._warmup_buf) == self.warmup:
                self._calibrate()
            return SPRTState(
                log_lr=0.0,
                n_observations=self._n,
                decision="continue",
                in_warmup=True,
            )

        self._post_warmup_n += 1
        self._post_warmup_sum += float(score)
        n = self._post_warmup_n
        x_bar = self._post_warmup_sum / n

        # Λ_n = sqrt(σ² / (σ² + n τ² σ²))
        #         · exp( n² (x̄ − μ0)² τ² / (2 (σ² + n τ² σ²)) )
        # Working in log-space for stability.
        sigma2 = self._sigma2
        denom = sigma2 + n * self.tau**2 * sigma2
        log_factor = 0.5 * math.log(sigma2 / denom)
        diff2 = (x_bar - self._mu0) ** 2
        exp_arg = (n * n * diff2 * self.tau**2) / (2.0 * denom)
        self._log_lambda = log_factor + exp_arg

        # mSPRT only rejects (one-sided in evidence). Acceptance of H0
        # requires a futility boundary — in always-valid inference,
        # one typically runs until either rejection or a pre-specified
        # max-n. We expose only the "reject" decision; callers can
        # impose their own futility cap externally.
        if self._log_lambda >= self._log_threshold:
            self._decision = "h1"

        return SPRTState(
            log_lr=self._log_lambda,
            n_observations=self._n,
            decision=self._decision,
            in_warmup=False,
        )

    def _calibrate(self) -> None:
        buf = self._warmup_buf
        n = len(buf)
        self._mu0 = sum(buf) / n
        variance = sum((x - self._mu0) ** 2 for x in buf) / max(n - 1, 1)
        self._sigma2 = max(variance, 1e-12)

    def reset(self) -> None:
        self._reset_state()

    @property
    def decision(self) -> str:
        return self._decision

    @property
    def n_observations(self) -> int:
        return self._n

    @property
    def log_lambda(self) -> float:
        return self._log_lambda

    @property
    def threshold(self) -> float:
        """Log-rejection threshold log(1/α)."""
        return self._log_threshold

    def __repr__(self) -> str:
        return (
            f"MSPRTDetector(alpha={self.alpha}, tau={self.tau}, "
            f"n={self._n}, decision={self._decision!r})"
        )


class MultiSPRT:
    """Independent SPRT detector per behavioral axis.

    Maintains one ``SPRTDetector`` per key in ``axis_names``.  Reports
    a drift decision when *any* axis crosses the H1 boundary (union
    bound, conservative but simple).

    Useful in the bisect corner-scorer: updates all axis detectors
    simultaneously per replay run, stops when any axis rejects H0.
    """

    def __init__(
        self,
        axis_names: list[str],
        alpha: float = 0.05,
        beta: float = 0.20,
        effect_size: float = 0.5,
        warmup: int = 5,
    ) -> None:
        self._detectors: dict[str, SPRTDetector] = {
            name: SPRTDetector(alpha=alpha, beta=beta, effect_size=effect_size, warmup=warmup)
            for name in axis_names
        }

    def update(self, scores: dict[str, float]) -> dict[str, SPRTState]:
        """Update each axis detector with its score. Returns per-axis states."""
        return {
            name: det.update(scores[name])
            for name, det in self._detectors.items()
            if name in scores
        }

    @property
    def any_drift_detected(self) -> bool:
        """True when any axis has crossed the H1 (drift) boundary."""
        return any(det.decision == "h1" for det in self._detectors.values())

    @property
    def all_null_accepted(self) -> bool:
        """True when all axes have crossed the H0 (no drift) boundary."""
        return all(det.decision == "h0" for det in self._detectors.values())

    def reset_all(self) -> None:
        for det in self._detectors.values():
            det.reset()


class MSPRTtDetector:
    """Variance-adaptive mixture SPRT for unknown σ.

    The standard :class:`MSPRTDetector` requires σ to be known (or
    well-estimated from a long warmup) for Robbins's always-valid bound
    to be exact. With a short warmup, plug-in σ̂ noise inflates the
    Type-I rate.

    This detector uses the **running sample variance** ``s²_n`` instead
    of a fixed σ̂, updating the variance estimate after every observation
    via Welford's algorithm. The mixture statistic becomes

        Λ_n = sqrt(s²_n / (s²_n + n τ² s²_n)) ·
              exp( n² (x̄_n − μ0)² τ² / (2 (s²_n + n τ² s²_n)) )

    which is equivalent to running the Gaussian mSPRT statistic with
    the most recent variance estimate. This is the practical "t-mixture
    spirit" approach (Lai & Xing 2010): adapt to the data's actual
    variability instead of committing to a single warmup estimate.

    **Important caveats**:

    1. The always-valid bound is **asymptotic**, not exact. Welford-
       updated s²_n is consistent for σ² but not Bayesian-conjugate;
       the running statistic is no longer a strict martingale under H0.
       Empirical Type-I rate matches α only as warmup → ∞.
    2. For *exact* always-valid inference under unknown σ, use the
       Robbins-Siegmund (1973) variance-adaptive SPRT or the time-
       uniform sub-Gaussian bound from Howard, Ramdas, McAuliffe &
       Sekhon (2021) — both require more numerical machinery (Bessel
       functions / nonparametric supermartingale tests) than this
       module currently includes.
    3. For *Type-I control* in production, the recommended path
       remains: use :class:`MSPRTDetector` with a large warmup
       (≥100 observations) so the plug-in σ̂ is essentially the truth.
       This adaptive detector is for exploratory / variance-uncertain
       use cases where the long warmup isn't available.

    Parameters
    ----------
    alpha : float
        Type-I error bound. P(false positive at any n) ≤ alpha
        asymptotically.
    tau : float
        Prior std on δ in units of σ. Same role as in MSPRTDetector.
    warmup : int
        Minimum observations before the statistic starts accumulating.
        Smaller warmup is acceptable here than for MSPRTDetector
        because the variance adapts to new data, but ≥5 is recommended
        so s²_n is at least loosely defined.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        tau: float = 1.0,
        warmup: int = 5,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0; got {tau}")
        if warmup < 2:
            raise ValueError(f"warmup must be >= 2; got {warmup}")

        self.alpha = alpha
        self.tau = tau
        self.warmup = warmup
        self._log_threshold = math.log(1.0 / alpha)

        self._reset_state()

    def _reset_state(self) -> None:
        # Welford running statistics across ALL observations (warmup + post).
        self._n_total: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from mean
        # Post-warmup running mean for the test statistic numerator.
        self._post_n: int = 0
        self._post_sum: float = 0.0
        # Frozen at warmup completion: μ0 reference.
        self._mu0: float = 0.0
        self._log_lambda: float = 0.0
        self._decision: str = "continue"

    def _welford_update(self, x: float) -> None:
        """Online update for running mean and variance (Welford, 1962)."""
        self._n_total += 1
        delta = x - self._mean
        self._mean += delta / self._n_total
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def _running_variance(self) -> float:
        """Sample variance (Bessel-corrected). 0 when fewer than 2 obs."""
        if self._n_total < 2:
            return 0.0
        return self._m2 / (self._n_total - 1)

    def update(self, score: float) -> SPRTState:
        """Process one observation. Returns the current state.

        Decisions are absorbing — once H0 is rejected, subsequent
        updates do not change the decision.
        """
        x = float(score)
        self._welford_update(x)

        if self._decision != "continue":
            return SPRTState(
                log_lr=self._log_lambda,
                n_observations=self._n_total,
                decision=self._decision,
                in_warmup=False,
            )

        if self._n_total <= self.warmup:
            # Still in warmup: lock μ0 to the running mean, but don't
            # accumulate the test statistic yet.
            if self._n_total == self.warmup:
                self._mu0 = self._mean
            return SPRTState(
                log_lr=0.0,
                n_observations=self._n_total,
                decision="continue",
                in_warmup=True,
            )

        # Post-warmup: accumulate sum and recompute statistic.
        self._post_n += 1
        self._post_sum += x
        n = self._post_n
        x_bar = self._post_sum / n
        sigma2 = max(self._running_variance, 1e-12)

        denom = sigma2 + n * self.tau**2 * sigma2
        log_factor = 0.5 * math.log(sigma2 / denom)
        diff2 = (x_bar - self._mu0) ** 2
        exp_arg = (n * n * diff2 * self.tau**2) / (2.0 * denom)
        self._log_lambda = log_factor + exp_arg

        if self._log_lambda >= self._log_threshold:
            self._decision = "h1"

        return SPRTState(
            log_lr=self._log_lambda,
            n_observations=self._n_total,
            decision=self._decision,
            in_warmup=False,
        )

    def reset(self) -> None:
        self._reset_state()

    @property
    def decision(self) -> str:
        return self._decision

    @property
    def n_observations(self) -> int:
        return self._n_total

    @property
    def log_lambda(self) -> float:
        return self._log_lambda

    @property
    def threshold(self) -> float:
        return self._log_threshold

    @property
    def running_variance(self) -> float:
        """Public accessor for the Welford variance estimate."""
        return self._running_variance

    def __repr__(self) -> str:
        return (
            f"MSPRTtDetector(alpha={self.alpha}, tau={self.tau}, "
            f"n={self._n_total}, sigma2_hat={self._running_variance:.4f}, "
            f"decision={self._decision!r})"
        )


__all__ = [
    "MSPRTDetector",
    "MSPRTtDetector",
    "MultiSPRT",
    "SPRTDetector",
    "SPRTState",
]
