# Sequential Probability Ratio Test — Wald and mixture variants

**Module:** `shadow.statistical.sprt`
**Classes:** `SPRTDetector`, `MSPRTDetector`, `MSPRTtDetector`, `MultiSPRT`

## What they compute

Three sequential detectors for streaming behavioral scores:

| Detector | Tests | When to use |
|---|---|---|
| `SPRTDetector` | H0: μ=μ0 vs H1: μ=μ0+δ (point alternative) | Pre-specified effect size; want fastest stopping |
| `MSPRTDetector` | H0: μ=μ0 vs H1: μ≠μ0 (Gaussian prior over δ, known σ) | Continuous monitoring; can't commit to a δ |
| `MSPRTtDetector` | Same as mSPRT but with running sample variance | Unknown σ; can't run a long warmup |

All three accept a `score` per-update and return an `SPRTState` with
the current `decision` (`"continue"` / `"h0"` / `"h1"`) and the
log-likelihood ratio.

## Guarantees

### Wald SPRT

For known μ0 and σ, with boundaries `log A = log(β/(1−α))` and
`log B = log((1−β)/α)`:

> P(reject H0 | H0) ≤ α and P(accept H0 | H1, true effect = δ) ≤ β

Decisions are **absorbing**: once a boundary is crossed, the test
stops. Continuing to accumulate after a decision invalidates the
(α, β) bounds.

### Mixture SPRT (Robbins 1970)

With a Gaussian prior δ ~ N(0, τ²σ²) and known σ²:

  Λ_n = sqrt(σ²/(σ² + nτ²σ²)) · exp(n²(x̄−μ0)²τ²/(2(σ² + nτ²σ²)))

Λ_n is a non-negative martingale under H0, so by Ville's inequality:

> P(sup_{n ≥ 1} Λ_n ≥ 1/α) ≤ α

This holds **simultaneously over all n** — no multiple-testing
penalty for peeking. The standard choice for production A/B testing
(Johari, Pekelis & Walsh 2017).

### t-mixture mSPRT (variance-adaptive)

Uses Welford-updated running sample variance s²_n in the mSPRT
formula. The bound is **asymptotic**, not exact: for finite warmup
the running variance estimator breaks the strict martingale
property. Use when σ is unknown and a long warmup isn't available;
otherwise prefer `MSPRTDetector` with warmup ≥ 100.

## Algorithm (mSPRT)

1. Estimate (μ̂0, σ̂²) from the first `warmup` observations.
2. After each post-warmup observation x_t:
   - Update running sum and mean x̄_t.
   - Compute log Λ_t via the closed form above.
   - If log Λ_t ≥ log(1/α), reject H0 and freeze the decision.

## References

- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses."
  Annals of Mathematical Statistics 16.
- Robbins, H. (1970). "Statistical methods related to the law of the
  iterated logarithm." Ann. Math. Statist. 41(5).
- Johari, Pekelis & Walsh (2017). "Always Valid Inference: Bringing
  Sequential Analysis to A/B Testing." KDD 2017.
- Lai, T. L. & Xing, H. (2010). "Sequential Analysis: Some Classical
  Problems and New Challenges." Statistica Sinica.

## Caveats — plug-in σ̂

Both Wald SPRT and mSPRT have **exact** error bounds only when σ is
known. Shadow estimates σ̂ from the warmup buffer, so:

- With warmup ≥ 100 and post-warmup streams ≤ a few hundred
  observations, empirical Type-I rate is within ~2× nominal α.
- With warmup = 20, Type-I can be substantially inflated.
- For accurate Type-I control in production, use warmup ≥ 100.

The empirical validation suite (`@pytest.mark.slow`) verifies
Type-I, power, and always-valid bounds at the asymptotic regime
where these guarantees hold tightly.

For exact always-valid inference under unknown σ, the literature
(Lai & Xing 2010; Howard et al. 2021 nonparametric supermartingales)
gives more sophisticated tests. They are not currently implemented;
`MSPRTtDetector` is the closest practical approximation Shadow ships.
