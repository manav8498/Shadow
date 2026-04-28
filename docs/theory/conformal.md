# Conformal prediction — distribution-free coverage bounds

**Module:** `shadow.conformal`
**Functions:** `conformal_calibrate`, `build_parametric_estimate`
**Class:** `ACIDetector`

## What it computes

Given a calibration set of nonconformity scores `s_1, …, s_n`, the
conformal quantile

  q̂ = s_{⌈(n+1)(1−α)⌉}   (sorted ascending)

defines a prediction interval such that, for a future exchangeable
score s_{n+1}:

> P(s_{n+1} ≤ q̂) ≥ 1 − α

The probability is over the joint distribution of all (n+1) scores
under the exchangeability assumption.

## Guarantee

**Distribution-free.** No assumption on the score distribution
beyond exchangeability of the calibration set with future
observations. The guarantee is **exact in finite samples**, not
asymptotic.

This is the headline property: even if the score distribution is
heavy-tailed, multimodal, or anything else, the bound holds.

## Algorithm

1. Sort calibration scores: s_(1) ≤ s_(2) ≤ … ≤ s_(n).
2. q̂ ← s_(⌈(n+1)(1−α)⌉) (1-indexed).
3. For a future score s_{n+1}, declare it "in-distribution" iff
   s_{n+1} ≤ q̂; "out-of-distribution" otherwise.

Shadow's `conformal_calibrate(per_axis_scores, target_coverage,
confidence)` runs this per axis and returns a
`ConformalCoverageReport` with `is_distribution_free=True`.

## ACI — adaptive variant for distribution shift

Standard split conformal assumes exchangeability. Under production
distribution drift, the calibration quantile becomes stale and
coverage degrades.

`ACIDetector` implements **Adaptive Conformal Inference** (Gibbs &
Candès 2021):

  α_{t+1} = α_t + γ · (α_target − I[breach_t])

By the Gibbs-Candès theorem:

> | (1/T) Σ_t I[breach_t] − α_target | ≤ 1/(γT)

The empirical miscoverage converges to the target at rate O(1/T)
under **arbitrary** distribution shift — no exchangeability assumed.

## References

- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.*
- Angelopoulos, A. & Bates, S. (2022). "A Gentle Introduction to
  Conformal Prediction and Distribution-Free Uncertainty Quantification."
- Gibbs, I. & Candès, E. (2021). "Adaptive Conformal Inference Under
  Distribution Shift." NeurIPS 2021.

## Caveats

- **Exchangeability** is the only assumption, but it's a real
  assumption. If the calibration set was collected under conditions
  systematically different from production (different model, different
  load, different time of day), the bound may not hold.
- **Marginal, not conditional.** The guarantee is `P(s ≤ q̂) ≥ 1−α`
  averaged over all future scores. It does NOT mean each individual
  prediction has probability 1-α of being correct. For
  conditional / individual guarantees, look at conformalized
  quantile regression (Romano, Patterson & Candès 2019).
- `build_parametric_estimate` is a **separate** code path (not
  conformal). It synthesizes a Gaussian calibration set from summary
  statistics when the per-run scores have already been aggregated.
  The result is parametric and `is_distribution_free=False`. Don't
  use it where a distribution-free guarantee is contractual; use
  `conformal_calibrate` with real per-run scores.
