# Hotelling T² — multivariate two-sample test

**Module:** `shadow.statistical.hotelling`
**Function:** `hotelling_t2(x1, x2, alpha=0.05, permutations=0)`

## What it computes

Given two matrices `X1` (`n1 × D`) and `X2` (`n2 × D`) representing
behavioral fingerprints from a baseline and a candidate trace,
`hotelling_t2` tests:

> **H0**: μ1 = μ2 (no behavioral shift on any axis)
>
> **H1**: μ1 ≠ μ2 (some axis has shifted)

It returns a `HotellingResult` with the T² statistic, its F
approximation, a p-value, and a `reject_null` decision at the given
α level.

## Guarantee

Under multivariate normality with equal covariances, the F transform

  F = ((n1 + n2 − D − 1) / ((n1 + n2 − 2) D)) · T²

is **exactly** F-distributed with (D, n1+n2−D−1) degrees of freedom
under H0. So `p_value < α` is an exact size-α test.

When n1+n2−2 is small relative to D (the "small-sample / high-D"
regime), the pooled covariance is poorly conditioned. Shadow applies
the **Oracle Approximating Shrinkage** estimator (OAS, Chen, Wiesel,
Eldar & Hero 2010):

  Σ̂_OAS = (1 − ρ̂) S + ρ̂ (tr(S)/D) I

with ρ̂ computed in closed form. After shrinkage the F null is no
longer exact — the test is asymptotically valid but has finite-sample
bias. For exact size control at small n, pass `permutations=N` and
the function returns a Monte-Carlo permutation p-value with the
Phipson-Smyth (2010) correction `(b+1)/(B+1)`.

## Algorithm

1. Compute sample means μ̂1, μ̂2 and pooled covariance S.
2. Apply OAS shrinkage to S → Σ̂_OAS.
3. Invert Σ̂_OAS (fall back to pseudo-inverse on singular).
4. T² = (n1·n2)/(n1+n2) · (μ̂1−μ̂2)ᵀ Σ̂_OAS⁻¹ (μ̂1−μ̂2).
5. F-statistic and p-value via `scipy.stats.f.sf` (or permutation
   loop if `permutations > 0`).

## References

- Hotelling, H. (1931). "The generalization of Student's ratio."
- Chen, Wiesel, Eldar & Hero (2010). "Shrinkage Algorithms for MMSE
  Covariance Estimation." IEEE Trans. Signal Processing.
- Anderson, T. W. (2003). *An Introduction to Multivariate
  Statistical Analysis* (3rd ed.). Permutation T² in §5.3.
- Phipson, B. & Smyth, G. K. (2010). "Permutation P-values Should
  Never Be Zero." Statistical Applications in Genetics and
  Molecular Biology 9(1).

## Caveats

- The F-test assumes multivariate normality with equal covariances.
  Real fingerprint dimensions are bounded [0,1] and may be
  non-Gaussian; in practice the test still has reasonable Type-I
  control thanks to the central limit theorem at moderate n, but for
  small n with clearly non-normal data, switch to the permutation
  path.
- "Equal covariances" is the standard pooled-covariance assumption.
  If the candidate has substantially different variance than the
  baseline (heteroscedasticity), the test is anti-conservative —
  consider Box's M test as a pre-check.
- The shrinkage path improves conditioning but does not magically
  recover power when n is genuinely too small. With df2 ≤ 0 the
  function returns p=1.0 (cannot reject) rather than crashing.
