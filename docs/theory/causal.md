# Causal attribution — intervention-based ATE

**Module:** `shadow.causal`
**Function:** `causal_attribution`
**Status:** Production. Bootstrap CIs, back-door adjustment, and
E-value sensitivity ship in v2.7.

## What it computes

Given a baseline configuration, a candidate configuration, and a list
of named deltas that distinguish them, attribute each delta's
contribution to a per-axis behavioral divergence.

For each delta `d_i` where `baseline[d_i] ≠ candidate[d_i]`:

> ATE(d_i) = E[Y | do(d_i = candidate)] − E[Y | do(d_i = baseline)]

The `do` operator (Pearl) means "intervene on `d_i` alone, holding
all other deltas fixed." This is **causation**, not correlation: the
ATE measures the effect of changing only `d_i`, not the regression
coefficient when many things change at once.

## Why this replaces LASSO

The legacy `shadow.bisect` fits a LASSO regression from delta-presence
vectors to per-axis divergences and ranks deltas by coefficient
magnitude. That's correlational:

- **Confounded deltas** that ride with a real cause score high.
- **Correlated deltas** split credit (multicollinearity).
- The answer is descriptive — "delta_3 was associated with the
  regression" — not causal.

Pearl's do-calculus framework gives the right answer: actually
intervene, replay, measure. The ATE is unbiased under the random-
intervention design.

`shadow.bisect` is still useful when you can't afford the
intervention budget (each replay costs an LLM call). In that regime
the v2.5+ `rank_attributions_with_ci` adds Meinshausen-Bühlmann
stability selection and residual-bootstrap CIs to the LASSO
coefficients — descriptive but with honest uncertainty. For a
causal claim, use `shadow.causal`.

## Algorithm

### 1. Point ATE

1. Identify target deltas: D = {d : baseline[d] ≠ candidate[d], d ∉ confounders}.
2. **Control runs:** replay the baseline `n_replays` times → mean
   divergence vector Y_base.
3. For each target delta d ∈ D:
   - Build `intervened = dict(baseline) | {d: candidate[d]}`.
   - Replay `intervened` `n_replays` times → mean Y_d.
   - ATE(d) = Y_d − Y_base, per axis.

### 2. Bootstrap CIs (`n_bootstrap > 0`)

Efron (1979) percentile bootstrap on the per-arm replay outputs:

1. For each bootstrap replicate b = 1..B:
   - Resample the control runs and intervention runs **with
     replacement, within each stratum** (for back-door-adjusted
     estimates the resampling preserves stratum identity).
   - Compute the resampled ATE.
2. Report the (α/2, 1−α/2) percentiles across B replicates as the CI.

### 3. Back-door adjustment (`confounders=[...]`)

When configured confounders C ⊆ keys differ between baseline and
candidate, naive single-delta interventions either (a) leave the
confounders at baseline values (which may not generalise) or
(b) flip them along with the delta (which conflates effects).
Pearl's back-door criterion gives the unbiased estimator:

> ATE_BD(d) = Σ_c P(C=c) · [E[Y | do(d=cand), C=c] − E[Y | do(d=base), C=c]]

Implementation: enumerate the 2^|C| combinations of (baseline, candidate)
values across the confounders, run a per-stratum ATE on each, and
average with uniform weights. (Other weighting schemes — propensity-
score-derived, sampling-frequency-weighted — can be layered on top
by computing per-stratum ATEs externally; the built-in estimator
keeps the inspectable uniform-weight default.)

### 4. E-value sensitivity (`sensitivity=True`)

VanderWeele & Ding (2017) E-value: the smallest unmeasured-
confounder effect size that could explain away the observed ATE.

For continuous outcomes:

  d   = ATE / SD_pooled                    (Cohen's d, equal-n)
  RR  ≈ exp(0.91 · |d|)                    (continuous-to-RR transform)
  E   = RR + sqrt(RR · (RR − 1))           (VanderWeele-Ding Eq. 4)

Larger effects → larger E-values → harder to explain away. An
E-value of 1.0 means the observed effect is null (no confounding
strength needed). The pooled SD is computed across the union of
control and intervention runs.

## API

```python
from shadow.causal import causal_attribution

result = causal_attribution(
    baseline_config={"prompt": "v1", "model": "haiku", "temp": 0.2},
    candidate_config={"prompt": "v2", "model": "sonnet", "temp": 0.7},
    replay_fn=my_replay,
    n_replays=10,            # 10 replays per cell for noise reduction
    n_bootstrap=500,         # 500-replicate bootstrap → percentile CI
    confounders=["model"],   # back-door adjust over model
    sensitivity=True,        # compute E-value
)

# Point ATE per (delta, axis)
print(result.ate["prompt"]["semantic"])

# Bootstrap 95% CI
print(result.ci_low["prompt"]["semantic"], result.ci_high["prompt"]["semantic"])

# Sensitivity: how much unmeasured confounding would explain this away?
print(result.e_values["prompt"]["semantic"])
```

The `replay_fn` is user-supplied so callers wire whatever backend
they use (mock, Rust replay, live LLM).

## Caveats and out-of-scope

- **Front-door adjustment.** Rare in practice for this domain; the
  back-door identification covers the typical agent-config
  confounding pattern.
- **Optimal experiment design.** When intervention budget is
  limited, the algorithm should prioritise high-information
  interventions. Currently it runs all combinations naively.
  Sequential / Bayesian-optimal designs are deferred.
- **Rosenbaum bounds for matched designs.** The E-value subsumes the
  common use case for unmatched continuous-outcome work; matched-pair
  Rosenbaum bounds are not implemented.

## References

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Efron, B. (1979). "Bootstrap methods: another look at the
  jackknife". *Annals of Statistics* 7(1): 1-26.
- VanderWeele, T. J. & Ding, P. (2017). "Sensitivity analysis in
  observational research: introducing the E-value". *Annals of
  Internal Medicine* 167(4): 268-274.
- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If.*
- Imbens, G. W. & Rubin, D. B. (2015). *Causal Inference for
  Statistics, Social, and Biomedical Sciences.*
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why.*
