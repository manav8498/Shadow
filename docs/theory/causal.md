# Causal attribution — intervention-based ATE

**Module:** `shadow.causal`
**Function:** `causal_attribution`
**Status:** Foundation. Production-grade extensions (back-door
adjustment, bootstrap CI, optimal experiment design) deferred.

## What it computes

Given a baseline configuration, a candidate configuration, and a list
of named deltas that distinguish them, attribute each delta's
contribution to a per-axis behavioral divergence.

For each delta d_i where `baseline[d_i] ≠ candidate[d_i]`:

> ATE(d_i) = E[Y | do(d_i = candidate)] − E[Y | do(d_i = baseline)]

The `do` operator (Pearl) means "intervene on d_i alone, holding all
other deltas fixed." This is **causation**, not correlation: the ATE
measures the effect of changing only d_i, not the regression
coefficient when many things change at once.

## Why this replaces LASSO

The current `shadow.bisect` fits a LASSO regression from delta-
presence vectors to per-axis divergences and ranks deltas by
coefficient magnitude. That's correlational:

- **Confounded deltas** that ride with a real cause score high.
- **Correlated deltas** split credit (multicollinearity).
- The answer is descriptive — "delta_3 was associated with the
  regression" — not causal.

Pearl's do-calculus framework gives the right answer: actually
intervene, replay, measure. The ATE is unbiased under the random-
intervention design.

## Algorithm

1. Identify differing deltas: D = {d : baseline[d] ≠ candidate[d]}.
2. **Control runs**: replay `baseline` `n_replays` times → mean
   divergence vector Y_base.
3. For each delta d ∈ D:
   - Build `intervened = dict(baseline) | {d: candidate[d]}`.
   - Replay `intervened` `n_replays` times → mean Y_d.
   - ATE(d) = Y_d − Y_base, per axis.
4. Return per-(delta, axis) ATE estimates.

The `replay_fn` is user-supplied so callers wire whatever backend
they use (mock, Rust replay, live LLM).

## References

- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge.
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why.*
- Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If.*
- Imbens, G. W. & Rubin, D. B. (2015). *Causal Inference for
  Statistics, Social, and Biomedical Sciences.*

## Caveats

The current implementation is **foundation-stage**. It computes the
single-delta ATE correctly, but doesn't yet handle:

- **Confounding when deltas have shared parents** in the config DAG.
  E.g., changing the model also changes the default temperature; the
  per-delta intervention then conflates two effects. Back-door /
  front-door adjustment is the standard fix.
- **Variance estimates on the ATE.** The current API returns point
  estimates; bootstrap CIs and Welch-style standard errors are
  needed for confidence statements.
- **Optimal experiment design.** When intervention budget is
  limited (each replay costs an LLM call), the algorithm should
  prioritise high-information interventions. Currently it runs all
  N interventions naively.
- **Sensitivity analysis** (Rosenbaum bounds for unmeasured
  confounding).

These are real research-grade extensions documented in the literature
above. The foundation API is stable; the extensions plug in over
time without breaking callers.

For the current narrow case (deterministic `MockLLM` replays, no
hidden confounding, n_replays = 1), the simple difference-of-means
ATE is unbiased and the algorithm works correctly — verified by the
synthetic 3-real-deltas + 5-noise-deltas ground-truth test.
