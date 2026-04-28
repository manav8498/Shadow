"""Causal attribution for behavioral regressions (Pearl-style do-calculus).

Replaces the LASSO-based bisection (correlational) with proper
intervention-based attribution. The current ``shadow.bisect``
fits a linear model from config-deltas to per-axis divergences and
ranks deltas by their LASSO coefficient — that's correlation, not
causation. Confounded deltas score high; truly-irrelevant deltas
sometimes score high too.

Pearl's do-calculus framework gives the right answer: for each
candidate delta, we *intervene* on that delta alone (all others held
constant), replay, and measure the resulting effect. The attribution
weight is the average treatment effect:

    ATE(delta_i) = E[Y | do(delta_i = candidate)] − E[Y | do(delta_i = baseline)]

Computed as a difference-of-means across replays. With the existing
counterfactual primitives in :mod:`shadow.counterfactual_loop`
(``branch_at_turn``, ``replace_tool_result``, ``replace_tool_args``),
the intervention is mechanical: substitute the single delta into the
baseline trace, replay, measure the diff vector.

**Status: production.** Ships:

1. Point ATE via single-delta interventions, with ``n_replays`` for
   noise reduction.
2. Bootstrap percentile CIs on the ATE when ``n_bootstrap > 0``
   (Efron 1979). The resampling is stratum-aware so back-door-adjusted
   estimates carry honest CIs.
3. Back-door adjustment for named confounders via uniform-weighted
   stratification over confounder-value combinations (Pearl 2009 §3.3).
4. E-value sensitivity to unmeasured confounding (VanderWeele &
   Ding 2017) — reports the smallest unmeasured-confounder effect
   that could explain away the observed ATE.

Out of scope (deferred):
- Front-door adjustment (rare in practice; back-door covers the
  typical agent-config confounding pattern).
- Optimal experiment design (which deltas to intervene on first
  under a fixed budget).
- Rosenbaum bounds for matched designs (E-value subsumes the
  common use case for unmatched, continuous-outcome work).

References
----------
Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
Efron, B. (1979). "Bootstrap methods: Another look at the jackknife".
  *Ann. Statist.* 7(1): 1-26.
VanderWeele, T. J. & Ding, P. (2017). "Sensitivity analysis in
  observational research: introducing the E-value". *Annals of
  Internal Medicine* 167(4): 268-274.
Hernán, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
"""

from shadow.causal.attribution import (
    CausalAttribution,
    InterventionResult,
    causal_attribution,
)

__all__ = [
    "CausalAttribution",
    "InterventionResult",
    "causal_attribution",
]
