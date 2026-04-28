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

**Status: foundation.** This commit ships the API + a synthetic
ground-truth test that exercises the intervention loop. The
production-grade implementation needs:

1. Larger-scale design matrix (currently we run 1 intervention per
   delta; production needs averaging over multiple replays per cell
   for noise reduction).
2. Confound-adjusted estimands (front-door / back-door correction
   when deltas have shared parents in the config DAG).
3. Integration with the existing ``shadow bisect`` CLI command so
   users opt in via ``--engine causal`` instead of LASSO.

The full causal engine is multi-week scope (4-8 weeks for solid
research-grade implementation matching ICML/NeurIPS standards).
This foundation establishes the API contract and the simplest valid
intervention-based ATE estimator so the production work has a clear
starting point and an end-to-end test to regress against.

References
----------
Pearl, J. (2009). "Causality" (2nd ed.). Cambridge.
Pearl, J. & Mackenzie, D. (2018). "The Book of Why."
Hernán, M. A. & Robins, J. M. (2020). "Causal Inference: What If."
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
