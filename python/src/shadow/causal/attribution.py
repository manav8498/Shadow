"""Intervention-based attribution: Average Treatment Effect (ATE).

Given a baseline trace, a candidate trace, and a list of named deltas
that distinguish the two, this module:

1. For each delta, constructs a "single-delta candidate" by injecting
   only that delta into the baseline configuration.
2. Replays each single-delta candidate (using a user-supplied replay
   function — the module is interface-only here).
3. Computes the resulting per-axis divergence vector.
4. Reports the ATE per (delta, axis) pair, sorted by absolute effect.

Compared to LASSO (the legacy ``shadow.bisect``):

- LASSO regresses divergence on a binary delta-presence matrix and
  ranks deltas by coefficient. Confounders inflate scores; correlated
  deltas split credit; the answer is descriptive, not causal.
- Intervention-based ATE asks "what if THIS delta hadn't fired?" and
  measures the actual effect. Confounders need explicit treatment
  (front-door / back-door adjustment) but are no longer hidden in the
  coefficient.

For now, ATE is computed as a single-replay difference per delta.
The production-grade variant averages multiple replays per delta
to reduce variance — the API supports it via the ``n_replays`` arg
but the simple version with n_replays=1 is correct on noise-free
deterministic replays (which is the common case for Shadow's
``MockLLM`` backend).

Foundation scope
----------------
This module ships:
- The API (``causal_attribution``, ``CausalAttribution``,
  ``InterventionResult``).
- A pluggable ``replay_fn`` parameter so callers wire in their own
  replay backend (Rust, Python, mock, real LLM).
- A simple n_replays=1 difference-of-means ATE estimator.
- An end-to-end test that exercises the API on a synthetic scenario
  with 3 real deltas + 5 noise deltas, where the algorithm correctly
  attributes effect to the 3 real deltas.

Not in scope (for follow-up work):
- Confound-adjusted estimands (back-door / front-door).
- Bootstrap CI on ATE estimates.
- Integration with the ``shadow bisect`` CLI command (separate commit).
- Optimal experiment design (which deltas to intervene on first when
  the budget is limited).
- Sensitivity analysis (Rosenbaum bounds for unmeasured confounding).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Type aliases for clarity.
DeltaName = str
AxisName = str
DivergenceVector = dict[AxisName, float]


@dataclass(frozen=True)
class InterventionResult:
    """Result of running ONE intervention (replay with one delta toggled)."""

    delta: DeltaName
    """The delta that was injected from candidate config into baseline."""
    divergence: DivergenceVector
    """Per-axis divergence of the resulting replay vs the baseline run."""

    def to_dict(self) -> dict[str, Any]:
        return {"delta": self.delta, "divergence": dict(self.divergence)}


@dataclass(frozen=True)
class CausalAttribution:
    """Per-(delta, axis) ATE estimate, sorted by absolute effect."""

    # Mapping: delta_name -> axis_name -> ATE
    ate: dict[DeltaName, dict[AxisName, float]] = field(default_factory=dict)
    """Average treatment effect per delta, per axis."""
    interventions: list[InterventionResult] = field(default_factory=list)
    """Raw per-intervention divergence vectors (for diagnostics)."""

    def top(self, axis: AxisName, k: int = 5) -> list[tuple[DeltaName, float]]:
        """Return top-k deltas by |ATE| on the given axis."""
        scored = [(d, abs(self.ate.get(d, {}).get(axis, 0.0))) for d in self.ate]
        scored.sort(key=lambda x: -x[1])
        return [
            (d, self.ate.get(d, {}).get(axis, 0.0))
            for d, _ in scored[:k]
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ate": {d: dict(axes) for d, axes in self.ate.items()},
            "interventions": [iv.to_dict() for iv in self.interventions],
        }


# A replay function takes a "config" (a dict of named deltas) and
# returns a per-axis divergence vector relative to the baseline.
# Callers wire this to whatever backend they use — mock, Rust replay,
# live LLM.
ReplayFn = Callable[[dict[DeltaName, Any]], DivergenceVector]


def causal_attribution(
    *,
    baseline_config: dict[DeltaName, Any],
    candidate_config: dict[DeltaName, Any],
    replay_fn: ReplayFn,
    n_replays: int = 1,
    axes: list[AxisName] | None = None,
) -> CausalAttribution:
    """Estimate per-delta ATE via single-delta interventions.

    For each delta where ``baseline_config[d] != candidate_config[d]``:

    1. Build ``intervened_config = dict(baseline_config)`` then
       set ``intervened_config[d] = candidate_config[d]``.
    2. Run ``replay_fn(intervened_config)`` ``n_replays`` times and
       average the per-axis divergence vectors.
    3. Run ``replay_fn(baseline_config)`` ``n_replays`` times for
       the control mean.
    4. ATE on each axis = intervened_mean − control_mean.

    Returns
    -------
    CausalAttribution with per-(delta, axis) ATE values and the raw
    intervention results for diagnostics.

    Raises
    ------
    ValueError
        If baseline_config and candidate_config have no differing
        keys (nothing to attribute), or n_replays < 1.
    """
    if n_replays < 1:
        raise ValueError(f"n_replays must be >= 1; got {n_replays}")

    # Identify the deltas (keys where baseline and candidate differ).
    differing = [
        k for k in candidate_config if baseline_config.get(k) != candidate_config[k]
    ]
    if not differing:
        raise ValueError(
            "no differing keys between baseline_config and candidate_config — "
            "nothing to attribute. Did you pass two identical configs?"
        )

    # Control: replay the baseline n_replays times and average.
    control_runs = [replay_fn(dict(baseline_config)) for _ in range(n_replays)]
    control_mean = _mean_divergence(control_runs)

    interventions: list[InterventionResult] = []
    ate: dict[DeltaName, dict[AxisName, float]] = {}

    for delta in differing:
        intervened_config = dict(baseline_config)
        intervened_config[delta] = candidate_config[delta]
        intervention_runs = [
            replay_fn(intervened_config) for _ in range(n_replays)
        ]
        intervened_mean = _mean_divergence(intervention_runs)

        # ATE per axis = intervened mean − control mean.
        all_axes = set(intervened_mean.keys()) | set(control_mean.keys())
        if axes is not None:
            all_axes &= set(axes)
        per_axis_ate = {
            ax: intervened_mean.get(ax, 0.0) - control_mean.get(ax, 0.0)
            for ax in all_axes
        }
        ate[delta] = per_axis_ate
        interventions.append(
            InterventionResult(delta=delta, divergence=intervened_mean)
        )

    return CausalAttribution(ate=ate, interventions=interventions)


def _mean_divergence(runs: list[DivergenceVector]) -> DivergenceVector:
    """Element-wise mean of a list of divergence vectors."""
    if not runs:
        return {}
    keys: set[str] = set()
    for run in runs:
        keys.update(run.keys())
    return {k: sum(run.get(k, 0.0) for run in runs) / len(runs) for k in keys}
