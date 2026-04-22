"""Causal bisection over atomic config deltas.

Phase 4 of the Shadow v0.1 plan: parse two YAML configs into typed deltas,
generate a factorial / Plackett-Burman design, replay the corners, and
fit a LASSO attribution.
"""

from shadow.bisect.attribution import rank_attributions
from shadow.bisect.deltas import Delta, diff_configs
from shadow.bisect.design import full_factorial, plackett_burman
from shadow.bisect.runner import run_bisect

__all__ = [
    "Delta",
    "diff_configs",
    "full_factorial",
    "plackett_burman",
    "rank_attributions",
    "run_bisect",
]
