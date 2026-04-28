"""Causal bisection over atomic config deltas.

Three modes, in priority order:

1. `lasso_over_corners` — real LASSO-over-corners scorer. Requires a
   live LLM backend. Runs every corner of the category-level design
   matrix and fits LASSO per axis on the resulting divergence matrix.
   Produces principled per-category attribution.
2. `heuristic_kind_allocator` — when no backend is available but both
   baseline and candidate traces exist. Computes real divergence from
   the two traces, allocates across delta categories by
   `DELTA_KIND_AFFECTS`.
3. `lasso_placeholder_zero` — neither backend nor candidate traces.
   Returns shape-correct zero attributions and a warning.
"""

from shadow.bisect.apply import (
    CONFIG_CATEGORIES,
    active_categories,
    apply_config_to_request,
    build_intermediate_config,
)
from shadow.bisect.attribution import (
    AXIS_NAMES,
    rank_attributions,
    rank_attributions_with_ci,
    rank_attributions_with_interactions,
)
from shadow.bisect.corner_scorer import (
    replay_with_config,
    score_corners,
)
from shadow.bisect.corner_scorer import (
    run_sync as run_corner_bisect,
)
from shadow.bisect.deltas import Delta, diff_configs
from shadow.bisect.design import full_factorial, plackett_burman
from shadow.bisect.runner import run_bisect

__all__ = [
    "AXIS_NAMES",
    "CONFIG_CATEGORIES",
    "Delta",
    "active_categories",
    "apply_config_to_request",
    "build_intermediate_config",
    "diff_configs",
    "full_factorial",
    "plackett_burman",
    "rank_attributions",
    "rank_attributions_with_ci",
    "rank_attributions_with_interactions",
    "replay_with_config",
    "run_bisect",
    "run_corner_bisect",
    "score_corners",
]
