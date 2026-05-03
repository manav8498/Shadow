"""Causal Regression Forensics for AI Agents.

`shadow diagnose-pr` answers, in one PR comment:

  1. Did agent behavior change?
  2. How many real or production-like traces are affected?
  3. Which exact candidate change caused the regression?
  4. How confident are we?
  5. What fix should be verified before merge?

This package composes existing Shadow internals (the 9-axis Rust
differ, `shadow.causal.attribution`, `shadow.hierarchical` policy
checker, `shadow.mine` representative selection) into one PR-time
command surface. It does not reinvent any of them.

The `v0.1` schema is intentionally narrow:

  * verdict: ship / probe / hold / stop
  * blast_radius: affected / total
  * dominant_cause: a single ConfigDelta with ATE + bootstrap CI +
    E-value, or None when causal attribution wasn't run.

Forward compatibility: future versions add fields without renaming
existing ones; readers must tolerate unknown keys.
"""

from __future__ import annotations

SCHEMA_VERSION = "diagnose-pr/v0.1"
"""Schema identifier embedded in every diagnose-pr report.json.

Bumped only on breaking field renames or removals; additive changes
(new optional fields) keep the same schema_version.
"""

DEFAULT_MAX_TRACES = 200
"""Cap on traces fed to the per-trace replay/diff loop. When the input
corpus exceeds this, `shadow.mine.mine` selects a representative
sample using the failure-mode score (errors / refusals / high latency
rank highest)."""

DEFAULT_N_BOOTSTRAP = 500
"""Default bootstrap resample count for confidence intervals on
causal ATE. 500 is a common floor; raise via --n-bootstrap when
corpora are small and CI width matters."""

DEFAULT_CONFIDENCE = 0.95
"""Default CI level. Anything tighter loses interpretability for the
PR audience; anything looser stops being meaningful."""


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAX_TRACES",
    "DEFAULT_N_BOOTSTRAP",
    "SCHEMA_VERSION",
]
