"""Frozen dataclasses for the `shadow diagnose-pr` v0.1 report.

These are the public report shape. Every consumer — PR comment
renderer, JSON writer, future `verify-fix` command — reads these
fields by name. Adding a field is safe; renaming or removing one
is a v0.2 schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["ship", "probe", "hold", "stop"]
"""Four-level verdict gradient. v1 keeps it small to avoid analysis
paralysis on the PR audience; future versions may sub-bin `probe`."""

DeltaKind = Literal[
    "prompt",
    "model",
    "tool_schema",
    "retriever",
    "temperature",
    "policy",
    "unknown",
]
"""Coarse classification of a config-level change. The extractor in
`deltas.py` assigns this; the renderer uses it to phrase the cause.
`unknown` is the safe fallback — better to surface 'unknown delta at
config.X' than to mis-classify."""


@dataclass(frozen=True)
class ConfigDelta:
    """One atomic, named change between baseline and candidate config.

    `id` is the stable identifier the renderer prints (e.g.
    `system_prompt.md:47` for a prompt-file diff or `model:gpt-4.1->
    gpt-4.1-mini` for a top-level field flip). `path` is the
    config-key path or file path the change lives at. Hash fields
    are hex sha256 over canonical bytes when comparable, `None` when
    the side wasn't a hashable artefact (e.g. when the file didn't
    exist on one side).
    """

    id: str
    kind: DeltaKind
    path: str
    old_hash: str | None
    new_hash: str | None
    display: str


@dataclass(frozen=True)
class TraceDiagnosis:
    """Per-trace diagnosis result. Carries the smallest amount of
    state the PR comment + JSON report need; richer analysis lives
    in side artefacts (per-trace diff reports under `.shadow/`).
    """

    trace_id: str
    affected: bool
    risk: float  # 0..100 — corpus-relative severity ranking
    worst_axis: str | None
    first_divergence: dict[str, Any] | None
    policy_violations: list[dict[str, Any]]


@dataclass(frozen=True)
class CauseEstimate:
    """One cause-estimate from causal_attribution, normalised for
    presentation.

    `ci_low`/`ci_high` are `None` when bootstrap wasn't run (e.g.
    `--n-bootstrap 0`). `e_value` is `None` when sensitivity wasn't
    requested. `confidence` is a coarse v1 marker — 1.0 when the CI
    excludes zero, 0.5 otherwise.
    """

    delta_id: str
    axis: str
    ate: float
    ci_low: float | None
    ci_high: float | None
    e_value: float | None
    confidence: float


@dataclass(frozen=True)
class DiagnosePrReport:
    """The full diagnose-pr v0.1 report.

    `dominant_cause` is the single cause the verdict and PR comment
    headline. `top_causes` is the ranked list (typically up to 5)
    surfaced in the JSON for tooling consumers. `flags` carries
    advisory warnings — the v1 set is just `["low_power"]` for n<30.
    """

    schema_version: str
    verdict: Verdict
    total_traces: int
    affected_traces: int
    blast_radius: float
    dominant_cause: CauseEstimate | None
    top_causes: list[CauseEstimate]
    trace_diagnoses: list[TraceDiagnosis]
    affected_trace_ids: list[str]
    new_policy_violations: int
    worst_policy_rule: str | None
    suggested_fix: str | None
    flags: list[str]


__all__ = [
    "CauseEstimate",
    "ConfigDelta",
    "DeltaKind",
    "DiagnosePrReport",
    "TraceDiagnosis",
    "Verdict",
]
