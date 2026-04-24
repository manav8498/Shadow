"""Type stubs for the `shadow._core` Rust extension module.

The implementation lives in `crates/shadow-core/src/python.rs`; this file
exists so mypy --strict can see the surface.
"""

from __future__ import annotations

from typing import Any, TypedDict

__version__: str
SPEC_VERSION: str

class AxisStat(TypedDict):
    """One axis's statistical result. Shape of every element in
    the `rows` list returned by `compute_diff_report`.

    `axis` is one of: semantic, trajectory, safety, verbosity, latency,
    cost, reasoning, judge, conformance.

    `severity` is one of: none, minor, moderate, severe. See
    `Severity::classify` / `classify_rate` in `crates/shadow-core/src/diff/axes.rs`.

    `flags` may contain: `low_power` (n<5) or `ci_crosses_zero`
    (the bootstrap 95% CI strictly straddles zero — a boundary-touching
    CI like [0.0, 1.0] is NOT flagged).

    This TypedDict is documentation of the shape; the runtime value is
    a plain dict.
    """

    axis: str
    baseline_median: float
    candidate_median: float
    delta: float
    ci95_low: float
    ci95_high: float
    severity: str
    n: int
    flags: list[str]

class FirstDivergence(TypedDict):
    """First turn at which the candidate meaningfully diverged from
    the baseline.

    `kind` ∈ {`style_drift`, `decision_drift`, `structural_drift`}:
      - `style_drift`: cosmetic wording only (semantic similarity ≥ 0.9,
        same tool shape, same stop reason).
      - `decision_drift`: same structure, different decision (different
        arg values, refusal flipped, meaningful semantic shift).
      - `structural_drift`: tool sequence differs (insertion, deletion,
        reorder, or tool-name mismatch at the same position).

    `primary_axis` names which of the nine diff axes carries the
    strongest signal for this divergence (semantic / trajectory /
    safety / conformance).

    `confidence` is 0..1; callers can gate display on >= 0.5.
    """

    baseline_turn: int
    candidate_turn: int
    kind: str
    primary_axis: str
    explanation: str
    confidence: float

class DiffReport(TypedDict):
    """Shape of the dict returned by `compute_diff_report`.

    Documentation only — the runtime return value is a plain
    `dict[str, Any]`. Consumers may cast to this TypedDict for static
    access if they want.

    `first_divergence` is `None` when the two traces agree end-to-end,
    otherwise a `FirstDivergence` describing the first meaningful
    behavioural delta (in alignment order).

    `divergences` is the top-K ranked list of divergences, sorted by
    importance (Structural > Decision > Style, then by confidence).
    Empty when the two traces agree. Capped at `DEFAULT_K=5` entries.

    `recommendations` is the list of prescriptive fix suggestions
    derived from the divergences and axis rows. Sorted by severity
    (Error > Warning > Info), capped at 8. Empty when no action is
    warranted.
    """

    rows: list[AxisStat]
    baseline_trace_id: str
    candidate_trace_id: str
    pair_count: int
    first_divergence: FirstDivergence | None
    divergences: list[FirstDivergence]
    recommendations: list[Recommendation]
    drill_down: list[PairDrilldown]

class PairAxisScore(TypedDict):
    """One axis's contribution to a per-pair drill-down row.

    `baseline_value` and `candidate_value` are in raw axis units (ms,
    tokens, USD, similarity ratio, …). `delta` is candidate - baseline.
    `normalized_delta` is `|delta| / axis_scale` clamped to `[0, 4]` —
    1.0 corresponds roughly to one severity-severe-sized movement on
    that axis, so values are summable across axes into a single
    regression score.
    """

    axis: str
    baseline_value: float
    candidate_value: float
    delta: float
    normalized_delta: float

class PairDrilldown(TypedDict):
    """Per-pair breakdown of which axes regressed on a single turn.

    `pair_index` is the 0-based position in the paired-responses list
    (baseline and candidate turn numbers are kept separately for
    future cases where the alignment is non-identity).

    `axis_scores` carries one `PairAxisScore` per axis (Judge excluded
    — the Rust core never populates it).

    `regression_score` is the sum of per-axis `normalized_delta` and is
    the ranking key used to produce the top-K drill-down list.

    `dominant_axis` is the single axis that contributed the most to
    `regression_score`; useful for one-line headlines like
    "the regression at pair #3 was trajectory-driven."
    """

    pair_index: int
    baseline_turn: int
    candidate_turn: int
    axis_scores: list[PairAxisScore]
    regression_score: float
    dominant_axis: str

class Recommendation(TypedDict):
    """A prescriptive fix recommendation derived from a divergence.

    `severity` ∈ {`error`, `warning`, `info`}:
      - `error`: likely a real regression that should block merge
        (structural drift, refusal flip to `content_filter`, trace-
        wide severe axis shift).
      - `warning`: decision drift worth reviewing before merge
        (arg value change, semantic shift).
      - `info`: style drift or low-confidence signal; FYI only.

    `action` ∈ {`restore`, `remove`, `revert`, `review`, `verify`}:
      - `restore`: bring back something the candidate dropped.
      - `remove`: remove something the candidate added without
        justification (e.g. duplicate tool call).
      - `revert`: change a value back to the baseline.
      - `review`: human judgement required; candidate change may be
        intentional.
      - `verify`: low-signal event that might be noise; confirm before
        acting.

    `message` is a one-line imperative action ("Restore X at turn N.").
    `rationale` is a one-line explanation of the triggering signal.
    """

    severity: str
    action: str
    turn: int
    message: str
    rationale: str
    axis: str
    confidence: float

def parse_agentlog(data: bytes) -> list[dict[str, Any]]:
    """Parse a `.agentlog` byte blob into a list of record dicts.

    Each record has keys: `version`, `id`, `kind`, `ts`, `parent`, `meta`
    (optional), `payload`. See SPEC §3.
    """

def write_agentlog(records: list[dict[str, Any]]) -> bytes:
    """Serialize a list of record dicts into `.agentlog` bytes."""

def canonical_bytes(payload: dict[str, Any] | list[Any] | str | int | float | bool | None) -> bytes:
    """Canonical-JSON byte sequence for a payload (SPEC §5)."""

def content_id(payload: dict[str, Any] | list[Any] | str | int | float | bool | None) -> str:
    """Content id for a payload: `"sha256:" + hex(sha256(canonical_bytes))` (SPEC §6)."""

def compute_diff_report(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    pricing: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute a nine-axis DiffReport.

    Parameters
    ----------
    baseline, candidate:
        Parsed `.agentlog` record lists (from [`parse_agentlog`]).
    pricing:
        Optional per-model `{"model-id": (input_$per_mtok, output_$per_mtok)}`
        used by the `cost` axis. When omitted, the cost axis reports 0.
    seed:
        Bootstrap RNG seed. Use a fixed value for reproducible CIs.

    Returns
    -------
    dict[str, Any]
        Shape is documented by the `DiffReport` TypedDict in this stub.
        Keys: `rows` (list of per-axis results, each shaped like
        `AxisStat`), `baseline_trace_id`, `candidate_trace_id`,
        `pair_count`.
    """
