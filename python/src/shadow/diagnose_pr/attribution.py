"""Cause attribution for `shadow diagnose-pr`.

Two paths, one entry point per path:

  * `simple_attribution(deltas, has_divergence)` — used when no
    replayable backend is available. Returns one CauseEstimate per
    delta with `confidence=0.5` (no CI). When there's exactly one
    delta AND we observed divergence, that delta gets
    `confidence=1.0` by construction (only one possible explanation).

  * `causal_from_replay(baseline_config, candidate_config,
    replay_fn, ...)` — wraps `shadow.causal.attribution.
    causal_attribution` and converts its output to a list of
    CauseEstimate. Includes bootstrap CI + E-value when those are
    requested.

  * `pick_dominant(causes)` — rank top_causes by `|ATE| * confidence`
    (a coarse v1 ranking; Week 4 may incorporate blast-radius). One
    cause wins or None when no causes exceed a noise floor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from shadow.diagnose_pr.models import CauseEstimate, ConfigDelta

ReplayFn = Callable[[dict[str, Any]], dict[str, float]]
"""Match the contract `shadow.causal.attribution.causal_attribution`
expects: takes a config dict, returns a 5-axis divergence dict with
keys in {semantic, trajectory, safety, verbosity, latency}."""

# Symbolic ATE for simple-attribution (no real causal inference).
# Non-zero so renderers don't drop the cause; small enough that any
# real causal_attribution number dwarfs it.
_SIMPLE_ATE_PLACEHOLDER = 0.01


def _cause_from_delta(d: ConfigDelta, *, confidence: float) -> CauseEstimate:
    """Build a CauseEstimate that propagates the delta's blame text.

    Centralises the blame-pass-through so simple_attribution and
    causal_from_replay both surface file:line + removed/added text
    when the underlying delta has them. The replay path matches by
    `delta_id`; this helper is the simple-path equivalent.
    """
    return CauseEstimate(
        delta_id=d.id,
        axis="trajectory",
        ate=_SIMPLE_ATE_PLACEHOLDER,
        ci_low=None,
        ci_high=None,
        e_value=None,
        confidence=confidence,
        file_path=d.file_path,
        line_no=d.line_no,
        removed_text=d.removed_text,
        added_text=d.added_text,
    )


def simple_attribution(
    *,
    deltas: list[ConfigDelta],
    has_divergence: bool,
) -> list[CauseEstimate]:
    """Best-effort cause attribution without a replay backend.

    No CI is computed (confidence is the only useful signal). With
    exactly one delta and observed divergence, attribution is
    confident-by-construction (1.0); with multiple deltas we can't
    rank them so each gets 0.5. With no observed divergence,
    everything stays at 0.5 — the deltas exist but didn't move
    behavior in the available data.
    """
    if not deltas:
        return []
    if len(deltas) == 1 and has_divergence:
        return [_cause_from_delta(deltas[0], confidence=1.0)]
    return [_cause_from_delta(d, confidence=0.5) for d in deltas]


def causal_from_replay(
    *,
    baseline_config: dict[str, Any],
    candidate_config: dict[str, Any],
    replay_fn: ReplayFn,
    n_replays: int = 1,
    n_bootstrap: int = 0,
    sensitivity: bool = False,
    confidence_level: float = 0.95,
    seed: int = 42,
    deltas: list[ConfigDelta] | None = None,
) -> list[CauseEstimate]:
    """Wrap `shadow.causal.attribution.causal_attribution` and
    convert the multi-dimensional CausalAttribution into a flat
    list of CauseEstimate (one per (delta, axis) pair).

    Each CauseEstimate.confidence is set by whether its bootstrap
    CI excludes zero — 1.0 if it does, 0.5 otherwise. This is the
    coarse v1 marker; Week 4 may refine it.
    """
    from shadow.causal.attribution import causal_attribution

    result = causal_attribution(
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        replay_fn=replay_fn,
        n_replays=n_replays,
        n_bootstrap=n_bootstrap,
        sensitivity=sensitivity,
        confidence_level=confidence_level,
        seed=seed,
    )

    # Build a delta_id -> ConfigDelta lookup so we can propagate
    # file:line blame onto the resulting CauseEstimate. The causal
    # backend identifies deltas by the same `delta.id` extract_deltas
    # produced; matching also picks up the `path/to/prompt.md:LINE`
    # ids when blame was attached upstream.
    delta_by_id: dict[str, ConfigDelta] = {d.id: d for d in (deltas or [])}

    out: list[CauseEstimate] = []
    for delta_id, per_axis_ate in result.ate.items():
        d = delta_by_id.get(delta_id)
        for axis, ate_val in per_axis_ate.items():
            ci_low = result.ci_low.get(delta_id, {}).get(axis) if result.ci_low else None
            ci_high = result.ci_high.get(delta_id, {}).get(axis) if result.ci_high else None
            e_val = result.e_values.get(delta_id, {}).get(axis) if result.e_values else None
            ci_excludes_zero = (
                ci_low is not None and ci_high is not None and (ci_low > 0.0 or ci_high < 0.0)
            )
            out.append(
                CauseEstimate(
                    delta_id=delta_id,
                    axis=axis,
                    ate=ate_val,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    e_value=e_val,
                    confidence=1.0 if ci_excludes_zero else 0.5,
                    file_path=d.file_path if d is not None else None,
                    line_no=d.line_no if d is not None else None,
                    removed_text=d.removed_text if d is not None else None,
                    added_text=d.added_text if d is not None else None,
                )
            )
    # Sort deterministically. Underlying causal_attribution uses sets
    # internally, so dict iteration order can vary across runs even
    # with a fixed RNG seed. We pin (delta_id, axis) order so the
    # report.json is byte-identical across reruns of identical input.
    out.sort(key=lambda c: (c.delta_id, c.axis))
    return out


def pick_dominant(causes: list[CauseEstimate]) -> CauseEstimate | None:
    """Pick the single cause with the largest |ATE| * confidence
    weight, but only when it is *strictly* higher-scoring than every
    other cause.

    Returns None when:
      * the list is empty;
      * the top cause is tied with another cause on score (we have
        multiple plausible explanations, no evidence to crown one).

    The strict-greater rule is what prevents simple_attribution's
    multi-delta tied-at-0.5 case from misleadingly displaying
    "appears to be the main cause" in the PR comment when in fact
    the signal is "any one of these N deltas could be the cause."
    """
    if not causes:
        return None
    scored = [(abs(c.ate) * c.confidence, c) for c in causes]
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1]
    if scored[0][0] > scored[1][0]:
        return scored[0][1]
    return None  # tied at the top — no evidence to crown one


_FIX_HINTS: dict[str, str] = {
    "prompt": (
        "Review the prompt change at `{delta}` — "
        "restore the instruction or constraint it removed."
    ),
    "model": (
        "Pin the model back: revert `{delta}` " "until the candidate model's behavior is verified."
    ),
    "temperature": (
        "Revert `{delta}` to the baseline value, or rerun "
        "with the candidate value against more traces."
    ),
    "tool_schema": (
        "Review the tool schema change at `{delta}` — "
        "confirm the candidate respects the same call protocol as baseline."
    ),
    "retriever": (
        "Review the retriever change at `{delta}` — "
        "confirm document set + ranking match baseline."
    ),
    "policy": (
        "Review the policy change at `{delta}` — " "restore the rule or update affected callers."
    ),
    "unknown": (
        "Investigate the change at `{delta}` — kind is unrecognized, "
        "rerun with --changed-files for better attribution."
    ),
}


def suggested_fix_for(
    dominant: CauseEstimate | None,
    *,
    deltas: list[ConfigDelta],
) -> str | None:
    """Generate a one-sentence fix hint for the dominant cause.

    The hint is deliberately advisory — v1 cannot generate a real
    patch. The text picks a template by `DeltaKind`, names the
    delta, and suggests the recovery action.

    Match by either `delta.id` OR `delta.path` — when
    `--changed-files` overrides `id` to a filename (e.g.
    `prompts/system.md`), the causal layer still references the
    flattened config path (e.g. `prompt.system`), and we need to
    resolve back to the same kind.
    """
    if dominant is None:
        return None
    kind = next(
        (d.kind for d in deltas if d.id == dominant.delta_id or d.path == dominant.delta_id),
        "unknown",
    )
    template = _FIX_HINTS.get(kind, _FIX_HINTS["unknown"])
    # Prefer the delta.id as the displayed name; if our flattened-
    # path delta_id matches a delta with a filename id, use the
    # filename id (it's friendlier in PR comments).
    display_id = next(
        (d.id for d in deltas if d.path == dominant.delta_id and d.id != dominant.delta_id),
        dominant.delta_id,
    )
    return template.format(delta=display_id)


__all__ = [
    "ReplayFn",
    "causal_from_replay",
    "pick_dominant",
    "simple_attribution",
    "suggested_fix_for",
]
