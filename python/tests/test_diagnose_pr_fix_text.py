"""Tests for `shadow.diagnose_pr.attribution.suggested_fix_for`.

The fix text is a hint, not a patch. v1 produces one short
sentence per DeltaKind, plus an optional policy-yaml snippet when
the dominant cause is a policy violation."""

from __future__ import annotations

from shadow.diagnose_pr.attribution import suggested_fix_for
from shadow.diagnose_pr.models import CauseEstimate, ConfigDelta


def _cause(delta_id: str, axis: str = "trajectory", confidence: float = 1.0) -> CauseEstimate:
    return CauseEstimate(
        delta_id=delta_id,
        axis=axis,
        ate=0.3,
        ci_low=0.2,
        ci_high=0.4,
        e_value=2.0,
        confidence=confidence,
    )


def _delta(delta_id: str, kind: str) -> ConfigDelta:
    return ConfigDelta(
        id=delta_id,
        kind=kind,  # type: ignore[arg-type]
        path=delta_id,
        old_hash=None,
        new_hash=None,
        display=delta_id,
    )


def test_no_dominant_cause_returns_none() -> None:
    assert suggested_fix_for(None, deltas=[]) is None


def test_prompt_kind_suggests_restoring_instruction() -> None:
    cause = _cause("prompts/system.md")
    deltas = [_delta("prompts/system.md", "prompt")]
    fix = suggested_fix_for(cause, deltas=deltas)
    assert fix is not None
    assert "prompt" in fix.lower() or "instruction" in fix.lower()
    assert "prompts/system.md" in fix


def test_temperature_kind_suggests_reverting() -> None:
    cause = _cause("params.temperature:0.2->0.7")
    deltas = [_delta("params.temperature:0.2->0.7", "temperature")]
    fix = suggested_fix_for(cause, deltas=deltas)
    assert fix is not None
    assert "temperature" in fix.lower()


def test_model_kind_suggests_pinning_back() -> None:
    cause = _cause("model:gpt-4.1->gpt-4.1-mini")
    deltas = [_delta("model:gpt-4.1->gpt-4.1-mini", "model")]
    fix = suggested_fix_for(cause, deltas=deltas)
    assert fix is not None
    assert "model" in fix.lower()


def test_tool_schema_kind_suggests_reviewing_schema() -> None:
    cause = _cause("tools")
    deltas = [_delta("tools", "tool_schema")]
    fix = suggested_fix_for(cause, deltas=deltas)
    assert fix is not None
    assert "schema" in fix.lower() or "tool" in fix.lower()


def test_unknown_kind_returns_generic_hint() -> None:
    """Even an `unknown` delta kind produces a non-empty hint —
    the renderer always wants something to show."""
    cause = _cause("custom.weird")
    deltas = [_delta("custom.weird", "unknown")]
    fix = suggested_fix_for(cause, deltas=deltas)
    assert fix is not None
    assert len(fix) > 10


def test_dominant_cause_with_no_matching_delta_falls_back_gracefully() -> None:
    """If the dominant cause's delta_id doesn't appear in `deltas`
    (defensive against external callers passing inconsistent data),
    we still produce a generic hint instead of crashing."""
    cause = _cause("ghost.delta")
    fix = suggested_fix_for(cause, deltas=[])
    assert fix is not None
