"""Tests for shadow.diagnose_pr.attribution.

Two attribution paths:

  * simple_attribution(deltas, *, has_divergence) — used when no
    replayable backend is available. Lists deltas as candidates
    with confidence=0.5 (no CI). When there's exactly one delta
    AND has_divergence=True, it gets confidence=1.0 by construction
    (only one explanation possible).

  * causal_from_replay(baseline_config, candidate_config, replay_fn,
    n_bootstrap, sensitivity) — wraps the existing
    `shadow.causal.attribution.causal_attribution` and converts its
    output into the diagnose-pr CauseEstimate dataclass. Includes
    bootstrap CI + E-value when those are requested.

  * pick_dominant(causes) — rank top_causes by |ATE| * confidence
    multiplier; return None if no causes exceed the noise floor.
"""

from __future__ import annotations

from typing import Any


def test_simple_attribution_with_no_deltas_returns_empty() -> None:
    from shadow.diagnose_pr.attribution import simple_attribution

    result = simple_attribution(deltas=[], has_divergence=False)
    assert result == []


def test_simple_attribution_one_delta_with_divergence_is_dominant_cause() -> None:
    """One delta + observed divergence ⇒ confidence=1.0 (only one
    possible explanation). The CI fields stay None — no causal
    intervention was actually run."""
    from shadow.diagnose_pr.attribution import simple_attribution
    from shadow.diagnose_pr.models import ConfigDelta

    delta = ConfigDelta(
        id="prompt.system",
        kind="prompt",
        path="prompt.system",
        old_hash="a" * 64,
        new_hash="b" * 64,
        display="prompt.system (changed)",
    )
    causes = simple_attribution(deltas=[delta], has_divergence=True)
    assert len(causes) == 1
    c = causes[0]
    assert c.delta_id == "prompt.system"
    assert c.confidence == 1.0
    assert c.ate != 0.0  # symbolic: any nonzero placeholder
    assert c.ci_low is None
    assert c.ci_high is None


def test_simple_attribution_multiple_deltas_returns_low_confidence_list() -> None:
    """Multiple deltas with no backend: we can't rank them, so each
    gets confidence=0.5 and ATE is symbolic."""
    from shadow.diagnose_pr.attribution import simple_attribution
    from shadow.diagnose_pr.models import ConfigDelta

    deltas = [
        ConfigDelta(
            id="prompt.system",
            kind="prompt",
            path="prompt.system",
            old_hash=None,
            new_hash=None,
            display="x",
        ),
        ConfigDelta(
            id="model:a->b",
            kind="model",
            path="model",
            old_hash=None,
            new_hash=None,
            display="y",
        ),
    ]
    causes = simple_attribution(deltas=deltas, has_divergence=True)
    assert len(causes) == 2
    for c in causes:
        assert c.confidence == 0.5
        assert c.ci_low is None and c.ci_high is None


def test_pick_dominant_returns_highest_ate_times_confidence() -> None:
    from shadow.diagnose_pr.attribution import pick_dominant
    from shadow.diagnose_pr.models import CauseEstimate

    weak = CauseEstimate(
        delta_id="weak",
        axis="verbosity",
        ate=0.1,
        ci_low=-0.1,
        ci_high=0.3,
        e_value=1.1,
        confidence=0.5,
    )
    strong = CauseEstimate(
        delta_id="strong",
        axis="trajectory",
        ate=0.4,
        ci_low=0.3,
        ci_high=0.5,
        e_value=2.5,
        confidence=1.0,
    )
    assert pick_dominant([weak, strong]) is strong


def test_pick_dominant_returns_none_for_empty_list() -> None:
    from shadow.diagnose_pr.attribution import pick_dominant

    assert pick_dominant([]) is None


def test_causal_from_replay_with_known_real_delta_identifies_it() -> None:
    """Two deltas, only one moves the divergence. Real
    causal_attribution() with bootstrap CI must pick it as the
    largest-effect cause."""
    from shadow.diagnose_pr.attribution import causal_from_replay

    baseline = {"system_prompt": "old", "temperature": 0.1}
    candidate = {"system_prompt": "new", "temperature": 0.5}

    def fake_replay(config: dict[str, Any]) -> dict[str, float]:
        # Only system_prompt change moves trajectory; temperature is noise.
        if config.get("system_prompt") == "new":
            return {
                "semantic": 0.0,
                "trajectory": 0.5,
                "safety": 0.0,
                "verbosity": 0.0,
                "latency": 0.0,
            }
        return {
            "semantic": 0.0,
            "trajectory": 0.0,
            "safety": 0.0,
            "verbosity": 0.0,
            "latency": 0.0,
        }

    causes = causal_from_replay(
        baseline_config=baseline,
        candidate_config=candidate,
        replay_fn=fake_replay,
        n_bootstrap=200,
        sensitivity=True,
    )
    assert len(causes) >= 2
    by_id = {c.delta_id: c for c in causes}
    assert "system_prompt" in by_id
    # The system_prompt cause must have a strictly larger ATE on
    # the trajectory axis than the temperature noise cause.
    sys_cause = next(c for c in causes if c.delta_id == "system_prompt" and c.axis == "trajectory")
    temp_cause = next(c for c in causes if c.delta_id == "temperature" and c.axis == "trajectory")
    assert abs(sys_cause.ate) > abs(temp_cause.ate)


def test_causal_from_replay_ci_excludes_zero_for_real_delta() -> None:
    """Stronger property: the real delta's bootstrap CI must
    exclude zero with enough replays. This is what promotes the
    verdict from `probe` to `hold`."""
    from shadow.diagnose_pr.attribution import causal_from_replay

    baseline = {"x": "off"}
    candidate = {"x": "on"}

    def replay(config: dict[str, Any]) -> dict[str, float]:
        return {
            "semantic": 0.0,
            "trajectory": 0.6 if config.get("x") == "on" else 0.0,
            "safety": 0.0,
            "verbosity": 0.0,
            "latency": 0.0,
        }

    causes = causal_from_replay(
        baseline_config=baseline,
        candidate_config=candidate,
        replay_fn=replay,
        n_bootstrap=300,
        sensitivity=False,
    )
    cause = next(c for c in causes if c.delta_id == "x" and c.axis == "trajectory")
    assert cause.ci_low is not None and cause.ci_high is not None
    # CI is well-bracketed around the strong effect
    assert cause.ci_low > 0.0
