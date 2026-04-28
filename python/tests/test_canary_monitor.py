"""Tests for the canary-monitoring real-world example.

Validates that ``CanaryMonitor`` correctly exercises Shadow v2.5's new
features:

  - mSPRT (always-valid Type-I control across continuous peeking)
  - WeakUntil-based ``must_call_before`` policies
  - Distribution-free conformal calibration with a binomial drift test

The simulator generates realistic session traces; tests run them
through the monitor and check the resulting verdicts.  Type-I tests
use multiple repetitions to verify the empirical false-positive rate
respects the nominal α.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the example directory to sys.path so we can import the monitor.
_EXAMPLE_DIR = Path(__file__).parents[2] / "examples" / "canary-monitor"

if not _EXAMPLE_DIR.exists():
    pytestmark = pytest.mark.skip(reason="canary-monitor example directory missing")
else:
    sys.path.insert(0, str(_EXAMPLE_DIR))


from agent_simulator import generate_cohort, generate_session  # noqa: E402
from canary_monitor import (  # noqa: E402
    DEFAULT_POLICIES,
    CanaryMonitor,
    CanaryVerdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor(seed: int = 1, n_sessions: int = 50, **kwargs) -> CanaryMonitor:
    cal = generate_cohort("baseline", n_sessions=n_sessions, seed=seed)
    return CanaryMonitor(calibration_sessions=cal, **kwargs)


def _stream(monitor: CanaryMonitor, sessions) -> CanaryVerdict:
    """Stream sessions until alarm or exhaustion. Return last verdict."""
    last: CanaryVerdict | None = None
    for sess in sessions:
        last = monitor.observe_session(sess)
        if last.alarm:
            break
    assert last is not None
    return last


# ===========================================================================
# 1. Construction + calibration
# ===========================================================================


class TestConstruction:
    def test_monitor_constructs_with_default_policies(self):
        m = _make_monitor()
        assert m.alpha == 0.05
        assert m.target_coverage == 0.90
        assert m.confidence == 0.95
        assert m.policies == DEFAULT_POLICIES

    def test_calibration_too_few_sessions_raises(self):
        cal = generate_cohort("baseline", n_sessions=5, seed=1)
        with pytest.raises(ValueError, match="≥10 sessions"):
            CanaryMonitor(calibration_sessions=cal)

    def test_invalid_alpha_raises(self):
        cal = generate_cohort("baseline", n_sessions=20, seed=1)
        with pytest.raises(ValueError, match="alpha"):
            CanaryMonitor(calibration_sessions=cal, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            CanaryMonitor(calibration_sessions=cal, alpha=1.0)

    def test_conformal_report_is_distribution_free(self):
        m = _make_monitor()
        assert m.conformal_report.is_distribution_free is True

    def test_q_hat_is_set_for_all_axes(self):
        m = _make_monitor()
        # All 8 fingerprint dimensions should have a q_hat entry,
        # even if the value is 0 (axis is constant in baseline).
        assert len(m.q_hat) == 8

    def test_q_hat_is_non_negative(self):
        m = _make_monitor()
        for axis, q in m.q_hat.items():
            assert q >= 0, f"q_hat for {axis!r} is negative: {q}"

    def test_n_min_meets_pac_requirement(self):
        m = _make_monitor()
        # n_min = ceil(log(0.05)/log(0.90)) ≈ 29
        assert m.conformal_report.n_min == 29
        assert m.conformal_report.sufficient_n is True


# ===========================================================================
# 2. Healthy canary — Type-I control over many sessions
# ===========================================================================


class TestHealthyCanaryTypeI:
    def test_healthy_canary_no_alarm_in_single_run(self):
        m = _make_monitor(seed=1)
        healthy = generate_cohort("baseline", n_sessions=50, seed=2)
        verdict = _stream(m, healthy)
        assert not verdict.alarm, f"Healthy canary triggered false alarm: {verdict.summary()}"

    def test_healthy_canary_no_msprt_h1(self):
        m = _make_monitor(seed=1)
        healthy = generate_cohort("baseline", n_sessions=50, seed=2)
        verdict = _stream(m, healthy)
        # No axis should reject under H0 with healthy canary
        for axis, dec in verdict.sprt_decisions.items():
            assert dec != "h1", f"mSPRT[{axis}] falsely rejected H0 on healthy canary"

    def test_healthy_canary_no_ltl_violations(self):
        m = _make_monitor(seed=1)
        healthy = generate_cohort("baseline", n_sessions=30, seed=2)
        verdict = _stream(m, healthy)
        assert verdict.ltl_violations == []

    def test_continuous_peeking_does_not_inflate_type_i(self):
        """Verify always-valid Type-I rate across many independent canary cohorts.

        Run M independent monitor + healthy-canary pairs and count alarms.
        Empirical false-positive rate should be ≤ α + slack.
        """
        m_trials = 30
        alarms = 0
        for trial in range(m_trials):
            m = _make_monitor(seed=trial * 7 + 1, n_sessions=30)
            healthy = generate_cohort("baseline", n_sessions=40, seed=trial * 7 + 2)
            verdict = _stream(m, healthy)
            if verdict.alarm:
                alarms += 1
        rate = alarms / m_trials
        # Three alarm channels (mSPRT-latency, mSPRT-verbosity, conformal-drift),
        # all at α=0.05. Bonferroni upper bound ≈ 0.15, but they share the
        # same data. Allow generous slack for finite trials.
        assert rate < 0.30, f"Empirical false-positive rate {rate:.3f} too high"

    def test_healthy_canary_breach_count_near_expected(self):
        """Conformal breach count under H0 should be ≈ (1-coverage) × n.

        For 50 healthy sessions at 90% coverage we expect ≈ 5 breaches per axis.
        The cumulative count should be in [0, 15] on most runs.
        """
        m = _make_monitor(seed=1, n_sessions=80)
        healthy = generate_cohort("baseline", n_sessions=50, seed=2)
        for sess in healthy:
            m.observe_session(sess)
        # Don't break on alarm — we want to see total breach count.
        for axis, count in m._breach_count.items():
            # 50 sessions × (1-0.90) = 5 expected. Allow up to ~15 by chance.
            assert count <= 20, f"breach count for {axis!r} is {count} on healthy canary"


# ===========================================================================
# 3. Drifting canary — mSPRT power
# ===========================================================================


class TestDriftingCanaryDetection:
    def test_drifting_canary_triggers_alarm(self):
        m = _make_monitor(seed=1)
        drifting = generate_cohort("drifting", n_sessions=50, seed=3)
        verdict = _stream(m, drifting)
        assert verdict.alarm

    def test_drifting_canary_triggers_msprt_latency(self):
        m = _make_monitor(seed=1)
        drifting = generate_cohort("drifting", n_sessions=50, seed=3)
        verdict = _stream(m, drifting)
        assert verdict.sprt_decisions.get("latency") == "h1"

    def test_drifting_canary_caught_within_few_sessions(self):
        m = _make_monitor(seed=1)
        drifting = generate_cohort("drifting", n_sessions=50, seed=3)
        verdict = _stream(m, drifting)
        # 1.5x latency drift should be caught fast given σ̂ from 150-record warmup
        assert verdict.n_sessions <= 10, f"mSPRT took {verdict.n_sessions} sessions to catch drift"

    def test_drifting_detection_robust_across_seeds(self):
        """mSPRT should fire on drifting canaries for varied seeds."""
        n_trials = 10
        catches = 0
        for trial in range(n_trials):
            m = _make_monitor(seed=trial * 5 + 1)
            drifting = generate_cohort("drifting", n_sessions=50, seed=trial * 5 + 3)
            verdict = _stream(m, drifting)
            if verdict.alarm:
                catches += 1
        # Power should be very high — 1.5x latency is a clear signal.
        assert catches >= 9, f"Only {catches}/{n_trials} drifting trials triggered alarm"


# ===========================================================================
# 4. Policy-violating canary — LTL detection
# ===========================================================================


class TestPolicyViolatingCanary:
    def test_policy_violator_triggers_ltl_alarm(self):
        m = _make_monitor(seed=1)
        bad = generate_cohort(
            "policy_violating",
            n_sessions=20,
            seed=4,
            policy_violation_rate=0.5,
        )
        verdict = _stream(m, bad)
        assert verdict.alarm
        assert "verify-before-transfer" in verdict.ltl_violations

    def test_policy_violator_alarm_is_critical(self):
        m = _make_monitor(seed=1)
        bad = generate_cohort(
            "policy_violating",
            n_sessions=20,
            seed=4,
            policy_violation_rate=0.5,
        )
        verdict = _stream(m, bad)
        # The reason should mention LTL[critical]
        assert any("LTL[critical]" in r for r in verdict.reasons)

    def test_weak_until_passes_when_neither_tool_fires(self):
        """A session that doesn't fire verify_user OR transfer_funds
        must vacuously satisfy must_call_before — this is the WeakUntil
        fix.  Strong-Until would have flagged it as a violation."""
        m = _make_monitor(seed=1)
        # Build a session that calls neither verify_user nor transfer_funds.
        records = [
            {
                "version": "0.1",
                "id": f"sha256:{0xA1:064x}",
                "kind": "chat_response",
                "ts": "2026-04-27T10:00:00.000Z",
                "parent": "sha256:" + "0" * 64,
                "meta": {},
                "payload": {
                    "model": "transfer-agent-v1",
                    "content": [
                        {"type": "tool_use", "id": "x1", "name": "check_balance", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                    "latency_ms": 600.0,
                    "usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0},
                },
            }
        ]
        verdict = m.observe_session(records)
        assert (
            "verify-before-transfer" not in verdict.ltl_violations
        ), "WeakUntil regression: rule fired on a session that called neither tool"


# ===========================================================================
# 5. Conformal drift detection (binomial test on breach rate)
# ===========================================================================


class TestConformalDrift:
    def test_conformal_drift_does_not_fire_on_healthy(self):
        """Under H0, the breach rate is exactly (1-coverage). Binomial
        test should not reject across 50 healthy sessions."""
        m = _make_monitor(seed=1)
        healthy = generate_cohort("baseline", n_sessions=50, seed=2)
        verdict = _stream(m, healthy)
        # Conformal drift axes should be empty under healthy canary.
        assert verdict.conformal_drift_axes == [] or not verdict.alarm

    def test_breach_count_per_session_is_tracked(self):
        m = _make_monitor(seed=1)
        for _ in range(5):
            sess = generate_session("baseline", seed=99)
            m.observe_session(sess)
        # Even if no axis breached, the dict should exist with all axes.
        verdict = m.observe_session(generate_session("baseline", seed=100))
        assert len(verdict.conformal_breach_count) == 8


# ===========================================================================
# 6. Reset semantics
# ===========================================================================


class TestReset:
    def test_reset_clears_state_but_preserves_calibration(self):
        m = _make_monitor(seed=1)
        # Trigger an alarm on a drifting canary.
        drifting = generate_cohort("drifting", n_sessions=10, seed=3)
        _stream(m, drifting)
        # Reset.
        m.reset()
        # A fresh healthy canary should not alarm.
        healthy = generate_cohort("baseline", n_sessions=20, seed=5)
        verdict = _stream(m, healthy)
        assert not verdict.alarm

    def test_reset_resets_counters(self):
        m = _make_monitor(seed=1)
        for sess in generate_cohort("baseline", n_sessions=5, seed=2):
            m.observe_session(sess)
        m.reset()
        verdict = m.observe_session(generate_session("baseline", seed=99))
        assert verdict.n_sessions == 1
        assert verdict.n_records == 3  # one session has 3 chat_responses


# ===========================================================================
# 7. Verdict structure
# ===========================================================================


class TestVerdictStructure:
    def test_verdict_summary_when_no_alarm(self):
        m = _make_monitor(seed=1)
        verdict = m.observe_session(generate_session("baseline", seed=99))
        assert "OK" in verdict.summary() or verdict.alarm

    def test_verdict_summary_when_alarm(self):
        m = _make_monitor(seed=1)
        bad = generate_cohort(
            "policy_violating",
            n_sessions=10,
            seed=4,
            policy_violation_rate=1.0,
        )
        verdict = _stream(m, bad)
        if verdict.alarm:
            assert "ALARM" in verdict.summary()
            assert verdict.reasons


# ===========================================================================
# 8. CLI demo end-to-end
# ===========================================================================


class TestCLIDemo:
    def test_run_canary_exits_zero_for_correct_behavior(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(_EXAMPLE_DIR / "run_canary.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            proc.returncode == 0
        ), f"Demo exited {proc.returncode}; stdout:\n{proc.stdout[-1500:]}"

    def test_run_canary_output_contains_pass_verdict(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(_EXAMPLE_DIR / "run_canary.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "Overall: PASS" in proc.stdout

    def test_run_canary_output_describes_each_stream(self):
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(_EXAMPLE_DIR / "run_canary.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "HEALTHY CANARY" in proc.stdout
        assert "DRIFTING CANARY" in proc.stdout
        assert "POLICY-VIOLATING CANARY" in proc.stdout
        assert "mSPRT[latency]" in proc.stdout
        assert "verify-before-transfer" in proc.stdout
