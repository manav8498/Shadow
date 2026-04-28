"""Canary monitor for production agent rollouts.

Real-world scenario: a financial-services team rolls out a new
"transfer agent" model behind a 1% canary.  Production traffic streams
through both the baseline and the canary; the team wants to halt the
canary the *moment* it shows quality drift or a safety policy
violation, **without paying a multiple-testing penalty for every peek
at the dashboard**.

This monitor combines three of Shadow v2.5's primitives:

  1. **mSPRT** (mixture SPRT) — Robbins (1970) / Johari–Pekelis–Walsh
     (2017).  A non-negative martingale under H0 with the always-valid
     bound

         P( sup_{n ≥ 1} Λ_n ≥ 1/α ) ≤ α

     so the team can refresh the dashboard at any cadence — every
     request, every minute, never — and the false-positive rate
     stays at α.  Two axes are tracked independently: per-turn
     latency and per-turn output cost (token count).

  2. **LTL safety policies** — encoded with the new ``WeakUntil``
     operator.  The rule "transfer_funds may only fire after
     verify_user has fired" is

         (¬tool_call:transfer_funds) W tool_call:verify_user

     which correctly accepts sessions that fire neither tool
     (a help-desk question that never reaches a transfer).  The old
     strong-Until encoding rejected those sessions.

     LTL is checked **per session** so cross-session interleaving
     does not accidentally trip per-session invariants.

  3. **Distribution-free conformal** — calibrated on a held-out
     cohort of *known-safe* baseline sessions.  For each behavioral
     axis we compute the conformal quantile q̂ at 90% coverage of
     the **session-mean deviation** from the cohort mean.  Per-session
     breaches of q̂ are tracked but do not by themselves raise an
     alarm — under H0 we *expect* (1 − coverage) of sessions to
     breach.  Instead, the monitor performs a one-sided binomial test
     on the cumulative breach count: when the breach rate is
     significantly above (1 − coverage) at level α we raise a
     ``conformal_drift`` alarm.

**Family-wise error control.**  Because the monitor runs K independent
alarm channels in parallel (one mSPRT per tracked axis + one
conformal-drift test per fingerprint axis), each individual channel
is tested at the **Bonferroni-corrected level** α/K.  This bounds the
overall (family-wise) false-positive rate at α regardless of how many
times the operator peeks at the dashboard.

The monitor is stateful: ``observe_session(records)`` ingests one
canary session at a time and returns an updated :class:`CanaryVerdict`.
Once any axis or policy fires an alarm, the verdict is sticky — it
stays bad until ``reset()``.

Usage
-----
    monitor = CanaryMonitor(
        calibration_sessions=[s1, s2, ..., s50],  # baseline cohort
        safety_policies=DEFAULT_POLICIES,
        alpha=0.05,
    )
    for session in stream_canary_sessions():
        verdict = monitor.observe_session(session)
        if verdict.alarm:
            print(verdict.summary())
            halt_canary()
            break
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadow.conformal import ConformalCoverageReport, conformal_calibrate
from shadow.hierarchical import PolicyRule, check_policy
from shadow.statistical.fingerprint import (
    DEFAULT_CONFIG,
    FingerprintConfig,
    fingerprint_trace,
)
from shadow.statistical.sprt import MSPRTDetector

# ---------------------------------------------------------------------------
# Default safety policies for the transfer-agent use case
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: list[PolicyRule] = [
    PolicyRule(
        id="verify-before-transfer",
        kind="must_call_before",
        params={"first": "verify_user", "then": "transfer_funds"},
        severity="critical",
        description=(
            "transfer_funds must only be called after verify_user has "
            "succeeded earlier in the session.  Encoded as weak-until so "
            "sessions that never transfer are vacuously safe."
        ),
    ),
    PolicyRule(
        id="no-pii-in-text",
        kind="forbidden_text",
        params={"text": "SSN:"},
        severity="error",
        description=(
            "Never echo the literal string 'SSN:' in assistant text — "
            "guards against accidental PII leakage in logs."
        ),
    ),
]


# Fingerprint dimension names matching shadow.statistical.fingerprint.BehavioralVector.
_DIM_NAMES = [
    "tool_call_rate",
    "distinct_tool_frac",
    "stop_end_turn",
    "stop_tool_use",
    "stop_other",
    "verbosity",
    "latency",
    "refusal_flag",
]

# Axes monitored by mSPRT.  Each must be present in _DIM_NAMES.
_SPRT_AXES = ("latency", "verbosity")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class CanaryVerdict:
    """Snapshot of monitor state after one observed session."""

    n_sessions: int
    """Total canary sessions observed so far."""
    n_records: int
    """Total chat_response records ingested across all sessions."""
    alarm: bool
    """True iff any axis or policy has fired an alarm."""
    sprt_decisions: dict[str, str] = field(default_factory=dict)
    """Per-axis mSPRT decision: 'continue' or 'h1'."""
    ltl_violations: list[str] = field(default_factory=list)
    """Rule IDs of LTL policies any session has violated."""
    conformal_breach_count: dict[str, int] = field(default_factory=dict)
    """Per-axis count of sessions whose mean deviation exceeded q̂.
    Under H0 we expect ~(1 − target_coverage) × n_sessions; the alarm
    only fires when this count is *significantly* high."""
    conformal_drift_axes: list[str] = field(default_factory=list)
    """Axes where the cumulative breach rate is significantly above
    (1 − target_coverage) at level α — these triggered a real alarm."""
    reasons: list[str] = field(default_factory=list)
    """Human-readable reasons the alarm fired (empty if alarm=False)."""

    def summary(self) -> str:
        if not self.alarm:
            return (
                f"OK — {self.n_sessions} sessions / {self.n_records} records "
                f"observed, no alarms."
            )
        bullets = "\n  - " + "\n  - ".join(self.reasons)
        return (
            f"CANARY ALARM after {self.n_sessions} sessions "
            f"({self.n_records} records):{bullets}"
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class CanaryMonitor:
    """Streaming canary monitor with always-valid Type-I control.

    Calibration is one-shot (a list of baseline sessions, each a list of
    agentlog records).  After construction:

    - ``observe_session(records)`` ingests one canary session.  The
      session's chat_responses are fed one at a time to per-axis
      mSPRT detectors (continuous peeking is safe by Robbins's bound).
    - The session's mean fingerprint is compared to the baseline
      cohort mean; any axis whose deviation exceeds q̂ trips a
      conformal alarm (finite-sample, distribution-free).
    - LTL safety policies are checked on the session's records alone,
      so per-session invariants are not corrupted by cross-session
      interleaving.

    Once an alarm fires the verdict is sticky.  Call :meth:`reset`
    to start a fresh canary cohort against the same calibration.
    """

    def __init__(
        self,
        calibration_sessions: list[list[dict[str, Any]]],
        safety_policies: list[PolicyRule] | None = None,
        alpha: float = 0.05,
        target_coverage: float = 0.90,
        confidence: float = 0.95,
        fingerprint_config: FingerprintConfig | None = None,
        msprt_tau: float = 1.0,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if len(calibration_sessions) < 10:
            raise ValueError(
                f"calibration cohort must have ≥10 sessions; "
                f"got {len(calibration_sessions)}"
            )

        self.alpha = alpha
        self.target_coverage = target_coverage
        self.confidence = confidence
        self.fingerprint_config = fingerprint_config or DEFAULT_CONFIG
        self.policies = (
            safety_policies if safety_policies is not None else DEFAULT_POLICIES
        )

        # Bonferroni correction across K alarm channels:
        #   K = #(mSPRT axes) + #(fingerprint axes for conformal drift)
        # Each channel is tested at α/K so the family-wise error rate
        # of "any channel falsely fires" is bounded by α.
        n_channels = len(_SPRT_AXES) + len(_DIM_NAMES)
        self.alpha_per_channel = alpha / n_channels

        # Flatten calibration to a single matrix for cohort statistics.
        flat_records: list[dict[str, Any]] = []
        for sess in calibration_sessions:
            flat_records.extend(sess)
        self._baseline_mat = fingerprint_trace(
            flat_records, self.fingerprint_config
        )
        if self._baseline_mat.shape[0] < 10:
            raise ValueError(
                f"calibration sessions yielded {self._baseline_mat.shape[0]} "
                f"chat_responses; need ≥10"
            )
        self._baseline_mean: NDArray[np.float64] = self._baseline_mat.mean(axis=0)

        # mSPRT: warm up each tracked-axis detector with the entire cohort
        # so σ̂ is essentially the truth (asymptotic always-valid regime).
        self._msprt_warmup_n = self._baseline_mat.shape[0]
        self._msprt_tau = msprt_tau
        self._sprt: dict[str, MSPRTDetector] = {}
        for axis in _SPRT_AXES:
            idx = _DIM_NAMES.index(axis)
            det = MSPRTDetector(
                alpha=self.alpha_per_channel,
                tau=msprt_tau,
                warmup=self._msprt_warmup_n,
            )
            for v in self._baseline_mat[:, idx]:
                det.update(float(v))
            self._sprt[axis] = det

        # Conformal: per-session calibration scores = session-mean deviation
        # from cohort mean.  This is exchangeable with future canary sessions
        # under the null and gives a finite-sample 90% coverage bound.
        per_session_scores: dict[str, list[float]] = {ax: [] for ax in _DIM_NAMES}
        for sess in calibration_sessions:
            sess_mat = fingerprint_trace(sess, self.fingerprint_config)
            if sess_mat.shape[0] == 0:
                continue
            sess_mean = sess_mat.mean(axis=0)
            for i, axis in enumerate(_DIM_NAMES):
                per_session_scores[axis].append(
                    float(abs(sess_mean[i] - self._baseline_mean[i]))
                )
        self._conformal: ConformalCoverageReport = conformal_calibrate(
            per_session_scores,
            target_coverage=target_coverage,
            confidence=confidence,
        )
        self._q_hat: dict[str, float] = {
            ax.axis: ax.q_hat for ax in self._conformal.axes
        }

        # Streaming state.
        self._n_sessions = 0
        self._n_records = 0
        self._latched_reasons: list[str] = []
        self._latched_alarm = False
        self._latched_ltl_violations: list[str] = []
        # Per-axis cumulative breach counts (one count per session).
        self._breach_count: dict[str, int] = {ax: 0 for ax in _DIM_NAMES}
        self._latched_drift_axes: list[str] = []

    # ---- read-only access --------------------------------------------------

    @property
    def conformal_report(self) -> ConformalCoverageReport:
        return self._conformal

    @property
    def q_hat(self) -> dict[str, float]:
        return dict(self._q_hat)

    # ---- main entry point --------------------------------------------------

    def observe_session(
        self, session_records: list[dict[str, Any]]
    ) -> CanaryVerdict:
        """Ingest one canary session and return the current verdict.

        ``session_records`` is the list of agentlog records for a single
        user interaction.  All chat_responses in the session are fed
        sequentially to mSPRT detectors; the session's mean fingerprint
        is checked against the conformal q̂; LTL policies are evaluated
        on the session's records.
        """
        self._n_sessions += 1
        sess_mat = fingerprint_trace(session_records, self.fingerprint_config)

        # Stream each chat_response into the per-axis mSPRT detectors.
        for row in sess_mat:
            self._n_records += 1
            for axis, det in self._sprt.items():
                idx = _DIM_NAMES.index(axis)
                det.update(float(row[idx]))

        # Per-session conformal: count breaches; alarm only when the
        # cumulative breach rate is significantly above (1 − coverage).
        if sess_mat.shape[0] > 0:
            sess_mean = sess_mat.mean(axis=0)
            for i, axis in enumerate(_DIM_NAMES):
                q = self._q_hat.get(axis)
                if q is None:
                    continue
                dev = float(abs(sess_mean[i] - self._baseline_mean[i]))
                if dev > q + 1e-12:
                    self._breach_count[axis] += 1

            self._check_conformal_drift()

        # Per-session LTL: only the records from this session.
        if self.policies:
            violations = check_policy(session_records, self.policies)
            for v in violations:
                severity = self._severity_for(v.rule_id)
                if severity not in ("critical", "error"):
                    continue
                if v.rule_id not in self._latched_ltl_violations:
                    self._latched_ltl_violations.append(v.rule_id)
                    self._latched_reasons.append(
                        f"LTL[{severity}]: {v.rule_id} violated "
                        f"(session #{self._n_sessions})"
                    )
                if severity == "critical":
                    self._latched_alarm = True

        # mSPRT alarm check.
        sprt_decisions = {ax: det.decision for ax, det in self._sprt.items()}
        for axis, decision in sprt_decisions.items():
            if decision == "h1":
                msg = (
                    f"mSPRT[{axis}] rejected H0 at "
                    f"n={self._sprt[axis].n_observations}"
                )
                if msg not in self._latched_reasons:
                    self._latched_reasons.append(msg)
                self._latched_alarm = True

        return CanaryVerdict(
            n_sessions=self._n_sessions,
            n_records=self._n_records,
            alarm=self._latched_alarm,
            sprt_decisions=sprt_decisions,
            ltl_violations=list(self._latched_ltl_violations),
            conformal_breach_count=dict(self._breach_count),
            conformal_drift_axes=list(self._latched_drift_axes),
            reasons=list(self._latched_reasons),
        )

    def _check_conformal_drift(self) -> None:
        """One-sided binomial test on cumulative breach rate per axis.

        Under H0 (canary ~ baseline), session breaches are
        Binomial(n, 1−coverage).  We reject H0 in favour of "canary's
        true breach rate > 1−coverage" when the observed count is so
        large that ``P(Bin(n, 1-cov) ≥ observed) < α``.
        """
        n = self._n_sessions
        # Need at least a handful of sessions for the test to be meaningful.
        if n < 5:
            return
        p0 = 1.0 - self.target_coverage
        for axis, count in self._breach_count.items():
            if axis in self._latched_drift_axes:
                continue
            if axis not in self._q_hat:
                continue
            p_value = _binomial_sf(count - 1, n, p0) if count > 0 else 1.0
            if p_value < self.alpha_per_channel:
                self._latched_drift_axes.append(axis)
                self._latched_reasons.append(
                    f"conformal_drift on axis {axis!r}: {count}/{n} sessions "
                    f"breached q̂ (binomial p={p_value:.4f} < "
                    f"α_per_channel={self.alpha_per_channel:.4f})"
                )
                self._latched_alarm = True

    # ---- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear streaming state. Calibration is preserved."""
        for axis, det in self._sprt.items():
            det.reset()
            idx = _DIM_NAMES.index(axis)
            for v in self._baseline_mat[:, idx]:
                det.update(float(v))
        self._n_sessions = 0
        self._n_records = 0
        self._latched_reasons = []
        self._latched_alarm = False
        self._latched_ltl_violations = []
        self._breach_count = {ax: 0 for ax in _DIM_NAMES}
        self._latched_drift_axes = []

    # ---- internal -----------------------------------------------------------

    def _severity_for(self, rule_id: str) -> str:
        for rule in self.policies:
            if rule.id == rule_id:
                return rule.severity
        return "warning"


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X > k) for X ~ Binomial(n, p).

    Uses scipy.stats.binom when available; falls back to a normal
    approximation for n ≥ 30 otherwise.  Used by the conformal-drift
    detector to test "is the breach count significantly high?"
    """
    try:
        from scipy.stats import binom  # type: ignore[import-untyped]

        return float(binom.sf(k, n, p))
    except ImportError:
        # Normal approximation: X ≈ N(np, np(1-p)).
        if n < 30:
            return 0.5
        import math

        mean = n * p
        std = math.sqrt(n * p * (1.0 - p))
        if std < 1e-12:
            return 0.0
        z = (k + 0.5 - mean) / std  # continuity correction
        # P(X > k) ≈ 1 - Φ(z) using erfc
        return 0.5 * math.erfc(z / math.sqrt(2.0))


__all__ = [
    "DEFAULT_POLICIES",
    "CanaryMonitor",
    "CanaryVerdict",
]
