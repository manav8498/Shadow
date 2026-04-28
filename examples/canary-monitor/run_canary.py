"""End-to-end canary-monitoring demo.

Runs three streams through CanaryMonitor:

  1. A healthy canary (same distribution as baseline) — the monitor
     must NOT fire an alarm even though the operator peeks every
     session (always-valid Type-I control).

  2. A drifting canary (~1.5× higher latency) — the monitor must
     reject H0 via mSPRT after a handful of sessions.

  3. A policy-violating canary (some sessions skip verify_user before
     transfer_funds) — the monitor must fire an LTL alarm.

Outputs a per-stream summary and exits 0 if all three behave as
expected, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from agent_simulator import generate_cohort  # noqa: E402
from canary_monitor import CanaryMonitor  # noqa: E402


def _run(monitor: CanaryMonitor, sessions: list[list[dict]]) -> tuple[int, str]:
    """Stream sessions through monitor; return (sessions_consumed, summary)."""
    for i, sess in enumerate(sessions, 1):
        verdict = monitor.observe_session(sess)
        if verdict.alarm:
            return i, verdict.summary()
    return len(sessions), "OK — stream exhausted with no alarm"


def main() -> int:
    print("=" * 72)
    print("  SHADOW CANARY MONITOR — Transfer Agent Rollout")
    print("=" * 72)

    # 50 baseline sessions for calibration.
    calibration = generate_cohort("baseline", n_sessions=50, seed=1)
    n_records = sum(len(s) for s in calibration)
    print(
        f"\nCalibrated on {len(calibration)} baseline sessions "
        f"({n_records} chat_responses)"
    )

    monitor = CanaryMonitor(
        calibration_sessions=calibration,
        alpha=0.05,
        target_coverage=0.90,
        confidence=0.95,
    )

    print(f"  conformal n_min  = {monitor.conformal_report.n_min}")
    print(f"  sufficient_n     = {monitor.conformal_report.sufficient_n}")
    print(f"  worst axis       = {monitor.conformal_report.worst_axis!r}")
    print(f"  q̂ on latency     = {monitor.q_hat.get('latency', 0):.4f}")
    print(f"  q̂ on verbosity   = {monitor.q_hat.get('verbosity', 0):.4f}")
    print(f"  q̂ on tool_call_rate = {monitor.q_hat.get('tool_call_rate', 0):.4f}")

    overall_pass = True

    # Stream 1: healthy canary — should NOT trip any alarm.
    print("\n[1] HEALTHY CANARY  (expect: no alarm)")
    healthy = generate_cohort("baseline", n_sessions=50, seed=2)
    n1, summary1 = _run(monitor, healthy)
    if "OK" in summary1:
        print(f"    {summary1}  (peeked at {n1} session boundaries)")
    else:
        print(f"    FALSE ALARM after {n1} sessions: {summary1}")
        overall_pass = False

    # Stream 2: drifting canary — should fire mSPRT[latency].
    monitor.reset()
    print("\n[2] DRIFTING CANARY  (expect: mSPRT[latency] alarm)")
    drifting = generate_cohort("drifting", n_sessions=50, seed=3)
    n2, summary2 = _run(monitor, drifting)
    print(f"    {summary2}")
    if "mSPRT[latency]" not in summary2:
        print("    WRONG: expected mSPRT[latency] to fire")
        overall_pass = False

    # Stream 3: policy violator — should fire LTL[critical].
    monitor.reset()
    print("\n[3] POLICY-VIOLATING CANARY  (expect: LTL[critical] alarm)")
    bad = generate_cohort(
        "policy_violating",
        n_sessions=30,
        seed=4,
        policy_violation_rate=0.5,
    )
    n3, summary3 = _run(monitor, bad)
    print(f"    {summary3}")
    if "LTL[critical]" not in summary3:
        print("    WRONG: expected LTL[critical] to fire")
        overall_pass = False

    print("\n" + "=" * 72)
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
