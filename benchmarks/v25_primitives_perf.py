"""Performance budget for v2.5 primitives.

Establishes the runtime envelope for the new statistical/LTL/conformal
modules so future regressions are caught before they hit users. Each
benchmark prints a single line with elapsed wall time; the
``BUDGETS`` dict at the bottom defines the budget per scenario, and
the script exits 1 if any scenario exceeds its budget.

Run: ``python benchmarks/v25_primitives_perf.py``

Targets are calibrated for typical agent-eval workloads (5-100 turn
traces, 1-20 LTL formula nodes, 8-dimensional fingerprints). For
production scale (10K+ turns) refer to the deferred-work notes in
docs/theory/ltl.md and docs/theory/conformal.md.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from shadow.causal import causal_attribution
from shadow.conformal import ACIDetector, conformal_calibrate
from shadow.ltl import Atom, Globally, Implies, Not, Until, eval_all_positions
from shadow.ltl.checker import TraceState
from shadow.statistical.hotelling import hotelling_t2
from shadow.statistical.sprt import MSPRTDetector, MSPRTtDetector, SPRTDetector


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _bench(name: str, fn: Callable[[], Any]) -> float:
    """Run ``fn`` once, return elapsed seconds. Prints "name: <ms>"."""
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {elapsed * 1000:.2f} ms")
    return elapsed


# ---------------------------------------------------------------------------
# Hotelling T² — agent-eval scale
# ---------------------------------------------------------------------------


def hotelling_small() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((30, 8))  # n=30, D=8 — typical fingerprint
    x2 = rng.standard_normal((30, 8)) + 0.3
    hotelling_t2(x1, x2)


def hotelling_perm_100() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((20, 8))
    x2 = rng.standard_normal((20, 8)) + 0.5
    hotelling_t2(x1, x2, permutations=100, rng=rng)


# ---------------------------------------------------------------------------
# SPRT / mSPRT streaming
# ---------------------------------------------------------------------------


def sprt_500_obs() -> None:
    rng = np.random.default_rng(0)
    det = SPRTDetector(alpha=0.05, beta=0.20, effect_size=0.5, warmup=20)
    for _ in range(500):
        det.update(float(rng.normal(0.5, 1)))


def msprt_500_obs() -> None:
    rng = np.random.default_rng(0)
    det = MSPRTDetector(alpha=0.05, tau=1.0, warmup=20)
    for _ in range(500):
        det.update(float(rng.normal(0.5, 1)))


def msprt_t_500_obs() -> None:
    rng = np.random.default_rng(0)
    det = MSPRTtDetector(alpha=0.05, tau=1.0, warmup=20)
    for _ in range(500):
        det.update(float(rng.normal(0.5, 1)))


# ---------------------------------------------------------------------------
# LTL — agent-eval and protocol-scale formulas
# ---------------------------------------------------------------------------


def _trace(n: int) -> list[TraceState]:
    return [
        TraceState(
            pair_index=i,
            tool_calls=["a"] if i % 3 == 0 else ["b"],
            stop_reason="end_turn",
        )
        for i in range(n)
    ]


def ltl_small() -> None:
    # Typical agent-eval formula: G(call_a → ¬X(call_b)) over 100 turns.
    phi = Globally(Implies(Atom("tool_call:a"), Not(Atom("tool_call:b"))))
    eval_all_positions(phi, _trace(100))


def ltl_long() -> None:
    # Long trace, complex formula: nested Until + G over 5000 turns.
    inner = Until(Not(Atom("tool_call:b")), Atom("tool_call:a"))
    phi = Globally(inner)
    eval_all_positions(phi, _trace(5000))


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------


def conformal_calibrate_500() -> None:
    rng = np.random.default_rng(0)
    scores = {f"axis_{i}": list(np.abs(rng.standard_normal(500))) for i in range(9)}
    conformal_calibrate(scores, target_coverage=0.90, confidence=0.95)


def aci_2000_steps() -> None:
    rng = np.random.default_rng(0)
    calib = sorted(np.abs(rng.standard_normal(200)).tolist())
    det = ACIDetector(calibration_scores=calib, alpha_target=0.10, gamma=0.005)
    for _ in range(2000):
        det.update(float(np.abs(rng.standard_normal())))


# ---------------------------------------------------------------------------
# Causal — single intervention loop
# ---------------------------------------------------------------------------


def causal_5_deltas() -> None:
    def replay(_cfg: dict[str, Any]) -> dict[str, float]:
        return {"a": 0.1, "b": 0.2, "c": 0.3}

    baseline = {f"d{i}": "low" for i in range(5)}
    candidate = {f"d{i}": "high" for i in range(5)}
    causal_attribution(
        baseline_config=baseline,
        candidate_config=candidate,
        replay_fn=replay,
    )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

# Wall-time budgets in ms. Calibrated empirically with ~3x safety
# margin over local laptop runs (M1 Air, Python 3.11). CI runners are
# typically slower; if a scenario blows the budget there, that's a
# real regression worth investigating.
BUDGETS_MS: dict[str, float] = {
    "hotelling_small": 50,
    "hotelling_perm_100": 250,
    "sprt_500_obs": 50,
    "msprt_500_obs": 50,
    "msprt_t_500_obs": 50,
    "ltl_small": 50,
    "ltl_long": 5000,  # 5K turns × full DP — pure-Python is ~1-2s
    "conformal_calibrate_500": 50,
    "aci_2000_steps": 200,
    "causal_5_deltas": 10,
}


def main() -> int:
    print("=" * 60)
    print("  Shadow v2.5 primitive perf benchmarks")
    print("=" * 60)
    print()

    scenarios: list[tuple[str, Callable[[], None]]] = [
        ("hotelling_small", hotelling_small),
        ("hotelling_perm_100", hotelling_perm_100),
        ("sprt_500_obs", sprt_500_obs),
        ("msprt_500_obs", msprt_500_obs),
        ("msprt_t_500_obs", msprt_t_500_obs),
        ("ltl_small", ltl_small),
        ("ltl_long", ltl_long),
        ("conformal_calibrate_500", conformal_calibrate_500),
        ("aci_2000_steps", aci_2000_steps),
        ("causal_5_deltas", causal_5_deltas),
    ]

    failures: list[str] = []
    for name, fn in scenarios:
        elapsed_ms = _bench(name, fn) * 1000
        budget = BUDGETS_MS[name]
        if elapsed_ms > budget:
            print(f"    BUDGET EXCEEDED: {elapsed_ms:.2f} ms > {budget:.0f} ms")
            failures.append(name)

    print()
    print("=" * 60)
    if failures:
        print(f"  FAIL: {len(failures)} scenario(s) over budget")
        for f in failures:
            print(f"    - {f}")
        print("=" * 60)
        return 1
    print(f"  OK: all {len(scenarios)} scenarios within budget")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
