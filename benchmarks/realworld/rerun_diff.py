"""Re-run the 15-scenario diffs against the already-saved .agentlog files.

Skips the expensive live-API step so we can verify Shadow fixes
without burning more API budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from scenarios import SCENARIOS  # noqa: E402

from shadow import _core  # noqa: E402

OUT = Path(__file__).parent / ".out"
SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def main() -> int:
    print(
        f"{'scenario':<26} {'expected':<18} {'severity':<12} {'observed':<12} "
        f"{'Δ':>10}  result"
    )
    print("-" * 96)
    caught = 0
    for sc in SCENARIOS:
        b_path = OUT / sc.name / "baseline.agentlog"
        c_path = OUT / sc.name / "candidate.agentlog"
        if not b_path.exists() or not c_path.exists():
            print(f"{sc.name:<26} (missing .agentlog — run run.py first)")
            continue
        baseline = _core.parse_agentlog(b_path.read_bytes())
        candidate = _core.parse_agentlog(c_path.read_bytes())
        report = _core.compute_diff_report(baseline, candidate, None, 42)
        target = next((r for r in report["rows"] if r["axis"] == sc.expected_axis), {})
        observed = target.get("severity", "none")
        hit = SEVERITY_RANK[observed] >= SEVERITY_RANK[sc.min_severity]
        delta = target.get("delta", 0.0)
        marker = "✓" if hit else "✗"
        if hit:
            caught += 1
        print(
            f"{sc.name:<26} {sc.expected_axis:<18} {sc.min_severity:<12} "
            f"{observed:<12} {delta:>+10.3f}  {marker}"
        )
    print(f"\n{caught}/{len(SCENARIOS)} scenarios caught correctly")
    return 0 if caught == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
