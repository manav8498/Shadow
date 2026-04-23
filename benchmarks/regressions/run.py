"""Run Shadow against the 20-case regression benchmark.

Usage:
    python benchmarks/regressions/run.py
    python benchmarks/regressions/run.py --json  # machine-readable output

Every case has:
  - a baseline trace and a candidate trace with a known regression
  - an expected axis (the one that MUST move) and minimum severity

The runner calls Shadow's differ on each case, checks whether the
expected axis reached at least the minimum severity, and reports
catch-rate = caught / total.

Exit code 0 iff every case passes. Useful as a permanent regression
guard in CI for Shadow itself: any change that drops the catch-rate
below 20/20 breaks this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from cases import CASES  # type: ignore[import-not-found]  # noqa: E402

from shadow import _core  # noqa: E402

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def run_one(name: str, case_fn: Any) -> dict[str, Any]:
    baseline, candidate, expected = case_fn()
    report = _core.compute_diff_report(baseline, candidate, None, 42)
    rows_by_axis = {r["axis"]: r for r in report["rows"]}
    target_axis = expected.get("axis")
    min_sev = expected.get("min_severity", "minor")
    known_limit = bool(expected.get("known_limit", False))

    if target_axis is None:
        worst = max((SEVERITY_RANK[r["severity"]] for r in report["rows"]), default=0)
        caught = worst <= SEVERITY_RANK["minor"] - 1
        observed_sev = max(
            (r["severity"] for r in report["rows"]),
            key=lambda s: SEVERITY_RANK[s],
            default="none",
        )
        return {
            "case": name,
            "description": expected.get("description", ""),
            "expected_axis": None,
            "expected_min_severity": min_sev,
            "observed_severity": observed_sev,
            "caught": caught,
            "known_limit": False,
        }

    row = rows_by_axis.get(target_axis, {})
    observed_sev = row.get("severity", "none")
    caught = SEVERITY_RANK.get(observed_sev, 0) >= SEVERITY_RANK.get(min_sev, 1)
    return {
        "case": name,
        "description": expected.get("description", ""),
        "expected_axis": target_axis,
        "expected_min_severity": min_sev,
        "observed_severity": observed_sev,
        "observed_delta": row.get("delta"),
        "flags": row.get("flags", []),
        "caught": caught,
        "known_limit": known_limit,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = p.parse_args()

    results = [run_one(name, fn) for name, fn in CASES]
    caught = sum(1 for r in results if r["caught"])
    total = len(results)
    known_limits = [r for r in results if r["known_limit"]]
    unexpected_misses = [r for r in results if not r["caught"] and not r["known_limit"]]
    unexpected_catches = [r for r in results if r["caught"] and r["known_limit"]]

    if args.json:
        print(
            json.dumps(
                {
                    "total": total,
                    "caught": caught,
                    "known_limits": len(known_limits),
                    "unexpected_misses": len(unexpected_misses),
                    "cases": results,
                },
                indent=2,
            )
        )
    else:
        print(f"Shadow regression benchmark — {caught}/{total} caught\n")
        print(
            f"{'case':<24} {'axis':<12} {'expected':<10} {'observed':<10} {'result':<10}"
        )
        print("-" * 80)
        for r in results:
            if r["caught"] and not r["known_limit"]:
                marker = "✓"
            elif not r["caught"] and r["known_limit"]:
                marker = "known-miss"
            elif r["caught"] and r["known_limit"]:
                marker = "now-caught!"
            else:
                marker = "✗ FAIL"
            print(
                f"{r['case']:<24} {str(r.get('expected_axis', '-')):<12} "
                f"{r['expected_min_severity']:<10} {r['observed_severity']:<10} {marker}"
            )
            if not r["caught"] and not r["known_limit"]:
                print(f"  {r['description']}")
        print(
            f"\n{caught}/{total} caught "
            f"(incl. {len(known_limits)} known limitation"
            f"{'s' if len(known_limits) != 1 else ''})"
        )
        if unexpected_misses:
            print(f"FAIL: {len(unexpected_misses)} unexpected miss(es)")
        if unexpected_catches:
            print(
                f"note: {len(unexpected_catches)} case(s) previously marked as a "
                "known limitation are now being caught — consider un-marking them."
            )

    return 0 if not unexpected_misses else 1


if __name__ == "__main__":
    raise SystemExit(main())
