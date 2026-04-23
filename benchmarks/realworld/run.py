"""Run all 15 real-world scenarios against live Anthropic.

For each scenario:
  1. Fire every `task` at both the baseline and candidate configs via
     the live Anthropic SDK (auto-instrumented by Shadow's Session).
  2. Produce two `.agentlog` files.
  3. Run the Rust 9-axis differ.
  4. Assert the expected axis reached at least the expected severity.
  5. Print per-scenario findings + a final summary table.

Usage:
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python benchmarks/realworld/run.py

Cost: 15 × 10 × 2 ≈ 300 calls on Haiku 4.5, ~$0.50 total.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from scenarios import SCENARIOS, AgentConfig, Scenario  # noqa: E402

from shadow import _core  # noqa: E402
from shadow.sdk import Session  # noqa: E402

OUT = Path(__file__).parent / ".out"
SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def run_config(scenario: Scenario, config: AgentConfig, label: str) -> Path:
    """Fire every scenario task at the config, record into an .agentlog."""
    out_path = OUT / scenario.name / f"{label}.agentlog"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anthropic_tools = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["input_schema"],
        }
        for t in config.tools
    ]
    with Session(
        output_path=out_path,
        tags={"scenario": scenario.name, "config": label},
        auto_instrument=True,
    ):
        import anthropic

        client = anthropic.Anthropic()
        for task in scenario.tasks:
            kwargs: dict[str, Any] = {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "messages": [{"role": "user", "content": task}],
                "system": config.system,
                "temperature": config.temperature,
            }
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            try:
                client.messages.create(**kwargs)
            except Exception as e:  # network / rate-limit transient
                print(f"   [warn] {scenario.name}/{label}: {e}")
    return out_path


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    print(f"→ {scenario.name}: {scenario.description}")
    b = run_config(scenario, scenario.baseline, "baseline")
    c = run_config(scenario, scenario.candidate, "candidate")

    baseline_records = _core.parse_agentlog(b.read_bytes())
    candidate_records = _core.parse_agentlog(c.read_bytes())
    report = _core.compute_diff_report(baseline_records, candidate_records, None, 42)

    target = next(
        (r for r in report["rows"] if r["axis"] == scenario.expected_axis), {}
    )
    observed_sev = target.get("severity", "none")
    caught = SEVERITY_RANK[observed_sev] >= SEVERITY_RANK[scenario.min_severity]

    # Collect every axis that moved for transparency.
    all_movers = [
        r
        for r in report["rows"]
        if r.get("delta") and abs(r["delta"]) > 1e-9 and r["severity"] != "none"
    ]

    findings = {
        "scenario": scenario.name,
        "expected_axis": scenario.expected_axis,
        "expected_severity": scenario.min_severity,
        "observed_severity": observed_sev,
        "observed_delta": target.get("delta"),
        "ci95_low": target.get("ci95_low"),
        "ci95_high": target.get("ci95_high"),
        "flags": target.get("flags", []),
        "n": target.get("n", 0),
        "caught": caught,
        "also_moved": [
            {"axis": r["axis"], "severity": r["severity"], "delta": r["delta"]}
            for r in all_movers
            if r["axis"] != scenario.expected_axis
        ],
    }
    marker = "✓" if caught else "✗"
    print(
        f"   {marker} expected {scenario.expected_axis}≥{scenario.min_severity}, "
        f"observed {observed_sev} (Δ={target.get('delta', 0):+.3f}, "
        f"CI [{target.get('ci95_low', 0):+.2f},{target.get('ci95_high', 0):+.2f}], "
        f"n={target.get('n', 0)})"
    )
    if all_movers:
        also = ", ".join(f"{m['axis']}={m['severity']}" for m in findings["also_moved"])
        if also:
            print(f"     other axes: {also}")
    return findings


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Aborting.", file=sys.stderr)
        return 2
    OUT.mkdir(exist_ok=True)

    findings = []
    for scenario in SCENARIOS:
        findings.append(run_scenario(scenario))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    caught = sum(1 for f in findings if f["caught"])
    total = len(findings)
    print(f"\n{caught}/{total} scenarios caught correctly\n")
    print(
        f"{'scenario':<26} {'expected':<18} {'severity':<12} {'observed':<12} {'result'}"
    )
    print("-" * 90)
    for f in findings:
        tick = "✓" if f["caught"] else "✗"
        print(
            f"{f['scenario']:<26} {f['expected_axis']:<18} "
            f"{f['expected_severity']:<12} {f['observed_severity']:<12} {tick}"
        )

    import json

    (OUT / "findings.json").write_text(json.dumps(findings, indent=2))
    print(f"\nFull findings: {OUT / 'findings.json'}")
    return 0 if caught == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
