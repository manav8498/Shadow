"""Re-run a single scenario against live Anthropic (cheap targeted retry)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run import run_scenario  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rerun_one.py <scenario_name>")
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required")
        return 2
    name = sys.argv[1]
    scenario = next((s for s in SCENARIOS if s.name == name), None)
    if scenario is None:
        print(f"unknown scenario: {name}")
        return 2
    run_scenario(scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
