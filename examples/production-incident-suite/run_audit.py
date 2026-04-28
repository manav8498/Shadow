"""End-to-end demo: full multi-incident regression audit.

Runs every v2.5+ feature against five real-world incident patterns
in one pass and prints a structured report. Exits 1 if any incident
is unsafe.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Force UTF-8 stdout so the unicode characters in this demo's output
# don't crash on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from audit import render_findings, run_audit  # noqa: E402
from scenarios import generate_baseline, generate_candidate  # noqa: E402


async def main() -> int:
    baseline = generate_baseline(seed=1)
    candidate = generate_candidate(seed=2)
    findings = await run_audit(baseline, candidate)
    print(render_findings(findings))
    return 0 if findings.is_safe else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
