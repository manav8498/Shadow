"""End-to-end test of `--backend live` against the real OpenAI API.

Skipped unless `SHADOW_RUN_NETWORK_TESTS=1` is set AND
`OPENAI_API_KEY` is in the environment. CI never sets this, so
the test is offline-by-default. To run locally:

    export SHADOW_RUN_NETWORK_TESTS=1
    export OPENAI_API_KEY=sk-...
    pytest python/tests/test_diagnose_pr_live_api_e2e.py -v

Cost: ~$0.05 per run on gpt-4o-mini at the default --n-bootstrap
and one delta. Bounded by the test's --max-cost flag.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("SHADOW_RUN_NETWORK_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="set SHADOW_RUN_NETWORK_TESTS=1 and OPENAI_API_KEY to run live OpenAI test",
)
def test_live_backend_against_openai_with_refund_demo(tmp_path: Path) -> None:
    """Real OpenAI calls against the refund demo. Asserts the
    structural contract — the API can produce non-zero divergence
    between baseline and candidate prompt, and the runner surfaces
    it as a dominant cause with a real bootstrap CI."""
    from typer.testing import CliRunner

    from shadow.cli.app import app

    demo = Path(__file__).resolve().parent.parent.parent / "examples" / "refund-causal-diagnosis"
    if not demo.is_dir():
        pytest.skip(f"refund demo not found at {demo}")

    out = tmp_path / "report.json"
    md = tmp_path / "comment.md"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "diagnose-pr",
            "--traces",
            str(demo / "baseline_traces"),
            "--candidate-traces",
            str(demo / "candidate_traces"),
            "--baseline-config",
            str(demo / "baseline.yaml"),
            "--candidate-config",
            str(demo / "candidate.yaml"),
            "--policy",
            str(demo / "policy.yaml"),
            "--out",
            str(out),
            "--pr-comment",
            str(md),
            "--backend",
            "live",
            "--n-bootstrap",
            "100",  # keep cost low; CI bound is ~$0.05
            "--max-cost",
            "0.50",  # hard ceiling
        ],
    )
    assert result.exit_code == 0, f"stdout:\n{result.stdout}"

    parsed = json.loads(out.read_text())
    assert parsed["verdict"] in {"hold", "stop"}, parsed["verdict"]
    assert parsed["affected_traces"] >= 1
    dom = parsed["dominant_cause"]
    assert dom is not None
    assert dom["delta_id"] in {"prompt.system", "prompts/candidate.md"}
    # Live ATE varies — assert structural contract, not magnitude.
    assert isinstance(dom["ate"], int | float)
    # CI must be present (we passed --n-bootstrap > 0).
    assert dom["ci_low"] is not None and dom["ci_high"] is not None
    # E-value must be present (sensitivity is on).
    assert dom["e_value"] is not None and dom["e_value"] > 1.0

    md_text = md.read_text()
    assert "Shadow verdict" in md_text
    assert "prompt" in md_text.lower()
    # Live runs must NOT have the synthetic-mock disclosure.
    assert "synthetic" not in md_text.lower()
