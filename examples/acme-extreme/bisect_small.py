"""Focused LASSO-over-corners bisect on a 10-request slice.

Uses the first 10 chat-pairs of the existing baseline.agentlog so we
don't spend more on OpenAI than necessary — with k=3 categories and 10
tickets per corner, this is 80 live calls (~$0.01 on gpt-4o-mini).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))

from shadow import _core
from shadow.bisect import run_bisect
from shadow.llm.openai_backend import OpenAILLM

OUT = Path(__file__).parent / ".out"


def main() -> int:
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"

    full_baseline = _core.parse_agentlog((OUT / "baseline.agentlog").read_bytes())

    # Trim to metadata + first 10 request/response pairs.
    meta = full_baseline[0]
    chat = [r for r in full_baseline if r["kind"] in ("chat_request", "chat_response")]
    trimmed = [meta] + chat[:20]  # 10 pairs
    small_path = OUT / "baseline_small.agentlog"
    small_path.write_bytes(_core.write_agentlog(trimmed))

    # configs already written by scenario.py
    config_a = OUT / "config_a.yaml"
    config_b = OUT / "config_b.yaml"

    backend = OpenAILLM(model_override="gpt-4o-mini", backend_id="bisect")
    result = run_bisect(config_a, config_b, small_path, backend=backend)

    print(f"mode: {result['mode']}")
    print(f"active categories: {result['active_categories']}")
    print(f"design runs: {result['design_runs']}")
    print()
    ci_attr = result.get("attributions_ci", {})
    if not ci_attr:
        print("FAIL: attributions_ci missing from result")
        return 1

    print(f"{'axis':<12} {'category':<10} {'weight':>8}  {'CI (95%)':<22} sig")
    print("-" * 62)
    for axis in (
        "trajectory",
        "conformance",
        "verbosity",
        "latency",
        "safety",
        "semantic",
    ):
        rows = ci_attr.get(axis, [])
        for i, r in enumerate(rows):
            star = "*" if r["significant"] else ""
            ci = f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
            prefix = axis if i == 0 else ""
            print(
                f"{prefix:<12} {r['category']:<10} {r['weight']:>8.3f}  {ci:<22} {star}"
            )

    # Save the full result too for inspection.
    (OUT / "bisect_result.json").write_text(json.dumps(_jsonable(result), indent=2))
    return 0


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
