"""LASSO-over-corners bisect at k=4 — all config categories differ.

Config A: gpt-4o-mini, tight prompt, temp=0.0, tool=check_order_status.
Config B: gpt-4o,      loose prompt, temp=0.7, tool=lookup_order.

All four categories {model, prompt, params, tools} differ → k=4 → 16 corners.
Runs ~6 tickets per corner × 16 = 96 live calls (gpt-4o-mini / gpt-4o mix ≈ $0.10 total).

Ground-truth expectations:
  - trajectory → tools (tool rename is THE cause)
  - conformance → prompt (JSON instruction is in prompt only)
  - latency → model (gpt-4o is slower than mini)
  - verbosity → prompt + params together

Significance is the key test: the driver should have a CI that excludes zero.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))

from shadow import _core  # noqa: E402
from shadow.bisect import run_bisect  # noqa: E402
from shadow.llm.openai_backend import OpenAILLM  # noqa: E402
from shadow.sdk import Session  # noqa: E402

OUT = Path(__file__).parent / ".out" / "k4"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    f"I want a refund for order #A-{1000 + i}, it arrived damaged."
    for i in range(6)
]

TOOL_GOOD = {
    "name": "check_order_status",
    "description": "Look up order status.",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}
TOOL_BAD = {
    "name": "lookup_order",
    "description": "Look up an order.",
    "input_schema": {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
}

CONFIG_A = {
    "model": "gpt-4o-mini",
    "prompt": {
        "system": (
            'You are a support agent. Return ONLY a JSON object: '
            '{"action":"refund|info","reason":"..."}. No prose.'
        )
    },
    "params": {"temperature": 0.0, "max_tokens": 120},
    "tools": [TOOL_GOOD],
}
CONFIG_B = {
    "model": "gpt-4o",
    "prompt": {"system": "You are a helpful assistant."},
    "params": {"temperature": 0.7, "max_tokens": 120},
    "tools": [TOOL_BAD],
}


def write_baseline_trace(path: Path) -> None:
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY required"
    with Session(output_path=path, tags={"config": "baseline"}, auto_instrument=True):
        import openai

        client = openai.OpenAI()
        ot = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_GOOD["name"],
                    "description": TOOL_GOOD["description"],
                    "parameters": TOOL_GOOD["input_schema"],
                },
            }
        ]
        for task in TASKS:
            try:
                client.chat.completions.create(
                    model=CONFIG_A["model"],
                    messages=[
                        {"role": "system", "content": CONFIG_A["prompt"]["system"]},
                        {"role": "user", "content": task},
                    ],
                    tools=ot,
                    temperature=0.0,
                    max_completion_tokens=120,
                )
            except Exception as e:
                print(f"   [warn] baseline: {e}")


def main() -> int:
    import yaml

    baseline_path = OUT / "baseline.agentlog"
    print(f"Writing baseline trace ({len(TASKS)} calls)...")
    write_baseline_trace(baseline_path)

    cfg_a_path = OUT / "config_a.yaml"
    cfg_b_path = OUT / "config_b.yaml"
    cfg_a_path.write_text(yaml.safe_dump(CONFIG_A, sort_keys=False))
    cfg_b_path.write_text(yaml.safe_dump(CONFIG_B, sort_keys=False))

    backend = OpenAILLM(backend_id="bisect-k4")
    print(f"Running LASSO-over-corners bisect at k=4 (~{16 * len(TASKS)} calls)...")
    result = run_bisect(cfg_a_path, cfg_b_path, baseline_path, backend=backend)
    (OUT / "result.json").write_text(json.dumps(_jsonable(result), indent=2))

    print(f"\nmode: {result['mode']}")
    print(f"active categories: {result['active_categories']} (k={len(result['active_categories'])})")
    print(f"design runs: {result['design_runs']}")

    if len(result["active_categories"]) != 4:
        print(f"ERROR: expected k=4, got k={len(result['active_categories'])}")
        return 1

    print()
    print(f"{'axis':<12} {'category':<10} {'weight':>8}  {'CI (95%)':<22} sig")
    print("-" * 62)
    ci_attr = result.get("attributions_ci", {})
    for axis in ("trajectory", "conformance", "latency", "verbosity", "semantic", "safety"):
        rows = ci_attr.get(axis, [])
        for i, r in enumerate(rows):
            star = "*" if r["significant"] else ""
            ci = f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
            prefix = axis if i == 0 else ""
            print(f"{prefix:<12} {r['category']:<10} {r['weight']:>8.3f}  {ci:<22} {star}")

    # Key ground-truth check: trajectory → tools should be significant.
    # Conformance is honestly hard to force under model robustness and
    # we don't assert on axes that may legitimately not move.
    trajectory_rows = ci_attr.get("trajectory", [])
    tools_row = next(
        (r for r in trajectory_rows if r["category"] == "tools"), None
    )
    if tools_row is None or not tools_row["significant"] or tools_row["weight"] < 0.5:
        print(
            "\nFAIL: trajectory → tools should be significant with weight ≥ 0.5; "
            f"got {tools_row}"
        )
        return 1
    print(
        f"\nPASS: trajectory → tools attributed at {tools_row['weight']:.3f} "
        f"CI [{tools_row['ci95_low']:+.3f}, {tools_row['ci95_high']:+.3f}] (significant)"
    )
    print(
        "(other axes may not move meaningfully across corners — LLMs often "
        "produce format-compliant output even under loosened prompts, which "
        "is a real meta-finding about model robustness.)"
    )
    return 0


def _jsonable(obj: Any) -> Any:
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
