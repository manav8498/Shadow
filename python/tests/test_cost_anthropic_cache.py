"""Regression test for bug_029 — Anthropic cache-read tokens must not be
billed at the output/reasoning rate.

The earlier `test_cached_input_tokens_billed_at_cheaper_rate` in Rust
passed because it hand-crafted a usage dict with `cached_input_tokens`
populated directly. But no shipped backend actually emitted that
field — `AnthropicLLM._from_provider` routed `cache_read_input_tokens`
into `thinking_tokens`, which the new cost formula bills at the
reasoning/output rate (~50x higher than the true cache-read rate).

This test goes through the real backend conversion path: it feeds a
mock-anthropic-response-shaped object through `_from_provider`, writes
it to an `.agentlog`, and runs the Rust differ with the bundled
`pricing.json`. The computed cost must match the hand-calculated
truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.llm.anthropic_backend import AnthropicLLM
from shadow.sdk import Session


class _FakeUsage:
    def __init__(self, input_t: int, output_t: int, cache_read: int) -> None:
        self.input_tokens = input_t
        self.output_tokens = output_t
        self.cache_read_input_tokens = cache_read


class _FakePart:
    def __init__(self) -> None:
        self.type = "text"
        self.text = "hi"


class _FakeResponse:
    def __init__(self, cache_read: int) -> None:
        self.model = "claude-opus-4-7"
        self.content = [_FakePart()]
        self.stop_reason = "end_turn"
        self.usage = _FakeUsage(input_t=1000, output_t=500, cache_read=cache_read)


def _load_bundled_pricing() -> dict[str, Any]:
    """Load the bundled pricing.json (same as `shadow diff --pricing pricing.json`)."""
    path = Path(__file__).resolve().parents[2] / "pricing.json"
    raw = json.loads(path.read_text())
    # Strip the comment/metadata fields.
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}


def test_anthropic_cache_read_tokens_bill_at_cache_read_rate(tmp_path: Path) -> None:
    """End-to-end: backend → agentlog → diff — cache reads priced correctly.

    Using bundled pricing for claude-opus-4-7:
      input: $15/Mtok, output: $75/Mtok, cached_input: $1.50/Mtok

    For a response with input=1000, output=500, cache_read=1000:
      true cost = 1000*15e-6 + 1000*1.5e-6 + 500*75e-6
                = 0.015 + 0.0015 + 0.0375
                = 0.054

    The buggy code routed cache_read into thinking_tokens, billed at
    reasoning_rate → output_rate = $75/Mtok:
      buggy cost = 1000*15e-6 + 500*75e-6 + 1000*75e-6
                 = 0.015 + 0.0375 + 0.075
                 = 0.1275   (2.36x overstated)
    """
    shadow_resp = AnthropicLLM._from_provider(_FakeResponse(cache_read=1000), latency_ms=100)
    # Sanity check: cache-reads must land in cached_input_tokens, not thinking_tokens.
    assert shadow_resp["usage"]["cached_input_tokens"] == 1000
    assert shadow_resp["usage"]["thinking_tokens"] == 0

    # Now pipe two identical responses through the differ; cost should equal truth.
    path = tmp_path / "t.agentlog"
    with Session(output_path=path, auto_instrument=False) as s:
        s.record_chat(
            request={
                "model": "claude-opus-4-7",
                "messages": [{"role": "user", "content": "hi"}],
                "params": {},
            },
            response=shadow_resp,
        )
    records = _core.parse_agentlog(path.read_bytes())

    pricing_raw = _load_bundled_pricing()
    pricing = {
        model: {
            "input": float(e["input"]),
            "output": float(e["output"]),
            "cached_input": float(e.get("cached_input", 0.0)),
            "reasoning": float(e.get("reasoning", 0.0)),
            "batch_discount": float(e.get("batch_discount", 0.0)),
        }
        for model, e in pricing_raw.items()
    }
    report = _core.compute_diff_report(records, records, pricing, 42)
    cost_row = next(r for r in report["rows"] if r["axis"] == "cost")
    # Self-diff: baseline == candidate → delta 0.
    assert abs(cost_row["delta"]) < 1e-9
    # Baseline cost is the correct figure: $0.054 per call.
    assert abs(cost_row["baseline_median"] - 0.054) < 1e-6, (
        f"expected baseline_median=0.054 got {cost_row['baseline_median']}; "
        "the cache-read over-billing regression is back."
    )


def test_anthropic_response_without_cache_reads_omits_cached_input_key() -> None:
    """No prompt caching → no cached_input_tokens in usage dict (keeps record tidy)."""
    shadow_resp = AnthropicLLM._from_provider(_FakeResponse(cache_read=0), latency_ms=100)
    assert "cached_input_tokens" not in shadow_resp["usage"]
    assert shadow_resp["usage"]["thinking_tokens"] == 0
