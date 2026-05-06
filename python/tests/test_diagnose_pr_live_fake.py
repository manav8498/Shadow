"""Deterministic live-backend smoke test using a FakeOpenAIClient.

The real `--backend live` path goes through `OpenAIReplayer`, which
requires `OPENAI_API_KEY` and hits the network. The live integration
test (`test_diagnose_pr_live_api_e2e.py`) exercises that path but is
gated on `SHADOW_RUN_NETWORK_TESTS=1` so CI never actually spends.

This file exercises the SAME live-replay code path
(`build_live_replay_fn_per_corpus` → `causal_from_replay`
→ `runner.run_diagnose_pr`) using a fake replayer factory whose
behavior is keyed on the `system_prompt` value. That gives us a
deterministic CI-runnable test for:

  * Per-trace anchor extraction.
  * Cost tracking across traces.
  * Bootstrap CI computation against synthesised intervention data.
  * `--max-cost` cap aborting before runaway spend.

The fake produces divergence values that vary by `system_prompt`
content — so removing the "Always confirm" instruction makes
trajectory divergence jump, exactly the shape causal_attribution
expects to attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow.causal.replay.types import ReplayResult
from shadow.diagnose_pr.live import (
    CostTracker,
    build_live_replay_fn_per_corpus,
)
from shadow.diagnose_pr.loaders import LoadedTrace
from shadow.diagnose_pr.runner import DiagnoseOptions, run_diagnose_pr

# ---------------------------------------------------------------------------
# FakeOpenAIClient — scripted, deterministic, GIL-free.
# ---------------------------------------------------------------------------


class FakeReplayer:
    """Drop-in replacement for `OpenAIReplayer` for diagnose-pr tests.

    Matches the real replayer's constructor + __call__ shape but
    never imports `openai` and never reads `OPENAI_API_KEY`. The
    `divergence` it returns is a deterministic function of the
    `system_prompt` text — when the prompt drops "Always confirm",
    `trajectory` jumps; when it keeps the instruction, divergence
    is zero. This is enough to exercise the causal_from_replay
    pipeline end-to-end including bootstrap CI.

    Output-token count is fixed so cost tracking is predictable.
    """

    def __init__(
        self,
        *,
        baseline_response_text: str,
        baseline_tool_calls: list[str] | None = None,
        baseline_stop_reason: str = "stop",
        baseline_latency_ms: float = 1000.0,
        baseline_output_tokens: int = 100,
    ) -> None:
        self.baseline_response_text = baseline_response_text
        self.baseline_tool_calls = list(baseline_tool_calls or [])
        self._cache: dict[str, ReplayResult] = {}

    def __call__(self, config: dict[str, Any]) -> ReplayResult:
        # Cache key: full config dict. The real replayer canonicalises;
        # for the fake, repr() is stable enough.
        key = repr(sorted(config.items()))
        cached = self._cache.get(key)
        if cached is not None:
            return ReplayResult(
                config=cached.config,
                response_text=cached.response_text,
                tool_calls=list(cached.tool_calls),
                stop_reason=cached.stop_reason,
                latency_ms=cached.latency_ms,
                output_tokens=cached.output_tokens,
                divergence=dict(cached.divergence),
                cached=True,
            )

        prompt = str(config.get("system_prompt", ""))
        # Deterministic "this prompt removed the key instruction"
        # detector. The refund-causal-diagnosis
        # baseline prompt mentions `confirm_refund_amount` (the
        # tool name + the confirmation step); the candidate prompt
        # doesn't. Use that token as the discriminator.
        dropped_confirmation = "confirm_refund_amount" not in prompt.lower()
        # Synth divergence vector. Trajectory jumps on the prompt
        # change; everything else stays zero. This mirrors the
        # shape causal_attribution sees for a real prompt regression.
        divergence = {
            "semantic": 0.0,
            "trajectory": 0.6 if dropped_confirmation else 0.0,
            "safety": 0.0,
            "verbosity": 0.0,
            "latency": 0.0,
        }
        result = ReplayResult(
            config=dict(config),
            response_text="" if dropped_confirmation else self.baseline_response_text,
            tool_calls=list(self.baseline_tool_calls),
            stop_reason="stop",
            latency_ms=1000.0,
            output_tokens=120,
            divergence=divergence,
            cached=False,
        )
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Helpers — synthesise a tiny trace corpus + configs without needing fixtures.
# ---------------------------------------------------------------------------


def _synth_trace(trace_id: str, user_prompt: str, response_text: str) -> LoadedTrace:
    """Build the smallest trace shape the live-replay anchor extractor
    accepts. We don't need a hash-correct .agentlog — runner only
    looks at trace_id + records when computing per-trace anchors."""
    records = [
        {
            "id": f"sha256:{trace_id}",
            "kind": "metadata",
            "payload": {"agent_id": "fake"},
        },
        {
            "id": f"sha256:{trace_id}-q",
            "kind": "chat_request",
            "payload": {
                "messages": [{"role": "user", "content": user_prompt}],
            },
        },
        {
            "id": f"sha256:{trace_id}-a",
            "kind": "chat_response",
            "payload": {
                "content": [{"type": "text", "text": response_text}],
            },
        },
    ]
    return LoadedTrace(
        path=Path(f"/tmp/{trace_id}.agentlog"),
        trace_id=f"sha256:{trace_id}",
        records=records,
    )


def _write_yaml(path: Path, body: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(body, sort_keys=True))


# ---------------------------------------------------------------------------
# Cost tracker tests — the easy half.
# ---------------------------------------------------------------------------


def test_cost_tracker_aborts_when_max_cost_exceeded() -> None:
    """`--max-cost` is the key safety on `--backend live`. Verify
    the tracker raises before total spend exceeds the cap."""
    tracker = CostTracker(max_usd=0.01)  # 1 cent ceiling
    # First call should fit (gpt-4o-mini @ ~$0.15/Mtok input → cents
    # per K tokens). 10K input + 10K output stays under 1 cent.
    tracker.record(model="gpt-4o-mini", input_tokens=1_000, output_tokens=1_000)
    # Pile on enough tokens that we breach. 200K output @ $0.60/Mtok
    # = $0.12, well past 1 cent.
    with pytest.raises(RuntimeError, match="--max-cost"):
        tracker.record(model="gpt-4o-mini", input_tokens=10_000, output_tokens=200_000)
    # The breaching call must still appear in breakdown so the report
    # tells the user which call pushed them over.
    assert tracker.calls == 2
    assert any("output_tokens" in entry for entry in tracker.breakdown)


# ---------------------------------------------------------------------------
# Per-corpus replay smoke — exercises the real aggregation code path.
# ---------------------------------------------------------------------------


def test_per_corpus_replay_means_divergence_across_traces() -> None:
    """Three baseline traces, FakeReplayer keyed on system_prompt.
    With a candidate prompt that drops 'Always confirm', every per-
    trace replay reports trajectory=0.6; the corpus mean is also
    0.6. Verify the aggregation is correct + cost is tracked."""
    traces = [
        _synth_trace("t1", "Refund for order #1", "Confirmed and refunded."),
        _synth_trace("t2", "Refund for order #2", "Confirmed and refunded."),
        _synth_trace("t3", "Refund for order #3", "Confirmed and refunded."),
    ]
    replay_fn, cost = build_live_replay_fn_per_corpus(
        baseline_traces=traces,
        max_cost_usd=10.00,
        replayer_factory=FakeReplayer,
    )
    # Candidate prompt: dropped the "Always confirm" instruction.
    div = replay_fn(
        {
            "prompt.system": "You are a refund agent. Issue refunds promptly.",
            "model": "gpt-4o-mini",
            "params.temperature": 0.0,
        }
    )
    assert div["trajectory"] == pytest.approx(0.6, abs=1e-6)
    assert div["semantic"] == pytest.approx(0.0, abs=1e-6)
    # Each trace burned one API call's worth of fake spend.
    assert cost.calls == 3
    assert cost.total_usd > 0.0


def test_per_corpus_replay_cached_calls_dont_double_charge() -> None:
    """The replayer caches per-config results. Re-issuing the same
    candidate config should NOT increment the cost tracker."""
    traces = [_synth_trace("t1", "Refund for order #1", "ok")]
    replay_fn, cost = build_live_replay_fn_per_corpus(
        baseline_traces=traces,
        max_cost_usd=1.00,
        replayer_factory=FakeReplayer,
    )
    cfg = {
        "prompt.system": "You are a refund agent.",
        "model": "gpt-4o-mini",
        "params.temperature": 0.0,
    }
    replay_fn(cfg)
    first_calls = cost.calls
    replay_fn(cfg)
    # Same config — the per-trace replayer's cache should short-circuit
    # so cost.calls doesn't go up.
    assert cost.calls == first_calls


# ---------------------------------------------------------------------------
# End-to-end through run_diagnose_pr — the integration shape we promise.
# ---------------------------------------------------------------------------


def test_diagnose_pr_live_path_produces_bootstrap_ci_via_fake(tmp_path: Path) -> None:
    """Wire the FakeReplayer into run_diagnose_pr via monkey-patch
    so the production code paths are exercised end-to-end with a
    deterministic backend. Asserts that --backend live produces a
    cause with bootstrap CI populated (the headline claim of the
    backend honesty fix in P1).

    Reuses the refund-causal-diagnosis fixtures rather than
    synthesising .agentlog from scratch — the parser is strict
    about envelope schema and we'd otherwise be testing the parser
    instead of the live pipeline."""
    import shutil

    repo_root = Path(__file__).resolve().parents[2]
    fixtures = repo_root / "examples" / "refund-causal-diagnosis"
    if not fixtures.is_dir():
        pytest.skip("refund-causal-diagnosis example not in tree")

    # Copy traces + configs into tmp_path so we don't touch the
    # committed example state.
    shutil.copytree(fixtures / "baseline_traces", tmp_path / "traces")
    shutil.copy(fixtures / "baseline.yaml", tmp_path / "baseline.yaml")
    shutil.copy(fixtures / "candidate.yaml", tmp_path / "candidate.yaml")

    # Monkey-patch the live module to swap in FakeReplayer where the
    # real OpenAIReplayer would otherwise be loaded.
    import shadow.diagnose_pr.live as _live_mod

    real_factory = _live_mod.build_live_replay_fn_per_corpus

    def _patched_factory(*, baseline_traces: list[LoadedTrace], **kwargs: Any) -> Any:
        kwargs["replayer_factory"] = FakeReplayer
        return real_factory(baseline_traces=baseline_traces, **kwargs)

    _live_mod.build_live_replay_fn_per_corpus = _patched_factory  # type: ignore[assignment]
    try:
        result = run_diagnose_pr(
            DiagnoseOptions(
                traces=[tmp_path / "traces"],
                candidate_traces=None,
                baseline_config=tmp_path / "baseline.yaml",
                candidate_config=tmp_path / "candidate.yaml",
                backend="live",
                n_bootstrap=50,  # tight bootstrap to keep test fast
                max_cost_usd=10.00,
            )
        )
    finally:
        _live_mod.build_live_replay_fn_per_corpus = real_factory  # type: ignore[assignment]

    # The dominant cause should be the prompt change.
    assert result.report.dominant_cause is not None
    cause = result.report.dominant_cause
    # Backend-honesty assertion: live mode populates bootstrap CI.
    assert cause.ci_low is not None and cause.ci_high is not None
    # E-value should be present too (sensitivity=True in causal_from_replay).
    assert cause.e_value is not None
    # Cost was tracked through real CostTracker.
    assert result.cost_usd is not None
    assert result.cost_usd > 0.0
    # And the report did NOT flag this as synthetic_mock — it's live
    # (just a fake client). The synthetic_mock disclosure is only
    # for `--backend mock`.
    assert "synthetic_mock" not in result.report.flags
