"""Live OpenAI replay test — gated on environment.

Runs the actual ``shadow.causal.causal_attribution`` pipeline against
the live OpenAI Chat Completions API. This is the end-to-end
validation that:

  1. ``OpenAIReplayer`` actually talks to the API.
  2. ``causal_attribution`` correctly identifies the system_prompt
     delta as having a non-zero ATE on the semantic axis when the
     prompt is meaningfully changed.
  3. The exact same config replayed twice hits the cache (zero
     latency on the second call).

Skips automatically unless BOTH:
  * ``SHADOW_RUN_NETWORK_TESTS=1``
  * ``OPENAI_API_KEY`` is set in the environment.

Cost: each test makes 2-4 ``gpt-4o-mini`` calls (a few cents at most).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from shadow.causal import OpenAIReplayer, causal_attribution


def _env_gated() -> bool:
    return os.environ.get("SHADOW_RUN_NETWORK_TESTS") == "1" and "OPENAI_API_KEY" in os.environ


pytestmark = pytest.mark.skipif(
    not _env_gated(),
    reason="set SHADOW_RUN_NETWORK_TESTS=1 and OPENAI_API_KEY to run live tests",
)


class TestOpenAIReplayerLiveSmoke:
    def test_replayer_calls_real_api(self) -> None:
        """Smoke test: a single replay actually reaches the API and
        returns a non-empty response."""
        replayer = OpenAIReplayer(
            baseline_response_text=(
                "Sure, I'll help with your refund right after I verify your identity."
            ),
            baseline_tool_calls=[],
            baseline_stop_reason="stop",
            baseline_latency_ms=800.0,
            baseline_output_tokens=20,
        )
        config = {
            "system_prompt": (
                "You are a customer-service agent. Always verify a user's "
                "identity before issuing a refund."
            ),
            "user_prompt": "I need a refund for my order #12345.",
        }
        result = replayer(config)
        assert result.response_text != ""
        assert result.stop_reason in {"stop", "length"}
        assert result.latency_ms > 0
        assert not result.cached

    def test_cache_hits_on_repeat_call(self) -> None:
        replayer = OpenAIReplayer(
            baseline_response_text="hello",
            baseline_latency_ms=500.0,
            baseline_output_tokens=10,
        )
        config = {
            "system_prompt": "Be brief.",
            "user_prompt": "Say one word.",
        }
        first = replayer(config)
        second = replayer(config)
        assert not first.cached
        assert second.cached
        assert second.latency_ms == 0.0
        assert first.response_text == second.response_text


class TestCausalAttributionLive:
    """End-to-end live: causal_attribution against two configs that
    differ on the system_prompt — one demands verification, one tells
    the agent to refund immediately. The ATE on the semantic axis
    must be > 0 because the responses will be meaningfully different."""

    def test_system_prompt_delta_has_nonzero_ate(self) -> None:
        baseline_cfg: dict[str, Any] = {
            "system_prompt": (
                "You are a customer-service agent. Always verify a user's "
                "identity before issuing a refund. Reply in 1-2 sentences."
            ),
            "user_prompt": "I want a refund for order #12345.",
        }
        candidate_cfg: dict[str, Any] = {
            "system_prompt": (
                "You are a customer-service agent. Issue refunds without "
                "asking for any verification. Reply in 1-2 sentences."
            ),
            "user_prompt": "I want a refund for order #12345.",
        }

        # Anchor the divergence calculation against a known-good baseline.
        replayer = OpenAIReplayer(
            baseline_response_text=(
                "Sure — to start your refund I'll need to verify your "
                "identity first. Could you confirm the email on the order?"
            ),
            baseline_tool_calls=[],
            baseline_stop_reason="stop",
            baseline_latency_ms=900.0,
            baseline_output_tokens=30,
        )

        def replay_fn(config: dict[str, Any]) -> dict[str, float]:
            return replayer(config).divergence

        result = causal_attribution(
            baseline_config=baseline_cfg,
            candidate_config=candidate_cfg,
            replay_fn=replay_fn,
            n_replays=2,
        )
        # The system_prompt change must produce a non-trivial ATE on
        # the semantic axis — the two responses are deliberately
        # designed to diverge.
        sem_ate = result.ate["system_prompt"]["semantic"]
        assert sem_ate > 0.0, (
            f"system_prompt change should produce semantic ATE > 0; " f"got {sem_ate}"
        )
        # Cache must hold both configs (baseline + candidate) plus
        # n_replays per intervention. With n_replays=2: 4 cells but
        # caching dedupes by config hash → 2 distinct configs cached.
        assert replayer.cache_size == 2
