"""Offline tests for the replay backends.

The :class:`RecordedReplayer` exercises the entire
:func:`causal_attribution` pipeline against pre-recorded responses.
The :class:`OpenAIReplayer`'s constructor and divergence math are
also unit-tested here without a live API call.

Live API tests live in ``test_causal_replay_live.py`` and gate on
``SHADOW_RUN_NETWORK_TESTS=1`` + ``OPENAI_API_KEY``.
"""

from __future__ import annotations

from typing import Any

import pytest

from shadow.causal import (
    OpenAIReplayer,
    RecordedReplayer,
    ReplayResult,
    causal_attribution,
)
from shadow.causal.replay.openai_replayer import (
    _canonical_config_hash as _live_hash,
)
from shadow.causal.replay.openai_replayer import (
    _normalised_levenshtein,
    _relative_delta,
)

# ---------------------------------------------------------------------------
# RecordedReplayer
# ---------------------------------------------------------------------------


class TestRecordedReplayerLookup:
    def test_returns_recorded_result(self) -> None:
        cfg = {"system_prompt": "verify before refund", "user_prompt": "I want a refund"}
        rec = ReplayResult(
            config=cfg,
            response_text="I'll verify your identity first.",
            tool_calls=["verify_user"],
            stop_reason="stop",
            latency_ms=420.0,
            output_tokens=18,
            divergence={"safety": 0.0, "trajectory": 0.0},
        )
        replayer = RecordedReplayer.from_results([(cfg, rec)])
        out = replayer(cfg)
        assert out.response_text == "I'll verify your identity first."
        assert out.divergence == {"safety": 0.0, "trajectory": 0.0}

    def test_missing_config_raises_keyerror(self) -> None:
        replayer = RecordedReplayer({})
        with pytest.raises(KeyError, match="no recording"):
            replayer({"system_prompt": "X"})

    def test_canonical_hash_matches_across_replayers(self) -> None:
        # The recorded and live replayers must hash the same config to
        # the same key so a recorded table is portable.
        from shadow.causal.replay.recorded import _canonical_config_hash as recorded_hash

        cfg = {"a": 1, "b": "x", "c": [1, 2]}
        assert recorded_hash(cfg) == _live_hash(cfg)


# ---------------------------------------------------------------------------
# OpenAIReplayer construction guards
# ---------------------------------------------------------------------------


class TestOpenAIReplayerConstruction:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIReplayer(baseline_response_text="x")

    def test_constructor_does_not_accept_api_key_param(self) -> None:
        """Defensive: the class must NOT take an api_key parameter so
        keys cannot leak into stack frames or test fixtures."""
        import inspect

        sig = inspect.signature(OpenAIReplayer.__init__)
        param_names = set(sig.parameters)
        assert "api_key" not in param_names
        assert "key" not in param_names

    def test_constructor_succeeds_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Doesn't make any API calls — just verifies the class
        # constructs cleanly when the env is set.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-construction-only")
        replayer = OpenAIReplayer(baseline_response_text="hello")
        assert replayer.cache_size == 0


# ---------------------------------------------------------------------------
# Divergence math (unit-test the helper functions directly)
# ---------------------------------------------------------------------------


class TestDivergenceHelpers:
    def test_levenshtein_identical_is_zero(self) -> None:
        assert _normalised_levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0.0

    def test_levenshtein_disjoint_is_one(self) -> None:
        assert _normalised_levenshtein(["a", "b"], ["c", "d"]) == 1.0

    def test_levenshtein_one_substitution_is_third(self) -> None:
        # 1 edit out of 3 positions → 1/3.
        assert _normalised_levenshtein(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(1 / 3)

    def test_levenshtein_empty_lists_are_zero(self) -> None:
        assert _normalised_levenshtein([], []) == 0.0

    def test_levenshtein_one_empty_is_one(self) -> None:
        assert _normalised_levenshtein([], ["a"]) == 1.0
        assert _normalised_levenshtein(["a"], []) == 1.0

    def test_relative_delta_identical_is_zero(self) -> None:
        assert _relative_delta(100.0, 100.0) == 0.0

    def test_relative_delta_baseline_zero_uses_unit_denom(self) -> None:
        # Avoids divide-by-zero; 1.0 / max(0, 1.0) = 1.0, capped to 1.
        assert _relative_delta(1.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# Causal attribution end-to-end with RecordedReplayer
# ---------------------------------------------------------------------------


class TestCausalAttributionWithRecordedReplayer:
    """Full end-to-end: causal_attribution accepts a Replayer (which
    is a Protocol of ``__call__(config) -> ReplayResult``) but the
    public API takes a ``replay_fn`` returning a divergence dict.
    Build an adapter that pipes ReplayResult.divergence through."""

    def test_causal_attribution_with_replayer_adapter(self) -> None:
        # Two configs differing on `system_prompt`. The recorded
        # replayer returns specific divergence vectors; causal_attr
        # must compute the right ATE.
        cfg_baseline = {"system_prompt": "verify_first"}
        cfg_intervened = {"system_prompt": "refund_first"}

        replayer = RecordedReplayer.from_results(
            [
                (
                    cfg_baseline,
                    ReplayResult(
                        config=cfg_baseline,
                        response_text="Verifying...",
                        tool_calls=["verify_user"],
                        stop_reason="stop",
                        latency_ms=400.0,
                        output_tokens=20,
                        divergence={"safety": 0.0, "trajectory": 0.0},
                    ),
                ),
                (
                    cfg_intervened,
                    ReplayResult(
                        config=cfg_intervened,
                        response_text="Issuing refund...",
                        tool_calls=["refund_order"],
                        stop_reason="stop",
                        latency_ms=380.0,
                        output_tokens=22,
                        divergence={"safety": 0.7, "trajectory": 0.6},
                    ),
                ),
            ]
        )

        def replay_fn(config: dict[str, Any]) -> dict[str, float]:
            return replayer(config).divergence

        result = causal_attribution(
            baseline_config=cfg_baseline,
            candidate_config=cfg_intervened,
            replay_fn=replay_fn,
            n_replays=1,
        )
        # ATE = intervened - baseline = 0.7 - 0.0 = 0.7 on safety;
        # 0.6 - 0.0 = 0.6 on trajectory.
        assert result.ate["system_prompt"]["safety"] == pytest.approx(0.7)
        assert result.ate["system_prompt"]["trajectory"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# OpenAIReplayer never logs the API key
# ---------------------------------------------------------------------------


class TestOpenAIReplayerNeverLogsKey:
    def test_repr_does_not_contain_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-leak-check-XYZ")
        replayer = OpenAIReplayer(baseline_response_text="x")
        rep = repr(replayer)
        assert "sk-test-leak-check-XYZ" not in rep
        assert "OPENAI_API_KEY" not in rep
