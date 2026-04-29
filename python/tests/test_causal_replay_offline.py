"""Offline tests for the replay backends.

The :class:`RecordedReplayer` exercises the entire
:func:`causal_attribution` pipeline against pre-recorded responses.
The :class:`OpenAIReplayer`'s constructor and divergence math are
also unit-tested here without a live API call.

Live API tests live in ``test_causal_replay_live.py`` and gate on
``SHADOW_RUN_NETWORK_TESTS=1`` + ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
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

try:
    import openai as _openai  # noqa: F401

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

_needs_openai = pytest.mark.skipif(
    not _OPENAI_AVAILABLE,
    reason="install shadow-diff[openai] to run replayer construction tests",
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
        # Doesn't make any API calls - just verifies the class
        # constructs cleanly when the env is set.
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("install shadow-diff[openai] to construct OpenAIReplayer")
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


@_needs_openai
class TestOpenAIReplayerNeverLogsKey:
    def test_repr_does_not_contain_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-leak-check-XYZ")
        replayer = OpenAIReplayer(baseline_response_text="x")
        rep = repr(replayer)
        assert "sk-test-leak-check-XYZ" not in rep
        assert "OPENAI_API_KEY" not in rep


def _len_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic 2-D embedder for hand-rolled validation."""
    return [[float(len(t)), float(len(t.split()))] for t in texts]


def _disjoint_class_embedder(texts: list[str]) -> list[list[float]]:
    """Two-class embedder: 'A'-prefixed → [1,0], else → [0,1]. Lets
    us force a specific cosine value in tests."""
    out: list[list[float]] = []
    for t in texts:
        if t.upper().startswith("A"):
            out.append([1.0, 0.0])
        else:
            out.append([0.0, 1.0])
    return out


@pytest.fixture(autouse=True)
def _set_dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenAIReplayer constructor refuses to run without
    OPENAI_API_KEY in the env; for unit tests we just need the env
    var to be present (no API calls happen in these tests)."""
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-unit-only")


@_needs_openai
class TestDefaultJaccardPath:
    def test_no_embedder_uses_jaccard(self) -> None:
        """When embedder is omitted, semantic divergence is Jaccard
        on whitespace tokens (v2.8 behaviour preserved)."""
        replayer = OpenAIReplayer(
            baseline_response_text="alpha beta gamma",
            embedder=None,
        )
        # Direct test of the divergence path — no API call needed.
        div = replayer._compute_divergence(
            response_text="alpha beta gamma",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Identical → Jaccard divergence 0.
        assert div["semantic"] == 0.0

    def test_no_embedder_disjoint_text_is_one(self) -> None:
        replayer = OpenAIReplayer(
            baseline_response_text="alpha beta",
            embedder=None,
        )
        div = replayer._compute_divergence(
            response_text="delta epsilon",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Disjoint → Jaccard divergence 1.0.
        assert div["semantic"] == 1.0


@_needs_openai
class TestEmbedderPath:
    def test_identical_text_gives_zero(self) -> None:
        replayer = OpenAIReplayer(
            baseline_response_text="alpha beta gamma",
            embedder=_len_embedder,
        )
        div = replayer._compute_divergence(
            response_text="alpha beta gamma",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Same text → same vectors → cosine 1.0 → divergence 0.0.
        assert div["semantic"] == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_yield_divergence_one(self) -> None:
        """Force the embedder to produce orthogonal vectors. cosine
        = 0, divergence = 1.0."""
        replayer = OpenAIReplayer(
            baseline_response_text="alpha-baseline",  # → [1, 0]
            embedder=_disjoint_class_embedder,
        )
        div = replayer._compute_divergence(
            response_text="zeta-candidate",  # → [0, 1]
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        assert div["semantic"] == pytest.approx(1.0, abs=1e-9)

    def test_paraphrase_jaccard_zero_but_embedder_nonzero(self) -> None:
        """The headline of the v2.9 closure: a baseline/candidate pair
        with disjoint tokens (Jaccard = 1.0) where a smart embedder
        produces a non-trivial cosine (divergence < 1.0). Without the
        embedder, the agent regression detection would lie about the
        semantic shift; with the embedder, it reports honestly."""

        def near_paraphrase_embedder(texts: list[str]) -> list[list[float]]:
            # Both texts map to the same vector → cosine 1, divergence 0.
            return [[1.0, 0.0, 0.0] for _ in texts]

        replayer = OpenAIReplayer(
            baseline_response_text="please confirm before refunding",
            embedder=near_paraphrase_embedder,
        )
        # Token sets are disjoint between baseline and candidate.
        div = replayer._compute_divergence(
            response_text="verify identity prior to issuing refund",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Embedder treats them as paraphrases → divergence ≈ 0.
        assert div["semantic"] < 0.01

        # Compare with the Jaccard fallback — the SAME text pair
        # under no embedder produces divergence 1.0 (zero token overlap).
        replayer_no_emb = OpenAIReplayer(
            baseline_response_text="please confirm before refunding",
            embedder=None,
        )
        div_no_emb = replayer_no_emb._compute_divergence(
            response_text="verify identity prior to issuing refund",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        assert div_no_emb["semantic"] == 1.0
        # Embedder-aware divergence is strictly closer to "no shift" than
        # the Jaccard divergence on this paraphrase pair.
        assert div["semantic"] < div_no_emb["semantic"]


@_needs_openai
class TestEmbedderFailureModes:
    def test_embedder_returns_wrong_count_treated_as_max_divergence(self) -> None:
        def bad(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]  # always one vector regardless of input

        replayer = OpenAIReplayer(
            baseline_response_text="x",
            embedder=bad,
        )
        div = replayer._compute_divergence(
            response_text="y",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Misshapen embedder output → fail-loud, max divergence.
        # We don't silently produce "0 divergence" because that would
        # mask the embedder bug as "no regression."
        assert div["semantic"] == 1.0

    def test_embedder_dim_mismatch_treated_as_max(self) -> None:
        def bad(texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0], [3.0, 4.0, 5.0]]  # mismatched dimension

        replayer = OpenAIReplayer(
            baseline_response_text="x",
            embedder=bad,
        )
        div = replayer._compute_divergence(
            response_text="y",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        assert div["semantic"] == 1.0

    def test_embedder_raises_treated_as_max(self) -> None:
        def bad(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("network failure")

        replayer = OpenAIReplayer(
            baseline_response_text="x",
            embedder=bad,
        )
        div = replayer._compute_divergence(
            response_text="y",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Exception → divergence 1.0 (fail-loud, no silent 0).
        assert div["semantic"] == 1.0

    def test_both_zero_vectors_treated_as_identical(self) -> None:
        def all_zero(texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in texts]

        replayer = OpenAIReplayer(
            baseline_response_text="x",
            embedder=all_zero,
        )
        div = replayer._compute_divergence(
            response_text="y",
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        # Both-zero → divergence 0 (matches the Rust cosine convention).
        assert div["semantic"] == 0.0


@_needs_openai
class TestAgreementWithRustSemanticAxis:
    """Cross-validation: the embedder-aware semantic divergence in
    the OpenAIReplayer must produce the SAME numeric result as the
    Rust nine-axis semantic axis on the same input pair."""

    def test_replayer_matches_rust_cosine(self) -> None:
        from shadow import _core

        # Use a deterministic 3-D embedder — easy to reason about.
        def emb(texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                vowels = sum(1 for c in t if c.lower() in "aeiou")
                consonants = sum(1 for c in t if c.isalpha() and c.lower() not in "aeiou")
                out.append([float(vowels), float(consonants), float(len(t))])
            return out

        baseline_text = "verify identity before refund"
        candidate_text = "issue the refund immediately"

        # Replayer-side semantic divergence (1 - cosine).
        replayer = OpenAIReplayer(
            baseline_response_text=baseline_text,
            embedder=emb,
        )
        div = replayer._compute_divergence(
            response_text=candidate_text,
            tool_calls=[],
            stop_reason="stop",
            latency_ms=0.0,
            output_tokens=0,
        )
        replayer_sem = div["semantic"]

        # Rust-side: same texts as response records, same embedder.
        def _record(idx: int, text: str) -> dict[str, Any]:
            return {
                "version": "0.1",
                "id": f"sha256:{idx:064x}",
                "kind": "chat_response",
                "ts": "2026-04-28T00:00:00.000Z",
                "parent": "sha256:" + "0" * 64,
                "meta": {},
                "payload": {
                    "model": "x",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "latency_ms": 0,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "thinking_tokens": 0,
                    },
                },
            }

        rust_stat = _core.compute_semantic_axis_with_embedder(
            [_record(1, baseline_text)],
            [_record(2, candidate_text)],
            emb,
            seed=0,
        )
        # Rust reports candidate_median as the cosine SIMILARITY (not
        # divergence). Convert to divergence to compare.
        rust_div = max(0.0, 1.0 - rust_stat["candidate_median"])
        # Within float tolerance.
        assert replayer_sem == pytest.approx(rust_div, abs=1e-6), (
            f"replayer semantic divergence ({replayer_sem}) should match "
            f"Rust nine-axis semantic divergence ({rust_div}) on the same "
            f"(text pair, embedder) inputs"
        )
