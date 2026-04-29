"""Tests for the v2.9 embedder path on OpenAIReplayer.

The v2.8 OpenAIReplayer computed semantic divergence via Jaccard on
whitespace tokens. The v2.9 ``embedder=`` parameter routes through
dense-vector cosine instead, matching the Rust nine-axis semantic
axis byte-for-byte.

These tests cover:
  1. The default path (no embedder) is unchanged Jaccard.
  2. With an embedder, identical text → divergence 0.
  3. With an embedder, paraphrase pairs that score 0 under Jaccard
     score < 1 under a neural embedder.
  4. Embedder failures fall back to divergence = 1.0 (fail-loud)
     rather than 0.0 (fail-silent).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from shadow.causal import OpenAIReplayer


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
