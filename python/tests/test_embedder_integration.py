"""End-to-end integration test for the Embedder trait via PyO3 callback.

The Rust ``Embedder`` trait is the v2.7 extension point that lets users
plug any embedding source into the semantic axis without bundling
heavy ML dependencies in the wheel. Up through v2.7 the trait was
unit-tested in pure Rust but the cross-language call path
(Python → Rust → Python embedder callback → Rust cosine math) was
unverified.

This test exercises that full round trip:

    1. A pure-Python embedder closure is constructed (deterministic
       length/length-of-words vectors, plus a closure backed by
       ``sentence-transformers/all-MiniLM-L6-v2`` if the optional
       extra is installed — see ``test_sentence_transformers_path``).
    2. ``shadow._core.compute_semantic_axis_with_embedder`` is called
       with that closure.
    3. The Rust side calls back into Python for each text, gets the
       embedding vectors, and computes cosine similarity in Rust.
    4. The returned AxisStat is asserted against a hand-computed
       reference so any drift in the cosine math (Rust vs Python) is
       caught.

All tests skip cleanly when sentence-transformers is unavailable —
the deterministic-closure path runs unconditionally and is the
canonical end-to-end check.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from shadow import _core


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
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    }


def _length_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic 2-D embedder: (len_chars, n_words). Lets us hand-
    compute the expected cosine similarities and verify the Rust path
    returns matching values."""
    return [[float(len(t)), float(len(t.split()))] for t in texts]


def _expected_cosine(b: list[float], c: list[float]) -> float:
    """Reference cosine for hand-rolled validation."""
    nb = math.sqrt(sum(x * x for x in b))
    nc = math.sqrt(sum(x * x for x in c))
    if nb < 1e-12 and nc < 1e-12:
        return 1.0
    if nb < 1e-12 or nc < 1e-12:
        return 0.0
    dot = sum(x * y for x, y in zip(b, c, strict=True))
    return max(-1.0, min(1.0, dot / (nb * nc)))


class TestPyO3EmbedderCallback:
    def test_function_is_exposed(self) -> None:
        """The PyO3 binding must be discoverable on shadow._core."""
        assert hasattr(_core, "compute_semantic_axis_with_embedder")
        assert callable(_core.compute_semantic_axis_with_embedder)

    def test_non_callable_embedder_raises(self) -> None:
        baseline = [_record(1, "alpha")]
        candidate = [_record(2, "alpha")]
        with pytest.raises(ValueError, match="callable"):
            _core.compute_semantic_axis_with_embedder(
                baseline, candidate, embedder="not a function"
            )

    def test_identical_text_yields_similarity_1(self) -> None:
        baseline = [_record(1, "alpha beta gamma")]
        candidate = [_record(2, "alpha beta gamma")]
        result = _core.compute_semantic_axis_with_embedder(
            baseline, candidate, _length_embedder, seed=1
        )
        # Identical text → identical vectors → cosine 1.0.
        assert result["candidate_median"] == pytest.approx(1.0, abs=1e-9)
        assert result["axis"] == "semantic"

    def test_orthogonal_vectors_score_zero(self) -> None:
        # Same words but engineered embedder that returns orthogonal
        # vectors per text (overrides the length-based heuristic).
        def orthogonal(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i, _ in enumerate(texts)]

        # 5 baseline-vs-candidate pairs; baseline is index 0 (always [1,0]),
        # candidate is index 0 of its own list (also always [1,0] under
        # this embedder because each call is independent).
        # To force orthogonality, we exploit the per-call independence:
        # the embedder gets baselines on one call, candidates on another;
        # both at index 0 → [1,0]. So we need a stateful embedder.
        state = {"i": 0}

        def alternating(texts: list[str]) -> list[list[float]]:
            base = state["i"]
            state["i"] += 1
            return [[1.0, 0.0] if base == 0 else [0.0, 1.0] for _ in texts]

        baseline = [_record(1, "x"), _record(2, "y"), _record(3, "z")]
        candidate = [_record(11, "x"), _record(12, "y"), _record(13, "z")]
        result = _core.compute_semantic_axis_with_embedder(baseline, candidate, alternating, seed=2)
        # Baseline call → [[1,0]]*3, candidate call → [[0,1]]*3.
        # All pairwise cosines = 0.
        assert result["candidate_median"] == pytest.approx(0.0, abs=1e-9)

    def test_cosine_matches_hand_computed_reference(self) -> None:
        """For three pairs with known length embeddings, verify the
        Rust cosine output equals the Python reference within
        1e-6 relative tolerance."""
        baseline_texts = ["a", "ab", "abc"]
        candidate_texts = ["abcde", "abc", "ab"]
        baseline_records = [_record(i, t) for i, t in enumerate(baseline_texts)]
        candidate_records = [_record(100 + i, t) for i, t in enumerate(candidate_texts)]

        # Reference cosine per pair under _length_embedder.
        b_vecs = _length_embedder(baseline_texts)
        c_vecs = _length_embedder(candidate_texts)
        per_pair = [_expected_cosine(b, c) for b, c in zip(b_vecs, c_vecs, strict=True)]
        per_pair_sorted = sorted(per_pair)
        n = len(per_pair_sorted)
        expected_median = (
            per_pair_sorted[n // 2]
            if n % 2 == 1
            else (per_pair_sorted[n // 2 - 1] + per_pair_sorted[n // 2]) / 2.0
        )

        result = _core.compute_semantic_axis_with_embedder(
            baseline_records, candidate_records, _length_embedder, seed=42
        )
        assert result["candidate_median"] == pytest.approx(expected_median, abs=1e-6), (
            f"Rust cosine median {result['candidate_median']} != "
            f"Python reference {expected_median} on per-pair {per_pair}"
        )

    def test_misshapen_embedder_returns_empty_axis(self) -> None:
        """An embedder that returns the wrong number of vectors must
        produce an empty axis, not crash."""

        def wrong_count(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]  # always 1, regardless of input length

        baseline = [_record(1, "x"), _record(2, "y")]
        candidate = [_record(11, "x"), _record(12, "y")]
        result = _core.compute_semantic_axis_with_embedder(baseline, candidate, wrong_count, seed=3)
        # Empty axis: severity none, n_pairs 0.
        assert result["severity"] == "none"


class TestSentenceTransformersIntegration:
    """Optional: validate against the real sentence-transformers
    embedder. Skips cleanly when the package isn't installed."""

    def test_sentence_transformers_round_trip(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip(
                "sentence-transformers not installed; install with "
                "`pip install -e python/[embeddings]` to run this test"
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def st_embed(texts: list[str]) -> list[list[float]]:
            arr = model.encode(texts, show_progress_bar=False)
            return [vec.tolist() for vec in arr]

        # Paraphrase pair: TF-IDF would score ~0; sentence-transformers
        # scores high because the embeddings capture meaning.
        baseline = [_record(1, "yes"), _record(2, "I agree")]
        candidate = [_record(11, "yes"), _record(12, "I agree")]
        result = _core.compute_semantic_axis_with_embedder(baseline, candidate, st_embed, seed=7)
        # Identical text on both sides → similarity ≈ 1.
        assert result["candidate_median"] > 0.99, (
            f"identical text under sentence-transformers should score ≈1; "
            f"got {result['candidate_median']}"
        )

        # Cross paraphrase: baseline says "yes", candidate says "I agree".
        # The interesting comparison is "neural embedder produces a
        # higher score than TF-IDF would on the same disjoint-vocabulary
        # pair." TF-IDF on these inputs scores 0 (zero token overlap);
        # MiniLM gives a positive cosine because the embedding space
        # captures some semantic relatedness. We don't assert a high
        # absolute threshold (MiniLM scores "yes"/"I agree" around 0.2,
        # not 0.7-0.8 — the model is small and the texts are short),
        # but the cosine MUST be strictly positive, which is the
        # "end-to-end neural path produces non-trivial output"
        # guarantee.
        baseline_para = [_record(1, "yes"), _record(2, "yes")]
        candidate_para = [_record(11, "I agree"), _record(12, "I agree")]
        result_para = _core.compute_semantic_axis_with_embedder(
            baseline_para, candidate_para, st_embed, seed=8
        )
        median = result_para["candidate_median"]
        assert 0.0 < median < 1.0, (
            f"neural embedder paraphrase cosine should be in (0, 1) — "
            f"strictly positive (vs TF-IDF's 0 on disjoint vocab) and "
            f"strictly less than 1 (different surface forms); got {median}"
        )
