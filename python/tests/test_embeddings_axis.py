"""Tests for the optional sentence-transformer semantic axis.

Gated on `sentence-transformers` being installed. In CI without the
extra, the tests are collected and skipped rather than erroring.
"""

from __future__ import annotations

from typing import Any

import pytest

st_installed: bool
try:
    import sentence_transformers  # type: ignore[import-not-found]  # noqa: F401

    st_installed = True
except ImportError:
    st_installed = False


pytestmark = pytest.mark.skipif(
    not st_installed,
    reason="sentence-transformers not installed (shadow[embeddings] extra)",
)


def _record(kind: str, idx: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{kind[0]}{idx:063d}",
        "kind": kind,
        "ts": "2026-04-21T10:00:00.000Z",
        "parent": None,
        "payload": payload,
    }


def _resp(idx: int, text: str) -> dict[str, Any]:
    return _record(
        "chat_response",
        idx,
        {
            "model": "x",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "latency_ms": 0,
            "usage": {"input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0},
        },
    )


def test_embedder_imports_and_encodes() -> None:
    from shadow.embeddings import SemanticEmbedder

    emb = SemanticEmbedder()
    vectors = emb.encode(["hello world", "goodbye world"])
    assert vectors.shape == (2, 384)


def test_recompute_semantic_axis_paraphrase_scores_high() -> None:
    """Two paraphrases of the same meaning should score high on embeddings,
    which lexical BM25 can miss."""
    from shadow.embeddings import SemanticEmbedder, recompute_semantic_axis

    meta = _record("metadata", 0, {})
    b = [meta] + [_resp(i, "Your refund has been issued.") for i in range(10)]
    c = [meta] + [_resp(i, "The refund for your order has been processed.") for i in range(10)]
    row = recompute_semantic_axis(b, c, SemanticEmbedder(), seed=1, n_bootstrap=100)
    assert row["axis"] == "semantic"
    assert row["candidate_median"] > 0.7  # real embeddings see paraphrase as similar
    assert row["severity"] in {"none", "minor", "moderate"}


def test_recompute_semantic_axis_unrelated_text_scores_low() -> None:
    from shadow.embeddings import SemanticEmbedder, recompute_semantic_axis

    meta = _record("metadata", 0, {})
    b = [meta] + [_resp(i, "Your refund has been issued.") for i in range(10)]
    c = [meta] + [_resp(i, "The capital of France is Paris.") for i in range(10)]
    row = recompute_semantic_axis(b, c, SemanticEmbedder(), seed=1, n_bootstrap=100)
    assert row["candidate_median"] < 0.5
    assert row["severity"] in {"severe", "moderate", "minor"}
