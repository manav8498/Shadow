"""Sentence-transformer path for Shadow's semantic axis.

Replaces the default TF-IDF cosine with `all-MiniLM-L6-v2` cosine — a
384-d dense embedding that captures paraphrase similarity ("I'll refund
your order" ≈ "Your refund is being processed") that lexical methods
miss. Gated behind the optional `[embeddings]` extra to keep the base
install dependency-light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadow.errors import ShadowBackendError

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticEmbedder:
    """Wraps `sentence-transformers` for Shadow's semantic axis.

    Lazy-imports the SDK on first use so importing this module doesn't
    require the extra to be installed.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]
                SentenceTransformer,
            )
        except ImportError as e:
            raise ShadowBackendError(
                "sentence-transformers not installed\n"
                "hint: pip install 'shadow[embeddings]' "
                "(or add sentence-transformers>=3.3.1)"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        """Encode `texts` to a dense matrix of shape `(len(texts), dim)`."""
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float64)


def _cosine_row_wise(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise cosine similarity of two (n, dim) matrices."""
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    denom = np.where((norms_a * norms_b) < 1e-12, 1.0, norms_a * norms_b)
    dots = np.sum(a * b, axis=1)
    out: NDArray[np.float64] = (dots / denom).clip(0.0, 1.0).astype(np.float64)
    return out


def recompute_semantic_axis(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    embedder: SemanticEmbedder,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Return a DiffReport-shaped semantic row, using real embeddings.

    Splice the result into the DiffReport in place of axis 1 to
    override the Rust-side TF-IDF default.
    """
    b_texts, c_texts = _extract_pairs(baseline_records, candidate_records)
    if not b_texts:
        return _empty_row()
    b_vec = embedder.encode(b_texts)
    c_vec = embedder.encode(c_texts)
    similarities = _cosine_row_wise(b_vec, c_vec)
    baseline_median = 1.0
    candidate_median = float(np.median(similarities))
    delta = candidate_median - baseline_median

    rng = np.random.default_rng(seed)
    resampled = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(similarities), size=len(similarities))
        resampled[i] = float(np.median(similarities[idx])) - baseline_median
    ci_low = float(np.percentile(resampled, 2.5))
    ci_high = float(np.percentile(resampled, 97.5))

    return {
        "axis": "semantic",
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "delta": delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "severity": _severity(delta, baseline_median, ci_low, ci_high),
        "n": len(similarities),
        "flags": _flags(ci_low, ci_high, len(similarities)),
        "embedder": embedder.model_name,
    }


def _extract_pairs(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    b = [_response_text(r["payload"]) for r in baseline if r.get("kind") == "chat_response"]
    c = [_response_text(r["payload"]) for r in candidate if r.get("kind") == "chat_response"]
    n = min(len(b), len(c))
    return b[:n], c[:n]


def _response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content") or []
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts)


def _empty_row() -> dict[str, Any]:
    return {
        "axis": "semantic",
        "baseline_median": 0.0,
        "candidate_median": 0.0,
        "delta": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "severity": "none",
        "n": 0,
        "flags": [],
    }


def _severity(delta: float, baseline_median: float, ci_low: float, ci_high: float) -> str:
    if abs(delta) < 1e-9:
        return "none"
    ci_crosses_zero = ci_low <= 0.0 <= ci_high and not (ci_low == 0.0 and ci_high == 0.0)
    if ci_crosses_zero and abs(delta) < max(abs(baseline_median) * 0.05, 1e-9):
        return "none"
    if abs(baseline_median) < 1e-9:
        base = "none" if abs(delta) < 1e-9 else "minor"
    else:
        rel = abs(delta / baseline_median)
        if rel < 0.10:
            base = "minor"
        elif rel < 0.30:
            base = "moderate"
        else:
            base = "severe"
    if ci_crosses_zero and base in ("moderate", "severe"):
        return "minor"
    return base


def _flags(ci_low: float, ci_high: float, n: int) -> list[str]:
    flags: list[str] = []
    if 0 < n < 5:
        flags.append("low_power")
    if ci_low <= 0.0 <= ci_high and not (ci_low == 0.0 and ci_high == 0.0):
        flags.append("ci_crosses_zero")
    return flags


__all__ = ["SemanticEmbedder", "recompute_semantic_axis"]
