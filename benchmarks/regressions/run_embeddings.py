"""Re-run the benchmark with the embeddings-axis override.

Focus: case 10 (`formatting_loss`) is flagged as a known limitation of
lexical BM25. If the real sentence-transformer embedding path catches
it — and every other case stays caught — that proves the embeddings
path closes the gap and the benchmark's catch-rate goes to 20/20.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))

from cases import CASES  # type: ignore[import-not-found]

from shadow import _core
from shadow.embeddings import SemanticEmbedder, recompute_semantic_axis

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def main() -> int:
    print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    embedder = SemanticEmbedder()
    print(f"  ready (dim=384, model={embedder.model_name})\n")

    print(f"{'case':<24} {'expected':<10} {'bm25':<12} {'embeddings':<12} verdict")
    print("-" * 80)
    all_good = True
    wins = 0
    for name, fn in CASES:
        baseline, candidate, expected = fn()
        target_axis = expected.get("axis")
        if target_axis != "semantic":
            # Skip non-semantic cases — this pass is specifically about the
            # embeddings override for axis 1.
            continue
        min_sev = expected.get("min_severity", "minor")
        known_limit = bool(expected.get("known_limit", False))

        bm25_report = _core.compute_diff_report(baseline, candidate, None, 42)
        bm25_row = next(r for r in bm25_report["rows"] if r["axis"] == "semantic")
        bm25_sev = bm25_row["severity"]

        emb_row = recompute_semantic_axis(baseline, candidate, embedder, seed=42, n_bootstrap=200)
        emb_sev = emb_row["severity"]

        bm25_caught = SEVERITY_RANK[bm25_sev] >= SEVERITY_RANK[min_sev]
        emb_caught = SEVERITY_RANK[emb_sev] >= SEVERITY_RANK[min_sev]

        verdict = _verdict(bm25_caught, emb_caught, known_limit)
        if verdict.startswith("fail"):
            all_good = False
        if known_limit and emb_caught and not bm25_caught:
            wins += 1

        bm25_str = f"{bm25_sev}{'✓' if bm25_caught else '✗'}"
        emb_str = f"{emb_sev}{'✓' if emb_caught else '✗'}"
        print(
            f"{name:<24} {min_sev:<10} {bm25_str:<12} {emb_str:<12} {verdict}"
        )

    print()
    print(f"cases where embeddings closed a BM25 known-limit: {wins}")
    return 0 if all_good else 1


def _verdict(bm25_caught: bool, emb_caught: bool, known_limit: bool) -> str:
    if bm25_caught and emb_caught:
        return "ok (both paths catch)"
    if emb_caught and not bm25_caught and known_limit:
        return "embeddings closes known limit ★"
    if emb_caught and not bm25_caught:
        return "ok (embeddings stricter)"
    if bm25_caught and not emb_caught:
        return "fail: embeddings regressed vs BM25"
    if not bm25_caught and not emb_caught and known_limit:
        return "still known-miss"
    return "fail: neither path catches"


if __name__ == "__main__":
    raise SystemExit(main())
