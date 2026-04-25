"""Cross-modal semantic diff axis for ``blob_ref`` records.

Two-tier comparison aligned with what RAGAS / TruLens / DeepEval
ship as their multimodal eval baseline:

- **Cheap tier (always available)**: 64-bit dHash Hamming distance
  on image blobs. Embedded in every ``blob_ref`` record by default
  (see :mod:`shadow.v02_records`). Threshold ≤10/64 = "near-duplicate,"
  ≥16 = "different."
- **Semantic tier (opt-in)**: cosine similarity over an
  ``embedding.vec`` field on the ``blob_ref`` record. Production
  defaults: CLIP ViT-B/32 for images (LangSmith / Langfuse default
  ~0.8 for "same image"), Whisper-embed for audio. Embeddings
  themselves are computed by the user — Shadow ships the diff
  axis, not the model.

Match strategy: positional. Baseline blob #N is compared to
candidate blob #N. Unmatched blobs (one side has more than the
other) are flagged. This mirrors how :mod:`shadow.diff` axis
alignment works for chat pairs — keeps the axis cheap and
deterministic. Best-match alignment via Hungarian assignment is
out of scope; the chat-pair aligner already does that, and it's
not what users ask for first on multimodal traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shadow.v02_records import phash_distance


@dataclass
class BlobDelta:
    """One blob-pair comparison result."""

    pair_index: int
    """Position in the per-trace blob_ref sequence."""
    baseline_blob_id: str | None
    candidate_blob_id: str | None
    mime: str | None
    """The candidate's mime, or baseline's if candidate is missing."""
    cheap_tier: dict[str, Any] | None = None
    """``{"algo": "dhash64", "distance": int, "threshold_near_dup": 10}``
    when both sides have a phash; None otherwise."""
    semantic_tier: dict[str, Any] | None = None
    """``{"model": str, "cosine": float, "threshold_same": float}`` when
    both sides have an embedding; None otherwise."""
    severity: str = "none"
    """One of: ``none`` (≤cheap_threshold or ≥cosine_threshold),
    ``minor``, ``moderate``, ``severe``. Computed from whichever tier
    is available, with semantic taking precedence when both are."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "baseline_blob_id": self.baseline_blob_id,
            "candidate_blob_id": self.candidate_blob_id,
            "mime": self.mime,
            "cheap_tier": self.cheap_tier,
            "semantic_tier": self.semantic_tier,
            "severity": self.severity,
        }


@dataclass
class MultimodalDiff:
    """The full cross-modal diff axis output."""

    deltas: list[BlobDelta] = field(default_factory=list)
    baseline_blob_count: int = 0
    candidate_blob_count: int = 0
    worst_severity: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "deltas": [d.to_dict() for d in self.deltas],
            "baseline_blob_count": self.baseline_blob_count,
            "candidate_blob_count": self.candidate_blob_count,
            "worst_severity": self.worst_severity,
        }


# Default thresholds. Per RAGAS / TruLens / DeepEval / LangSmith /
# Langfuse conventions and the imagehash author's recommendation.
DEFAULT_DHASH_THRESHOLD_NEAR_DUP = 10
"""Hamming distance ≤ this on a 64-bit dHash = near-duplicate (no severity).
≥16 = visibly different (moderate). In between = minor."""

DEFAULT_COSINE_THRESHOLD_SAME = 0.85
"""Cosine >= this = "same content." 0.75 to 0.85 = "same subject,"
reported as minor. <0.75 = different, reported as severe."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine. Returns 0.0 on length mismatch or zero norm —
    callers expecting "same" should treat 0.0 as "no signal," not
    "definitely different.\" """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _extract_blob_refs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return blob_ref payloads in file order."""
    return [rec.get("payload") or {} for rec in records if rec.get("kind") == "blob_ref"]


def multimodal_diff(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    dhash_threshold_near_dup: int = DEFAULT_DHASH_THRESHOLD_NEAR_DUP,
    cosine_threshold_same: float = DEFAULT_COSINE_THRESHOLD_SAME,
) -> MultimodalDiff:
    """Compute per-blob deltas across two traces' ``blob_ref`` records.

    Pairs blobs by position. Unmatched blobs (one side has more) get
    one-sided deltas with severity ``severe`` (the candidate either
    introduced an unrelated blob or lost a recorded one).

    Severity decision tree per pair:

    1. If both sides have an ``embedding`` of the same model dim:
       cosine ≥ ``cosine_threshold_same`` → none, ≥0.75 → minor,
       ≥0.5 → moderate, else severe.
    2. Else if both sides have a matching-algo phash: distance ≤
       ``dhash_threshold_near_dup`` → none, ≤16 → minor, else
       moderate. Lexical-tier-only never goes to severe (we don't
       have enough signal to claim it).
    3. Else: blob_id-equal → none; not equal → minor (the most
       conservative thing to say without semantic signal).
    """
    base_blobs = _extract_blob_refs(baseline)
    cand_blobs = _extract_blob_refs(candidate)

    out = MultimodalDiff(
        baseline_blob_count=len(base_blobs),
        candidate_blob_count=len(cand_blobs),
    )
    n = max(len(base_blobs), len(cand_blobs))
    severity_rank = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    worst = 0
    for i in range(n):
        b = base_blobs[i] if i < len(base_blobs) else None
        c = cand_blobs[i] if i < len(cand_blobs) else None
        delta = _compare_blob_pair(
            i,
            b,
            c,
            dhash_threshold_near_dup=dhash_threshold_near_dup,
            cosine_threshold_same=cosine_threshold_same,
        )
        out.deltas.append(delta)
        worst = max(worst, severity_rank.get(delta.severity, 0))
    out.worst_severity = next((k for k, v in severity_rank.items() if v == worst), "none")
    return out


def _compare_blob_pair(
    pair_index: int,
    base: dict[str, Any] | None,
    cand: dict[str, Any] | None,
    *,
    dhash_threshold_near_dup: int,
    cosine_threshold_same: float,
) -> BlobDelta:
    if base is None and cand is None:
        return BlobDelta(
            pair_index=pair_index, baseline_blob_id=None, candidate_blob_id=None, mime=None
        )
    if base is None:
        # Candidate-only blob. Severe — the candidate emitted something
        # the baseline didn't.
        return BlobDelta(
            pair_index=pair_index,
            baseline_blob_id=None,
            candidate_blob_id=str(cand.get("blob_id") or "") if cand else None,
            mime=str((cand or {}).get("mime") or ""),
            severity="severe",
        )
    if cand is None:
        return BlobDelta(
            pair_index=pair_index,
            baseline_blob_id=str(base.get("blob_id") or ""),
            candidate_blob_id=None,
            mime=str(base.get("mime") or ""),
            severity="severe",
        )

    base_id = str(base.get("blob_id") or "")
    cand_id = str(cand.get("blob_id") or "")
    mime = str(cand.get("mime") or base.get("mime") or "")

    # Identical blobs short-circuit — content-addressing means same id
    # = same bytes.
    if base_id == cand_id and base_id:
        return BlobDelta(
            pair_index=pair_index,
            baseline_blob_id=base_id,
            candidate_blob_id=cand_id,
            mime=mime,
            severity="none",
        )

    # Semantic tier: both sides have an embedding of the same model
    # dim. Cosine drives severity.
    base_emb = base.get("embedding")
    cand_emb = cand.get("embedding")
    semantic: dict[str, Any] | None = None
    semantic_severity: str | None = None
    if (
        isinstance(base_emb, dict)
        and isinstance(cand_emb, dict)
        and base_emb.get("model") == cand_emb.get("model")
        and isinstance(base_emb.get("vec"), list)
        and isinstance(cand_emb.get("vec"), list)
    ):
        cos = cosine_similarity(base_emb["vec"], cand_emb["vec"])
        semantic = {
            "model": str(base_emb.get("model") or ""),
            "cosine": cos,
            "threshold_same": cosine_threshold_same,
        }
        if cos >= cosine_threshold_same:
            semantic_severity = "none"
        elif cos >= 0.75:
            semantic_severity = "minor"
        elif cos >= 0.5:
            semantic_severity = "moderate"
        else:
            semantic_severity = "severe"

    # Cheap tier: dHash Hamming distance.
    base_phash = base.get("phash") if isinstance(base.get("phash"), dict) else None
    cand_phash = cand.get("phash") if isinstance(cand.get("phash"), dict) else None
    cheap: dict[str, Any] | None = None
    cheap_severity: str | None = None
    if base_phash and cand_phash:
        dist = phash_distance(base_phash, cand_phash)
        if dist is not None:
            cheap = {
                "algo": base_phash.get("algo"),
                "distance": dist,
                "threshold_near_dup": dhash_threshold_near_dup,
            }
            if dist <= dhash_threshold_near_dup:
                cheap_severity = "none"
            elif dist <= 16:
                cheap_severity = "minor"
            else:
                cheap_severity = "moderate"

    # Semantic wins when present; cheap tier is the fallback. If
    # neither is available, lexical fallback: different blob ids
    # without semantic signal = minor (conservative).
    severity = semantic_severity or cheap_severity or "minor"

    return BlobDelta(
        pair_index=pair_index,
        baseline_blob_id=base_id,
        candidate_blob_id=cand_id,
        mime=mime,
        cheap_tier=cheap,
        semantic_tier=semantic,
        severity=severity,
    )


# ---- terminal renderer ----------------------------------------------


def render_terminal(diff: MultimodalDiff) -> str:
    """Compact terminal rendering of a multimodal diff."""
    if not diff.deltas:
        return "multimodal: no blob_ref records in either trace."
    lines = []
    lines.append(
        f"multimodal: {diff.baseline_blob_count} baseline / "
        f"{diff.candidate_blob_count} candidate blob(s); worst="
        f"{diff.worst_severity}"
    )
    for d in diff.deltas:
        if d.severity == "none":
            continue
        bid = d.baseline_blob_id[:19] if d.baseline_blob_id else "<missing>"
        cid = d.candidate_blob_id[:19] if d.candidate_blob_id else "<missing>"
        line = f"  pair {d.pair_index} [{d.severity}] {bid} → {cid}"
        if d.semantic_tier:
            line += f"  cosine={d.semantic_tier['cosine']:.3f}"
        elif d.cheap_tier:
            line += f"  dHash dist={d.cheap_tier['distance']}/64"
        lines.append(line)
    return "\n".join(lines)


def render_markdown(diff: MultimodalDiff) -> str:
    """Markdown for PR comments."""
    if not diff.deltas:
        return "_No blob_ref records in either trace._"
    rows = ["| pair | severity | baseline | candidate | comparison |", "|---|---|---|---|---|"]
    for d in diff.deltas:
        bid = (d.baseline_blob_id or "—")[:19] + ("…" if d.baseline_blob_id else "")
        cid = (d.candidate_blob_id or "—")[:19] + ("…" if d.candidate_blob_id else "")
        cmp_text = "—"
        if d.semantic_tier:
            cmp_text = (
                f"cosine `{d.semantic_tier['cosine']:.3f}` "
                f"(model: `{d.semantic_tier.get('model', '?')}`)"
            )
        elif d.cheap_tier:
            cmp_text = (
                f"dHash distance `{d.cheap_tier['distance']}/64` "
                f"(near-dup ≤ `{d.cheap_tier['threshold_near_dup']}`)"
            )
        sev_label = {"none": "✓", "minor": "🟡", "moderate": "🟠", "severe": "🔴"}.get(
            d.severity, d.severity
        )
        rows.append(
            f"| {d.pair_index} | {sev_label} {d.severity} | `{bid}` | `{cid}` | {cmp_text} |"
        )
    rows.append(
        f"\n_baseline: {diff.baseline_blob_count} blob(s) · "
        f"candidate: {diff.candidate_blob_count} · "
        f"worst severity: **{diff.worst_severity}**_"
    )
    return "\n".join(rows)


__all__ = [
    "DEFAULT_COSINE_THRESHOLD_SAME",
    "DEFAULT_DHASH_THRESHOLD_NEAR_DUP",
    "BlobDelta",
    "MultimodalDiff",
    "cosine_similarity",
    "multimodal_diff",
    "render_markdown",
    "render_terminal",
]
