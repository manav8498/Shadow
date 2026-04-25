"""Tests for v2.4 renderers: cross-modal diff axis + harness-event renderer."""

from __future__ import annotations

from typing import Any

from shadow.harness_diff_render import (
    render_markdown as render_harness_markdown,
)
from shadow.harness_diff_render import (
    render_terminal as render_harness_terminal,
)
from shadow.multimodal_diff import (
    DEFAULT_COSINE_THRESHOLD_SAME,
    cosine_similarity,
    multimodal_diff,
)
from shadow.multimodal_diff import (
    render_markdown as render_mm_markdown,
)
from shadow.multimodal_diff import (
    render_terminal as render_mm_terminal,
)
from shadow.v02_records import HarnessEventDelta

# ---- helpers ----------------------------------------------------------


def _meta() -> dict[str, Any]:
    return {
        "version": "0.1",
        "kind": "metadata",
        "id": "sha256:m",
        "ts": "t",
        "parent": None,
        "payload": {},
    }


def _blob_ref(
    *,
    blob_id: str,
    mime: str = "image/png",
    phash_hex: str | None = None,
    embedding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mime": mime,
        "size_bytes": 100,
        "blob_id": f"sha256:{blob_id}",
        "uri": f"agentlog-blob://default/{blob_id}",
    }
    if phash_hex is not None:
        payload["phash"] = {"algo": "dhash64", "hex": phash_hex}
    if embedding is not None:
        payload["embedding"] = embedding
    return {
        "version": "0.1",
        "kind": "blob_ref",
        "id": f"sha256:br-{blob_id}",
        "ts": "t",
        "parent": "sha256:m",
        "payload": payload,
    }


# ====================================================================
# multimodal_diff
# ====================================================================


def test_cosine_similarity_identity_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_opposite_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_zero_norm_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_length_mismatch_returns_zero() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_multimodal_diff_no_blobs_returns_empty() -> None:
    out = multimodal_diff([_meta()], [_meta()])
    assert out.deltas == []
    assert out.worst_severity == "none"


def test_multimodal_diff_identical_blobs_severity_none() -> None:
    base = [_meta(), _blob_ref(blob_id="abc")]
    cand = [_meta(), _blob_ref(blob_id="abc")]
    out = multimodal_diff(base, cand)
    assert len(out.deltas) == 1
    assert out.deltas[0].severity == "none"
    assert out.worst_severity == "none"


def test_multimodal_diff_phash_near_dup_severity_none() -> None:
    """dHash distance ≤ 10 = near-duplicate, severity none."""
    # 0x0...01 XOR 0x0...07 = 0b110 = 2 bits different
    base = [_meta(), _blob_ref(blob_id="b1", phash_hex="0000000000000001")]
    cand = [_meta(), _blob_ref(blob_id="c1", phash_hex="0000000000000007")]
    out = multimodal_diff(base, cand)
    assert out.deltas[0].severity == "none"
    assert out.deltas[0].cheap_tier["distance"] == 2


def test_multimodal_diff_phash_far_severity_moderate() -> None:
    """dHash distance > 16 = moderate (no semantic signal to claim severe)."""
    base = [_meta(), _blob_ref(blob_id="b1", phash_hex="0000000000000000")]
    cand = [_meta(), _blob_ref(blob_id="c1", phash_hex="ffffffffffffffff")]
    out = multimodal_diff(base, cand)
    assert out.deltas[0].cheap_tier["distance"] == 64
    assert out.deltas[0].severity == "moderate"


def test_multimodal_diff_semantic_high_cosine_severity_none() -> None:
    """Cosine ≥ 0.85 = same content, severity none even when blob_ids differ."""
    base = [
        _meta(),
        _blob_ref(
            blob_id="b1",
            embedding={"model": "clip-vit-b32", "dim": 3, "vec": [0.9, 0.1, 0.0]},
        ),
    ]
    cand = [
        _meta(),
        _blob_ref(
            blob_id="c1",
            embedding={"model": "clip-vit-b32", "dim": 3, "vec": [0.91, 0.11, 0.01]},
        ),
    ]
    out = multimodal_diff(base, cand)
    assert out.deltas[0].severity == "none"
    assert out.deltas[0].semantic_tier["cosine"] >= DEFAULT_COSINE_THRESHOLD_SAME


def test_multimodal_diff_semantic_low_cosine_severity_severe() -> None:
    """Cosine < 0.5 = severe."""
    base = [
        _meta(),
        _blob_ref(
            blob_id="b1",
            embedding={"model": "clip-vit-b32", "dim": 2, "vec": [1.0, 0.0]},
        ),
    ]
    cand = [
        _meta(),
        _blob_ref(
            blob_id="c1",
            embedding={"model": "clip-vit-b32", "dim": 2, "vec": [0.0, 1.0]},
        ),
    ]
    out = multimodal_diff(base, cand)
    assert out.deltas[0].severity == "severe"


def test_multimodal_diff_semantic_takes_precedence_over_phash() -> None:
    """When both tiers are present, semantic wins. A high cosine
    overrides a moderate phash distance — embeddings are higher
    signal."""
    base = [
        _meta(),
        _blob_ref(
            blob_id="b1",
            phash_hex="ffffffffffffffff",  # max distance from candidate
            embedding={"model": "clip", "dim": 2, "vec": [1.0, 0.0]},
        ),
    ]
    cand = [
        _meta(),
        _blob_ref(
            blob_id="c1",
            phash_hex="0000000000000000",
            embedding={"model": "clip", "dim": 2, "vec": [0.99, 0.01]},
        ),
    ]
    out = multimodal_diff(base, cand)
    # Semantic tier reports near-perfect; cheap tier reports max distance.
    # Final severity is none (semantic wins).
    assert out.deltas[0].severity == "none"


def test_multimodal_diff_unmatched_blob_severe() -> None:
    """Candidate has 1 blob, baseline has 0 → severe (unmatched)."""
    base = [_meta()]
    cand = [_meta(), _blob_ref(blob_id="extra")]
    out = multimodal_diff(base, cand)
    assert len(out.deltas) == 1
    assert out.deltas[0].severity == "severe"
    assert out.deltas[0].baseline_blob_id is None


def test_multimodal_diff_no_phash_no_embedding_moderate() -> None:
    """Different blob_ids without any signal = moderate. A swapped
    blob with no similarity signal is more likely a real regression
    than a near-duplicate; calling it minor under-flagged the case."""
    base = [_meta(), _blob_ref(blob_id="b1")]
    cand = [_meta(), _blob_ref(blob_id="c1")]
    out = multimodal_diff(base, cand)
    assert out.deltas[0].severity == "moderate"


def test_multimodal_diff_worst_severity_aggregation() -> None:
    """worst_severity is the max across all deltas."""
    base = [_meta(), _blob_ref(blob_id="a"), _blob_ref(blob_id="b")]
    cand = [_meta(), _blob_ref(blob_id="a"), _blob_ref(blob_id="c")]
    # First pair: identical → none. Second pair: blob_id differs, no signal → moderate.
    out = multimodal_diff(base, cand)
    assert out.worst_severity == "moderate"


def test_render_terminal_handles_empty() -> None:
    out = multimodal_diff([_meta()], [_meta()])
    text = render_mm_terminal(out)
    assert "no blob_ref" in text


def test_render_terminal_includes_severity_marker() -> None:
    base = [_meta(), _blob_ref(blob_id="b1", phash_hex="0000000000000000")]
    cand = [_meta(), _blob_ref(blob_id="c1", phash_hex="ffffffffffffffff")]
    text = render_mm_terminal(multimodal_diff(base, cand))
    assert "moderate" in text
    assert "dHash dist=64" in text


def test_render_markdown_table_shape() -> None:
    base = [_meta(), _blob_ref(blob_id="b1", phash_hex="0000000000000000")]
    cand = [_meta(), _blob_ref(blob_id="c1", phash_hex="ffffffffffffffff")]
    md = render_mm_markdown(multimodal_diff(base, cand))
    assert "| pair | severity | baseline | candidate | comparison |" in md
    assert "moderate" in md
    assert "worst severity" in md


# ====================================================================
# harness_diff_render
# ====================================================================


def test_harness_terminal_empty_returns_notice() -> None:
    out = render_harness_terminal([])
    assert "no events" in out


def test_harness_terminal_separates_regressions_from_fixes() -> None:
    deltas = [
        HarnessEventDelta(
            category="retry",
            name="retry.attempted",
            severity="warning",
            baseline_count=2,
            candidate_count=8,
            count_delta=6,
            first_occurrence_baseline=1,
            first_occurrence_candidate=0,
        ),
        HarnessEventDelta(
            category="cache",
            name="cache.hit",
            severity="info",
            baseline_count=10,
            candidate_count=2,
            count_delta=-8,
            first_occurrence_baseline=0,
            first_occurrence_candidate=0,
        ),
    ]
    text = render_harness_terminal(deltas)
    # Regressions section appears before fixes.
    re_pos = text.find("regressions")
    fx_pos = text.find("fixes")
    assert re_pos != -1 and fx_pos != -1
    assert re_pos < fx_pos
    assert "+6" in text  # signed delta
    assert "-8" in text


def test_harness_terminal_severity_ordering() -> None:
    """Within regressions, errors come before warnings come before info."""
    deltas = [
        HarnessEventDelta(
            category="cache",
            name="cache.miss",
            severity="info",
            baseline_count=0,
            candidate_count=3,
            count_delta=3,
            first_occurrence_baseline=None,
            first_occurrence_candidate=0,
        ),
        HarnessEventDelta(
            category="guardrail",
            name="guardrail.blocked",
            severity="error",
            baseline_count=0,
            candidate_count=1,
            count_delta=1,
            first_occurrence_baseline=None,
            first_occurrence_candidate=2,
        ),
    ]
    text = render_harness_terminal(deltas)
    err_idx = text.find("guardrail.blocked")
    info_idx = text.find("cache.miss")
    assert err_idx < info_idx, "errors must render before info"


def test_harness_markdown_renders_two_tables_when_both_present() -> None:
    deltas = [
        HarnessEventDelta(
            category="retry",
            name="retry.attempted",
            severity="warning",
            baseline_count=1,
            candidate_count=5,
            count_delta=4,
            first_occurrence_baseline=0,
            first_occurrence_candidate=1,
        ),
        HarnessEventDelta(
            category="cache",
            name="cache.hit",
            severity="info",
            baseline_count=10,
            candidate_count=3,
            count_delta=-7,
            first_occurrence_baseline=0,
            first_occurrence_candidate=0,
        ),
    ]
    md = render_harness_markdown(deltas)
    assert "regressions" in md.lower()
    assert "fixes" in md.lower()
    assert "+4" in md
    assert "-7" in md
    assert "first at" in md.lower()


def test_harness_markdown_empty_returns_notice() -> None:
    md = render_harness_markdown([])
    assert "_No" in md


# ====================================================================
# CLI integration: --harness-diff and --multimodal-diff flags
# ====================================================================


def test_cli_harness_diff_flag_surfaces_count_deltas(tmp_path: object) -> None:
    """`shadow diff --harness-diff` emits the harness-event renderer
    output. Tested via the typer test runner against two synthetic
    traces with known harness_event deltas."""
    import json
    from pathlib import Path

    from typer.testing import CliRunner

    from shadow.cli.app import app

    runner = CliRunner()
    base_dir: Path = tmp_path  # type: ignore[assignment]

    def _write(p: Path, records: list[dict[str, Any]]) -> None:
        p.write_bytes(b"".join(json.dumps(r).encode("utf-8") + b"\n" for r in records))

    base_records = [
        {
            "version": "0.1",
            "kind": "metadata",
            "id": "sha256:m",
            "ts": "2026-04-25T00:00:00.000Z",
            "parent": None,
            "payload": {"sdk": {"name": "test"}},
        },
        {
            "version": "0.1",
            "kind": "harness_event",
            "id": "sha256:e1",
            "ts": "2026-04-25T00:00:00.001Z",
            "parent": "sha256:m",
            "payload": {
                "category": "retry",
                "name": "retry.attempted",
                "severity": "warning",
                "attributes": {},
            },
        },
    ]
    cand_records = [
        base_records[0],
        # 4 retries vs 1 in baseline
        *[
            {
                "version": "0.1",
                "kind": "harness_event",
                "id": f"sha256:e{i}",
                "ts": "2026-04-25T00:00:00.001Z",
                "parent": "sha256:m",
                "payload": {
                    "category": "retry",
                    "name": "retry.attempted",
                    "severity": "warning",
                    "attributes": {"i": i},
                },
            }
            for i in range(4)
        ],
    ]
    base_path = base_dir / "b.agentlog"
    cand_path = base_dir / "c.agentlog"
    _write(base_path, base_records)
    _write(cand_path, cand_records)

    result = runner.invoke(app, ["diff", str(base_path), str(cand_path), "--harness-diff"])
    assert result.exit_code == 0, result.output
    assert "harness events" in result.output
    assert "retry.attempted" in result.output
    assert "+3" in result.output
