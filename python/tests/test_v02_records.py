"""Tests for v0.2 record kinds: ``chunk``, ``harness_event``, ``blob_ref``."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from shadow import _core
from shadow.errors import ShadowConfigError
from shadow.sdk.session import Session
from shadow.v02_records import (
    HARNESS_CATEGORIES,
    BlobStore,
    compute_phash_dhash64,
    harness_event_diff,
    phash_distance,
    record_blob_ref,
    record_chunk,
    record_harness_event,
    replay_chunk_timing,
    replay_chunks_async,
)

# ---- harness_event ----------------------------------------------------


def test_record_harness_event_appends_record_with_canonical_payload(tmp_path: Path) -> None:
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        rid = record_harness_event(
            s,
            category="retry",
            name="retry.attempted",
            severity="warning",
            attributes={"attempt_index": 2, "error_class": "RateLimitError"},
        )
    assert rid.startswith("sha256:")
    records = _core.parse_agentlog(out.read_bytes())
    he = next(r for r in records if r["kind"] == "harness_event")
    assert he["payload"]["category"] == "retry"
    assert he["payload"]["name"] == "retry.attempted"
    assert he["payload"]["severity"] == "warning"
    assert he["payload"]["attributes"]["attempt_index"] == 2


def test_record_harness_event_rejects_unknown_category(tmp_path: Path) -> None:
    with (
        Session(output_path=tmp_path / "t.agentlog", auto_instrument=False) as s,
        pytest.raises(ShadowConfigError, match="unknown harness_event category"),
    ):
        record_harness_event(
            s,
            category="garbage",  # type: ignore[arg-type]
            name="x",
            severity="info",
        )


def test_record_harness_event_rejects_unknown_severity(tmp_path: Path) -> None:
    with (
        Session(output_path=tmp_path / "t.agentlog", auto_instrument=False) as s,
        pytest.raises(ShadowConfigError, match="unknown harness_event severity"),
    ):
        record_harness_event(
            s,
            category="retry",
            name="retry.attempted",
            severity="catastrophic",  # type: ignore[arg-type]
        )


def test_harness_categories_cover_otel_alignment() -> None:
    # Sanity: the closed taxonomy matches what we documented in SPEC §4.9.
    expected = {
        "retry",
        "rate_limit",
        "model_switch",
        "context_trim",
        "cache",
        "guardrail",
        "budget",
        "stream_interrupt",
        "tool_lifecycle",
    }
    assert expected == HARNESS_CATEGORIES


# ---- chunk ------------------------------------------------------------


def test_record_chunk_carries_absolute_time_unix_nano(tmp_path: Path) -> None:
    out = tmp_path / "stream.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        before = time.time_ns()
        record_chunk(s, chunk_index=0, delta={"type": "text_delta", "text": "Hello "})
        record_chunk(s, chunk_index=1, delta={"type": "text_delta", "text": "world"})
        record_chunk(s, chunk_index=2, delta={"type": "text_delta", "text": "!"}, is_final=True)
        after = time.time_ns()
    records = _core.parse_agentlog(out.read_bytes())
    chunks = [r for r in records if r["kind"] == "chunk"]
    assert len(chunks) == 3
    for c in chunks:
        assert before <= c["payload"]["time_unix_nano"] <= after
    assert chunks[2]["payload"]["is_final"] is True


def test_replay_chunk_timing_returns_sleep_intervals() -> None:
    chunks = [
        {"payload": {"time_unix_nano": 1_000_000_000, "chunk_index": 0}},
        {"payload": {"time_unix_nano": 1_500_000_000, "chunk_index": 1}},
        {"payload": {"time_unix_nano": 1_750_000_000, "chunk_index": 2}},
    ]
    waits = replay_chunk_timing(chunks)
    assert waits == [0.0, 0.5, 0.25]


def test_replay_chunks_async_preserves_relative_timing() -> None:
    """Replay must preserve inter-chunk gaps, NOT cumulative — that's
    the deadline-loop guarantee. We use 5x speed so the test runs fast."""
    chunks = [
        {"payload": {"time_unix_nano": 0, "chunk_index": 0, "delta": {"text": "A"}}},
        {"payload": {"time_unix_nano": 200_000_000, "chunk_index": 1, "delta": {"text": "B"}}},
        {"payload": {"time_unix_nano": 400_000_000, "chunk_index": 2, "delta": {"text": "C"}}},
    ]
    seen: list[tuple[int, float]] = []
    start = time.monotonic()

    def collector(payload: dict) -> None:
        seen.append((payload["chunk_index"], time.monotonic() - start))

    asyncio.run(replay_chunks_async(chunks, collector, speed=5.0))
    assert [s[0] for s in seen] == [0, 1, 2]
    # 200ms gap at 5x = 40ms. Allow generous tolerance for CI jitter.
    assert seen[1][1] >= 0.030
    assert seen[2][1] >= 0.060
    # Total real time should be roughly 80ms — under 1 second easily.
    assert seen[2][1] < 1.0


def test_replay_chunks_async_handles_non_monotonic_timestamps() -> None:
    """If timestamps go backwards (clock skew, mis-recorded), the
    deadline loop must NOT sleep forever — it floors gap at 0."""
    chunks = [
        {"payload": {"time_unix_nano": 1_000_000_000, "chunk_index": 0, "delta": {"text": "A"}}},
        {
            "payload": {
                "time_unix_nano": 500_000_000,  # backwards!
                "chunk_index": 1,
                "delta": {"text": "B"},
            }
        },
    ]
    seen: list[int] = []
    asyncio.run(replay_chunks_async(chunks, lambda p: seen.append(p["chunk_index"]), speed=10.0))
    assert seen == [0, 1]


def test_replay_chunks_empty_input_is_noop() -> None:
    asyncio.run(replay_chunks_async([], lambda _p: None))


# ---- blob_ref + BlobStore --------------------------------------------


def test_blob_store_put_get_roundtrip(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path / "blobs")
    blob_id = store.put(b"hello world")
    assert blob_id.startswith("sha256:")
    assert store.get(blob_id) == b"hello world"
    assert store.has(blob_id)


def test_blob_store_dedupes_identical_content(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path / "blobs")
    a = store.put(b"same")
    b = store.put(b"same")
    assert a == b


def test_blob_store_uri_is_agentlog_blob_scheme(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path / "blobs")
    blob_id = store.put(b"x")
    uri = store.uri_for(blob_id)
    assert uri.startswith("agentlog-blob://default/")
    assert uri.endswith(blob_id[len("sha256:") :])


def test_record_blob_ref_writes_record_and_blob(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path / "blobs")
    out = tmp_path / "t.agentlog"
    with Session(output_path=out, auto_instrument=False) as s:
        rid, bid = record_blob_ref(
            s, blob=b"binary-data", mime="application/octet-stream", store=store
        )
    records = _core.parse_agentlog(out.read_bytes())
    br = next(r for r in records if r["kind"] == "blob_ref")
    assert br["payload"]["mime"] == "application/octet-stream"
    assert br["payload"]["size_bytes"] == len(b"binary-data")
    assert br["payload"]["blob_id"] == bid
    assert store.get(bid) == b"binary-data"


def test_record_blob_ref_skips_phash_for_non_image(tmp_path: Path) -> None:
    """Audio / arbitrary binary doesn't get a dHash."""
    store = BlobStore(root=tmp_path / "blobs")
    with Session(output_path=tmp_path / "t.agentlog", auto_instrument=False) as s:
        record_blob_ref(s, blob=b"\x00\x01\x02", mime="audio/wav", store=store)
    records = _core.parse_agentlog((tmp_path / "t.agentlog").read_bytes())
    br = next(r for r in records if r["kind"] == "blob_ref")
    assert "phash" not in br["payload"]


def test_record_blob_ref_compute_phash_false_skips_phash(tmp_path: Path) -> None:
    store = BlobStore(root=tmp_path / "blobs")
    with Session(output_path=tmp_path / "t.agentlog", auto_instrument=False) as s:
        record_blob_ref(
            s,
            blob=b"\x89PNG\r\n\x1a\n",  # PNG header (not a real image)
            mime="image/png",
            store=store,
            compute_phash=False,
        )
    records = _core.parse_agentlog((tmp_path / "t.agentlog").read_bytes())
    br = next(r for r in records if r["kind"] == "blob_ref")
    assert "phash" not in br["payload"]


def test_compute_phash_returns_none_when_imagehash_missing_or_corrupt() -> None:
    """If the optional dep is missing OR the bytes aren't a valid
    image, we get None — never an exception."""
    out = compute_phash_dhash64(b"not an image at all")
    # Either None (lib missing) or None (lib present but bytes broken) — both fine.
    assert out is None or out["algo"] == "dhash64"


def test_phash_distance_zero_for_identical_hashes() -> None:
    a = {"algo": "dhash64", "hex": "abcdef0123456789"}
    b = {"algo": "dhash64", "hex": "abcdef0123456789"}
    assert phash_distance(a, b) == 0


def test_phash_distance_counts_hamming_bits() -> None:
    a = {"algo": "dhash64", "hex": "0000000000000000"}
    b = {"algo": "dhash64", "hex": "0000000000000007"}  # last 3 bits set
    assert phash_distance(a, b) == 3


def test_phash_distance_returns_none_on_algo_mismatch() -> None:
    a = {"algo": "dhash64", "hex": "abc"}
    b = {"algo": "phash64", "hex": "abc"}
    assert phash_distance(a, b) is None


# ---- harness_event_diff ----------------------------------------------


def _wrap(records: list[dict]) -> list[dict]:
    """Add minimal metadata so records pass v0.1 invariants."""
    return [
        {
            "version": "0.1",
            "kind": "metadata",
            "id": "sha256:m",
            "ts": "t",
            "parent": None,
            "payload": {},
        },
        *records,
    ]


def _he(category: str, name: str, *, severity: str = "info") -> dict:
    return {
        "version": "0.1",
        "kind": "harness_event",
        "id": f"sha256:he-{category}-{name}",
        "ts": "t",
        "parent": "sha256:m",
        "payload": {"category": category, "name": name, "severity": severity, "attributes": {}},
    }


def test_harness_event_diff_count_delta() -> None:
    baseline = _wrap([_he("retry", "retry.attempted")])
    candidate = _wrap(
        [
            _he("retry", "retry.attempted"),
            _he("retry", "retry.attempted"),
            _he("retry", "retry.attempted"),
        ]
    )
    deltas = harness_event_diff(baseline, candidate)
    assert len(deltas) == 1
    assert deltas[0].count_delta == 2
    assert deltas[0].baseline_count == 1
    assert deltas[0].candidate_count == 3


def test_harness_event_diff_picks_up_new_categories() -> None:
    baseline = _wrap([])
    candidate = _wrap(
        [
            _he("retry", "retry.attempted", severity="warning"),
            _he("guardrail", "guardrail.blocked", severity="error"),
        ]
    )
    deltas = harness_event_diff(baseline, candidate)
    cats = {(d.category, d.name) for d in deltas}
    assert cats == {("retry", "retry.attempted"), ("guardrail", "guardrail.blocked")}


def test_harness_event_diff_sorted_by_absolute_count_delta() -> None:
    baseline = _wrap([_he("retry", "retry.attempted")])
    candidate = _wrap(
        [_he("retry", "retry.attempted")] * 5
        + [_he("cache", "cache.hit")]
        + [_he("model_switch", "model_switch.fallback")] * 2
    )
    deltas = harness_event_diff(baseline, candidate)
    assert [abs(d.count_delta) for d in deltas] == sorted(
        [abs(d.count_delta) for d in deltas], reverse=True
    )
    # The retry delta (4) is the biggest.
    assert deltas[0].name == "retry.attempted"
    assert deltas[0].count_delta == 4
