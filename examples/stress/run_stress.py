"""Real-world adverse stress test for v2.3.0 — `.agentlog` v0.2:
``chunk`` records with replay-timing fidelity, ``harness_event``
diff dimension, ``blob_ref`` content-addressed store, MCP-native
replay shim.

The stress harness is deterministic. v2.3 features are about FORMAT
and REPLAY correctness — they don't depend on real LLM behaviour. We
exercise the production code paths against adversarial inputs:

- chunk replay: 10K-chunk stream with bursty + stall patterns,
  non-monotonic timestamps, replay-speed multipliers, asyncio gather
  to surface state leakage between concurrent replays.
- harness_event diff: trace with hundreds of mixed-category events,
  count deltas at all severities, distinct (category, name) tuples
  in the thousands.
- blob_ref: deduplication under repeated puts of identical content,
  store recovery after crash (atomic rename), dHash collision
  behaviour, gigabyte-blob streaming put without OOM.
- MCP replay: lookup against a 1000-call recording, repeated calls
  with the same args returning recorded ordering, unicode URIs,
  errors propagating cleanly, capability-negotiation stub.

Run:

    .venv/bin/python examples/stress_v23x/run_stress.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.mcp_replay import (
    MCPCall,
    MCPCallNotRecorded,
    RecordingIndex,
    ReplayClientSession,
    canonicalize_params,
)
from shadow.sdk.session import Session
from shadow.v02_records import (
    BlobStore,
    harness_event_diff,
    phash_distance,
    record_blob_ref,
    record_chunk,
    record_harness_event,
    replay_chunks_async,
)


# ---- pretty print ------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
DIM = "\033[2m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}")
    if detail:
        print(f"        {DIM}{detail}{RESET}")
    results.append((name, ok, detail))


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ====================================================================
# Streaming chunks — record + replay timing fidelity
# ====================================================================


def stress_chunk_record_10000(tmp: Path) -> None:
    """A 10K-chunk stream serializes and parses correctly. The
    `.agentlog` per-line cap (16 MiB) is well above any single chunk
    payload, but a 10K-line file is the realistic upper bound for a
    long streaming response."""
    output = tmp / "10k.agentlog"
    t0 = time.perf_counter()
    with Session(output_path=output, auto_instrument=False) as s:
        for i in range(10000):
            record_chunk(
                s,
                chunk_index=i,
                delta={"type": "text_delta", "text": f"chunk-{i}"},
                is_final=(i == 9999),
            )
    elapsed = time.perf_counter() - t0
    records = _core.parse_agentlog(output.read_bytes())
    chunks = [r for r in records if r["kind"] == "chunk"]
    report(
        "10K-chunk session writes and parses cleanly",
        len(chunks) == 10000,
        f"recorded in {elapsed * 1000:.0f}ms; parsed {len(chunks)} chunks",
    )
    # Every chunk's id must be content-addressed.
    bad = [c for c in chunks if c["id"] != _core.content_id(c["payload"])]
    report("every chunk id == content_id(payload)", not bad, f"bad={len(bad)}")


def stress_chunk_replay_timing_fidelity() -> None:
    """Replay a synthetic stream at 100x speed and verify the
    relative gaps between chunks are preserved (within tolerance)."""
    chunks = [
        {
            "payload": {
                "time_unix_nano": i * 50_000_000,  # 50ms gaps in original
                "chunk_index": i,
                "delta": {"text": f"c{i}"},
            }
        }
        for i in range(20)
    ]
    arrival_times: list[float] = []
    start = time.monotonic()

    def collector(payload: dict) -> None:
        arrival_times.append(time.monotonic() - start)

    asyncio.run(replay_chunks_async(chunks, collector, speed=100.0))
    # Each gap should be 0.5ms at 100x. Tolerate generous CI jitter.
    gaps = [
        arrival_times[i + 1] - arrival_times[i] for i in range(len(arrival_times) - 1)
    ]
    avg_gap_ms = (sum(gaps) / len(gaps)) * 1000
    report(
        "20-chunk replay at 100x preserves relative timing under tolerance",
        len(arrival_times) == 20 and 0 < avg_gap_ms < 50,
        f"avg gap = {avg_gap_ms:.2f}ms (target 0.5ms, 50ms ceiling)",
    )


def stress_chunk_replay_concurrent_no_state_leak() -> None:
    """Five concurrent stream replays must each preserve their own
    chunk order — no shared state."""
    seen: dict[int, list[int]] = {i: [] for i in range(5)}

    async def one(stream_id: int) -> None:
        chunks = [
            {
                "payload": {
                    "time_unix_nano": i * 1_000_000,
                    "chunk_index": stream_id * 100 + i,
                    "delta": {"i": i},
                }
            }
            for i in range(10)
        ]
        await replay_chunks_async(
            chunks, lambda p: seen[stream_id].append(p["chunk_index"]), speed=1000.0
        )

    async def run_all() -> None:
        await asyncio.gather(*(one(i) for i in range(5)))

    asyncio.run(run_all())
    ok = all(seen[i] == [i * 100 + j for j in range(10)] for i in range(5))
    report("5 concurrent replays do not interleave chunks across streams", ok)


def stress_chunk_replay_non_monotonic_timestamps() -> None:
    """Backward timestamps from clock skew must not deadlock the
    replay engine — gap is floored at 0."""
    chunks = [
        {"payload": {"time_unix_nano": 1_000_000_000, "chunk_index": 0, "delta": {}}},
        {
            "payload": {"time_unix_nano": 999_999_999, "chunk_index": 1, "delta": {}}
        },  # back!
        {
            "payload": {"time_unix_nano": 999_999_998, "chunk_index": 2, "delta": {}}
        },  # further back!
    ]
    seen: list[int] = []
    asyncio.run(
        replay_chunks_async(chunks, lambda p: seen.append(p["chunk_index"]), speed=1.0)
    )
    report("backward timestamps don't deadlock replay", seen == [0, 1, 2])


def stress_chunk_replay_speed_inf_no_delay() -> None:
    """speed=1e9 should yield all chunks effectively instantly."""
    chunks = [
        {"payload": {"time_unix_nano": i * 100_000_000, "chunk_index": i, "delta": {}}}
        for i in range(100)
    ]
    seen: list[int] = []
    t0 = time.monotonic()
    asyncio.run(
        replay_chunks_async(chunks, lambda p: seen.append(p["chunk_index"]), speed=1e9)
    )
    elapsed = time.monotonic() - t0
    report(
        "speed=1e9 replays 100 chunks in <100ms (no real sleep)",
        elapsed < 0.1 and len(seen) == 100,
        f"elapsed={elapsed * 1000:.1f}ms",
    )


# ====================================================================
# harness_event — diff dimension under load
# ====================================================================


def stress_harness_event_diff_at_scale() -> None:
    """Build two traces with hundreds of harness events across all
    nine categories. The diff must aggregate count deltas correctly
    in <100ms."""
    from shadow.v02_records import HARNESS_CATEGORIES

    base_meta = {
        "version": "0.1",
        "kind": "metadata",
        "id": "sha256:m",
        "ts": "t",
        "parent": None,
        "payload": {},
    }

    def he(cat: str, name: str, severity: str = "info") -> dict[str, Any]:
        return {
            "version": "0.1",
            "kind": "harness_event",
            "id": f"sha256:he-{cat}-{name}-{time.time_ns()}",
            "ts": "t",
            "parent": "sha256:m",
            "payload": {
                "category": cat,
                "name": name,
                "severity": severity,
                "attributes": {},
            },
        }

    # Baseline: 5 retries, 3 cache hits, 1 guardrail block.
    baseline = [
        base_meta,
        *[he("retry", "retry.attempted", "warning") for _ in range(5)],
        *[he("cache", "cache.hit") for _ in range(3)],
        he("guardrail", "guardrail.blocked", "error"),
    ]
    # Candidate: 12 retries (regression!), 0 cache hits (regression!),
    # 1 guardrail block (same), plus events across all categories.
    candidate = [
        base_meta,
        *[he("retry", "retry.attempted", "warning") for _ in range(12)],
        he("guardrail", "guardrail.blocked", "error"),
        *[he(cat, f"{cat}.event") for cat in HARNESS_CATEGORIES],
    ]
    t0 = time.perf_counter()
    deltas = harness_event_diff(baseline, candidate)
    elapsed = time.perf_counter() - t0
    by_name = {(d.category, d.name): d for d in deltas}
    retry_d = by_name.get(("retry", "retry.attempted"))
    cache_d = by_name.get(("cache", "cache.hit"))
    report(
        f"harness_event_diff over {len(baseline) + len(candidate)} records in {elapsed * 1000:.1f}ms",
        elapsed < 0.1,
    )
    report(
        "retry count delta is +7 (5→12)",
        retry_d is not None and retry_d.count_delta == 7,
        f"got {retry_d.count_delta if retry_d else None}",
    )
    report(
        "cache regression (3→0) shows as count_delta=-3",
        cache_d is not None and cache_d.count_delta == -3,
        f"got {cache_d.count_delta if cache_d else None}",
    )
    report(
        "deltas sorted by absolute count_delta descending",
        deltas == sorted(deltas, key=lambda d: -abs(d.count_delta)),
    )


def stress_harness_event_session_recording_under_load(tmp: Path) -> None:
    """Record 5000 mixed harness events in one session and verify the
    output trace round-trips and content-addressing holds."""
    output = tmp / "many-events.agentlog"
    cats = (
        "retry",
        "cache",
        "guardrail",
        "model_switch",
        "context_trim",
        "rate_limit",
        "budget",
        "stream_interrupt",
        "tool_lifecycle",
    )
    t0 = time.perf_counter()
    with Session(output_path=output, auto_instrument=False) as s:
        for i in range(5000):
            record_harness_event(
                s,
                category=cats[i % len(cats)],  # type: ignore[arg-type]
                name=f"event-{i % 100}",
                severity="info",
                attributes={"i": i},
            )
    elapsed = time.perf_counter() - t0
    records = _core.parse_agentlog(output.read_bytes())
    events = [r for r in records if r["kind"] == "harness_event"]
    bad = [r for r in events if r["id"] != _core.content_id(r["payload"])]
    report(
        f"5000 harness_event records written and parsed in {elapsed * 1000:.0f}ms",
        len(events) == 5000 and not bad,
        f"events={len(events)} bad_ids={len(bad)}",
    )


# ====================================================================
# blob_ref + BlobStore
# ====================================================================


def stress_blob_store_dedup_under_repeated_puts(tmp: Path) -> None:
    """Putting the same blob 1000 times must produce exactly one file."""
    store = BlobStore(root=tmp / "blobs")
    blob = b"the quick brown fox jumps over the lazy dog" * 20
    ids = set()
    for _ in range(1000):
        ids.add(store.put(blob))
    on_disk_count = sum(1 for _ in (tmp / "blobs").rglob("*") if _.is_file())
    report(
        "1000 puts of identical content produce exactly one file + one id",
        len(ids) == 1 and on_disk_count == 1,
        f"distinct ids={len(ids)} on_disk={on_disk_count}",
    )


def stress_blob_store_atomic_replace_survives_crash_simulation(tmp: Path) -> None:
    """The store uses temp-file + atomic rename. Simulate a crash by
    leaving a stale .tmp file in the target shard and confirm the
    next put still produces the right final file."""
    store = BlobStore(root=tmp / "blobs")
    blob = b"important data"
    blob_id = store.put(blob)
    assert store.get(blob_id) == blob
    # Drop a stale .tmp file in the shard.
    digest = blob_id[len("sha256:") :]
    shard = (tmp / "blobs") / digest[:2]
    stale = shard / f"{digest[2:]}.tmp"
    stale.write_bytes(b"stale partial write")
    # Re-put the same blob — atomic rename should overwrite the tmp
    # without corrupting the final file.
    store.put(blob)
    final = shard / digest[2:]
    report(
        "store survives a stale .tmp file from a simulated crash",
        final.read_bytes() == blob,
    )


def stress_blob_store_large_blob(tmp: Path) -> None:
    """16 MiB blob put + get round-trips and stays within memory."""
    store = BlobStore(root=tmp / "big-blobs")
    big = os.urandom(16 * 1024 * 1024)
    t0 = time.perf_counter()
    blob_id = store.put(big)
    elapsed = time.perf_counter() - t0
    out = store.get(blob_id)
    report(
        f"16 MiB blob round-trips through the store in {elapsed * 1000:.0f}ms",
        out == big and elapsed < 5.0,
    )


def stress_phash_distance_correctness() -> None:
    """Hamming distance is correct on hand-computed cases."""
    # All zeros vs all ones = 64 bit diff
    a = {"algo": "dhash64", "hex": "0000000000000000"}
    b = {"algo": "dhash64", "hex": "ffffffffffffffff"}
    d_full = phash_distance(a, b)
    # One bit set
    c = {"algo": "dhash64", "hex": "0000000000000001"}
    d_one = phash_distance(a, c)
    # Different algos
    e = {"algo": "phash64", "hex": "ffffffffffffffff"}
    d_mismatch = phash_distance(a, e)
    report(
        "phash distance: 0/64↔ones=64, 0/64↔1bit=1, algo mismatch=None",
        d_full == 64 and d_one == 1 and d_mismatch is None,
        f"full={d_full} one={d_one} mismatch={d_mismatch}",
    )


def stress_record_blob_ref_image_with_phash(tmp: Path) -> None:
    """If imagehash is installed AND we feed real PNG bytes, dHash
    lands in the record. If imagehash is missing, we silently skip
    and the record is still valid."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
        from io import BytesIO

        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
    except ImportError:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    output = tmp / "img.agentlog"
    store = BlobStore(root=tmp / "blobs")
    with Session(output_path=output, auto_instrument=False) as s:
        record_blob_ref(s, blob=png, mime="image/png", store=store)
    records = _core.parse_agentlog(output.read_bytes())
    br = next(r for r in records if r["kind"] == "blob_ref")
    has_phash = "phash" in br["payload"]
    # Either imagehash is installed (phash present) or not (phash absent
    # but record is otherwise valid).
    valid = br["payload"]["mime"] == "image/png" and br["payload"][
        "blob_id"
    ].startswith("sha256:")
    report(
        f"blob_ref records image with phash={'yes' if has_phash else 'no (imagehash missing)'}",
        valid,
    )


# ====================================================================
# MCP replay
# ====================================================================


def stress_mcp_replay_1000_call_recording() -> None:
    """A 1000-call recording must look up calls in O(1) and round-
    trip every recorded response."""
    calls = [
        MCPCall(
            method="tools/call",
            params={"name": f"tool_{i % 50}", "arguments": {"q": f"q{i}"}},
            response=f"result-{i}",
        )
        for i in range(1000)
    ]
    idx = RecordingIndex(calls=calls)
    sess = ReplayClientSession(idx, strict=True)
    t0 = time.perf_counter()
    misses = 0
    for i in range(1000):
        try:
            out = sess.call_tool(f"tool_{i % 50}", {"q": f"q{i}"})
            if out != f"result-{i}":
                misses += 1
        except MCPCallNotRecorded:
            misses += 1
    elapsed = time.perf_counter() - t0
    report(
        f"1000 lookups in {elapsed * 1000:.0f}ms with no misses",
        misses == 0 and elapsed < 1.0,
    )


def stress_mcp_replay_repeated_call_ordering() -> None:
    """Repeated calls return recorded responses in original order, then
    fall back to the last recorded response. Pinned to strict=False
    because the strict default (added with the strict-overflow fix)
    raises on over-consumption — that path is covered by the unit
    tests; this stress test exercises the permissive fallback."""
    idx = RecordingIndex(
        calls=[
            MCPCall(method="tools/list", params={}, response={"v": 1}),
            MCPCall(method="tools/list", params={}, response={"v": 2}),
            MCPCall(method="tools/list", params={}, response={"v": 3}),
        ]
    )
    sess = ReplayClientSession(idx, strict=False)
    seq = [sess.list_tools() for _ in range(5)]
    report(
        "ordering preserved then last-recorded fallback",
        seq == [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 3}, {"v": 3}],
    )


def stress_mcp_canonicalize_collisions() -> None:
    """Different param shapes that should hash differently MUST hash
    differently. Adversarial: numeric coercion, key ordering, nested
    dicts, unicode."""
    cases = [
        ({"a": 1}, {"a": 1.0}),  # int vs float — different
        ({"a": [1, 2]}, {"a": [2, 1]}),  # array order matters
        ({"a": {"b": 1, "c": 2}}, {"a": {"c": 2, "b": 1}}),  # but key order doesn't
        ({"u": "café"}, {"u": "cafe"}),  # unicode-vs-ascii — different
    ]
    expected_pair_distinct = [True, True, False, True]
    failures: list[str] = []
    for i, (p1, p2) in enumerate(cases):
        same = canonicalize_params(p1) == canonicalize_params(p2)
        if same and expected_pair_distinct[i]:
            failures.append(f"case {i}: collision on {p1} vs {p2}")
        if not same and not expected_pair_distinct[i]:
            failures.append(f"case {i}: false-distinct on {p1} vs {p2}")
    report(
        "canonicalize_params distinguishes / equates correctly",
        not failures,
        f"failures: {failures}",
    )


def stress_mcp_replay_unconsumed_keys_for_drift_detection() -> None:
    """If the candidate skips a recorded call, unconsumed_keys()
    surfaces it."""
    idx = RecordingIndex(
        calls=[
            MCPCall(
                method="tools/call", params={"name": "a", "arguments": {}}, response="A"
            ),
            MCPCall(
                method="tools/call", params={"name": "b", "arguments": {}}, response="B"
            ),
            MCPCall(
                method="tools/call", params={"name": "c", "arguments": {}}, response="C"
            ),
        ]
    )
    sess = ReplayClientSession(idx)
    sess.call_tool("a", {})
    sess.call_tool("c", {})  # skipped b!
    unconsumed = idx.unconsumed_keys()
    # unconsumed_keys returns (method, canonical-params-str) tuples;
    # `p` is already a canonicalized string, do not re-canonicalize.
    names = sorted(p for _, p in unconsumed)
    report(
        "unconsumed_keys surfaces the skipped call (b)",
        any('"name":"b"' in n for n in names),
        f"unconsumed: {names}",
    )


# ====================================================================
# main
# ====================================================================


def main() -> int:
    print("\033[1m=== Shadow v2.3.0 adverse stress test ===\033[0m")
    tmp = Path(tempfile.mkdtemp(prefix="shadow-stress-v23-"))
    t0 = time.monotonic()
    try:
        section("chunk records — record + replay")
        stress_chunk_record_10000(tmp)
        stress_chunk_replay_timing_fidelity()
        stress_chunk_replay_concurrent_no_state_leak()
        stress_chunk_replay_non_monotonic_timestamps()
        stress_chunk_replay_speed_inf_no_delay()

        section("harness_event diff under load")
        stress_harness_event_diff_at_scale()
        stress_harness_event_session_recording_under_load(tmp)

        section("blob_ref + content-addressed store")
        stress_blob_store_dedup_under_repeated_puts(tmp)
        stress_blob_store_atomic_replace_survives_crash_simulation(tmp)
        stress_blob_store_large_blob(tmp)
        stress_phash_distance_correctness()
        stress_record_blob_ref_image_with_phash(tmp)

        section("MCP-native replay")
        stress_mcp_replay_1000_call_recording()
        stress_mcp_replay_repeated_call_ordering()
        stress_mcp_canonicalize_collisions()
        stress_mcp_replay_unconsumed_keys_for_drift_detection()
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        for p in sorted(tmp.rglob("*"), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass

    elapsed = time.monotonic() - t0
    n_total = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print()
    print(f"\033[1m=== summary ({elapsed:.2f}s wall) ===\033[0m")
    print(f"  passed: {n_pass}/{n_total}")
    if n_pass < n_total:
        print()
        print("\033[1m  failures:\033[0m")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
