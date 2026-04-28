"""Property-based fuzz tests for the agentlog parser/writer round-trip.

Hypothesis is a project dependency but underutilized — most existing
tests use hand-crafted records. This module fuzzes the parser/writer
boundary with shrinking property tests so format-level regressions
(canonical-JSON serialization drift, hash-stability bugs, parser
edge cases) are caught at random inputs the developer didn't think of.

The core property: ``parse(write(records)) == records`` — and
``content_id`` of every record is stable across the round trip.
"""

from __future__ import annotations

import string
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shadow import _core

_NULL_PARENT = "sha256:" + "0" * 64


# Strategy: a payload-free record with valid envelope. The agentlog
# format requires version, id, kind, ts, parent, payload (which can be
# None for some kinds). Hypothesis generates random valid envelopes;
# the parser must accept them and the writer must reproduce them.

_id_strategy = st.from_regex(r"^sha256:[0-9a-f]{64}$", fullmatch=True)
_ts_strategy = st.from_regex(
    r"^20[0-9]{2}-[01][0-9]-[0-3][0-9]T[012][0-9]:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}Z$",
    fullmatch=True,
)
_text_strategy = st.text(alphabet=string.ascii_letters + string.digits + " -_.,!?", max_size=200)


@st.composite
def _chat_response_record(draw: Any) -> dict[str, Any]:
    """Generate a syntactically valid chat_response record."""
    return {
        "version": "0.1",
        "id": draw(_id_strategy),
        "kind": "chat_response",
        "ts": draw(_ts_strategy),
        "parent": draw(st.one_of(st.just(_NULL_PARENT), _id_strategy)),
        "meta": {},
        "payload": {
            "model": draw(st.sampled_from(["gpt-4", "claude-haiku", "test-model"])),
            "content": [{"type": "text", "text": draw(_text_strategy)}],
            "stop_reason": draw(st.sampled_from(["end_turn", "tool_use", "content_filter", "stop"])),
            "latency_ms": draw(st.integers(min_value=0, max_value=120000)),
            "usage": {
                "input_tokens": draw(st.integers(min_value=0, max_value=10000)),
                "output_tokens": draw(st.integers(min_value=0, max_value=10000)),
                "thinking_tokens": draw(st.integers(min_value=0, max_value=10000)),
            },
        },
    }


@given(records=st.lists(_chat_response_record(), min_size=0, max_size=20))
@settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_parse_write_roundtrip_preserves_records(records: list[dict[str, Any]]) -> None:
    """Records → write → bytes → parse → records identity holds for any
    valid chat_response record list."""
    written = _core.write_agentlog(records)
    parsed = _core.parse_agentlog(written)
    assert parsed == records, "parse(write(records)) != records"


@given(records=st.lists(_chat_response_record(), min_size=1, max_size=10))
@settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_content_id_stable_across_roundtrip(records: list[dict[str, Any]]) -> None:
    """Each record's content_id (computed on its payload) must be the
    same after a parse/write round trip — i.e. canonical JSON
    serialization is deterministic across the cycle."""
    before = [_core.content_id(r["payload"]) for r in records]
    written = _core.write_agentlog(records)
    parsed = _core.parse_agentlog(written)
    after = [_core.content_id(r["payload"]) for r in parsed]
    assert before == after


@given(records=st.lists(_chat_response_record(), min_size=2, max_size=10))
@settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_write_format_is_one_record_per_line(records: list[dict[str, Any]]) -> None:
    """JSONL invariant: bytes contain exactly one record per line, and
    the record count equals the line count (no embedded newlines)."""
    written = _core.write_agentlog(records)
    text = written.decode("utf-8")
    # Strip trailing newline if any; count newline-separated records.
    lines = [line for line in text.split("\n") if line]
    assert len(lines) == len(records)


@given(records=st.lists(_chat_response_record(), min_size=1, max_size=5))
@settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_canonical_bytes_deterministic(records: list[dict[str, Any]]) -> None:
    """canonical_bytes() of the same payload twice must be byte-identical."""
    for rec in records:
        a = _core.canonical_bytes(rec["payload"])
        b = _core.canonical_bytes(rec["payload"])
        assert a == b
