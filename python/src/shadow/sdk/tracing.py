"""Distributed trace-context propagation across Shadow Sessions.

Many real agents are multi-process / multi-agent — a planner calls a
worker calls a tool, each potentially in a different process. Shadow
supports joining those traces into one logical trace via W3C trace
context-style propagation:

- The parent Session sets `SHADOW_TRACE_ID=<128-bit hex>` and
  `SHADOW_PARENT_SPAN_ID=<64-bit hex>` in the env it spawns children
  with.
- Each child Session detects these env vars on `__enter__` and stamps
  them on every record's `meta.trace_id` / `meta.parent_span_id`.
- `shadow join a.agentlog b.agentlog ... -o merged.agentlog` re-links
  child traces under their parents and writes one merged `.agentlog`.

W3C `traceparent` header is ALSO recognised: if the parent set
`$TRACEPARENT` (standard OTel/W3C), Shadow reads the same IDs from it.
"""

from __future__ import annotations

import os
import secrets
from typing import Any

TRACE_ID_ENV = "SHADOW_TRACE_ID"
PARENT_SPAN_ENV = "SHADOW_PARENT_SPAN_ID"
W3C_TRACEPARENT_ENV = "TRACEPARENT"


def current_trace_id() -> str | None:
    """Return the active trace_id from env, or None if not in a distributed trace."""
    direct = os.environ.get(TRACE_ID_ENV)
    if direct:
        return direct
    tp = os.environ.get(W3C_TRACEPARENT_ENV)
    if tp:
        parsed = _parse_traceparent(tp)
        if parsed is not None:
            return parsed[0]
    return None


def current_parent_span_id() -> str | None:
    direct = os.environ.get(PARENT_SPAN_ENV)
    if direct:
        return direct
    tp = os.environ.get(W3C_TRACEPARENT_ENV)
    if tp:
        parsed = _parse_traceparent(tp)
        if parsed is not None:
            return parsed[1]
    return None


def new_trace_id() -> str:
    """Generate a fresh 128-bit hex trace_id."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Generate a fresh 64-bit hex span_id."""
    return secrets.token_hex(8)


def env_for_child(trace_id: str, span_id: str) -> dict[str, str]:
    """Build an env dict a child process should inherit to continue the trace.

    Sets both the Shadow-native env vars and the W3C `traceparent`
    header so non-Shadow children (OTel-instrumented) can join too.
    """
    traceparent = f"00-{trace_id}-{span_id}-01"
    return {
        TRACE_ID_ENV: trace_id,
        PARENT_SPAN_ENV: span_id,
        W3C_TRACEPARENT_ENV: traceparent,
    }


_HEX_CHARS = frozenset("0123456789abcdef")


def _parse_traceparent(header: str) -> tuple[str, str] | None:
    """Parse a W3C traceparent header: `00-<trace_id>-<span_id>-<flags>`.

    Strict per-spec: trace_id and span_id must be lowercase hex of the
    right length, and cannot be all-zeros (reserved per RFC 9110 §11.2.2).
    """
    parts = header.strip().split("-")
    if len(parts) != 4:
        return None
    version, trace_id, span_id, _flags = parts
    if version != "00":
        return None
    if len(trace_id) != 32 or len(span_id) != 16:
        return None
    if not (set(trace_id) <= _HEX_CHARS and set(span_id) <= _HEX_CHARS):
        return None
    # All-zero IDs are reserved as "invalid" per W3C spec.
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id, span_id


def stamp_meta(meta: dict[str, Any] | None, trace_id: str, span_id: str) -> dict[str, Any]:
    """Return a meta dict stamped with trace_id + span_id, preserving existing keys."""
    out = dict(meta or {})
    out["trace_id"] = trace_id
    out["span_id"] = span_id
    return out


__all__ = [
    "PARENT_SPAN_ENV",
    "TRACE_ID_ENV",
    "W3C_TRACEPARENT_ENV",
    "current_parent_span_id",
    "current_trace_id",
    "env_for_child",
    "new_span_id",
    "new_trace_id",
    "stamp_meta",
]
