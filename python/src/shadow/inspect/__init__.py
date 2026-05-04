"""Terminal renderer for `.agentlog` files — Shadow's daily-debug surface.

`shadow diff` produces the report; `shadow gate-pr` produces the
verdict; `shadow inspect` is the thing engineers reach for during
debugging. One terminal table, no JSON, fast enough that you'd run
it twenty times in a session.

Single-file mode:

    shadow inspect trace.agentlog

shows: turn, kind, role/tool, tokens (in/out), latency-ms, cost-usd,
redactions count, plus a brief stop-reason / first content snippet.

Comparison mode (two files):

    shadow inspect baseline.agentlog candidate.agentlog

walks paired chat turns side-by-side and highlights the first turn
that diverged in red. Mirrors the `first_divergence` semantics from
`shadow.align` without needing the full 9-axis differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadow import _core

__all__ = ["TraceRow", "first_divergence_index", "load_trace_rows"]


@dataclass(frozen=True)
class TraceRow:
    """One row in the inspect view. Carries everything the table
    renderer needs without the renderer having to know the
    `.agentlog` envelope shape."""

    turn: int
    kind: str
    role_or_tool: str
    """`user` / `assistant` for chat records; tool name for tool_call /
    tool_result; empty string for envelope records (metadata, error)."""
    summary: str
    """First 80 chars of meaningful payload — text snippet for chat
    responses, tool argument hash for tool calls, etc."""
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    cost_usd: float | None
    redactions: int
    """How many `[REDACTED:...]` markers appear in this record's payload.
    Lets the user see at a glance whether sensitive fields were
    swept by the Session's Redactor."""
    stop_reason: str
    record_id: str
    """First 16 chars of the content id — enough to grep for in
    larger fixture sets."""


def _summarise(payload: object) -> str:
    """Pull a short human-readable line out of an arbitrary record
    payload. Each record kind has its own preferred field; we fall
    back to the JSON repr capped at 80 chars."""
    if not isinstance(payload, dict):
        snippet = str(payload)
        return snippet if len(snippet) <= 80 else snippet[:77] + "..."

    # chat_request: first user message content
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c if len(c) <= 80 else c[:77] + "..."

    # chat_response: first text content block, or stop_reason fallback
    content = payload.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str) and t:
                    return t if len(t) <= 80 else t[:77] + "..."
        # No text block — name the tools that fired.
        names: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                n = b.get("name")
                if isinstance(n, str):
                    names.append(n)
        if names:
            return "tools: " + ", ".join(names)

    # tool_call / tool_result
    if "name" in payload and isinstance(payload["name"], str):
        return f"tool: {payload['name']}"
    if "result" in payload:
        r = payload["result"]
        rs = str(r)
        return rs if len(rs) <= 80 else rs[:77] + "..."

    return ""


def _role_or_tool(kind: str, payload: object) -> str:
    """Best human label for the actor on this record."""
    if not isinstance(payload, dict):
        return ""
    if kind == "chat_request":
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and isinstance(m.get("role"), str):
                    return str(m["role"])
        return "user"
    if kind == "chat_response":
        return "assistant"
    if kind in {"tool_call", "tool_result"}:
        n = payload.get("name")
        return str(n) if isinstance(n, str) else "tool"
    return ""


def _redaction_count(payload: object) -> int:
    """Count `[REDACTED:` markers anywhere in the payload's JSON form.
    Approximate, but cheap and accurate for the common case where
    the Session's Redactor walked the value before write."""
    import json

    text = json.dumps(payload, ensure_ascii=False)
    return text.count("[REDACTED:")


def _token_fields(payload: object) -> tuple[int | None, int | None]:
    """`(input_tokens, output_tokens)` extracted from `payload.usage`.
    Returns `(None, None)` when the record didn't carry usage."""
    if not isinstance(payload, dict):
        return None, None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    inp = usage.get("input_tokens") or usage.get("prompt_tokens")
    out = usage.get("output_tokens") or usage.get("completion_tokens")
    return (
        int(inp) if isinstance(inp, int | float) else None,
        int(out) if isinstance(out, int | float) else None,
    )


def _latency_ms(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    v = payload.get("latency_ms")
    if isinstance(v, int | float):
        return int(v)
    return None


def _cost_usd(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    v = payload.get("cost_usd")
    if isinstance(v, int | float):
        return float(v)
    return None


def _stop_reason(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    v = payload.get("stop_reason")
    return str(v) if isinstance(v, str) else ""


def load_trace_rows(path: Path) -> list[TraceRow]:
    """Parse a `.agentlog` file and return one `TraceRow` per record.

    Numbers turns 1..N for the human reader; envelope records like
    `metadata` get turn `0` so they don't shift the count.
    """
    blob = path.read_bytes()
    records = _core.parse_agentlog(blob)
    rows: list[TraceRow] = []
    turn = 0
    for rec in records:
        kind = str(rec.get("kind", ""))
        # Bump turn on user-visible record kinds. Metadata and
        # error envelopes share the same turn number as the next
        # real record so the table reads cleanly.
        if kind in {"chat_request", "tool_call"}:
            turn += 1
        payload: Any = rec.get("payload")
        in_tok, out_tok = _token_fields(payload)
        rows.append(
            TraceRow(
                turn=turn,
                kind=kind,
                role_or_tool=_role_or_tool(kind, payload),
                summary=_summarise(payload),
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=_latency_ms(payload),
                cost_usd=_cost_usd(payload),
                redactions=_redaction_count(payload),
                stop_reason=_stop_reason(payload),
                record_id=str(rec.get("id", ""))[:24],
            )
        )
    return rows


def first_divergence_index(baseline: list[TraceRow], candidate: list[TraceRow]) -> int | None:
    """Return the index of the first row that meaningfully differs
    between the two traces. Used by the comparison renderer to
    paint the divergence line red.

    "Meaningful" here = the same minimum we use in the Rust
    `shadow_align::first_divergence`: kind change, tool name change,
    or response-text change. Latency / token / cost diffs don't
    count — they jitter on every run.
    """
    n = min(len(baseline), len(candidate))
    for i in range(n):
        b = baseline[i]
        c = candidate[i]
        if b.kind != c.kind:
            return i
        if b.kind in {"chat_response", "tool_call", "tool_result"}:
            if b.role_or_tool != c.role_or_tool:
                return i
            if b.summary != c.summary:
                return i
    if len(baseline) != len(candidate):
        return n
    return None
