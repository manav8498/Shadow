"""Braintrust export → Shadow `.agentlog` converter.

Braintrust's `braintrust export experiment` emits one row per
evaluation example as JSONL or a JSON array. The per-row shape:

    {
      "id": "<example-id>",
      "input": "what is the capital of France?"          # str or dict
            |  {"messages": [...], "model": "...", "params": {...}},
      "output": "Paris"                                  # str or dict
            |  {"content": "Paris", "finish_reason": "stop"}
            |  [{"type": "text", "text": "Paris"}],
      "metadata": {"model": "gpt-4.1", "temperature": 0.2, ...},
      "metrics": {
          "latency": 0.412,                              # seconds
          "total_tokens": 14,
          "prompt_tokens": 9,
          "completion_tokens": 5,
          "reasoning_tokens": 0
      },
      "tags": ["production"]
    }

We accept either a list-of-rows JSON or newline-delimited JSON
(`.jsonl`). Each row becomes one chat_request + chat_response pair.
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError

BRAINTRUST_FORMAT = "braintrust"


def braintrust_to_agentlog(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a Braintrust experiment-export row list → Shadow records."""
    if not isinstance(data, list):
        raise ShadowConfigError(
            "Braintrust export should be a JSON array (or JSONL) of example rows.\n"
            "hint: run `braintrust export experiment <id> > out.json`."
        )

    meta_payload = {
        "sdk": {"name": "shadow", "version": "0.1.0"},
        "imported_from": BRAINTRUST_FORMAT,
        "row_count": len(data),
    }
    meta_id = _core.content_id(meta_payload)
    records: list[dict[str, Any]] = [
        {
            "version": "0.1",
            "id": meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": meta_payload,
        }
    ]
    last_parent: str = meta_id

    for i, row in enumerate(data):
        req_payload = _row_to_request(row)
        req_id = _core.content_id({**req_payload, "_row": i})  # disambiguate identical inputs
        records.append(
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": _now_iso(),
                "parent": last_parent,
                "payload": req_payload,
            }
        )
        resp_payload = _row_to_response(row)
        resp_id = _core.content_id({**resp_payload, "_row": i})
        records.append(
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": _now_iso(),
                "parent": req_id,
                "payload": resp_payload,
            }
        )
        last_parent = resp_id

    return records


def _row_to_request(row: dict[str, Any]) -> dict[str, Any]:
    inp = row.get("input")
    meta = row.get("metadata") or {}
    model = meta.get("model") or row.get("model") or ""
    params_raw = meta.get("params") or {
        k: meta[k] for k in ("temperature", "top_p", "max_tokens", "stop") if k in meta
    }
    messages: list[dict[str, Any]]
    if isinstance(inp, str):
        messages = [{"role": "user", "content": inp}]
    elif isinstance(inp, dict):
        if "messages" in inp:
            messages = [_normalise_message(m) for m in inp["messages"] or []]
        else:
            messages = [{"role": "user", "content": _stringify(inp)}]
    elif isinstance(inp, list):
        # Assume already a messages list.
        messages = [_normalise_message(m) for m in inp if isinstance(m, dict)]
    else:
        messages = [{"role": "user", "content": _stringify(inp)}]
    return {
        "model": model,
        "messages": messages,
        "params": params_raw,
    }


def _row_to_response(row: dict[str, Any]) -> dict[str, Any]:
    out = row.get("output")
    meta = row.get("metadata") or {}
    metrics = row.get("metrics") or {}
    content: list[dict[str, Any]]
    if isinstance(out, str):
        content = [{"type": "text", "text": out}]
    elif isinstance(out, dict):
        text = out.get("content")
        if isinstance(text, str):
            content = [{"type": "text", "text": text}]
        elif isinstance(text, list):
            content = [p for p in text if isinstance(p, dict)]
        else:
            content = [{"type": "text", "text": _stringify(out)}]
    elif isinstance(out, list):
        content = [p for p in out if isinstance(p, dict)] or [
            {"type": "text", "text": _stringify(out)}
        ]
    else:
        content = [{"type": "text", "text": _stringify(out)}]

    latency_s = metrics.get("latency")
    latency_ms = int(float(latency_s) * 1000) if isinstance(latency_s, int | float) else 0

    usage = {
        "input_tokens": int(metrics.get("prompt_tokens", 0) or 0),
        "output_tokens": int(metrics.get("completion_tokens", 0) or 0),
        "thinking_tokens": int(metrics.get("reasoning_tokens", 0) or 0),
    }
    stop_reason = "end_turn"
    if isinstance(out, dict) and out.get("finish_reason"):
        raw = str(out["finish_reason"])
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }.get(raw, raw)
    return {
        "model": meta.get("model") or row.get("model") or "",
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _normalise_message(m: dict[str, Any]) -> dict[str, Any]:
    return {"role": str(m.get("role", "user")), "content": m.get("content", "")}


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        import json

        return json.dumps(v)
    except (TypeError, ValueError):
        return str(v)


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["BRAINTRUST_FORMAT", "braintrust_to_agentlog"]
