"""Langfuse export → Shadow `.agentlog` converter.

Langfuse's public export shape (as of 2026-04) is a JSON object like:

    {
      "traces": [
        {
          "id": "<trace-id>",
          "name": "agent-turn",
          "timestamp": "2026-04-21T10:00:00.000Z",
          "userId": "...",
          "observations": [
            {
              "id": "<span-id>",
              "type": "generation",            # or "span"
              "name": "chat",
              "startTime": "2026-04-21T10:00:00.000Z",
              "endTime":   "2026-04-21T10:00:00.150Z",
              "model": "gpt-4.1",
              "modelParameters": {"temperature": 0.2, "max_tokens": 256},
              "input": [{"role": "user", "content": "hi"}],
              "output": {"role": "assistant", "content": "hello"},
              "usage": {"input": 4, "output": 1, "total": 5},
              "level": "DEFAULT"
            }
          ]
        }
      ]
    }

We convert each `generation` observation into a Shadow `chat_request`
+ `chat_response` pair. Non-generation observations are dropped.
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowConfigError

LANGFUSE_FORMAT = "langfuse"


def langfuse_to_agentlog(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Langfuse export dict to a list of Shadow records."""
    traces = data.get("traces")
    if not isinstance(traces, list):
        raise ShadowConfigError(
            "Langfuse export missing top-level `traces` list.\n"
            'hint: the export should look like {"traces": [...]}'
        )

    meta_payload = {
        "sdk": {"name": "shadow", "version": __version__},
        "imported_from": LANGFUSE_FORMAT,
        "trace_count": len(traces),
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

    for trace in traces:
        observations = trace.get("observations") or []
        for obs in observations:
            if obs.get("type") != "generation":
                continue
            req_payload = _to_request_payload(obs)
            req_id = _core.content_id(req_payload)
            records.append(
                {
                    "version": "0.1",
                    "id": req_id,
                    "kind": "chat_request",
                    "ts": _normalise_ts(obs.get("startTime", _now_iso())),
                    "parent": last_parent,
                    "payload": req_payload,
                }
            )
            resp_payload = _to_response_payload(obs)
            resp_id = _core.content_id(resp_payload)
            records.append(
                {
                    "version": "0.1",
                    "id": resp_id,
                    "kind": "chat_response",
                    "ts": _normalise_ts(obs.get("endTime", _now_iso())),
                    "parent": req_id,
                    "payload": resp_payload,
                }
            )
            last_parent = resp_id

    return records


def _to_request_payload(obs: dict[str, Any]) -> dict[str, Any]:
    messages = obs.get("input") or []
    if isinstance(messages, dict):
        messages = [messages]
    params_raw = obs.get("modelParameters") or {}
    params: dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens", "stop"):
        if key in params_raw:
            params[key] = params_raw[key]
    return {
        "model": obs.get("model", ""),
        "messages": [_normalise_message(m) for m in messages],
        "params": params,
    }


def _to_response_payload(obs: dict[str, Any]) -> dict[str, Any]:
    output = obs.get("output")
    content: list[dict[str, Any]] = []
    if isinstance(output, str):
        content = [{"type": "text", "text": output}]
    elif isinstance(output, dict):
        text = output.get("content")
        if isinstance(text, str):
            content = [{"type": "text", "text": text}]
        elif isinstance(text, list):
            content = [p for p in text if isinstance(p, dict)]
    elif isinstance(output, list):
        content = [p for p in output if isinstance(p, dict)]
    usage_raw = obs.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("input", usage_raw.get("input_tokens", 0)) or 0),
        "output_tokens": int(usage_raw.get("output", usage_raw.get("output_tokens", 0)) or 0),
        "thinking_tokens": int(usage_raw.get("reasoning_tokens", 0) or 0),
    }
    return {
        "model": obs.get("model", ""),
        "content": content,
        "stop_reason": _map_level_to_stop(obs.get("level", "DEFAULT")),
        "latency_ms": _derive_latency_ms(obs.get("startTime"), obs.get("endTime")),
        "usage": usage,
    }


def _normalise_message(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(m.get("role", "user")),
        "content": m.get("content", ""),
    }


def _map_level_to_stop(level: str) -> str:
    return "content_filter" if level in ("ERROR", "WARNING") else "end_turn"


def _derive_latency_ms(start: Any, end: Any) -> int:
    if not isinstance(start, str) or not isinstance(end, str):
        return 0
    try:
        start_dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
        delta = (end_dt - start_dt).total_seconds()
        return max(0, int(delta * 1000))
    except ValueError:
        return 0


def _normalise_ts(ts: str) -> str:
    """Ensure a trailing Z and millisecond precision."""
    if not ts:
        return _now_iso()
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (
            dt.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{dt.microsecond // 1000:03d}Z"
        )
    except ValueError:
        return _now_iso()


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["LANGFUSE_FORMAT", "langfuse_to_agentlog"]
