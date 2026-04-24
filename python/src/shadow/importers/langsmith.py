"""LangSmith trace export → Shadow `.agentlog`.

LangSmith runs export a JSON array of run objects. Each run can be an
`"llm"` run (one LLM call) or `"chain"` / `"tool"` / `"retriever"` run.
We convert every `llm` run into a Shadow `chat_request` + `chat_response`
pair. Non-llm runs are dropped.

Canonical run shape (the fields we rely on):

    {
      "id": "<run-uuid>",
      "run_type": "llm",
      "name": "ChatOpenAI",
      "start_time": "2026-04-21T10:00:00.000000",
      "end_time":   "2026-04-21T10:00:00.150000",
      "inputs": { "messages": [[{"role": "user", "content": "hi"}]] },
      "outputs": { "generations": [[{"text": "hello",
                                     "generation_info": {"finish_reason": "stop"}}]] },
      "extra": { "invocation_params": { "model": "gpt-4o", "temperature": 0.2 } }
    }
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowConfigError

LANGSMITH_FORMAT = "langsmith"


def langsmith_to_agentlog(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a LangSmith runs-export list to Shadow records."""
    if not isinstance(data, list):
        raise ShadowConfigError(
            "LangSmith export must be a JSON array of run objects.\n"
            "hint: `langsmith runs list --project <name> --export` produces this."
        )
    meta_payload = {
        "sdk": {"name": "shadow", "version": __version__},
        "imported_from": LANGSMITH_FORMAT,
        "run_count": len(data),
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
    last_parent = meta_id

    for run in data:
        if run.get("run_type") != "llm":
            continue
        req_payload = _run_to_request(run)
        req_id = _core.content_id(req_payload)
        records.append(
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": _normalise_ts(run.get("start_time")),
                "parent": last_parent,
                "payload": req_payload,
            }
        )
        resp_payload = _run_to_response(run)
        resp_id = _core.content_id(resp_payload)
        records.append(
            {
                "version": "0.1",
                "id": resp_id,
                "kind": "chat_response",
                "ts": _normalise_ts(run.get("end_time")),
                "parent": req_id,
                "payload": resp_payload,
            }
        )
        last_parent = resp_id
    return records


def _run_to_request(run: dict[str, Any]) -> dict[str, Any]:
    extra = run.get("extra") or {}
    params_raw = extra.get("invocation_params") or {}
    model = str(params_raw.get("model") or params_raw.get("model_name") or run.get("name", ""))
    params: dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens"):
        if key in params_raw:
            params[key] = params_raw[key]
    inputs = run.get("inputs") or {}
    raw_messages = inputs.get("messages") or []
    # LangSmith wraps messages as [[...]] when a run processes a batch.
    # For a single-message batch-of-1 this unwraps cleanly; for multi-batch
    # runs (generate_messages over n examples) we FLATTEN all batches
    # rather than silently dropping later ones.
    if raw_messages and isinstance(raw_messages[0], list):
        flat: list[Any] = []
        for batch in raw_messages:
            if isinstance(batch, list):
                flat.extend(batch)
            elif isinstance(batch, dict):
                flat.append(batch)
        raw_messages = flat
    messages: list[dict[str, Any]] = []
    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or _infer_role_from_type(m.get("type", ""))
        content = m.get("content") or m.get("text") or ""
        messages.append({"role": role, "content": content})
    return {"model": model, "messages": messages, "params": params}


def _run_to_response(run: dict[str, Any]) -> dict[str, Any]:
    outputs = run.get("outputs") or {}
    generations = outputs.get("generations") or []
    if generations and isinstance(generations[0], list):
        generations = generations[0]
    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"
    for g in generations:
        if not isinstance(g, dict):
            continue
        text = g.get("text")
        if isinstance(text, str):
            content.append({"type": "text", "text": text})
        info = g.get("generation_info") or {}
        raw_stop = str(info.get("finish_reason", "stop"))
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }.get(raw_stop, raw_stop)
    # Usage lives in various places depending on LangSmith version.
    usage_raw = (
        (
            outputs.get("llm_output", {}).get("token_usage", {})
            if isinstance(outputs.get("llm_output"), dict)
            else {}
        )
        or outputs.get("usage")
        or run.get("extra", {}).get("token_usage", {})
        or {}
    )
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
        "output_tokens": int(
            usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0
        ),
        "thinking_tokens": int(usage_raw.get("reasoning_tokens", 0) or 0),
    }
    latency_ms = _duration_ms(run.get("start_time"), run.get("end_time"))
    extra = run.get("extra") or {}
    params_raw = extra.get("invocation_params") or {}
    model = str(params_raw.get("model") or params_raw.get("model_name") or run.get("name", ""))
    return {
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": latency_ms,
        "usage": usage,
    }


def _infer_role_from_type(t: str) -> str:
    return {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
    }.get(t.lower(), "user")


def _normalise_ts(ts: Any) -> str:
    if not isinstance(ts, str) or not ts:
        return _now_iso()
    try:
        # LangSmith timestamps are ISO 8601 without TZ (naive UTC).
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return (
            dt.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{dt.microsecond // 1000:03d}Z"
        )
    except ValueError:
        return _now_iso()


def _duration_ms(start: Any, end: Any) -> int:
    if not isinstance(start, str) or not isinstance(end, str):
        return 0
    try:
        s = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
        if s.tzinfo is None:
            s = s.replace(tzinfo=datetime.UTC)
        if e.tzinfo is None:
            e = e.replace(tzinfo=datetime.UTC)
        return max(0, int((e - s).total_seconds() * 1000))
    except ValueError:
        return 0


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["LANGSMITH_FORMAT", "langsmith_to_agentlog"]
