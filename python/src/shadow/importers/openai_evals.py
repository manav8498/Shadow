"""OpenAI Evals output → Shadow `.agentlog`.

The OpenAI evals library emits JSONL with `sample` + `match` events.
Each `sampling` event contains the prompt + sampled completion for a
single eval example. We convert each into a chat_request / chat_response
pair so Shadow can diff two eval runs.

Shape (one JSON object per line):

    {
      "run_id": "...",
      "event_id": "...",
      "sample_id": "...",
      "type": "sampling",
      "data": {
        "prompt": [{"role": "user", "content": "..."}],   (or a string)
        "sampled": ["the model's completion"],
        "options": {"model": "gpt-4o", "temperature": 0.2},
        "usage": {"prompt_tokens": ..., "completion_tokens": ...}
      },
      "created_at": "2026-04-21T10:00:00Z"
    }

Other event types (`match`, `metrics`, …) are dropped; they carry
scoring metadata rather than raw LLM I/O.
"""

from __future__ import annotations

import datetime
from typing import Any

from shadow import __version__, _core
from shadow.errors import ShadowConfigError

OPENAI_EVALS_FORMAT = "openai-evals"


def openai_evals_to_agentlog(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert an OpenAI-Evals JSONL event list to Shadow records."""
    if not isinstance(data, list):
        raise ShadowConfigError(
            "OpenAI-Evals input must be a list of event dicts (one JSON object per line)."
        )
    meta_payload = {
        "sdk": {"name": "shadow", "version": __version__},
        "imported_from": OPENAI_EVALS_FORMAT,
        "event_count": len(data),
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

    for ev in data:
        if ev.get("type") != "sampling":
            continue
        d = ev.get("data") or {}
        options = d.get("options") or {}
        model = str(options.get("model", ""))
        # Prompt can be a string or a list of messages.
        prompt = d.get("prompt")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = [
                {"role": str(m.get("role", "user")), "content": m.get("content", "")}
                for m in prompt
                if isinstance(m, dict)
            ]
        else:
            messages = []
        params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_tokens"):
            if key in options:
                params[key] = options[key]

        req_payload = {"model": model, "messages": messages, "params": params}
        req_id = _core.content_id(
            {**req_payload, "_sample_id": ev.get("sample_id", ev.get("event_id", ""))}
        )

        # Sampled completions. OpenAI Evals supports n>1 sampling; emit one
        # chat_response record per completion so no data is lost. Each
        # response shares the same parent chat_request.
        sampled = d.get("sampled") or []
        if not isinstance(sampled, list) or not sampled:
            sampled = [""]
        usage_raw = d.get("usage") or {}
        # Usage is reported for the batch — split evenly across samples.
        per_sample_usage = {
            "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "output_tokens": int((int(usage_raw.get("completion_tokens", 0) or 0)) // len(sampled)),
            "thinking_tokens": int(
                (int(usage_raw.get("reasoning_tokens", 0) or 0)) // len(sampled)
            ),
        }

        ts = _normalise_ts(ev.get("created_at", ""))
        records.append(
            {
                "version": "0.1",
                "id": req_id,
                "kind": "chat_request",
                "ts": ts,
                "parent": last_parent,
                "payload": req_payload,
            }
        )
        for sample_idx, raw_text in enumerate(sampled):
            text = raw_text if isinstance(raw_text, str) else str(raw_text)
            resp_payload = {
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "latency_ms": 0,
                "usage": per_sample_usage,
            }
            resp_id = _core.content_id(
                {
                    **resp_payload,
                    "_sample_id": ev.get("sample_id", ev.get("event_id", "")),
                    "_n": sample_idx,
                }
            )
            records.append(
                {
                    "version": "0.1",
                    "id": resp_id,
                    "kind": "chat_response",
                    "ts": ts,
                    "parent": req_id,
                    "payload": resp_payload,
                }
            )
            last_parent = resp_id
    return records


def _normalise_ts(ts: str) -> str:
    if not ts:
        return _now_iso()
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        return (
            dt.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{dt.microsecond // 1000:03d}Z"
        )
    except ValueError:
        return _now_iso()


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["OPENAI_EVALS_FORMAT", "openai_evals_to_agentlog"]
