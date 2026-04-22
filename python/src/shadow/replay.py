"""Python-side replay engine. Mirrors `shadow_core::replay::engine::run_replay`.

Kept in Python (instead of exposed through PyO3) because it's async and
uses the Python `LlmBackend` protocol — the Rust trait object can't be
called back into from PyO3 without ceremony. Semantics match SPEC §10.
"""

from __future__ import annotations

import datetime
import time
from typing import Any

from shadow import _core
from shadow.errors import ShadowBackendError, ShadowParseError
from shadow.llm.base import LlmBackend


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


async def run_replay(
    baseline: list[dict[str, Any]],
    backend: LlmBackend,
) -> list[dict[str, Any]]:
    """Walk `baseline`, dispatching every chat_request to `backend`.

    Returns a new trace: metadata root → re-emitted requests + backend
    responses → tool records copied through → replay_summary.
    """
    if not baseline:
        raise ShadowParseError("baseline trace is empty — need at least a metadata root")
    root = baseline[0]
    if root.get("kind") != "metadata":
        raise ShadowParseError(
            f"baseline root is {root.get('kind')!r}, expected 'metadata' (SPEC §3.3)"
        )

    out: list[dict[str, Any]] = []
    started = time.perf_counter()
    input_count = 0
    output_count = 0
    error_count = 0

    # 1. New metadata root pointing back at the baseline root.
    new_meta_payload = dict(root["payload"])
    new_meta_payload["baseline_of"] = root["id"]
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(
        {
            "version": _core.SPEC_VERSION,
            "id": new_meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": new_meta_payload,
        }
    )
    last_parent: str = new_meta_id

    # 2. Walk baseline.
    for i, record in enumerate(baseline):
        kind = record.get("kind")
        if kind == "metadata":
            # Only the root metadata; subsequent metadata is invalid but
            # we copy-through defensively.
            if i == 0:
                continue
            copy = dict(record)
            copy["parent"] = last_parent
            copy["ts"] = _now_iso()
            copy["id"] = _core.content_id(copy["payload"])
            last_parent = str(copy["id"])
            out.append(copy)
        elif kind == "chat_request":
            input_count += 1
            req = {
                "version": _core.SPEC_VERSION,
                "id": _core.content_id(record["payload"]),
                "kind": "chat_request",
                "ts": _now_iso(),
                "parent": last_parent,
                "payload": dict(record["payload"]),
            }
            out.append(req)
            try:
                response_payload = await backend.complete(record["payload"])
                resp = {
                    "version": _core.SPEC_VERSION,
                    "id": _core.content_id(response_payload),
                    "kind": "chat_response",
                    "ts": _now_iso(),
                    "parent": req["id"],
                    "payload": response_payload,
                }
                out.append(resp)
                last_parent = str(resp["id"])
                output_count += 1
            except ShadowBackendError as e:
                error_count += 1
                err_payload = {
                    "source": "llm",
                    "code": "backend_error",
                    "message": str(e),
                    "retriable": False,
                }
                err = {
                    "version": _core.SPEC_VERSION,
                    "id": _core.content_id(err_payload),
                    "kind": "error",
                    "ts": _now_iso(),
                    "parent": req["id"],
                    "payload": err_payload,
                }
                out.append(err)
                last_parent = str(err["id"])
        elif kind == "chat_response":
            # Skip baseline responses; backend produces them.
            continue
        elif kind in ("tool_call", "tool_result", "error"):
            copy = dict(record)
            copy["payload"] = dict(record["payload"])
            copy["parent"] = last_parent
            copy["ts"] = _now_iso()
            copy["id"] = _core.content_id(copy["payload"])
            out.append(copy)
            last_parent = str(copy["id"])
        else:
            # replay_summary or unknown: skip.
            continue

    duration_ms = int((time.perf_counter() - started) * 1000)
    summary_payload = {
        "baseline_trace_id": root["id"],
        "backend_id": backend.id,
        "input_count": input_count,
        "output_count": output_count,
        "error_count": error_count,
        "duration_ms": duration_ms,
    }
    out.append(
        {
            "version": _core.SPEC_VERSION,
            "id": _core.content_id(summary_payload),
            "kind": "replay_summary",
            "ts": _now_iso(),
            "parent": last_parent,
            "payload": summary_payload,
        }
    )
    return out
