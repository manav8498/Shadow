"""Python-side replay engines. Mirrors `shadow_core::replay::engine::run_replay`.

Kept in Python (instead of exposed through PyO3) because it's async and
uses the Python `LlmBackend` protocol — the Rust trait object can't be
called back into from PyO3 without ceremony. Semantics match SPEC §10.

Two engines live here:

- `run_replay` — re-sends every baseline chat_request through the
  backend, collecting fresh responses. The workhorse.

- `run_partial_replay` — branch-point variant. Copies the baseline
  prefix through verbatim, then switches to live replay at a specified
  turn index. Useful for asking "if the agent diverged at turn 3, what
  would turns 3+ look like under config B holding turns 0-2 fixed?"
  — a targeted experiment the full replay can't isolate.
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


async def run_partial_replay(
    baseline: list[dict[str, Any]],
    branch_at: int,
    backend: LlmBackend,
) -> list[dict[str, Any]]:
    """Replay a baseline trace starting at `branch_at`, prefix locked to baseline.

    The suffix (turns `branch_at..`) is sent fresh through `backend`; the
    prefix (turns `0..branch_at-1`) is copied from baseline verbatim and
    used as locked-in context. The resulting `.agentlog` is valid for
    Shadow's differ — it will align turn-by-turn against the original
    baseline and only show divergence from `branch_at` onward.

    `branch_at` indexes **chat_response turns**, 0-based. Passing
    `branch_at=0` is equivalent to `run_replay` (fully live). Passing
    `branch_at >= len(baseline_turns)` produces a verbatim baseline copy
    with no live calls — useful as a no-op benchmark.

    Semantics:
    - Every baseline record with index < the chat_request that starts
      turn `branch_at` is emitted verbatim (with fresh `ts` + re-linked
      `parent` chain, but payload + id are preserved for content-address
      stability).
    - At turn `branch_at` onward, the engine behaves like `run_replay`:
      re-emit request, send to backend, emit fresh response (or error).
    - A `replay_summary` with `branch_at` + `prefix_turn_count` trails.
    """
    if not baseline:
        raise ShadowParseError("baseline trace is empty — need at least a metadata root")
    if branch_at < 0:
        raise ShadowParseError(
            f"branch_at must be >= 0; got {branch_at}.\n"
            "hint: pass 0 for fully-live replay (same as `shadow replay`)."
        )
    root = baseline[0]
    if root.get("kind") != "metadata":
        raise ShadowParseError(
            f"baseline root is {root.get('kind')!r}, expected 'metadata' (SPEC §3.3)"
        )

    # Build an ordered list of baseline chat_request records so we know
    # which index corresponds to the Kth turn. `run_replay` indexes
    # chat_request/chat_response pairs; we do the same.
    turn_positions: list[int] = [
        i for i, r in enumerate(baseline) if r.get("kind") == "chat_request"
    ]
    if branch_at > len(turn_positions):
        branch_at = len(turn_positions)  # clamp — replay everything from baseline

    out: list[dict[str, Any]] = []
    started = time.perf_counter()
    prefix_turn_count = 0
    input_count = 0
    output_count = 0
    error_count = 0

    # 1. New metadata root pointing back at the baseline root, tagged
    #    with the branch point so downstream consumers can reason about
    #    where live execution started.
    new_meta_payload = {
        **root["payload"],
        "baseline_of": root["id"],
        "partial_replay": {"branch_at": int(branch_at)},
    }
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

    # 2. Walk baseline records. Before the branch point, copy verbatim
    #    (preserving original content-address but re-threading parents).
    #    After, switch to live mode.
    branch_rec_index: int = (
        turn_positions[branch_at] if branch_at < len(turn_positions) else len(baseline)
    )
    turn_seen = -1

    for i, record in enumerate(baseline):
        if i == 0:
            continue  # metadata root already emitted
        kind = record.get("kind")
        if kind == "metadata":
            # Secondary metadata (rare but legal): carry through verbatim.
            copy = {**record, "parent": last_parent, "ts": _now_iso()}
            out.append(copy)
            last_parent = str(copy["id"])
            continue

        if i < branch_rec_index:
            # PREFIX — verbatim copy, preserving id so content-addressing
            # is consistent between runs of the same branch_at. Only
            # `parent` + `ts` are refreshed.
            copy = {**record, "parent": last_parent, "ts": _now_iso()}
            out.append(copy)
            last_parent = str(copy["id"])
            if kind == "chat_request":
                input_count += 1
            elif kind == "chat_response":
                prefix_turn_count += 1
                output_count += 1
            continue

        # SUFFIX — live replay, matching `run_replay` semantics.
        if kind == "chat_request":
            turn_seen += 1
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
            continue  # backend produced a fresh one for this turn
        elif kind in ("tool_call", "tool_result", "error"):
            copy = {**record, "parent": last_parent, "ts": _now_iso()}
            out.append(copy)
            last_parent = str(copy["id"])

    duration_ms = int((time.perf_counter() - started) * 1000)
    summary_payload = {
        "baseline_trace_id": root["id"],
        "backend_id": backend.id,
        "input_count": input_count,
        "output_count": output_count,
        "error_count": error_count,
        "duration_ms": duration_ms,
        "branch_at": int(branch_at),
        "prefix_turn_count": int(prefix_turn_count),
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
