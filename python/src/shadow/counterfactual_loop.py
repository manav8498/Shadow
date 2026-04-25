"""Tool-loop counterfactual primitives (complement to :mod:`shadow.counterfactual`).

The existing :mod:`shadow.counterfactual` answers "what if we'd
applied delta X to every LLM call?" — config-level counterfactuals.
This module adds the tool-loop level: surgical "what-ifs" that pin a
single tool result, swap a single tool call's args, or branch the
agent loop at a chosen turn under an override. Together they form
the "confirm with `shadow replay`" rails that the bisect renderer's
caveat references.

All three primitives produce ordinary ``.agentlog`` records (Shadow's
content-addressed envelope), so every existing command (``diff``,
``check-policy``, ``mine``, ``mcp-serve``, ``bisect``) reads
counterfactual outputs without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shadow import _core
from shadow.errors import ShadowConfigError
from shadow.llm.base import LlmBackend
from shadow.replay_loop import (
    AgentLoopConfig,
    AgentLoopSummary,
    drive_loop_forward,
    run_agent_loop_replay,
)
from shadow.tools.base import ToolBackend, ToolCall
from shadow.tools.replay import ReplayToolBackend


@dataclass
class CounterfactualLoopResult:
    """Output of a tool-loop counterfactual replay.

    The trace is a real ``.agentlog`` (parseable, content-addressed,
    pipe-able into ``shadow diff``); ``summary`` carries the agent-loop
    statistics; ``override`` echoes the substitution that produced
    this run, so a downstream renderer can label the comparison
    "baseline vs. counterfactual where `<override>`."
    """

    trace: list[dict[str, Any]]
    summary: AgentLoopSummary
    override: dict[str, Any]


# ---- branch-at-turn ------------------------------------------------------


async def branch_at_turn(
    baseline: list[dict[str, Any]],
    *,
    turn: int,
    llm_backend: LlmBackend,
    tool_backend: ToolBackend,
    config: AgentLoopConfig | None = None,
) -> CounterfactualLoopResult:
    """Replay the baseline up through ``turn`` verbatim, then drive the
    agent loop forward against the supplied backends.

    The output trace contains:

    1. A re-emitted root metadata pointing back at ``baseline``.
    2. Verbatim copies of the first ``turn`` chat pairs (and any
       interleaved tool / metadata records) — content IDs preserved
       so the prefix is bit-identical to the baseline's.
    3. The forward-driven continuation from turn ``turn+1`` onward,
       seeded with the messages the agent would have seen at that
       point. The LLM backend gets fresh requests; the tool backend
       resolves whatever tool calls the candidate decides to make.
    4. A trailing replay-summary metadata.

    ``turn=0`` is a full forward-drive (equivalent to
    :func:`run_agent_loop_replay`).

    The LLM backend must be able to answer the post-prefix request
    the engine constructs — that's the messages array of the
    baseline's ``turn``-th chat_request, which a content-id-keyed
    :class:`~shadow.llm.MockLLM` can serve as long as the prefix is
    intact. For "what if the agent kept going past where the
    baseline stopped?" use a live backend or a positional mock.
    """
    if turn < 0:
        raise ShadowConfigError(f"turn must be >= 0, got {turn}")
    if not baseline or baseline[0].get("kind") != "metadata":
        raise ShadowConfigError("baseline must start with a metadata record")
    config = config or AgentLoopConfig()

    out: list[dict[str, Any]] = []
    summary = AgentLoopSummary()

    # 1. Root metadata with provenance.
    root = baseline[0]
    new_meta_payload = dict(root["payload"])
    new_meta_payload["baseline_of"] = root["id"]
    new_meta_payload["replay"] = {
        "engine": "agent_loop_branch",
        "branch_at_turn": turn,
        "llm_backend": getattr(llm_backend, "id", "unknown"),
        "tool_backend": getattr(tool_backend, "id", "unknown"),
    }
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(_envelope("metadata", new_meta_id, parent=None, payload=new_meta_payload))
    last_parent = new_meta_id

    if turn == 0:
        # No prefix — drive forward from the baseline's first chat_request.
        seed_request = _first_chat_request(baseline)
        if seed_request is None:
            return CounterfactualLoopResult(
                trace=out,
                summary=summary,
                override={"kind": "branch_at_turn", "turn": 0},
            )
        forward, stats, _ = await drive_loop_forward(
            seed_messages=seed_request.get("messages") or [],
            seed_model=str(seed_request.get("model") or ""),
            seed_params=seed_request.get("params") or {},
            seed_tools=seed_request.get("tools"),
            parent_id=last_parent,
            llm_backend=llm_backend,
            tool_backend=tool_backend,
            config=config,
        )
        out.extend(forward)
        _accumulate(summary, stats)
        return CounterfactualLoopResult(
            trace=out,
            summary=summary,
            override={"kind": "branch_at_turn", "turn": 0},
        )

    # 2. Copy the prefix verbatim (records up to and including turn N's response).
    prefix_records, post_prefix_request = _slice_prefix_at_turn(baseline, turn)
    if not prefix_records:
        raise ShadowConfigError(
            f"baseline has fewer than {turn} chat-response turns; " "cannot branch past the end."
        )
    for rec in prefix_records[1:]:  # skip the metadata; we re-emitted ours
        last_parent = _emit_copy(rec, parent=last_parent, out=out)

    if post_prefix_request is None:
        # Baseline ended exactly at turn N. Nothing to drive forward.
        summary.sessions_replayed = 1
        return CounterfactualLoopResult(
            trace=out,
            summary=summary,
            override={"kind": "branch_at_turn", "turn": turn},
        )

    # 3. Drive forward starting from turn N+1's seed messages.
    seed_messages = post_prefix_request.get("messages") or []
    forward, stats, _ = await drive_loop_forward(
        seed_messages=seed_messages,
        seed_model=str(post_prefix_request.get("model") or ""),
        seed_params=post_prefix_request.get("params") or {},
        seed_tools=post_prefix_request.get("tools"),
        parent_id=last_parent,
        llm_backend=llm_backend,
        tool_backend=tool_backend,
        config=config,
    )
    out.extend(forward)
    _accumulate(summary, stats)

    return CounterfactualLoopResult(
        trace=out,
        summary=summary,
        override={"kind": "branch_at_turn", "turn": turn},
    )


# ---- replace_tool_result -------------------------------------------------


async def replace_tool_result(
    baseline: list[dict[str, Any]],
    *,
    tool_call_id: str,
    new_output: str | dict[str, Any],
    new_is_error: bool = False,
    llm_backend: LlmBackend | None = None,
    config: AgentLoopConfig | None = None,
) -> CounterfactualLoopResult:
    """Produce a counterfactual trace where one tool result is swapped.

    Directly patches the baseline's recorded ``tool_result`` for the
    named call, leaving every other record alone. This is the core
    counterfactual: "what does the trace look like if this tool had
    returned X instead of Y?"

    Two modes:

    - Default (``llm_backend=None``): pure deterministic patch. The
      output trace is identical to the baseline except for the
      patched ``tool_result``. Useful when you want to ask the bisect
      / diff machinery whether the swap alone causes a behavioral
      regression downstream.
    - With an LLM backend: re-drives the agent loop from the patched
      tool result onward. The downstream LLM responses can diverge
      from baseline. Requires a content-aware backend (live API or
      positional mock) since a content-id mock will miss on the
      patched-messages content id.
    """
    target_call = _find_tool_call(baseline, tool_call_id)
    if target_call is None:
        raise ShadowConfigError(f"no tool_call with id {tool_call_id!r} found in baseline")

    if llm_backend is None:
        patched = _patch_tool_result(baseline, tool_call_id, new_output, new_is_error)
        summary = AgentLoopSummary(sessions_replayed=1, total_llm_calls=0, total_tool_calls=1)
        return CounterfactualLoopResult(
            trace=patched,
            summary=summary,
            override={
                "kind": "replace_tool_result",
                "tool_call_id": tool_call_id,
                "tool_name": target_call.name,
                "new_is_error": new_is_error,
                "mode": "patch",
            },
        )

    # Re-drive mode: emit the prefix verbatim through the patched
    # ``tool_result``, then drive the agent loop forward from a seed
    # whose last tool message carries the new output. This is the
    # difference from re-driving from turn 0 — the prefix's content-
    # addressed records are preserved, only the suffix diverges.
    config = config or AgentLoopConfig()
    if not baseline or baseline[0].get("kind") != "metadata":
        raise ShadowConfigError("baseline must start with a metadata record")
    out: list[dict[str, Any]] = []
    summary = AgentLoopSummary()
    root = baseline[0]
    new_meta_payload = dict(root["payload"])
    new_meta_payload["baseline_of"] = root["id"]
    new_meta_payload["replay"] = {
        "engine": "agent_loop_replace_tool_result",
        "tool_call_id": tool_call_id,
        "llm_backend": getattr(llm_backend, "id", "unknown"),
    }
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(_envelope("metadata", new_meta_id, parent=None, payload=new_meta_payload))
    last_parent = new_meta_id

    prefix_records, post_messages, model, params, tools = _slice_prefix_at_tool_result(
        baseline, tool_call_id, new_output, new_is_error
    )
    if prefix_records is None:
        raise ShadowConfigError(f"no tool_result with id {tool_call_id!r} found in baseline")
    for rec in prefix_records[1:]:  # skip metadata; we re-emitted ours
        last_parent = _emit_copy(rec, parent=last_parent, out=out)

    # Forward-drive from the messages-as-of-after-the-patched-tool.
    forward, stats, _ = await drive_loop_forward(
        seed_messages=post_messages,
        seed_model=model,
        seed_params=params,
        seed_tools=tools,
        parent_id=last_parent,
        llm_backend=llm_backend,
        tool_backend=ReplayToolBackend.from_trace(baseline),
        config=config,
    )
    out.extend(forward)
    _accumulate(summary, stats)
    summary.total_tool_calls += 1  # the patched call we baked into the prefix

    return CounterfactualLoopResult(
        trace=out,
        summary=summary,
        override={
            "kind": "replace_tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": target_call.name,
            "new_is_error": new_is_error,
            "mode": "redrive",
        },
    )


# ---- replace_tool_args ---------------------------------------------------


async def replace_tool_args(
    baseline: list[dict[str, Any]],
    *,
    tool_call_id: str,
    new_arguments: dict[str, Any],
    tool_backend: ToolBackend | None = None,
    llm_backend: LlmBackend | None = None,
    config: AgentLoopConfig | None = None,
) -> CounterfactualLoopResult:
    """Counterfactual: same tool, different arguments.

    Where :func:`replace_tool_result` overrides what a known call
    returned, this primitive overrides the *call itself* — as if the
    model had emitted different ``input`` values.

    Three modes:

    - **Default** (no backends): deterministic patch. The output
      trace mirrors the baseline but the named ``tool_call`` /
      ``tool_use`` block carries ``new_arguments``. The paired
      ``tool_result`` is left alone (because we can't know what the
      tool would have returned without re-running it). Useful for
      sensitivity-analysis questions like "did the args of this call
      change between baseline and candidate?"
    - **With ``tool_backend``**: re-dispatches the patched call
      against the supplied backend, replaces the ``tool_result``
      with whatever the backend returns. Pair with
      :class:`~shadow.tools.sandbox.SandboxedToolBackend` to actually
      run the user's tool function on the new args under sandbox.
    - **With both backends**: full re-drive of the agent loop from
      the patched call onward. The downstream LLM responses can
      diverge from baseline. Requires a content-aware LLM backend.
    """
    target_call = _find_tool_call(baseline, tool_call_id)
    if target_call is None:
        raise ShadowConfigError(f"no tool_call with id {tool_call_id!r} found in baseline")

    patched = _patch_tool_arguments(baseline, tool_call_id, new_arguments)

    if tool_backend is None and llm_backend is None:
        # Pure patch mode.
        summary = AgentLoopSummary(sessions_replayed=1, total_tool_calls=1)
        return CounterfactualLoopResult(
            trace=patched,
            summary=summary,
            override={
                "kind": "replace_tool_args",
                "tool_call_id": tool_call_id,
                "tool_name": target_call.name,
                "new_arguments": new_arguments,
                "mode": "patch",
            },
        )

    if tool_backend is not None and llm_backend is None:
        # Re-dispatch the patched call against the tool backend so the
        # paired tool_result reflects what the new args actually return.
        new_call = ToolCall(id=target_call.id, name=target_call.name, arguments=new_arguments)
        new_result = await tool_backend.execute(new_call)
        patched = _patch_tool_result(
            patched,
            tool_call_id,
            new_result.output,
            new_result.is_error,
        )
        summary = AgentLoopSummary(sessions_replayed=1, total_tool_calls=1)
        return CounterfactualLoopResult(
            trace=patched,
            summary=summary,
            override={
                "kind": "replace_tool_args",
                "tool_call_id": tool_call_id,
                "tool_name": target_call.name,
                "new_arguments": new_arguments,
                "mode": "redispatch",
            },
        )

    if tool_backend is None:
        raise ShadowConfigError(
            "replace_tool_args with llm_backend also requires tool_backend "
            "to dispatch the patched call"
        )
    assert llm_backend is not None  # narrowed by earlier branches
    trace, summary = await run_agent_loop_replay(
        patched, llm_backend=llm_backend, tool_backend=tool_backend, config=config
    )
    return CounterfactualLoopResult(
        trace=trace,
        summary=summary,
        override={
            "kind": "replace_tool_args",
            "tool_call_id": tool_call_id,
            "tool_name": target_call.name,
            "new_arguments": new_arguments,
            "mode": "redrive",
        },
    )


# ---- helpers ------------------------------------------------------------


def _envelope(
    kind: str, record_id: str, *, parent: str | None, payload: dict[str, Any]
) -> dict[str, Any]:
    """Shadow record envelope. Uses a fresh timestamp via _now_iso."""
    return {
        "version": _core.SPEC_VERSION,
        "id": record_id,
        "kind": kind,
        "ts": _now_iso(),
        "parent": parent,
        "payload": payload,
    }


def _now_iso() -> str:
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _emit_copy(rec: dict[str, Any], *, parent: str, out: list[dict[str, Any]]) -> str:
    """Re-emit a baseline record verbatim with a fresh parent.

    Content-addressed: ``rec["id"]`` is preserved (it's a hash of the
    payload, which we don't touch), so a downstream consumer reading
    both baseline and the counterfactual can match the prefix records
    by id. Only ``parent`` and ``ts`` get refreshed — everything else
    is the original record.
    """
    copy = dict(rec)
    copy["parent"] = parent
    copy["ts"] = _now_iso()
    out.append(copy)
    return str(copy["id"])


def _first_chat_request(baseline: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the first chat_request payload in a trace, or None."""
    for rec in baseline:
        if rec.get("kind") == "chat_request":
            return rec.get("payload") or {}
    return None


def _accumulate(summary: AgentLoopSummary, stats: Any) -> None:
    """Fold per-session stats into the aggregate summary."""
    summary.sessions_replayed += 1
    summary.total_llm_calls += stats.llm_calls
    summary.total_tool_calls += stats.tool_calls
    summary.total_tool_errors += stats.tool_errors
    if getattr(stats, "truncated", False):
        summary.sessions_truncated += 1


def _slice_prefix_at_turn(
    baseline: list[dict[str, Any]], turn: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (prefix-records, post-prefix request payload).

    The prefix includes every baseline record from index 0 up
    through turn N's ``chat_response`` *and* through any
    ``tool_call`` / ``tool_result`` records that follow it before
    the next ``chat_request``. That way the prefix carries the full
    state the agent had observed at the end of turn N — the LLM's
    response plus the resolved tool round-trips it triggered.

    The second element is the payload of the (``turn``+1)-th
    ``chat_request`` if one exists in baseline, else ``None``.

    If the baseline has fewer than ``turn`` chat_response records,
    returns ``([], None)`` — caller should error out.
    """
    response_count = 0
    response_idx: int | None = None
    for i, rec in enumerate(baseline):
        if rec.get("kind") == "chat_response":
            response_count += 1
            if response_count == turn:
                response_idx = i
                break
    if response_idx is None:
        return [], None
    # Extend the cut past any trailing tool_call / tool_result / error
    # / metadata records, stopping just before the next chat_request.
    cut_idx = response_idx + 1
    while cut_idx < len(baseline) and baseline[cut_idx].get("kind") != "chat_request":
        cut_idx += 1
    prefix = baseline[:cut_idx]
    post = (baseline[cut_idx].get("payload") or {}) if cut_idx < len(baseline) else None
    return prefix, post


def _slice_prefix_at_tool_result(
    baseline: list[dict[str, Any]],
    tool_call_id: str,
    new_output: str | dict[str, Any],
    new_is_error: bool,
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    str,
    dict[str, Any],
    list[dict[str, Any]] | None,
]:
    """Build the prefix + the post-tool seed messages for re-drive mode.

    Walks the baseline forward up to and including the named
    ``tool_result``. Replaces that tool_result with one carrying
    ``new_output`` / ``new_is_error``. Reconstructs the message list
    the agent would see at that point so the engine can drive
    forward.

    Returns ``(prefix, post_messages, model, params, tools)`` or
    ``(None, [], "", {}, None)`` if the named tool_call_id is missing.
    """
    prefix: list[dict[str, Any]] = []
    seen_tool_result = False
    last_request_payload: dict[str, Any] | None = None
    pending_tool_use: dict[str, Any] | None = None  # the tool_use block
    for rec in baseline:
        kind = rec.get("kind")
        if kind == "chat_request":
            last_request_payload = rec.get("payload") or {}
            prefix.append(rec)
        elif kind == "chat_response":
            payload = rec.get("payload") or {}
            for block in payload.get("content") or []:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and str(block.get("id") or "") == tool_call_id
                ):
                    pending_tool_use = block
            prefix.append(rec)
        elif kind == "tool_call":
            payload = rec.get("payload") or {}
            if str(payload.get("tool_call_id") or "") == tool_call_id:
                pending_tool_use = pending_tool_use or {
                    "id": tool_call_id,
                    "name": payload.get("tool_name"),
                    "input": payload.get("arguments") or {},
                }
            prefix.append(rec)
        elif kind == "tool_result":
            payload = rec.get("payload") or {}
            if str(payload.get("tool_call_id") or "") == tool_call_id:
                # Substitute the patched tool_result.
                patched_payload = dict(payload)
                patched_payload["output"] = new_output
                patched_payload["is_error"] = new_is_error
                # Re-content-id the patched record.
                patched_id = _core.content_id(patched_payload)
                prefix.append({**rec, "id": patched_id, "payload": patched_payload})
                seen_tool_result = True
                break
            prefix.append(rec)
        else:
            prefix.append(rec)

    if not seen_tool_result or last_request_payload is None:
        return None, [], "", {}, None

    # Reconstruct what the agent would see at the post-prefix point:
    # everything in last_request_payload.messages, plus the assistant
    # message carrying the tool_use block, plus a tool message with
    # the patched output.
    base_messages = list(last_request_payload.get("messages") or [])
    if pending_tool_use:
        from shadow.tools.base import ToolCall

        call = ToolCall.from_block(pending_tool_use)
        # Mirror what _build_assistant_message would produce.
        import json as _json

        base_messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": _json.dumps(call.arguments, sort_keys=True),
                        },
                    }
                ],
            }
        )
    # Tool message with the patched output. Mirrors _build_tool_message.
    if isinstance(new_output, str):
        content = new_output
    else:
        import json as _json

        try:
            content = _json.dumps(new_output, sort_keys=True, default=str)
        except (TypeError, ValueError):
            content = str(new_output)
    if new_is_error:
        content = f"[tool_error] {content}"
    base_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    return (
        prefix,
        base_messages,
        str(last_request_payload.get("model") or ""),
        dict(last_request_payload.get("params") or {}),
        last_request_payload.get("tools"),
    )


def _split_at_turn(
    baseline: list[dict[str, Any]], turn: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``baseline`` into (prefix, suffix) at the start of turn N.

    A "turn" here is one ``chat_response`` record (the agent
    finishing one round-trip with the LLM). Records that don't bump
    the response count — metadata, tool_call, tool_result, error —
    attach to whichever turn they precede.
    """
    response_count = 0
    split_idx = len(baseline)
    for i, rec in enumerate(baseline):
        if rec.get("kind") == "chat_response":
            response_count += 1
            if response_count == turn:
                split_idx = i + 1
                break
    return baseline[:split_idx], baseline[split_idx:]


def _find_tool_call(baseline: list[dict[str, Any]], tool_call_id: str) -> ToolCall | None:
    """Locate a tool call by id, whether it's a standalone record or a
    ``tool_use`` content block embedded in a ``chat_response``."""
    for rec in baseline:
        kind = rec.get("kind")
        if kind == "tool_call":
            payload = rec.get("payload") or {}
            if str(payload.get("tool_call_id") or "") == tool_call_id:
                return ToolCall.from_record(rec)
        elif kind == "chat_response":
            for block in (rec.get("payload") or {}).get("content") or []:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and str(block.get("id") or "") == tool_call_id
                ):
                    return ToolCall.from_block(block)
    return None


def _patch_tool_result(
    baseline: list[dict[str, Any]],
    tool_call_id: str,
    new_output: str | dict[str, Any],
    new_is_error: bool,
) -> list[dict[str, Any]]:
    """Return a copy of ``baseline`` with the named tool's result
    overridden.

    Walks the trace, replaces the ``tool_result`` whose
    ``tool_call_id`` matches; leaves every other record alone.
    """
    out: list[dict[str, Any]] = []
    for rec in baseline:
        if rec.get("kind") == "tool_result":
            payload = rec.get("payload") or {}
            if str(payload.get("tool_call_id") or "") == tool_call_id:
                new_payload = dict(payload)
                new_payload["output"] = new_output
                new_payload["is_error"] = new_is_error
                out.append({**rec, "payload": new_payload})
                continue
        out.append(rec)
    return out


def _patch_tool_arguments(
    baseline: list[dict[str, Any]],
    tool_call_id: str,
    new_arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a copy of ``baseline`` with the named tool call's args replaced.

    Patches both the standalone ``tool_call`` record (when present)
    and the ``tool_use`` content block embedded in the preceding
    ``chat_response``. Unaffected records pass through by reference;
    we don't deep-copy untouched payloads since the engine only reads
    them.
    """
    out: list[dict[str, Any]] = []
    for rec in baseline:
        kind = rec.get("kind")
        if kind == "tool_call":
            payload = rec.get("payload") or {}
            if str(payload.get("tool_call_id") or "") == tool_call_id:
                new_payload = dict(payload)
                new_payload["arguments"] = dict(new_arguments)
                out.append({**rec, "payload": new_payload})
                continue
        elif kind == "chat_response":
            payload = rec.get("payload") or {}
            content = payload.get("content") or []
            patched_content: list[Any] = []
            patched_any = False
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and str(block.get("id") or "") == tool_call_id
                ):
                    patched_block = dict(block)
                    patched_block["input"] = dict(new_arguments)
                    patched_content.append(patched_block)
                    patched_any = True
                else:
                    patched_content.append(block)
            if patched_any:
                new_payload = dict(payload)
                new_payload["content"] = patched_content
                out.append({**rec, "payload": new_payload})
                continue
        out.append(rec)
    return out


__all__ = [
    "CounterfactualLoopResult",
    "branch_at_turn",
    "replace_tool_args",
    "replace_tool_result",
]
