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

from shadow.errors import ShadowConfigError
from shadow.llm.base import LlmBackend
from shadow.replay_loop import (
    AgentLoopConfig,
    AgentLoopSummary,
    run_agent_loop_replay,
)
from shadow.tools.base import ToolBackend, ToolCall, ToolResult
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
    """Replay the baseline up through ``turn`` then drive the loop forward.

    The first ``turn`` chat-pairs are taken as a baseline prefix; the
    agent-loop engine then takes over and completes the trace under
    the supplied backends. Useful for sensitivity analysis: "if the
    agent diverged at turn 5, what would turns 5..N look like?"

    The LLM backend must be able to answer requests the engine
    constructs from the prefix's tail — typically a live backend or
    a positional mock keyed by turn rather than a content-id mock.
    """
    if turn < 0:
        raise ShadowConfigError(f"turn must be >= 0, got {turn}")

    prefix, _suffix = _split_at_turn(baseline, turn)
    if not prefix:
        # turn=0 means a full replay.
        trace, summary = await run_agent_loop_replay(
            baseline,
            llm_backend=llm_backend,
            tool_backend=tool_backend,
            config=config,
        )
        return CounterfactualLoopResult(
            trace=trace,
            summary=summary,
            override={"kind": "branch_at_turn", "turn": 0},
        )

    trace, summary = await run_agent_loop_replay(
        prefix,
        llm_backend=llm_backend,
        tool_backend=tool_backend,
        config=config,
    )
    return CounterfactualLoopResult(
        trace=trace,
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

    tool_backend = ReplayToolBackend.from_trace(baseline)
    sig = target_call.signature_hash()
    tool_backend._results[sig] = ToolResult(
        tool_call_id=target_call.id,
        output=new_output,
        is_error=new_is_error,
        latency_ms=0,
    )
    tool_backend._rebuild_secondary_index()
    trace, summary = await run_agent_loop_replay(
        baseline, llm_backend=llm_backend, tool_backend=tool_backend, config=config
    )
    return CounterfactualLoopResult(
        trace=trace,
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
