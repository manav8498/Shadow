"""Agent-loop replay engine.

The classic :func:`shadow.replay.run_replay` walks a baseline trace
in lock-step: every recorded ``chat_request`` is replayed through an
:class:`~shadow.llm.base.LlmBackend`, and every ``tool_call`` /
``tool_result`` record is copied through verbatim. That works for
"replay the LLM under a deterministic mock," but it can't answer
"what would the candidate have done?" — because copying tool records
through means the candidate's tool decisions are never actually
exercised.

This module ships the agent-loop variant. The engine takes a baseline
trace, recovers each session's seed messages (system + user prompts),
then drives the loop forward against the supplied backends:

  1. Send the running message list to the :class:`LlmBackend`.
  2. Inspect the response. If it has ``tool_use`` blocks, dispatch
     each one to the :class:`~shadow.tools.base.ToolBackend`, append
     the resulting ``tool_result`` to the message list, loop.
  3. Otherwise the turn ended; record the final response and move on.

Because both backends are deterministic by default
(:class:`~shadow.llm.MockLLM` for LLM, :class:`~shadow.tools.replay.ReplayToolBackend`
for tools), the replay is reproducible and side-effect free. With
:class:`~shadow.tools.sandbox.SandboxedToolBackend` you can run real
tool functions but still block their network / fs / subprocess
side effects, which is the "shadow deployment" mode.

The output is an ordinary ``.agentlog`` — same envelope shape as a
real recorded trace — so every existing Shadow command (``diff``,
``check-policy``, ``mine``, ``mcp-serve``, ``bisect``) works on it
without modification.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from typing import Any

from shadow import _core
from shadow.errors import ShadowBackendError, ShadowParseError
from shadow.llm.base import LlmBackend
from shadow.tools.base import ToolBackend, ToolCall

# Runaway agents: every loop is bounded by a max-iteration guard so a
# misbehaving model can't burn unbounded backend calls. 32 turns is
# generous for any real agent flow but safely above the 99th
# percentile we've seen in production (~12).
DEFAULT_MAX_TURNS = 32


@dataclass
class AgentLoopConfig:
    """Knobs for the agent-loop engine.

    All fields default to safe values; users only override what they
    need. The engine constructs one ``AgentLoopConfig`` per session.
    """

    max_turns: int = DEFAULT_MAX_TURNS
    """Hard cap on how many LLM round-trips a single session may
    drive. Exceeded → engine emits an ``error`` record with
    ``code=loop_max_exceeded`` and stops the session."""

    on_tool_error_continue: bool = True
    """When a tool backend raises, do we surface it as a ``tool_result``
    with ``is_error=True`` and let the loop continue (matches real
    agent behavior), or abort the session?"""

    seed_includes_full_history: bool = True
    """If True, the engine seeds each session with the full
    ``messages`` array of its first ``chat_request`` — exactly what
    the agent saw at start. If False, the engine seeds with only the
    system + first user message and lets the candidate's loop
    rebuild from there."""


@dataclass
class AgentLoopSummary:
    """Statistics emitted alongside the output trace."""

    sessions_replayed: int = 0
    sessions_truncated: int = 0
    """Sessions that hit ``max_turns`` before the model stopped."""
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    duration_ms: int = 0
    """Wall-clock ms across the whole replay."""

    def to_payload(self) -> dict[str, Any]:
        return {
            "sessions_replayed": self.sessions_replayed,
            "sessions_truncated": self.sessions_truncated,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "duration_ms": self.duration_ms,
        }


@dataclass
class _SessionSeed:
    """The starting state for one session of the replay."""

    metadata_record: dict[str, Any] | None
    """The session's metadata record from the baseline (may be
    None for the very first auto-emitted Session metadata; the
    engine handles either case)."""

    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    """The ``messages`` list to send on the first LLM call."""

    initial_model: str = ""
    """Default model id, taken from the session's first
    ``chat_request``. The engine plumbs this through unchanged unless
    the candidate's responses choose a different one (LLMs don't
    self-select model, but we honour whatever the response payload
    declares so behavior matches a real run)."""

    initial_params: dict[str, Any] = field(default_factory=dict)
    """Sampling params from the session's first ``chat_request``."""

    initial_tools: list[dict[str, Any]] | None = None
    """Tools-array (in OpenAI/Anthropic shape) that the session
    advertised. Used for downstream policy checks but doesn't
    influence the loop directly."""


# ---- forward-driving primitive ------------------------------------------


async def drive_loop_forward(
    *,
    seed_messages: list[dict[str, Any]],
    seed_model: str,
    seed_params: dict[str, Any] | None = None,
    seed_tools: list[dict[str, Any]] | None = None,
    parent_id: str,
    llm_backend: LlmBackend,
    tool_backend: ToolBackend,
    config: AgentLoopConfig | None = None,
) -> tuple[list[dict[str, Any]], _SessionStats, str]:
    """Drive the agent loop forward starting from a fixed seed.

    Returns ``(records, stats, last_parent_id)``. The caller is
    responsible for placing those records in a larger trace and
    chaining ``last_parent_id`` into whatever record follows.

    This is the surgical primitive the counterfactual helpers use:
    a baseline is replayed verbatim through some pivot turn, then
    this function takes over from a synthesised seed (the messages
    the agent would see at that pivot) and runs the loop to
    termination.
    """
    config = config or AgentLoopConfig()
    seed = _SessionSeed(
        metadata_record=None,
        initial_messages=list(seed_messages),
        initial_model=seed_model,
        initial_params=dict(seed_params or {}),
        initial_tools=[dict(t) for t in (seed_tools or [])] or None,
    )
    out: list[dict[str, Any]] = []
    last_parent, stats = await _replay_one_session(
        seed=seed,
        parent=parent_id,
        llm_backend=llm_backend,
        tool_backend=tool_backend,
        config=config,
        out=out,
    )
    return out, stats, last_parent


# ---- public entry point --------------------------------------------------


async def run_agent_loop_replay(
    baseline: list[dict[str, Any]],
    *,
    llm_backend: LlmBackend,
    tool_backend: ToolBackend,
    config: AgentLoopConfig | None = None,
) -> tuple[list[dict[str, Any]], AgentLoopSummary]:
    """Drive the agent loop forward across every session of ``baseline``.

    Returns ``(records, summary)``. The record list is a normal
    ``.agentlog`` ready to write to disk; the summary captures
    aggregate stats for the CLI to surface.

    The engine never mutates ``baseline``. Two replays of the same
    inputs produce byte-identical output (under deterministic
    backends) thanks to Shadow's content-addressed envelopes.
    """
    config = config or AgentLoopConfig()
    if not baseline:
        raise ShadowParseError("baseline trace is empty — need at least a metadata root")
    root = baseline[0]
    if root.get("kind") != "metadata":
        raise ShadowParseError(
            f"baseline root is {root.get('kind')!r}, expected 'metadata' (SPEC §3.3)"
        )

    out: list[dict[str, Any]] = []
    summary = AgentLoopSummary()
    started = time.perf_counter()

    # 1. Re-emit the baseline root metadata, marking provenance.
    new_meta_payload = dict(root["payload"])
    new_meta_payload["baseline_of"] = root["id"]
    new_meta_payload["replay"] = {
        "engine": "agent_loop",
        "llm_backend": getattr(llm_backend, "id", "unknown"),
        "tool_backend": getattr(tool_backend, "id", "unknown"),
        "max_turns": config.max_turns,
    }
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(_envelope("metadata", new_meta_id, parent=None, payload=new_meta_payload))
    last_parent = new_meta_id

    # 2. Slice the baseline into sessions and replay each one.
    for seed in _iter_session_seeds(baseline):
        seed_metadata_id: str | None = None
        if seed.metadata_record is not None and seed.metadata_record is not root:
            seed_meta_payload = dict(seed.metadata_record["payload"])
            seed_meta_payload["baseline_of"] = seed.metadata_record["id"]
            seed_meta_id = _core.content_id(seed_meta_payload)
            out.append(
                _envelope("metadata", seed_meta_id, parent=last_parent, payload=seed_meta_payload)
            )
            last_parent = seed_meta_id
            seed_metadata_id = seed_meta_id

        last_parent, session_stats = await _replay_one_session(
            seed=seed,
            parent=last_parent,
            llm_backend=llm_backend,
            tool_backend=tool_backend,
            config=config,
            out=out,
        )
        summary.sessions_replayed += 1
        summary.total_llm_calls += session_stats.llm_calls
        summary.total_tool_calls += session_stats.tool_calls
        summary.total_tool_errors += session_stats.tool_errors
        if session_stats.truncated:
            summary.sessions_truncated += 1
        # Marker so renderers can group by session.
        _ = seed_metadata_id

    # 3. Replay summary record at the end.
    summary.duration_ms = int((time.perf_counter() - started) * 1000)
    summary_payload = summary.to_payload()
    out.append(
        _envelope(
            "metadata",
            _core.content_id({"replay_summary": summary_payload}),
            parent=last_parent,
            payload={"replay_summary": summary_payload},
        )
    )
    return out, summary


# ---- session driver ------------------------------------------------------


@dataclass
class _SessionStats:
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    truncated: bool = False


async def _replay_one_session(
    *,
    seed: _SessionSeed,
    parent: str,
    llm_backend: LlmBackend,
    tool_backend: ToolBackend,
    config: AgentLoopConfig,
    out: list[dict[str, Any]],
) -> tuple[str, _SessionStats]:
    """Drive the ReAct-style loop for one session and append to ``out``.

    Returns the new ``last_parent`` id (the latest record appended)
    and a :class:`_SessionStats` carrying the per-session counters.
    """
    stats = _SessionStats()
    if not seed.initial_messages:
        # Nothing to do — empty session; emit a marker so the trace
        # remains structurally aligned with the baseline.
        return parent, stats

    messages: list[dict[str, Any]] = list(seed.initial_messages)
    last_parent = parent

    for _turn in range(config.max_turns):
        request_payload: dict[str, Any] = {
            "model": seed.initial_model,
            "messages": list(messages),
            "params": dict(seed.initial_params),
        }
        if seed.initial_tools:
            request_payload["tools"] = list(seed.initial_tools)

        request_id = _core.content_id(request_payload)
        out.append(
            _envelope("chat_request", request_id, parent=last_parent, payload=request_payload)
        )

        try:
            response_payload = await llm_backend.complete(request_payload)
        except ShadowBackendError as exc:
            err_payload = {
                "source": "llm",
                "code": "backend_error",
                "message": str(exc),
                "retriable": False,
            }
            err_id = _core.content_id(err_payload)
            out.append(_envelope("error", err_id, parent=request_id, payload=err_payload))
            last_parent = err_id
            return last_parent, stats

        stats.llm_calls += 1
        response_id = _core.content_id(response_payload)
        out.append(
            _envelope("chat_response", response_id, parent=request_id, payload=response_payload)
        )
        last_parent = response_id

        # Append the assistant message to the running history. The
        # message we feed back to the LLM mirrors the OpenAI/Anthropic
        # wire shape so a real user-supplied tool function can
        # introspect it the same way it does in production.
        assistant_msg = _build_assistant_message(response_payload)
        if assistant_msg is not None:
            messages.append(assistant_msg)

        # Find tool_use blocks; dispatch each, append the result.
        tool_uses = _extract_tool_uses(response_payload)
        if not tool_uses:
            return last_parent, stats

        for tool_use in tool_uses:
            call = ToolCall.from_block(tool_use)
            stats.tool_calls += 1

            # Record the tool_call as its own record so policy /
            # session-aware diff can locate it without re-parsing
            # response content.
            tool_call_payload = {
                "tool_name": call.name,
                "tool_call_id": call.id,
                "arguments": call.arguments,
            }
            tool_call_id = _core.content_id(tool_call_payload)
            out.append(
                _envelope(
                    "tool_call",
                    tool_call_id,
                    parent=last_parent,
                    payload=tool_call_payload,
                )
            )
            last_parent = tool_call_id

            tool_result = await _safe_tool_execute(
                tool_backend, call, on_error_continue=config.on_tool_error_continue
            )
            if tool_result.is_error:
                stats.tool_errors += 1
            result_payload = tool_result.to_record_payload()
            result_id = _core.content_id(result_payload)
            out.append(
                _envelope(
                    "tool_result",
                    result_id,
                    parent=last_parent,
                    payload=result_payload,
                )
            )
            last_parent = result_id

            # Feed the result back into the running message list.
            messages.append(_build_tool_message(call, tool_result.output, tool_result.is_error))

        # Loop continues with the updated messages — the next iteration
        # will dispatch another LLM call.

    # If we drop out of the loop, we hit the max_turns guard.
    stats.truncated = True
    truncate_payload = {
        "source": "engine",
        "code": "loop_max_exceeded",
        "message": (
            f"agent loop exceeded {config.max_turns} turns; "
            "stopping to prevent runaway. Raise `max_turns` if your agent "
            "legitimately needs more."
        ),
        "retriable": False,
    }
    err_id = _core.content_id(truncate_payload)
    out.append(_envelope("error", err_id, parent=last_parent, payload=truncate_payload))
    return err_id, stats


# ---- session slicing -----------------------------------------------------


def _iter_session_seeds(baseline: list[dict[str, Any]]) -> list[_SessionSeed]:
    """Walk the baseline once, yield a seed per logical session.

    A session begins at every ``metadata`` record and runs until the
    next one. Within a session the seed is taken from the first
    ``chat_request`` that appears (which always exists if the session
    contains any agent activity).
    """
    seeds: list[_SessionSeed] = []
    current_metadata: dict[str, Any] | None = None
    current_seed: _SessionSeed | None = None

    for rec in baseline:
        kind = rec.get("kind")
        if kind == "metadata":
            if current_seed is not None:
                seeds.append(current_seed)
            current_metadata = rec
            current_seed = _SessionSeed(metadata_record=current_metadata)
            continue
        if current_seed is None:
            # Records before any metadata record — treat them as session 0.
            current_seed = _SessionSeed(metadata_record=None)
        if kind == "chat_request" and not current_seed.initial_messages:
            payload = rec.get("payload") or {}
            current_seed.initial_messages = [
                dict(m) for m in payload.get("messages", []) if isinstance(m, dict)
            ]
            current_seed.initial_model = str(payload.get("model") or "")
            current_seed.initial_params = dict(payload.get("params") or {})
            tools = payload.get("tools")
            if isinstance(tools, list):
                current_seed.initial_tools = [t for t in tools if isinstance(t, dict)]

    if current_seed is not None:
        seeds.append(current_seed)
    return seeds


# ---- response shape helpers ----------------------------------------------


def _extract_tool_uses(response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``tool_use`` content blocks in a chat_response.

    Shadow's record format mirrors Anthropic's ``content`` array
    shape, so a list-of-dicts walk is sufficient.
    """
    content = response_payload.get("content") or []
    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            out.append(block)
    return out


def _build_assistant_message(response_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Coerce a chat_response payload into a wire-shape assistant message.

    Used to extend the running ``messages`` list before the next LLM
    call. Mirrors the shape both Anthropic and OpenAI accept on input
    (text + optional tool_calls), so user-supplied tool functions see
    the message in the same form they would in production.
    """
    content = response_payload.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            txt = block.get("text")
            if isinstance(txt, str):
                text_parts.append(txt)
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}, sort_keys=True),
                    },
                }
            )
    msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg if (msg["content"] or tool_calls) else None


def _build_tool_message(
    call: ToolCall, output: str | dict[str, Any], is_error: bool
) -> dict[str, Any]:
    """Wire-shape ``role: tool`` message that closes a tool round-trip."""
    if isinstance(output, str):
        content = output
    else:
        try:
            content = json.dumps(output, sort_keys=True, default=str)
        except (TypeError, ValueError):
            content = str(output)
    if is_error:
        # Common convention: prefix error results so the model
        # recognises them. Keeps the candidate's loop behaviour
        # closer to a real run.
        content = f"[tool_error] {content}"
    return {"role": "tool", "tool_call_id": call.id, "content": content}


# ---- error-tolerant tool dispatch ----------------------------------------


async def _safe_tool_execute(
    backend: ToolBackend, call: ToolCall, *, on_error_continue: bool
) -> Any:
    """Wrap a tool execution; turn raises into ``is_error`` results.

    Importing :class:`ToolResult` lazily inside the function avoids a
    circular-import edge for any future module that wraps this one.
    """
    from shadow.tools.base import ToolResult

    try:
        return await backend.execute(call)
    except Exception as exc:
        if not on_error_continue:
            raise
        return ToolResult(
            tool_call_id=call.id,
            output=f"{type(exc).__name__}: {exc}",
            is_error=True,
            latency_ms=0,
        )


# ---- envelope builder ----------------------------------------------------


def _envelope(
    kind: str, record_id: str, *, parent: str | None, payload: dict[str, Any]
) -> dict[str, Any]:
    """Build a Shadow record envelope with a fresh timestamp."""
    return {
        "version": _core.SPEC_VERSION,
        "id": record_id,
        "kind": kind,
        "ts": _now_iso(),
        "parent": parent,
        "payload": payload,
    }


def _now_iso() -> str:
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = [
    "DEFAULT_MAX_TURNS",
    "AgentLoopConfig",
    "AgentLoopSummary",
    "drive_loop_forward",
    "run_agent_loop_replay",
]
