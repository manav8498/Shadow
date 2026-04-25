"""``ReplayToolBackend`` — deterministic recorded-result lookup.

Mirrors :class:`shadow.llm.mock.MockLLM` for tools: build from one or
more baseline traces, index every recorded ``tool_result`` by its
originating tool call's ``(name, canonical_args_hash)``, serve them
back when the agent-loop engine asks for them.

When the candidate calls a tool the baseline never recorded, the
backend delegates to a configurable :class:`NovelCallPolicy`
(see :mod:`shadow.tools.novel`) so users can pick between failing
loud, returning a stub, fuzzy-matching the nearest baseline call,
or escalating to a user-provided fallback.

Performance notes
-----------------

- Indexing is O(n) across the baseline records; lookups are O(1)
  hash table reads.
- The index is built once at construction; ``execute`` only reads it.
- Result objects are returned by reference (frozen dataclass), so a
  long replay against a large baseline doesn't copy result text per
  call.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shadow import _core
from shadow.tools.base import ToolBackend, ToolCall, ToolResult, canonical_args_hash


class ReplayToolBackend(ToolBackend):
    """Indexed recorded-result lookup for the agent-loop replay engine.

    Parameters
    ----------
    results
        Pre-built ``signature_hash → ToolResult`` index. Most users
        construct via :meth:`from_trace` / :meth:`from_traces` /
        :meth:`from_path` rather than passing this directly.
    novel_policy
        What to do when ``execute`` is called with a tool the index
        doesn't know about. ``None`` (default) raises
        :class:`~shadow.errors.ShadowBackendError` for an early loud
        failure; production callers usually pass a
        :class:`~shadow.tools.novel.StubPolicy` /
        :class:`~shadow.tools.novel.FuzzyMatchPolicy` /
        :class:`~shadow.tools.novel.DelegatePolicy`.
    backend_id
        Stable identifier surfaced as ``self.id``.
    """

    def __init__(
        self,
        results: dict[str, ToolResult] | None = None,
        *,
        novel_policy: NovelCallPolicy | None = None,
        backend_id: str = "replay",
        recorded_calls: dict[str, ToolCall] | None = None,
    ) -> None:
        self._results: dict[str, ToolResult] = dict(results or {})
        self._recorded_calls: dict[str, ToolCall] = dict(recorded_calls or {})
        self._novel_policy = novel_policy
        self._id = backend_id
        # Secondary index: tool_name → list[(args_hash, result)].
        # Built from the recorded_calls map so fuzzy-match policies
        # can walk same-tool calls without scanning the whole index.
        self._by_name: dict[str, list[tuple[str, ToolResult]]] = {}
        self._rebuild_secondary_index()

    @property
    def id(self) -> str:
        return self._id

    def __len__(self) -> int:
        return len(self._results)

    def __contains__(self, call: ToolCall) -> bool:
        return call.signature_hash() in self._results

    # ---- construction --------------------------------------------------

    @classmethod
    def from_trace(
        cls,
        trace: list[dict[str, Any]],
        *,
        novel_policy: NovelCallPolicy | None = None,
        backend_id: str = "replay",
    ) -> ReplayToolBackend:
        """Build from a single parsed trace (list of record dicts).

        Walks the trace, pairs every ``tool_call`` record with the
        first subsequent ``tool_result`` that references its
        ``tool_call_id``, and indexes the result by the call's
        canonical signature hash.
        """
        results, calls = _index_trace(trace)
        return cls(
            results=results,
            recorded_calls=calls,
            novel_policy=novel_policy,
            backend_id=backend_id,
        )

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[list[dict[str, Any]]],
        *,
        novel_policy: NovelCallPolicy | None = None,
        backend_id: str = "replay",
    ) -> ReplayToolBackend:
        """Build from multiple traces (later traces win on conflicts)."""
        merged_results: dict[str, ToolResult] = {}
        merged_calls: dict[str, ToolCall] = {}
        for trace in traces:
            results, calls = _index_trace(trace)
            merged_results.update(results)
            merged_calls.update(calls)
        return cls(
            results=merged_results,
            recorded_calls=merged_calls,
            novel_policy=novel_policy,
            backend_id=backend_id,
        )

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        novel_policy: NovelCallPolicy | None = None,
        backend_id: str = "replay",
    ) -> ReplayToolBackend:
        """Load a single ``.agentlog`` file from disk."""
        data = Path(path).read_bytes()
        trace = _core.parse_agentlog(data)
        return cls.from_trace(trace, novel_policy=novel_policy, backend_id=backend_id)

    # ---- execution -----------------------------------------------------

    async def execute(self, call: ToolCall) -> ToolResult:
        sig = call.signature_hash()
        recorded = self._results.get(sig)
        if recorded is not None:
            # Replay records preserve the recorded latency; only the
            # tool_call_id needs to mirror the candidate's id so the
            # downstream pair stitching aligns.
            return ToolResult(
                tool_call_id=call.id,
                output=recorded.output,
                is_error=recorded.is_error,
                latency_ms=recorded.latency_ms,
            )
        if self._novel_policy is None:
            from shadow.errors import ShadowBackendError

            raise ShadowBackendError(
                f"ReplayToolBackend has no recorded result for "
                f"{call.name}({sorted(call.arguments)})\n"
                "hint: pass novel_policy=... or re-record the baseline "
                "with this call shape."
            )
        return await self._novel_policy.resolve(call, self)

    # ---- introspection used by fuzzy / delegate policies ---------------

    def candidates_for(self, tool_name: str) -> list[tuple[str, ToolResult]]:
        """All ``(args_hash, result)`` entries recorded for one tool name.

        Used by :class:`~shadow.tools.novel.FuzzyMatchPolicy` to walk
        the recorded calls when searching for the nearest match by
        arg shape.
        """
        return list(self._by_name.get(tool_name, ()))

    # ---- internals -----------------------------------------------------

    def _rebuild_secondary_index(self) -> None:
        """Group ``(args_hash, result)`` pairs by tool name.

        Fed by :attr:`_recorded_calls`, which the trace-based
        constructors populate. Users who pass raw ``results`` without
        ``recorded_calls`` get an empty by-name index — fuzzy match
        falls back to ``StubPolicy`` for them, which is the right
        behavior given there's no recoverable tool-name mapping from a
        bare result dict.
        """
        self._by_name.clear()
        for sig_hash, call in self._recorded_calls.items():
            result = self._results.get(sig_hash)
            if result is None:
                continue
            self._by_name.setdefault(call.name, []).append((sig_hash, result))

    def _lookup_recorded_call(self, sig_hash: str) -> ToolCall | None:
        """Recover the recorded ``ToolCall`` for a signature hash.

        Used by :class:`~shadow.tools.novel.FuzzyMatchPolicy` to walk
        recorded calls and compute distance against the candidate's
        call without us having to store full ToolCall objects on the
        secondary index.
        """
        return self._recorded_calls.get(sig_hash)


# ---- module-private helpers ----------------------------------------------


def _index_trace(
    trace: list[dict[str, Any]],
) -> tuple[dict[str, ToolResult], dict[str, ToolCall]]:
    """Pair tool_call → tool_result records and index by signature hash.

    Returns ``(results_by_sig, calls_by_sig)``. The two dicts share
    keys; the second exists so fuzzy-match policies can recover the
    recorded ``ToolCall`` for any given signature hash.
    """
    pending_calls: dict[str, ToolCall] = {}
    results: dict[str, ToolResult] = {}
    calls: dict[str, ToolCall] = {}
    for rec in trace:
        kind = rec.get("kind")
        if kind == "tool_call":
            call = ToolCall.from_record(rec)
            if call.id:
                pending_calls[call.id] = call
        elif kind == "chat_response":
            payload = rec.get("payload") or {}
            for block in payload.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    call = ToolCall.from_block(block)
                    if call.id:
                        pending_calls[call.id] = call
        elif kind == "tool_result":
            payload = rec.get("payload") or {}
            tcid = str(payload.get("tool_call_id") or "")
            if not tcid:
                continue
            if tcid not in pending_calls:
                continue
            call = pending_calls.pop(tcid)
            sig = canonical_args_hash(call.name, call.arguments)
            results[sig] = ToolResult(
                tool_call_id=tcid,
                output=payload.get("output", ""),
                is_error=bool(payload.get("is_error", False)),
                latency_ms=int(payload.get("latency_ms", 0) or 0),
            )
            calls[sig] = call
    return results, calls


# ---- forward-declared type for novel-call policy --------------------------


# Imported at the bottom to avoid a circular import: novel policies
# need the backend type for fuzzy lookups, the backend needs the
# policy type for the constructor signature.
from shadow.tools.novel import NovelCallPolicy  # noqa: E402

ReplayToolBackend.__init__.__annotations__["novel_policy"] = "NovelCallPolicy | None"


__all__ = ["ReplayToolBackend"]
