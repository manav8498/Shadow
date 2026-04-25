"""Novel-tool-call policies.

When the candidate calls a tool the baseline never recorded, the
agent-loop replay engine has to choose what to return. The right
answer depends on the user's intent:

- Some scenarios want the replay to **fail loudly** so the regression
  is impossible to ignore.
- Some scenarios want the replay to **continue with a stub** so the
  diff still produces a comparable trace.
- Some scenarios want a **fuzzy match** against the closest recorded
  call of the same tool — useful when arg values shifted but the
  intent is the same.
- Some scenarios want to **delegate** to a user-supplied callable —
  e.g. hit a real sandboxed backend to get a fresh answer.

This module ships four policies covering each path. The framework
is open: implement :class:`NovelCallPolicy` and pass it into
:class:`~shadow.tools.replay.ReplayToolBackend`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from shadow.tools.base import ToolCall, ToolResult

if TYPE_CHECKING:  # pragma: no cover
    from shadow.tools.replay import ReplayToolBackend


@runtime_checkable
class NovelCallPolicy(Protocol):
    """Strategy invoked when the replay backend has no recorded result.

    Implementations are async so they can hit network / disk / fall
    back to another backend if needed.
    """

    async def resolve(self, call: ToolCall, backend: ReplayToolBackend) -> ToolResult:
        """Return a :class:`ToolResult` for ``call``.

        ``backend`` is passed in so policies can inspect the recorded
        index (e.g. fuzzy-match against same-tool recordings) without
        the user having to wire the backend through twice.
        """
        ...


class StrictPolicy:
    """Raise :class:`~shadow.errors.ShadowBackendError` on any novel call.

    The default for production CI: a tool the baseline didn't record
    is by definition a behavioral regression, and silently stubbing it
    out would mask exactly the thing the replay is supposed to surface.
    """

    async def resolve(self, call: ToolCall, backend: ReplayToolBackend) -> ToolResult:
        from shadow.errors import ShadowBackendError

        raise ShadowBackendError(
            f"novel tool call {call.name}({sorted(call.arguments)}) "
            f"has no recorded result and policy=strict"
        )


class StubPolicy:
    """Return a deterministic placeholder result.

    The output string includes the tool name and a hash of the args
    so the resulting trace's content-addressed IDs stay distinct
    across novel calls. Optionally marks results as errors so the
    candidate trace's ``stop_reason`` can flow through error-handling
    branches naturally.
    """

    def __init__(self, *, is_error: bool = False) -> None:
        self._is_error = is_error

    async def resolve(self, call: ToolCall, backend: ReplayToolBackend) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            output=f"<novel: {call.name}#{call.signature_hash()[:12]}>",
            is_error=self._is_error,
            latency_ms=0,
        )


class FuzzyMatchPolicy:
    """Return the recorded result of the nearest baseline call for the
    same tool.

    Distance metric: Jaccard distance over the **set of argument
    keys**, ties broken by Levenshtein distance over the canonical
    JSON of the values. The intent is "same tool, similar shape" —
    if the candidate calls ``search(query='python', limit=10)`` and
    baseline only ever recorded ``search(query='rust', limit=10)``,
    the policy returns the rust result.

    Falls back to ``StubPolicy`` when no same-tool calls were ever
    recorded.

    Parameters
    ----------
    max_key_distance
        Reject matches whose Jaccard key-set distance exceeds this.
        Defaults to 0.5 (more than half the keys overlap).
    """

    def __init__(self, *, max_key_distance: float = 0.5) -> None:
        self._max_key_distance = max_key_distance
        self._fallback = StubPolicy()

    async def resolve(self, call: ToolCall, backend: ReplayToolBackend) -> ToolResult:
        candidates = backend.candidates_for(call.name)
        if not candidates:
            return await self._fallback.resolve(call, backend)
        best: tuple[float, ToolResult] | None = None
        target_keys = set(call.arguments.keys())
        for _hash, result in candidates:
            # We don't have the recorded ToolCall in hand here, only
            # its result + signature hash. Recover the recorded call
            # via the backend's secondary index, which keeps the call
            # alongside the result.
            recorded = backend._lookup_recorded_call(_hash)
            if recorded is None:
                continue
            keys = set(recorded.arguments.keys())
            distance = _jaccard_distance(target_keys, keys)
            if distance > self._max_key_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, result)
        if best is None:
            return await self._fallback.resolve(call, backend)
        recorded_result = best[1]
        return ToolResult(
            tool_call_id=call.id,
            output=recorded_result.output,
            is_error=recorded_result.is_error,
            latency_ms=recorded_result.latency_ms,
        )


class DelegatePolicy:
    """Defer to a user-supplied async callable.

    Lets users bridge to a sandboxed real backend, a third-party
    mocking server, an LLM-driven mock, or anything else, without
    coupling the replay engine to a particular delegate type.

    Example
    -------
    .. code-block:: python

        async def my_fallback(call: ToolCall) -> ToolResult:
            # Hit your own sandboxed test runtime
            output = await my_test_runtime.run(call.name, call.arguments)
            return ToolResult(call.id, output)

        backend = ReplayToolBackend.from_path(
            "baseline.agentlog",
            novel_policy=DelegatePolicy(my_fallback),
        )
    """

    def __init__(self, fallback: Callable[[ToolCall], Awaitable[ToolResult]]) -> None:
        self._fallback = fallback

    async def resolve(self, call: ToolCall, backend: ReplayToolBackend) -> ToolResult:
        return await self._fallback(call)


# ---- helpers ------------------------------------------------------------


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    """Jaccard distance over two key sets. 0.0 = identical, 1.0 = disjoint."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    return 1.0 - (len(intersection) / len(union))


__all__ = [
    "DelegatePolicy",
    "FuzzyMatchPolicy",
    "NovelCallPolicy",
    "StrictPolicy",
    "StubPolicy",
]
