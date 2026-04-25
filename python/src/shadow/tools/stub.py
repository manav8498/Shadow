"""Deterministic stub tool backend for tests and the ``stub`` novel-call policy.

Always succeeds and returns a fixed ``{"output": "<stub: ...>"}``-shaped
payload. Used in unit tests, in the agent-loop engine when the user
opts into ``--novel-tool-policy stub`` for unrecorded calls, and as a
sanity-check baseline that doesn't require any real tool definitions.
"""

from __future__ import annotations

from shadow.tools.base import ToolBackend, ToolCall, ToolResult


class StubToolBackend(ToolBackend):
    """Returns a deterministic placeholder for every call.

    The output string includes the tool name + canonical args hash so
    the value is unique per call signature, which keeps the resulting
    trace's content-addressed IDs distinct.

    Parameters
    ----------
    backend_id
        Stable identifier surfaced as ``self.id``. Defaults to
        ``"stub"``; tests sometimes pass ``"stub-foo"`` to assert the
        engine threaded the right backend through.
    is_error
        When True, every result has ``is_error=True``. Useful as a
        guard mode where any tool call is considered a regression.
    """

    def __init__(self, *, backend_id: str = "stub", is_error: bool = False) -> None:
        self._id = backend_id
        self._is_error = is_error

    @property
    def id(self) -> str:
        return self._id

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            output=f"<stub: {call.name}#{call.signature_hash()[:12]}>",
            is_error=self._is_error,
            latency_ms=0,
        )


__all__ = ["StubToolBackend"]
