"""``ToolBackend`` protocol + supporting dataclasses.

Mirrors :class:`shadow.llm.base.LlmBackend` exactly: single async
method, stable id property, runtime-checkable Protocol so the engine
can ``isinstance(backend, ToolBackend)`` for plugin validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from shadow import _core


@dataclass(frozen=True)
class ToolCall:
    """One agent-issued tool invocation.

    Mirrors the ``tool_use`` content block shape Shadow records:
    ``{type: "tool_use", id, name, input}``. Kept as a frozen
    dataclass so it can hash + index cleanly.

    Attributes
    ----------
    id
        Tool-call identifier the model assigned. Used to correlate
        the call with its eventual ``tool_result``. Provider-defined
        opaque string (Anthropic uses ``toolu_...``, OpenAI uses
        ``call_...``); shadow doesn't interpret it.
    name
        Tool function name as the model invoked it. Maps 1:1 to a
        Python callable in the user's tool registry.
    arguments
        The arguments dict the model emitted (already JSON-decoded).
        For matching purposes we hash a canonical-JSON form via
        :func:`canonical_args_hash` so semantically identical args
        with different key ordering match.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_block(cls, block: dict[str, Any]) -> ToolCall:
        """Build from a Shadow ``tool_use`` content block."""
        return cls(
            id=str(block.get("id") or ""),
            name=str(block.get("name") or ""),
            arguments=dict(block.get("input") or {}),
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ToolCall:
        """Build from a Shadow ``tool_call`` record (the standalone form)."""
        payload = record.get("payload") or {}
        return cls(
            id=str(payload.get("tool_call_id") or ""),
            name=str(payload.get("tool_name") or ""),
            arguments=dict(payload.get("arguments") or {}),
        )

    def signature_hash(self) -> str:
        """Stable hash over (name, canonical args). The pairing key
        the replay backend uses for indexed lookup."""
        return canonical_args_hash(self.name, self.arguments)


@dataclass(frozen=True)
class ToolResult:
    """The return of a tool execution, in Shadow's record shape.

    Attributes
    ----------
    tool_call_id
        Matches the originating ``ToolCall.id`` so renderers and the
        diff engine can splice them back together.
    output
        The tool's return value. Strings flow through verbatim; rich
        types (dict / list / object) are JSON-stringified by the
        runtime when the result is recorded.
    is_error
        True iff the tool raised. The next agent turn sees the
        stringified exception in ``output``.
    latency_ms
        Wall-clock cost of the call, integer milliseconds. Zero is
        the convention for "instantly resolved from cache."
    """

    tool_call_id: str
    output: str | dict[str, Any]
    is_error: bool = False
    latency_ms: int = 0

    def to_record_payload(self) -> dict[str, Any]:
        """Serialise as a ``tool_result`` record payload."""
        return {
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "is_error": self.is_error,
            "latency_ms": self.latency_ms,
        }


@runtime_checkable
class ToolBackend(Protocol):
    """An async backend that resolves agent tool calls into results.

    Implementations:
    - :class:`~shadow.tools.replay.ReplayToolBackend` — recorded-result lookup
    - :class:`~shadow.tools.sandbox.SandboxedToolBackend` — wraps real
      tool functions with side-effect isolation
    - :class:`~shadow.tools.stub.StubToolBackend` — deterministic mock
    """

    async def execute(self, call: ToolCall) -> ToolResult:
        """Map a :class:`ToolCall` to a :class:`ToolResult`."""
        ...

    @property
    def id(self) -> str:
        """Stable identifier, e.g. ``"replay"`` / ``"sandbox"``."""
        ...


def canonical_args_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable content hash of ``(tool_name, canonical-JSON(arguments))``.

    Uses Shadow's existing canonical-JSON machinery (sorted keys, no
    extra whitespace, RFC-8259 numbers) so two semantically identical
    argument dicts produce the same hash regardless of insertion order.
    The ``tool_name`` is folded in so different tools never collide
    even on identical args.

    Returns the same content-id format Shadow uses elsewhere
    (``sha256:<hex>``) so a hash that surfaces in a Shadow record is
    immediately recognisable as one.
    """
    payload = {"tool_name": tool_name, "arguments": arguments}
    result = _core.content_id(payload)
    if isinstance(result, str):
        return result
    return str(result)


__all__ = [
    "ToolBackend",
    "ToolCall",
    "ToolResult",
    "canonical_args_hash",
]
