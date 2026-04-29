"""Shared types for the replay backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ReplayResult:
    """One full replay outcome plus the divergence vector callers need.

    Backends populate this; the per-axis ``divergence`` dict is what
    :func:`shadow.causal.causal_attribution` consumes.
    """

    config: dict[str, Any]
    """The config the replay was run against."""
    response_text: str
    """The model's final response text. Empty string if the call
    errored or the response had no text content."""
    tool_calls: list[str]
    """Names of tool_use blocks emitted by the model, in order."""
    stop_reason: str
    """The stop_reason field from the response (or ``"error"`` on
    failure)."""
    latency_ms: float
    """Wall-clock latency of the API call. ``0.0`` for cached returns."""
    output_tokens: int
    """Output tokens reported by the API. ``0`` for errored calls."""
    divergence: dict[str, float]
    """Per-axis divergence vector — what causal_attribution needs.
    Keys are axis names (``"semantic"``, ``"trajectory"``, etc.);
    values are in [0, 1] for rate-like axes or unbounded for value-
    like axes. Backends compute this against a baseline reference
    that the caller supplies.
    """
    cached: bool = False
    """True iff the result came from the in-memory cache rather than
    a live API call."""


class Replayer(Protocol):
    """Replay backend Protocol.

    Implementations are callable: passing a config dict returns a
    :class:`ReplayResult`. The result's ``divergence`` field is the
    only thing :func:`shadow.causal.causal_attribution` needs; the
    other fields are diagnostic.
    """

    def __call__(self, config: dict[str, Any]) -> ReplayResult: ...
