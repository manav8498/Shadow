"""Behavioral fingerprint extraction.

A D=8 feature vector per agent turn, designed to capture behavioral
patterns that the nine-axis tests measure *individually* but not
*jointly*:

  0  tool_call_rate     — log(1 + tool_use_blocks) / log(1 + max_tools), clipped [0,1]
  1  distinct_tool_frac — distinct tool names / total tool calls (0..1)
  2  stop_end_turn      — 1.0 if stop_reason == "end_turn"
  3  stop_tool_use      — 1.0 if stop_reason == "tool_use"
  4  stop_other         — 1.0 if stop_reason is anything else
  5  output_len_log     — log(output_tokens+1) / log(token_scale+1), clipped
  6  latency_log        — log(latency_ms+1)   / log(latency_scale+1), clipped
  7  refusal_flag       — 1.0 if stop_reason == "content_filter"

All log-scaled features are bounded to [0, 1] so that no single axis
dominates the Hotelling T² statistic.  Scales are configurable via
:class:`FingerprintConfig` so that long-context models (``token_scale``)
and slow reasoning models (``latency_scale``) can be fingerprinted
without saturating to 1.0.

Note on ``tool_call_rate``: this is a log-scaled count of tool_use
blocks in the response, *not* a rate per content block.  The previous
linear cap at 1.0 was insensitive to differences between agents that
fan out 1 vs 5 tool calls per turn; the log transform recovers that
discrimination while keeping the dimension bounded.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

DIM = 8  # fingerprint dimension


@dataclass(frozen=True)
class FingerprintConfig:
    """Configuration for behavioral fingerprint scaling.

    The defaults target current-generation Anthropic / OpenAI agents.
    Override for environments with very long contexts or extended-
    thinking latencies (e.g. ``token_scale=200_000`` for 1M-context
    runs, ``latency_scale=120_000`` for thinking models).
    """

    token_scale: int = 4096
    """Output tokens at which the verbosity dimension saturates to 1.0."""
    latency_scale_ms: int = 30_000
    """Latency in ms at which the latency dimension saturates to 1.0."""
    max_tool_calls: int = 8
    """Tool-use blocks per response at which tool_call_rate saturates to 1.0."""


DEFAULT_CONFIG = FingerprintConfig()


def _log_scale(value: float, scale: float) -> float:
    """log(1 + value) / log(1 + scale), clipped to [0, 1]."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log(value + 1.0) / math.log(scale + 1.0))


_STOP_END_TURN = "end_turn"
_STOP_TOOL_USE = "tool_use"
_STOP_CONTENT_FILTER = "content_filter"
_KNOWN_STOPS = (_STOP_END_TURN, _STOP_TOOL_USE, _STOP_CONTENT_FILTER, "")


@dataclass
class BehavioralVector:
    """D=8 behavioral fingerprint for one response turn."""

    tool_call_rate: float
    distinct_tool_frac: float
    stop_end_turn: float
    stop_tool_use: float
    stop_other: float
    output_len_log: float
    latency_log: float
    refusal_flag: float

    def to_array(self) -> NDArray[np.float64]:
        return np.asarray(astuple(self), dtype=np.float64)


def _extract_response_vector(
    payload: dict[str, Any], config: FingerprintConfig
) -> BehavioralVector:
    """Feature-vector for one chat_response payload."""
    content = payload.get("content") or []
    tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    tool_count = len(tool_blocks)
    tool_names = {str(b.get("name") or "") for b in tool_blocks}

    # Log-scaled tool count: distinguishes 1 vs 5 vs 10 tool calls per turn
    # while remaining bounded.  At max_tool_calls the dimension saturates.
    tool_call_rate = _log_scale(tool_count, config.max_tool_calls)
    distinct_tool_frac = (len(tool_names) / tool_count) if tool_count > 0 else 0.0

    stop = str(payload.get("stop_reason") or "")
    stop_end_turn = 1.0 if stop == _STOP_END_TURN else 0.0
    stop_tool_use = 1.0 if stop == _STOP_TOOL_USE else 0.0
    refusal_flag = 1.0 if stop == _STOP_CONTENT_FILTER else 0.0
    stop_other = 1.0 if stop not in _KNOWN_STOPS else 0.0

    usage = payload.get("usage") or {}
    out_tokens = int(usage.get("output_tokens") or 0)
    output_len_log = _log_scale(float(out_tokens), float(config.token_scale))

    latency_ms = float(payload.get("latency_ms") or 0.0)
    latency_log = _log_scale(latency_ms, float(config.latency_scale_ms))

    return BehavioralVector(
        tool_call_rate=tool_call_rate,
        distinct_tool_frac=distinct_tool_frac,
        stop_end_turn=stop_end_turn,
        stop_tool_use=stop_tool_use,
        stop_other=stop_other,
        output_len_log=output_len_log,
        latency_log=latency_log,
        refusal_flag=refusal_flag,
    )


def fingerprint_trace(
    records: list[dict[str, Any]],
    config: FingerprintConfig | None = None,
) -> NDArray[np.float64]:
    """Return a ``(n_responses, DIM)`` fingerprint matrix for a trace.

    Rows correspond to chat_response records in order. Returns a
    ``(0, DIM)`` matrix when the trace has no chat_response records.

    Pass a custom :class:`FingerprintConfig` to adjust scales for
    long-context or thinking-mode agents.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    rows: list[NDArray[np.float64]] = []
    for rec in records:
        if rec.get("kind") != "chat_response":
            continue
        payload = rec.get("payload") or {}
        rows.append(_extract_response_vector(payload, cfg).to_array())
    if not rows:
        return np.empty((0, DIM), dtype=np.float64)
    return np.stack(rows, axis=0)


def mean_fingerprint(
    records: list[dict[str, Any]],
    config: FingerprintConfig | None = None,
) -> NDArray[np.float64]:
    """Trace-level mean fingerprint (D-dim vector).

    Returns a zero vector when the trace has no responses — the caller
    should check ``n == 0`` and skip the test rather than passing a
    zero vector as a valid observation.
    """
    mat = fingerprint_trace(records, config)
    if mat.shape[0] == 0:
        return np.zeros(DIM, dtype=np.float64)
    mean: NDArray[np.float64] = mat.mean(axis=0)
    return mean


__all__ = [
    "DEFAULT_CONFIG",
    "DIM",
    "BehavioralVector",
    "FingerprintConfig",
    "fingerprint_trace",
    "mean_fingerprint",
]
