"""Behavioral fingerprint extraction.

A D=12 feature vector per agent turn that captures both behavioral
**pattern** features (tool-call shape, stop reason, latency, length)
and **content-aware** features (numeric density, error markers,
text-character length, total argument complexity) so the Hotelling
T² null shifts when content drifts even if tool patterns stay
constant.

  Pattern features (carried from D=8):
    0  tool_call_rate     — log(1 + tool_use_blocks) / log(1 + max_tools)
    1  distinct_tool_frac — distinct tool names / total tool calls (0..1)
    2  stop_end_turn      — 1.0 if stop_reason == "end_turn"
    3  stop_tool_use      — 1.0 if stop_reason == "tool_use"
    4  stop_other         — 1.0 if stop_reason is anything else
    5  output_len_log     — log(output_tokens+1) / log(token_scale+1)
    6  latency_log        — log(latency_ms+1)   / log(latency_scale+1)
    7  refusal_flag       — 1.0 if stop_reason == "content_filter"

  Content-aware features (added v2.7+):
    8  text_chars_log     — log(text_char_count+1) / log(char_scale+1).
                            Distinct from output_tokens — long
                            reasoning-token responses can have small
                            text payloads, and vice versa.
    9  arg_keys_total_log — log(total_arg_keys_across_tools+1) /
                            log(arg_keys_scale+1). Proxy for
                            tool-call complexity beyond just count.
   10  error_token_flag   — 1.0 if response text contains any of
                            ``error``, ``exception``, ``failed``,
                            ``cannot``, ``unable``, ``not found``
                            (case-insensitive). Surfaces "agent
                            started saying it can't do things" as a
                            content shift even when refusal_flag is 0.
   11  numeric_token_density — fraction of whitespace-split tokens
                            in the response text that parse as numbers.
                            Catches "agent stopped producing
                            calculation output" or "agent started
                            citing fabricated dollar amounts."

All log-scaled features are bounded to [0, 1] so that no single axis
dominates the Hotelling T² statistic.  Scales are configurable via
:class:`FingerprintConfig`.

Why content-aware features matter: the prior D=8 fingerprint detected
behavioral pattern shifts (tools called differently, latency changed,
refusal rate moved) but was blind to "the agent started lying with
the same tool pattern, latency, and refusal rate." The new dimensions
fold light-weight content signals into the Hotelling T² so a content
regression with unchanged behavioral patterns still moves the joint
test statistic.

Note on ``tool_call_rate``: this is a log-scaled count of tool_use
blocks in the response, *not* a rate per content block.

Coverage cross-references
-------------------------
What the fingerprint catches (under Hotelling T² on D=12):

- Tool-call rate / shape / refusal / latency / cost shifts (D=8 base).
- Text-length and arg-complexity drift (text_chars_log,
  arg_keys_total_log).
- "Agent started saying it can't / errored" without changing
  stop_reason (error_token_flag).
- Numeric-density shift between tabular and prose responses
  (numeric_token_density).

What the fingerprint does NOT catch:

- Subtle paraphrase drift with similar length and density — the
  Rust semantic axis covers this with TF-IDF cosine, and a neural
  ``Embedder`` plugged into ``shadow_core::diff::semantic::compute_with_embedder``
  catches paraphrase-only content shifts.
- Tool-argument VALUE drift (e.g. ``delete_user(id="alice")`` →
  ``delete_user(id="bob")``) — the Rust trajectory axis covers
  this via the v2.7+ value digest in tokens; the alignment module
  catches the first divergence with W_ARGS=0.20.
- Sequential-policy violations (``verify_user`` before
  ``issue_refund``) — encoded as LTLf ``must_call_before`` rules
  in :mod:`shadow.ltl`.
- Domain-specific harm rubrics — :mod:`shadow.judge` with an
  LLM-as-judge rubric is the only surface that grades against a
  task-specific quality bar.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

DIM = 12  # fingerprint dimension (8 pattern + 4 content-aware)


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
    char_scale: int = 16_384
    """Text characters at which text_chars_log saturates to 1.0.
    Default tuned for typical chat responses; raise for long-form output."""
    arg_keys_scale: int = 32
    """Total arg keys across all tool calls at which arg_keys_total_log
    saturates. Default is generous for multi-step agents that fan out
    many tool calls per turn."""


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

# Tokens that often signal "agent is reporting an error / inability to
# proceed" without triggering the provider's content_filter stop reason.
# Lowercased substring match; the bar is "appears in the response text"
# because the rate-of-occurrence is the signal, not the exact wording.
_ERROR_TOKENS: tuple[str, ...] = (
    "error",
    "exception",
    "failed",
    "cannot",
    "unable",
    "not found",
    "no such",
    "i don't have",
    "i can't access",
)


@dataclass
class BehavioralVector:
    """D=12 behavioral fingerprint for one response turn."""

    # Pattern features
    tool_call_rate: float
    distinct_tool_frac: float
    stop_end_turn: float
    stop_tool_use: float
    stop_other: float
    output_len_log: float
    latency_log: float
    refusal_flag: float
    # Content-aware features
    text_chars_log: float
    arg_keys_total_log: float
    error_token_flag: float
    numeric_token_density: float

    def to_array(self) -> NDArray[np.float64]:
        return np.asarray(astuple(self), dtype=np.float64)


def _extract_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "\n".join(parts)


def _numeric_token_density(text: str) -> float:
    """Fraction of whitespace-split tokens that parse as numbers.

    Numbers include integers, floats, signed values, and currency-style
    figures (after stripping trailing/leading punctuation). Empty text
    returns 0.0 — the caller can distinguish "no text" from "no
    numbers" via the text_chars_log dimension.
    """
    if not text:
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = 0
    for raw in tokens:
        cleaned = raw.strip(".,;:!?()[]\"'$%")
        if not cleaned:
            continue
        try:
            float(cleaned)
            numeric += 1
        except ValueError:
            continue
    return numeric / len(tokens)


def _error_token_flag(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    for tok in _ERROR_TOKENS:
        if tok in lower:
            return 1.0
    return 0.0


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

    # Content-aware features.
    text = _extract_text(content)
    text_chars_log = _log_scale(float(len(text)), float(config.char_scale))

    arg_keys_total = 0
    for block in tool_blocks:
        input_obj = block.get("input")
        if isinstance(input_obj, dict):
            arg_keys_total += len(input_obj)
    arg_keys_total_log = _log_scale(float(arg_keys_total), float(config.arg_keys_scale))

    error_token_flag = _error_token_flag(text)
    numeric_density = _numeric_token_density(text)

    return BehavioralVector(
        tool_call_rate=tool_call_rate,
        distinct_tool_frac=distinct_tool_frac,
        stop_end_turn=stop_end_turn,
        stop_tool_use=stop_tool_use,
        stop_other=stop_other,
        output_len_log=output_len_log,
        latency_log=latency_log,
        refusal_flag=refusal_flag,
        text_chars_log=text_chars_log,
        arg_keys_total_log=arg_keys_total_log,
        error_token_flag=error_token_flag,
        numeric_token_density=numeric_density,
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
