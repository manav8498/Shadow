"""Live OpenAI replay backend for causal attribution.

Calls the OpenAI Chat Completions API with deterministic seeding,
exponential-backoff retry on rate limits, and per-config caching
keyed by a SHA-256 hash of the canonical-JSON config. The cache
makes re-running the same attribution free after the first pass.

API key
-------
Read **only** from the ``OPENAI_API_KEY`` environment variable. The
class refuses to accept a key as a constructor parameter so the key
cannot leak into call-site stack frames, log lines, or test
fixtures. Set the env var in your shell or a gitignored ``.env``
file before instantiating.

Determinism
-----------
The OpenAI API supports a ``seed`` parameter (best-effort
determinism — same seed + same prompt + same model usually
returns the same response, modulo provider-side cache eviction).
We pass a stable seed derived from the config hash so two runs of
the same config get the same response, on top of the in-memory
cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from shadow.causal.replay.types import ReplayResult


def _canonical_config_hash(config: dict[str, Any]) -> str:
    """Stable hash for cache keying. Sorts keys and serialises to UTF-8
    canonical JSON before hashing — same logical config produces the
    same hash regardless of dict iteration order."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class OpenAIReplayerConfig:
    """Tunables for the live OpenAI replayer."""

    model: str = "gpt-4o-mini"
    """Default model. Callers can override per-config by setting a
    ``model`` key in the config dict."""
    temperature: float = 0.0
    """Default sampling temperature. ``0.0`` plus the seed gives the
    most deterministic output the API can deliver."""
    max_tokens: int = 512
    """Output token cap per call."""
    max_retries: int = 5
    """Retry attempts on transient errors (rate limit, 5xx)."""
    initial_backoff_s: float = 1.0
    """Initial sleep on retry; doubles each attempt."""
    timeout_s: float = 30.0
    """Per-call timeout."""


class OpenAIReplayer:
    """Live OpenAI replayer backend.

    Use ``OpenAIReplayer(baseline_response_text=...)`` to anchor
    divergence computation against a recorded baseline. The replayer
    sends each candidate config to the API, computes per-axis
    divergence vs the baseline, and caches the response by config
    hash so repeat calls cost nothing.
    """

    def __init__(
        self,
        *,
        baseline_response_text: str,
        baseline_tool_calls: list[str] | None = None,
        baseline_stop_reason: str = "stop",
        baseline_latency_ms: float = 1000.0,
        baseline_output_tokens: int = 100,
        replayer_config: OpenAIReplayerConfig | None = None,
    ) -> None:
        # Refuse to even look at an api_key parameter — keep secrets
        # out of stack frames. The OpenAI client picks the env var up
        # automatically on construction.
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError(
                "OPENAI_API_KEY env var is not set. "
                "OpenAIReplayer never accepts the key as a parameter; "
                "export it in your shell or load a gitignored .env "
                "file before instantiating."
            )
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed. "
                "pip install 'shadow-diff[openai]' or 'pip install openai'."
            ) from e

        self._client = OpenAI()
        self._cfg = replayer_config or OpenAIReplayerConfig()
        self._baseline_text = baseline_response_text
        self._baseline_tool_calls = list(baseline_tool_calls or [])
        self._baseline_stop_reason = baseline_stop_reason
        self._baseline_latency_ms = baseline_latency_ms
        self._baseline_output_tokens = baseline_output_tokens

        self._cache: dict[str, ReplayResult] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def __call__(self, config: dict[str, Any]) -> ReplayResult:
        """Replay one config and return the per-axis divergence."""
        key = _canonical_config_hash(config)
        cached = self._cache.get(key)
        if cached is not None:
            # Return a copy with cached=True flag set.
            return ReplayResult(
                config=cached.config,
                response_text=cached.response_text,
                tool_calls=cached.tool_calls,
                stop_reason=cached.stop_reason,
                latency_ms=0.0,
                output_tokens=cached.output_tokens,
                divergence=dict(cached.divergence),
                cached=True,
            )

        # Build messages from config. Required keys: system_prompt
        # and user_prompt. model is optional; falls back to
        # OpenAIReplayerConfig.model.
        system_prompt = str(config.get("system_prompt", ""))
        user_prompt = str(config.get("user_prompt", ""))
        model = str(config.get("model", self._cfg.model))
        temperature = float(config.get("temperature", self._cfg.temperature))

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Deterministic seed derived from the config hash. Stable
        # across runs: same config → same seed → same response (on a
        # best-effort basis from the provider).
        seed = int(key[:8], 16) % (2**31)

        result = self._call_with_retry(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )
        self._cache[key] = result
        return result

    def _call_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        seed: int,
    ) -> ReplayResult:
        """Call the API with exponential-backoff retry on transient
        errors, then convert the response to a ReplayResult."""
        from openai import (  # type: ignore[import-not-found]
            APIError,
            APITimeoutError,
            RateLimitError,
        )

        backoff = self._cfg.initial_backoff_s
        last_error: Exception | None = None
        for _attempt in range(self._cfg.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=self._cfg.max_tokens,
                    seed=seed,
                    timeout=self._cfg.timeout_s,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return self._build_result(resp, latency_ms, original_config={})
            except (RateLimitError, APITimeoutError) as e:
                last_error = e
                time.sleep(backoff)
                backoff *= 2.0
            except APIError as e:
                # Server errors (5xx) — retry. Client errors (4xx) — break.
                if getattr(e, "status_code", 0) >= 500:
                    last_error = e
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    last_error = e
                    break

        # All retries exhausted — return an error-shaped result with
        # divergence vector saturated so the caller surfaces the
        # failure as a regression rather than silently failing
        # closed.
        return ReplayResult(
            config={},
            response_text=f"[error: {last_error}]" if last_error else "[error]",
            tool_calls=[],
            stop_reason="error",
            latency_ms=0.0,
            output_tokens=0,
            divergence={
                "semantic": 1.0,
                "trajectory": 1.0,
                "safety": 1.0,
                "verbosity": 1.0,
                "latency": 1.0,
            },
            cached=False,
        )

    def _build_result(
        self, resp: Any, latency_ms: float, original_config: dict[str, Any]
    ) -> ReplayResult:
        choice = resp.choices[0]
        msg = choice.message
        response_text = msg.content or ""
        # Extract tool_calls if present (OpenAI returns them on the
        # message in the v2 SDK).
        tool_calls: list[str] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls or []:
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None)
                    if isinstance(name, str):
                        tool_calls.append(name)
        stop_reason = choice.finish_reason or "stop"
        usage = resp.usage
        output_tokens = int(getattr(usage, "completion_tokens", 0))

        divergence = self._compute_divergence(
            response_text=response_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
        )
        return ReplayResult(
            config=original_config,
            response_text=response_text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
            divergence=divergence,
            cached=False,
        )

    def _compute_divergence(
        self,
        *,
        response_text: str,
        tool_calls: list[str],
        stop_reason: str,
        latency_ms: float,
        output_tokens: int,
    ) -> dict[str, float]:
        """Per-axis divergence vector vs the baseline.

        Lightweight, fast, deterministic divergence computation:
          * semantic   — 1 - jaccard(baseline_tokens, candidate_tokens)
                         on whitespace-tokenised lowercased text.
          * trajectory — Levenshtein-normalised on tool_calls.
          * safety     — 1.0 if stop_reason transitions to
                         ``content_filter``, else 0.0.
          * verbosity  — relative output-token delta.
          * latency    — relative latency delta.

        For deeper analysis, callers should pipe the returned
        ``response_text`` / ``tool_calls`` through the Rust diff
        module's full nine-axis pipeline.
        """
        # Semantic via jaccard
        b_tokens = set(self._baseline_text.lower().split())
        c_tokens = set(response_text.lower().split())
        if not b_tokens and not c_tokens:
            sem = 0.0
        elif not b_tokens or not c_tokens:
            sem = 1.0
        else:
            inter = len(b_tokens & c_tokens)
            union = len(b_tokens | c_tokens)
            sem = 1.0 - (inter / union if union else 0.0)

        # Trajectory via simple Levenshtein on tool sequences
        traj = _normalised_levenshtein(self._baseline_tool_calls, tool_calls)

        # Safety: did the candidate get content-filtered when the
        # baseline didn't, or vice versa?
        safety = (
            1.0
            if (stop_reason == "content_filter") != (self._baseline_stop_reason == "content_filter")
            else 0.0
        )

        # Verbosity / latency: relative deltas
        verbosity = _relative_delta(output_tokens, self._baseline_output_tokens)
        latency = _relative_delta(latency_ms, self._baseline_latency_ms)

        return {
            "semantic": float(min(1.0, sem)),
            "trajectory": float(min(1.0, traj)),
            "safety": float(safety),
            "verbosity": float(verbosity),
            "latency": float(latency),
        }


def _normalised_levenshtein(a: list[str], b: list[str]) -> float:
    """Levenshtein distance / max(len(a), len(b)). Returns 0 for two
    empty lists."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n] / max(m, n)


def _relative_delta(candidate: float, baseline: float) -> float:
    """|c - b| / max(b, 1.0), clipped to [0, 1]. ``baseline`` of zero
    falls back to denominator 1.0 so we don't divide by zero."""
    denom = max(abs(baseline), 1.0)
    return min(1.0, abs(candidate - baseline) / denom)
