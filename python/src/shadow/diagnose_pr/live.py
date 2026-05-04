"""Live OpenAI backend for `shadow diagnose-pr`.

Wraps `shadow.causal.replay.openai_replayer.OpenAIReplayer` as a
`replay_fn` matching the contract expected by `causal_from_replay`.

Two surfaces:

  1. `build_live_replay_fn(...)` — single-anchor (one baseline
     trace). Used by tests + simple-corpus runs.

  2. `build_live_replay_fn_per_corpus(traces, max_cost_usd=)` —
     **per-trace** anchored replay. Each baseline trace gets its
     own replayer; the corpus-level divergence vector is the mean
     across all per-trace divergence vectors. Includes a
     `CostTracker` that estimates spend per call against a v1
     pricing table and aborts at `max_cost_usd` so a runaway
     bootstrap doesn't burn money.

Key handling: never accepts `OPENAI_API_KEY` as a parameter. Reads
from the environment (consistent with the underlying replayer's
strict no-secrets-in-stack-frames model).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shadow.diagnose_pr.loaders import LoadedTrace

_DEFAULT_MODEL = "gpt-4o-mini"
"""Default model for the live backend. Cheap + capable enough for
behavior-diff tests. Override per-config via the `model` key."""


# v1 pricing table (USD per million tokens, input / output).
# Sourced from OpenAI's published rates; keep tight — adjust when
# OpenAI changes pricing. Numbers err on the high side so the
# --max-cost cap is conservative (won't surprise-overrun).
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
_PRICING_FALLBACK = (0.50, 2.00)
"""Fallback (input, output) USD/MTok when the model isn't in the
table. Conservative-high so unknown models trigger the cap sooner."""


@dataclass
class CostTracker:
    """Cumulative live-backend spend tracker.

    `breakdown` records per-call spend so the report can surface
    where the budget went. Mutates in place; raise via the
    `record()` method when the running total exceeds `max_usd`.
    """

    max_usd: float | None = None
    total_usd: float = 0.0
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0

    def record(self, *, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record one API call's spend. Raises RuntimeError if the
        new total would exceed `max_usd`. Always appends to
        `breakdown` for transparency in the report."""
        in_rate, out_rate = _PRICING_USD_PER_MTOK.get(model, _PRICING_FALLBACK)
        cost = (input_tokens / 1_000_000.0) * in_rate + (
            output_tokens / 1_000_000.0
        ) * out_rate
        self.calls += 1
        self.total_usd += cost
        self.breakdown.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )
        if self.max_usd is not None and self.total_usd > self.max_usd:
            raise RuntimeError(
                f"--max-cost ${self.max_usd:.2f} exceeded "
                f"(spent ${self.total_usd:.4f} across {self.calls} calls). "
                "Reduce --n-bootstrap or pin --max-traces lower."
            )


def build_live_replay_fn(
    *,
    baseline_user_prompt: str,
    baseline_response_text: str,
    baseline_tool_calls: list[str] | None = None,
    baseline_stop_reason: str = "stop",
    baseline_latency_ms: float = 1000.0,
    baseline_output_tokens: int = 100,
    replayer_factory: Callable[..., Any] | None = None,
    cost_tracker: CostTracker | None = None,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Build a `replay_fn(config) -> divergence` for diagnose-pr's
    `--backend live` path against a SINGLE baseline anchor.

    For multi-trace corpora, prefer
    `build_live_replay_fn_per_corpus` which runs a per-trace
    replayer and aggregates.

    `replayer_factory` is for testing — defaults to the real
    `OpenAIReplayer` (requires `OPENAI_API_KEY`).
    """
    if replayer_factory is None:
        from shadow.causal.replay.openai_replayer import OpenAIReplayer

        replayer_factory = OpenAIReplayer

    replayer = replayer_factory(
        baseline_response_text=baseline_response_text,
        baseline_tool_calls=baseline_tool_calls,
        baseline_stop_reason=baseline_stop_reason,
        baseline_latency_ms=baseline_latency_ms,
        baseline_output_tokens=baseline_output_tokens,
    )

    def _replay(flat_config: dict[str, Any]) -> dict[str, float]:
        translated = {
            "system_prompt": str(flat_config.get("prompt.system", "")),
            "user_prompt": baseline_user_prompt,
            "model": str(flat_config.get("model", _DEFAULT_MODEL)),
            "temperature": float(flat_config.get("params.temperature", 0.0)),
        }
        result = replayer(translated)
        if cost_tracker is not None and not getattr(result, "cached", False):
            cost_tracker.record(
                model=translated["model"],
                input_tokens=len(translated["system_prompt"]) // 4
                + len(translated["user_prompt"]) // 4,
                output_tokens=getattr(result, "output_tokens", 100) or 100,
            )
        div = result.divergence
        return {k: float(v) for k, v in div.items()}

    return _replay


def _extract_anchor(records: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (user_message, response_text) from the first chat
    pair in a parsed .agentlog. Empty strings for unrecorded fields.
    """
    user = ""
    resp = ""
    for rec in records:
        if rec.get("kind") == "chat_request":
            msgs = rec.get("payload", {}).get("messages") or []
            for m in msgs:
                if m.get("role") == "user":
                    user = str(m.get("content", ""))
                    break
            if user:
                break
    for rec in records:
        if rec.get("kind") == "chat_response":
            content = rec.get("payload", {}).get("content") or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    resp = str(block.get("text", ""))
                    break
            if resp:
                break
    return user, resp


def build_live_replay_fn_per_corpus(
    *,
    baseline_traces: list[LoadedTrace],
    max_cost_usd: float | None = None,
    replayer_factory: Callable[..., Any] | None = None,
) -> tuple[Callable[[dict[str, Any]], dict[str, float]], CostTracker]:
    """Build a replay_fn that runs the live OpenAIReplayer
    **per baseline trace** and returns the mean per-axis divergence
    across the corpus.

    Each baseline trace gets its own anchored replayer. For a given
    candidate-config perturbation, we invoke every per-trace
    replayer with the same flat-config + per-trace user prompt,
    then mean the divergence vectors.

    Returns (replay_fn, cost_tracker). The cost tracker is shared
    across all per-trace replayers so the cap applies to total
    spend across the corpus.
    """
    if replayer_factory is None:
        from shadow.causal.replay.openai_replayer import OpenAIReplayer

        replayer_factory = OpenAIReplayer

    if not baseline_traces:
        raise ValueError("build_live_replay_fn_per_corpus needs at least one trace")

    cost = CostTracker(max_usd=max_cost_usd)

    # One replayer per trace — each anchored to its own (user_prompt,
    # response_text) pair. We share `cost` so the cap applies to the
    # whole corpus's spend.
    per_trace = []
    for t in baseline_traces:
        user, resp = _extract_anchor(t.records)
        if not user:
            # Skip traces with no user message — the live replayer
            # has nothing to send. Logged at the boundary; v1 keeps
            # the rest of the corpus running rather than failing.
            continue
        per_trace.append(
            {
                "trace_id": t.trace_id,
                "user_prompt": user,
                "replayer": replayer_factory(
                    baseline_response_text=resp,
                    baseline_tool_calls=None,
                    baseline_stop_reason="stop",
                    baseline_latency_ms=1000.0,
                    baseline_output_tokens=100,
                ),
            }
        )
    if not per_trace:
        raise RuntimeError(
            "no baseline trace had an extractable user message — "
            "live replay needs at least one chat_request with role=user"
        )

    def _replay(flat_config: dict[str, Any]) -> dict[str, float]:
        # Aggregate = mean of per-trace divergence vectors.
        sums = {
            "semantic": 0.0,
            "trajectory": 0.0,
            "safety": 0.0,
            "verbosity": 0.0,
            "latency": 0.0,
        }
        n = 0
        model = str(flat_config.get("model", _DEFAULT_MODEL))
        temperature = float(flat_config.get("params.temperature", 0.0))
        system_prompt = str(flat_config.get("prompt.system", ""))
        for entry in per_trace:
            translated = {
                "system_prompt": system_prompt,
                "user_prompt": entry["user_prompt"],
                "model": model,
                "temperature": temperature,
            }
            result = entry["replayer"](translated)
            if not getattr(result, "cached", False):
                cost.record(
                    model=model,
                    input_tokens=(len(system_prompt) + len(entry["user_prompt"])) // 4,
                    output_tokens=getattr(result, "output_tokens", 100) or 100,
                )
            for ax, val in result.divergence.items():
                sums[ax] += float(val)
            n += 1
        if n == 0:
            return sums
        return {k: v / n for k, v in sums.items()}

    return _replay, cost


__all__ = [
    "CostTracker",
    "build_live_replay_fn",
    "build_live_replay_fn_per_corpus",
]
