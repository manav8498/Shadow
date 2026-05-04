"""Live OpenAI backend for `shadow diagnose-pr`.

Wraps `shadow.causal.replay.openai_replayer.OpenAIReplayer` as a
`replay_fn` matching the contract expected by `causal_from_replay`.

The wrapper handles two translations:

  1. **Key shape**: diagnose-pr uses flat dotted-path keys
     (`prompt.system`, `params.temperature`); OpenAIReplayer expects
     `system_prompt` / `user_prompt` / `model` / `temperature`.

  2. **Baseline anchoring**: OpenAIReplayer is constructed with a
     fixed baseline response. The CLI extracts a representative
     baseline trace, pulls its user prompt + response text, and
     hands them to `build_live_replay_fn`.

API key handling: never accepts the key as a parameter. Reads
`OPENAI_API_KEY` from the environment (consistent with the
underlying replayer's strict no-secrets-in-stack-frames model).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_DEFAULT_MODEL = "gpt-4o-mini"
"""Default model for the live backend. Cheap + capable enough for
behavior-diff tests. Override per-config via the `model` key."""


def build_live_replay_fn(
    *,
    baseline_user_prompt: str,
    baseline_response_text: str,
    baseline_tool_calls: list[str] | None = None,
    baseline_stop_reason: str = "stop",
    baseline_latency_ms: float = 1000.0,
    baseline_output_tokens: int = 100,
    replayer_factory: Callable[..., Any] | None = None,
) -> Callable[[dict[str, Any]], dict[str, float]]:
    """Build a `replay_fn(config) -> divergence` for diagnose-pr's
    `--backend live` path.

    `replayer_factory` is for testing — defaults to the real
    `OpenAIReplayer`, which requires `OPENAI_API_KEY`. Tests pass a
    stub that captures calls instead.
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
        # Translate diagnose-pr's flat dotted keys to the keys
        # OpenAIReplayer reads. Anything not recognized is dropped —
        # OpenAIReplayer ignores extra keys, but explicitly mapping
        # only what we use makes the contract obvious here.
        translated = {
            "system_prompt": str(flat_config.get("prompt.system", "")),
            "user_prompt": baseline_user_prompt,
            "model": str(flat_config.get("model", _DEFAULT_MODEL)),
            "temperature": float(flat_config.get("params.temperature", 0.0)),
        }
        result = replayer(translated)
        # OpenAIReplayer returns a ReplayResult dataclass; we want
        # just the 5-axis divergence dict.
        div = result.divergence
        # Defensive copy with float values so the caller can't mutate
        # internal state.
        return {k: float(v) for k, v in div.items()}

    return _replay


__all__ = ["build_live_replay_fn"]
