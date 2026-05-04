"""Tests for the `--backend live` wiring.

The live backend wraps `shadow.causal.replay.openai_replayer.
OpenAIReplayer` as a `replay_fn` for `causal_from_replay`. The
existing OpenAIReplayer requires `OPENAI_API_KEY` and an OpenAI
client — we mock those at the boundary so this test runs offline.

The actual end-to-end live test against real OpenAI lives in
`test_diagnose_pr_live_api_e2e.py` and is skipped unless
`SHADOW_RUN_NETWORK_TESTS=1` is set."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest


def test_build_live_replay_fn_translates_flat_keys_to_openai_shape() -> None:
    """The live wrapper must translate diagnose-pr's flat keys
    (e.g. `prompt.system`, `params.temperature`) to the keys
    OpenAIReplayer expects (`system_prompt`, `user_prompt`, `model`,
    `temperature`). Verified by capturing the config the underlying
    replayer sees."""
    from shadow.diagnose_pr.live import build_live_replay_fn

    captured: list[dict[str, Any]] = []

    class _StubReplayer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, config: dict[str, Any]) -> Any:
            captured.append(dict(config))

            class _R:
                divergence: dict[str, float] = {  # noqa: RUF012  # test stub
                    "semantic": 0.1,
                    "trajectory": 0.0,
                    "safety": 0.0,
                    "verbosity": 0.0,
                    "latency": 0.0,
                }

            return _R()

    replay_fn = build_live_replay_fn(
        baseline_user_prompt="Refund #123.",
        baseline_response_text="OK, confirmed.",
        replayer_factory=_StubReplayer,
    )
    flat_config = {
        "model": "gpt-4o-mini",
        "params.temperature": 0.7,
        "prompt.system": "You are a refund agent.",
    }
    div = replay_fn(flat_config)
    assert div == {
        "semantic": 0.1,
        "trajectory": 0.0,
        "safety": 0.0,
        "verbosity": 0.0,
        "latency": 0.0,
    }
    assert len(captured) == 1
    seen = captured[0]
    # Translation contract: flat → OpenAIReplayer keys.
    assert seen["system_prompt"] == "You are a refund agent."
    assert seen["user_prompt"] == "Refund #123."
    assert seen["model"] == "gpt-4o-mini"
    assert seen["temperature"] == pytest.approx(0.7)


def test_build_live_replay_fn_uses_default_model_when_missing() -> None:
    """If the config doesn't supply a model, the live wrapper falls
    back to a v1 default (gpt-4o-mini — cheap, capable enough for
    behavior diff tests)."""
    from shadow.diagnose_pr.live import build_live_replay_fn

    captured: list[dict[str, Any]] = []

    class _StubReplayer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, config: dict[str, Any]) -> Any:
            captured.append(dict(config))

            class _R:
                divergence: dict[str, float] = {  # noqa: RUF012  # test stub
                    "semantic": 0.0,
                    "trajectory": 0.0,
                    "safety": 0.0,
                    "verbosity": 0.0,
                    "latency": 0.0,
                }

            return _R()

    fn = build_live_replay_fn(
        baseline_user_prompt="hi",
        baseline_response_text="hello",
        replayer_factory=_StubReplayer,
    )
    fn({"prompt.system": "Be terse."})
    assert captured[0]["model"] == "gpt-4o-mini"


def test_build_live_replay_fn_raises_when_api_key_missing() -> None:
    """If OPENAI_API_KEY isn't set, the underlying OpenAIReplayer
    raises RuntimeError. The wrapper must surface this clearly so
    the CLI can convert it to a typed error."""
    from shadow.diagnose_pr.live import build_live_replay_fn

    # Real factory will refuse without the key.
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        build_live_replay_fn(
            baseline_user_prompt="x",
            baseline_response_text="y",
        )
