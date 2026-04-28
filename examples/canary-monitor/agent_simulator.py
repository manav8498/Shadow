"""Synthetic transfer-agent trace generator.

Produces realistic .agentlog records for a financial transfer agent
with three controllable behavior profiles:

  - ``"baseline"``: well-calibrated reference model.  Latencies are
    log-normal around ~600 ms, output ~150 tokens, always calls
    ``verify_user`` before ``transfer_funds``.

  - ``"drifting"``: like baseline but ~3× slower latency on average.
    Realistic regression after a model upgrade with longer reasoning
    chains.  Should be caught by mSPRT on the latency axis.

  - ``"policy_violating"``: mostly fine but occasionally calls
    ``transfer_funds`` without first calling ``verify_user`` — the
    safety bug we want LTL to catch.

The generator is deterministic given a seed so test fixtures are
reproducible without committing 100s of files.
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

Profile = Literal["baseline", "drifting", "policy_violating"]


_TS_BASE = "2026-04-27T10:00:00.000Z"
_NULL_ID = "sha256:" + "0" * 64


def _record(
    kind: str,
    payload: dict[str, Any] | None,
    idx: int,
    parent: str | None = None,
) -> dict[str, Any]:
    return {
        "version": "0.1",
        "id": f"sha256:{idx:064x}",
        "kind": kind,
        "ts": _TS_BASE,
        "parent": parent or _NULL_ID,
        "meta": {"session_tag": "synthetic"},
        "payload": payload,
    }


def _log_normal(rng: random.Random, mean_log: float, sigma_log: float) -> float:
    return math.exp(rng.gauss(mean_log, sigma_log))


def _make_response(
    rng: random.Random,
    idx: int,
    *,
    tools: list[str],
    text: str,
    stop_reason: str,
    output_tokens: int,
    latency_ms: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for i, tname in enumerate(tools):
        content.append(
            {
                "type": "tool_use",
                "id": f"t{idx}_{i}",
                "name": tname,
                "input": {},
            }
        )
    payload = {
        "model": "transfer-agent-v1",
        "content": content,
        "stop_reason": stop_reason,
        "latency_ms": float(latency_ms),
        "usage": {
            "input_tokens": 200,
            "output_tokens": int(output_tokens),
            "thinking_tokens": 0,
        },
    }
    return _record("chat_response", payload, idx)


def generate_session(
    profile: Profile,
    seed: int,
    *,
    inject_policy_violation: bool = False,
) -> list[dict[str, Any]]:
    """Return a single multi-turn session as a list of agentlog records.

    Sessions are 3 turns: lookup → confirm → transfer.  Profiles change
    the latency / cost distribution and (optionally) the tool order.
    """
    rng = random.Random(seed)

    if profile == "baseline":
        lat_mean, lat_sigma = math.log(600), 0.2
        tok_mean = 150
    elif profile == "drifting":
        # ~1.5× latency on average — small enough that the always-valid
        # property of mSPRT matters (a fixed-N test would need much more
        # data to detect this effect size reliably).
        lat_mean, lat_sigma = math.log(900), 0.22
        tok_mean = 150
    elif profile == "policy_violating":
        lat_mean, lat_sigma = math.log(600), 0.2
        tok_mean = 150
    else:
        raise ValueError(f"unknown profile: {profile!r}")

    records: list[dict[str, Any]] = []
    base_idx = seed * 100

    # Turn 1: greeting + verify_user (UNLESS this is a policy-violating session).
    if profile == "policy_violating" and inject_policy_violation:
        # Skip verify_user — go straight to a transfer in turn 2.
        records.append(
            _make_response(
                rng,
                base_idx + 1,
                tools=[],
                text="Hello, ready to help.",
                stop_reason="end_turn",
                output_tokens=int(rng.gauss(tok_mean, 30)),
                latency_ms=_log_normal(rng, lat_mean, lat_sigma),
            )
        )
    else:
        records.append(
            _make_response(
                rng,
                base_idx + 1,
                tools=["verify_user"],
                text="I'll verify your identity first.",
                stop_reason="tool_use",
                output_tokens=int(rng.gauss(tok_mean, 30)),
                latency_ms=_log_normal(rng, lat_mean, lat_sigma),
            )
        )

    # Turn 2: confirmation
    records.append(
        _make_response(
            rng,
            base_idx + 2,
            tools=["check_balance"],
            text="Confirming current balance.",
            stop_reason="tool_use",
            output_tokens=int(rng.gauss(tok_mean, 30)),
            latency_ms=_log_normal(rng, lat_mean, lat_sigma),
        )
    )

    # Turn 3: transfer
    records.append(
        _make_response(
            rng,
            base_idx + 3,
            tools=["transfer_funds"],
            text="Transferring funds now.",
            stop_reason="tool_use",
            output_tokens=int(rng.gauss(tok_mean, 30)),
            latency_ms=_log_normal(rng, lat_mean, lat_sigma),
        )
    )

    return records


def generate_cohort(
    profile: Profile,
    n_sessions: int,
    *,
    seed: int = 0,
    policy_violation_rate: float = 0.0,
) -> list[list[dict[str, Any]]]:
    """Generate ``n_sessions`` sessions as a list-of-lists.

    For ``profile="policy_violating"`` set ``policy_violation_rate``
    > 0 — that fraction of sessions will skip the verify_user step.
    """
    rng = random.Random(seed)
    out: list[list[dict[str, Any]]] = []
    for i in range(n_sessions):
        violate = profile == "policy_violating" and rng.random() < policy_violation_rate
        out.append(
            generate_session(
                profile,
                seed=seed * 10000 + i,
                inject_policy_violation=violate,
            )
        )
    return out


__all__ = ["generate_cohort", "generate_session"]
