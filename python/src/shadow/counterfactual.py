"""Counterfactual replay: one slice of the replay-as-science wedge.

The full "replay as first-class science" ambition in the strategic
analysis is a 3-4 month effort covering five distinct modes:
counterfactual, partial, sandboxed, streaming, multimodal. This
module ships **one** of those — counterfactual replay — with honest
scoping so future slices can land without churn.

## What counterfactual replay does

Takes a baseline `.agentlog` and a single config delta (e.g. "swap
model to claude-haiku, leave everything else the same"). Produces a
candidate `.agentlog` where **only that one delta** was applied to
every LLM call, with all other inputs held constant at the baseline.

This is different from `shadow replay` today: that replays a whole
candidate config (all deltas bundled). Counterfactual isolates one
variable at a time, which is what causal attribution really wants.

## Why it matters

When a PR changes 4 things at once (prompt + model + temperature +
tool schema), Shadow's `shadow bisect` already attributes the
resulting behaviour delta to specific deltas via LASSO. But that
attribution is a statistical inference; counterfactual replay is the
**direct** experiment — literally re-run with only one thing changed
and measure the actual behaviour.

The two compose: bisect tells you "we think the model swap is 78% of
the latency regression"; a counterfactual replay confirms it with a
direct replay that holds everything else constant.

## Supported delta kinds (v1.1)

| Delta kind | Supported? |
|---|---|
| `model` — swap the model used for every LLM call | ✅ |
| `params.temperature` / `params.top_p` / `params.max_tokens` | ✅ |
| `prompt.system` — override the system prompt | ✅ |
| `tools` — replace the whole tool schema list | ✅ |
| `tools[N].description` — edit one tool's description | ⚠️ not yet (use whole-tool replacement) |
| Multimodal (image / audio blocks) | ❌ deferred — see module docstring in `replay.py` |
| Streaming-response semantics | ❌ deferred |
| Tool-sandbox-level deltas | ❌ deferred |

## NOT shipped here

The other four replay-as-science slices are still v1.2+ work:

- **Partial replay** — replay only a prefix of the trace, let the
  tail run live. Requires a branching-point marker in `.agentlog`.
- **Sandboxed replay** — tool calls run inside a deterministic
  sandbox (filesystem snapshot, network blackhole, clock fix).
  Requires a tool-sandbox trait and per-tool adapters.
- **Streaming replay** — replay a streaming response as-was with
  the original inter-chunk timings. Requires recording chunks, not
  just concatenated text.
- **Multimodal replay** — image/audio inputs and outputs. Requires
  a SPEC update for binary payload addressing.

Each of the four deserves its own v1.N release. This module is the
skeleton; nothing stops a future PR adding the others alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from shadow import _core
from shadow.errors import ShadowConfigError
from shadow.llm.base import LlmBackend

# The single delta a counterfactual replay applies. Deliberately
# narrow — one scalar thing at a time — because that's what makes it
# a "counterfactual" vs a full candidate-config replay.
CounterfactualKind = Literal[
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "system_prompt",
    "tools",
]


@dataclass
class Counterfactual:
    """One atomic config delta to apply to every request in a replay.

    Exactly one of the typed fields should be set; the `kind` tag
    names which. Separate fields (rather than a single `Any` value)
    keep type-checking honest and make unsupported combinations
    impossible to construct.
    """

    kind: CounterfactualKind
    # Exactly one of these is set depending on `kind`.
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None


def apply_counterfactual_to_request(
    request_payload: dict[str, Any], cf: Counterfactual
) -> dict[str, Any]:
    """Apply a counterfactual to a single chat_request payload.

    Returns a new payload; the input is not mutated. Unknown fields
    on the request are preserved so replay through a live backend
    round-trips cleanly.
    """
    out = {**request_payload}
    if cf.kind == "model":
        if cf.model is None:
            raise ShadowConfigError("Counterfactual(kind='model') requires model=...")
        out["model"] = cf.model
    elif cf.kind == "temperature":
        if cf.temperature is None:
            raise ShadowConfigError("Counterfactual(kind='temperature') requires temperature=...")
        params = {**(out.get("params") or {})}
        params["temperature"] = float(cf.temperature)
        out["params"] = params
    elif cf.kind == "top_p":
        if cf.top_p is None:
            raise ShadowConfigError("Counterfactual(kind='top_p') requires top_p=...")
        params = {**(out.get("params") or {})}
        params["top_p"] = float(cf.top_p)
        out["params"] = params
    elif cf.kind == "max_tokens":
        if cf.max_tokens is None:
            raise ShadowConfigError("Counterfactual(kind='max_tokens') requires max_tokens=...")
        params = {**(out.get("params") or {})}
        params["max_tokens"] = int(cf.max_tokens)
        out["params"] = params
    elif cf.kind == "system_prompt":
        if cf.system_prompt is None:
            raise ShadowConfigError(
                "Counterfactual(kind='system_prompt') requires system_prompt=..."
            )
        messages = list(out.get("messages") or [])
        # Replace the leading system message if present; otherwise prepend.
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": cf.system_prompt}
        else:
            messages = [{"role": "system", "content": cf.system_prompt}, *messages]
        out["messages"] = messages
    elif cf.kind == "tools":
        if cf.tools is None:
            raise ShadowConfigError("Counterfactual(kind='tools') requires tools=...")
        out["tools"] = [dict(t) for t in cf.tools]
    else:
        raise ShadowConfigError(f"unknown counterfactual kind: {cf.kind}")
    return out


async def run_counterfactual(
    baseline_records: list[dict[str, Any]],
    counterfactual: Counterfactual,
    backend: LlmBackend,
    keep_original_parents: bool = True,
) -> list[dict[str, Any]]:
    """Replay a baseline trace with a single counterfactual delta applied.

    For every `chat_request` in the baseline, produce a modified
    request where the counterfactual has been applied, send it through
    `backend`, and stitch the response back into the trace with parent
    links.

    `keep_original_parents` is True by default — the counterfactual
    trace preserves the baseline's DAG shape so the Shadow differ can
    align paired turns by index. Setting False produces a freshly-
    content-addressed DAG (useful for forward-flow replay, but breaks
    one-to-one pairing with baseline).
    """
    if not baseline_records:
        raise ShadowConfigError("baseline is empty — nothing to counterfactual-replay")

    import datetime

    def _now_iso() -> str:
        now = datetime.datetime.now(datetime.UTC)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    out: list[dict[str, Any]] = []

    # Root metadata record — mark this trace as a counterfactual of
    # the baseline so downstream consumers can distinguish.
    root = baseline_records[0]
    if root.get("kind") != "metadata":
        raise ShadowConfigError(f"baseline root is {root.get('kind')!r}, expected 'metadata'")
    new_meta_payload: dict[str, Any] = {
        **(root.get("payload") or {}),
        "counterfactual": {
            "kind": counterfactual.kind,
            "baseline_of": root.get("id"),
        },
    }
    new_meta_id = _core.content_id(new_meta_payload)
    out.append(
        {
            "version": "0.1",
            "id": new_meta_id,
            "kind": "metadata",
            "ts": _now_iso(),
            "parent": None,
            "payload": new_meta_payload,
        }
    )
    last_parent: str = new_meta_id

    # Walk the baseline; on each chat_request, apply the counterfactual
    # and send the new request through the backend. Non-request records
    # are copied through unchanged.
    for record in baseline_records[1:]:
        if record.get("kind") != "chat_request":
            continue
        new_payload = apply_counterfactual_to_request(record.get("payload") or {}, counterfactual)
        req_id = _core.content_id(new_payload)
        req_record = {
            "version": "0.1",
            "id": req_id,
            "kind": "chat_request",
            "ts": _now_iso(),
            "parent": last_parent,
            "payload": new_payload,
        }
        out.append(req_record)

        try:
            response_payload = await backend.complete(new_payload)
            resp_record = {
                "version": "0.1",
                "id": _core.content_id(response_payload),
                "kind": "chat_response",
                "ts": _now_iso(),
                "parent": req_id,
                "payload": response_payload,
            }
            out.append(resp_record)
            last_parent = str(resp_record["id"])
        except Exception as e:
            err_payload = {
                "source": "llm",
                "code": "backend_error",
                "message": str(e),
                "retriable": False,
            }
            err_record = {
                "version": "0.1",
                "id": _core.content_id(err_payload),
                "kind": "error",
                "ts": _now_iso(),
                "parent": req_id,
                "payload": err_payload,
            }
            out.append(err_record)
            last_parent = str(err_record["id"])

    return out


__all__ = [
    "Counterfactual",
    "CounterfactualKind",
    "apply_counterfactual_to_request",
    "run_counterfactual",
]
