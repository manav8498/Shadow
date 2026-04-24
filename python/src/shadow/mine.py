"""Mine a production trace set for representative regression cases.

Teams hate writing eval suites by hand. The trace set they already
have from production is usually the best source of test cases, but
shipping the whole thing into CI is expensive and noisy. Mining picks
a minimal subset that covers the interesting parts.

## What it does

Given one or more `.agentlog` files (ideally a large production trace
set), `shadow mine` groups turn-pairs by a simple signature and picks
one or two representatives from each group. The signature keys off:

- the tool-call sequence (what tools did the agent use, in what order)
- the stop reason (end_turn, tool_use, content_filter, error)
- a coarse response-length bucket
- a coarse latency bucket

Within a cluster, the picker prefers turns that look interesting:

- error stop reason (likely worth a regression test)
- content_filter stop reason (refusal behaviour)
- high cost or latency (budget-sensitive)
- high thinking-token count (reasoning-heavy)
- very short or very long responses (boundary cases)

The output is a new `.agentlog` file containing the selected turns and
their parent metadata, usable directly as a baseline for `shadow diff`
or as an input to CI.

## What it does not do

- It does not invoke an LLM to summarise or judge traces.
- It does not redact beyond what the source SDK already redacted.
- It does not invent test assertions. Use `shadow diff --policy` for
  contracts; use the mined traces as your fixture set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from shadow import __version__, _core


@dataclass
class MinedCase:
    """One mined turn-pair with its cluster signature and score."""

    cluster: str
    score: float
    reason: str
    request_record: dict[str, Any]
    response_record: dict[str, Any]
    baseline_source: str  # trace id the pair was sampled from


@dataclass
class MiningResult:
    """Aggregate output of a mining run."""

    total_input_pairs: int
    clusters_found: int
    cases: list[MinedCase]

    def to_agentlog(self) -> list[dict[str, Any]]:
        """Render the mined cases as a new .agentlog record sequence."""
        meta_payload = {
            "sdk": {"name": "shadow", "version": __version__},
            "source": {"format": "shadow-mine", "notes": "mined from production traces"},
            "mining": {
                "total_input_pairs": self.total_input_pairs,
                "clusters_found": self.clusters_found,
                "cases_selected": len(self.cases),
            },
        }
        meta_id = _core.content_id(meta_payload)
        records: list[dict[str, Any]] = [
            {
                "version": "0.1",
                "id": meta_id,
                "kind": "metadata",
                "ts": _now_iso(),
                "parent": None,
                "payload": meta_payload,
            }
        ]
        last_parent = meta_id
        for case in self.cases:
            req = dict(case.request_record)
            req["parent"] = last_parent
            req["ts"] = _now_iso()
            req["id"] = _core.content_id(req["payload"])
            records.append(req)
            resp = dict(case.response_record)
            resp["parent"] = req["id"]
            resp["ts"] = _now_iso()
            resp["id"] = _core.content_id(resp["payload"])
            records.append(resp)
            last_parent = resp["id"]
        return records


def mine(
    traces: list[list[dict[str, Any]]],
    max_cases: int = 50,
    per_cluster: int = 1,
    pricing: dict[str, Any] | None = None,
) -> MiningResult:
    """Mine a list of parsed traces for representative cases.

    `traces` is a list of record lists (one per source .agentlog file).
    `max_cases` caps the total output; `per_cluster` caps how many
    pairs to keep from each cluster. `pricing` is the same dict used
    by the cost axis; if supplied, the picker prefers high-cost pairs
    within a cluster.
    """
    pricing = pricing or {}
    all_pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    for trace in traces:
        trace_id = trace[0]["id"] if trace and trace[0].get("kind") == "metadata" else ""
        pending_req: dict[str, Any] | None = None
        for rec in trace:
            kind = rec.get("kind")
            if kind == "chat_request":
                pending_req = rec
            elif kind == "chat_response" and pending_req is not None:
                all_pairs.append((pending_req, rec, trace_id))
                pending_req = None

    clusters: dict[str, list[tuple[dict[str, Any], dict[str, Any], str]]] = {}
    for req, resp, trace_id in all_pairs:
        sig = _signature(req, resp)
        clusters.setdefault(sig, []).append((req, resp, trace_id))

    cases: list[MinedCase] = []
    for cluster_sig, members in clusters.items():
        members_scored = [
            (_score_pair(req, resp, pricing), _reason(req, resp), (req, resp, trace_id))
            for req, resp, trace_id in members
        ]
        members_scored.sort(key=lambda t: -t[0])
        for score, reason, (req, resp, trace_id) in members_scored[:per_cluster]:
            cases.append(
                MinedCase(
                    cluster=cluster_sig,
                    score=score,
                    reason=reason,
                    request_record=req,
                    response_record=resp,
                    baseline_source=trace_id,
                )
            )

    # Global cap: keep the highest-score cases across clusters
    cases.sort(key=lambda c: -c.score)
    cases = cases[: max(0, max_cases)]
    # Stable secondary sort by cluster signature for deterministic output
    cases.sort(key=lambda c: (c.cluster, -c.score))

    return MiningResult(
        total_input_pairs=len(all_pairs),
        clusters_found=len(clusters),
        cases=cases,
    )


# ---- signatures + scoring -------------------------------------------------


def _signature(req: dict[str, Any], resp: dict[str, Any]) -> str:
    """Stable cluster key for a turn-pair."""
    payload = resp.get("payload") or {}
    tools = _tool_sequence(payload)
    stop = payload.get("stop_reason", "end_turn")
    verbosity_bucket = _bucket(_output_tokens(payload), [0, 10, 50, 200, 1000, 5000])
    latency_bucket = _bucket(int(payload.get("latency_ms") or 0), [0, 100, 500, 2000, 10000])
    body = json.dumps(
        {"tools": tools, "stop": stop, "v": verbosity_bucket, "l": latency_bucket},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:12]


def _tool_sequence(payload: dict[str, Any]) -> list[str]:
    return [
        block.get("name", "?")
        for block in payload.get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _output_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage") or {}
    return int(usage.get("output_tokens") or 0)


def _bucket(value: int, edges: list[int]) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def _score_pair(req: dict[str, Any], resp: dict[str, Any], pricing: dict[str, Any]) -> float:
    """Higher score means more interesting. Range roughly 0..5."""
    payload = resp.get("payload") or {}
    score = 0.0

    stop = payload.get("stop_reason")
    if stop == "error":
        score += 2.5
    elif stop == "content_filter":
        score += 2.0
    elif stop == "max_tokens":
        score += 1.0

    latency = int(payload.get("latency_ms") or 0)
    if latency > 5000:
        score += 1.5
    elif latency > 2000:
        score += 0.8

    usage = payload.get("usage") or {}
    thinking = int(usage.get("thinking_tokens") or 0)
    if thinking > 1000:
        score += 1.0

    out_toks = int(usage.get("output_tokens") or 0)
    if out_toks == 0:
        score += 0.5
    elif out_toks > 2000:
        score += 0.7

    # Price estimate when pricing table is supplied
    model = (payload.get("model") or "").lower()
    if pricing and model:
        pinfo = pricing.get(model)
        if isinstance(pinfo, dict):
            cost = (
                int(usage.get("input_tokens") or 0) * float(pinfo.get("input", 0.0))
                + int(usage.get("output_tokens") or 0) * float(pinfo.get("output", 0.0))
                + int(usage.get("thinking_tokens") or 0) * float(pinfo.get("reasoning", 0.0))
            )
            if cost > 0.1:
                score += 1.0
            elif cost > 0.01:
                score += 0.3

    return score


def _reason(req: dict[str, Any], resp: dict[str, Any]) -> str:
    """Short human-readable tag for why the pair was selected."""
    payload = resp.get("payload") or {}
    stop = payload.get("stop_reason")
    if stop == "error":
        return "error_stop_reason"
    if stop == "content_filter":
        return "refusal"
    if stop == "max_tokens":
        return "output_truncated"
    usage = payload.get("usage") or {}
    if int(usage.get("thinking_tokens") or 0) > 1000:
        return "heavy_reasoning"
    if int(payload.get("latency_ms") or 0) > 5000:
        return "high_latency"
    if int(usage.get("output_tokens") or 0) > 2000:
        return "long_response"
    if int(usage.get("output_tokens") or 0) == 0:
        return "empty_response"
    return "cluster_representative"


def _now_iso() -> str:
    import datetime

    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


__all__ = ["MinedCase", "MiningResult", "mine"]
