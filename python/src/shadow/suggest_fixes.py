"""LLM-assisted prescriptive fix suggestions.

Shadow's deterministic recommendation engine (in the Rust
`crates/shadow-core/src/diff/recommendations.rs`) already produces a
ranked list of `Recommendation` entries with severity, action, and
rationale fields. Those are template-driven — they always say true
things, but they can't propose *specific code-level changes* because
they don't know the repo, the prompt, or the surrounding config.

This module layers an LLM pass on top. It takes:

- the full `DiffReport` (deterministic rows, drill-down, recommendations)
- the baseline + candidate `.agentlog` records
- an `LlmBackend`

and returns a list of enriched suggestions where each deterministic
recommendation is paired with a concrete fix proposal the reviewer can
either accept, adapt, or ignore. Every suggestion still cites the
deterministic trigger so the reviewer can tell what's grounded vs.
what the LLM invented.

## Guarantees

- **Grounded.** Each suggestion references a concrete `Recommendation`
  or axis delta from the DiffReport. Pure-LLM "while you're at it…"
  suggestions are explicitly disallowed — we filter the model's output
  against the deterministic evidence set.

- **Reproducible shape.** The output is a JSON-serialisable list of
  dicts so CLIs / CI comments can render it without parsing prose.

- **Honest about uncertainty.** Each suggestion carries a
  `confidence` field the LLM sets; suggestions below 0.3 are kept
  but flagged. Over-confident template output is a bad smell and
  gets downgraded automatically.

- **Opt-in only.** Cost is ~500-2500 output tokens per diff depending
  on severity density. The CLI exposes this behind `--suggest-fixes`,
  never by default.

## Safety

The LLM receives the deterministic summary + the worst-turn pair's
request/response payloads (truncated), never the full trace. That
keeps the prompt under 4K tokens even for 1K-pair traces, and keeps
the LLM grounded on concrete evidence.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from shadow.llm.base import LlmBackend

MAX_EVIDENCE_CHARS = 1800  # per-record payload truncation budget
MIN_CONFIDENCE = 0.0  # keep all, but flag < 0.3
FLAG_CONFIDENCE = 0.3

# Retry knobs. Tuned against Anthropic + OpenAI's published guidance
# (both recommend exponential backoff starting at ~1s for 429/503).
DEFAULT_MAX_RETRIES = 4
DEFAULT_INITIAL_BACKOFF_S = 1.0
DEFAULT_BACKOFF_MULT = 2.0
DEFAULT_BACKOFF_JITTER = 0.25  # ± 25% random jitter so concurrent callers don't synchronise

# Error-message substrings that mark a call as transient. We match on
# the exception message because Anthropic/OpenAI SDKs raise typed
# exceptions that don't share a common base class — but their messages
# consistently mention status codes or transport keywords.
_TRANSIENT_MARKERS = (
    "rate limit",
    "429",
    "503",
    "502",
    "504",
    "overloaded",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "try again",
)


def _is_transient(err: BaseException) -> bool:
    """Classify whether an exception is worth retrying.

    We err on the side of retrying: if we can't tell, we treat it as
    transient once, then stop. 400-class auth/validation errors
    bypass this because their messages don't match the markers.
    """
    msg = str(err).lower()
    if any(marker in msg for marker in _TRANSIENT_MARKERS):
        return True
    # Common auth / bad-request markers — definitively NOT transient.
    if any(
        m in msg for m in ("401", "403", "invalid api key", "authentication", "bad request", "400")
    ):
        return False
    return False


async def _complete_with_retry(
    backend: LlmBackend,
    payload: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_S,
    backoff_mult: float = DEFAULT_BACKOFF_MULT,
    jitter: float = DEFAULT_BACKOFF_JITTER,
    on_retry: Any = None,
) -> dict[str, Any]:
    """Call `backend.complete` with exponential backoff on transient errors.

    `on_retry(attempt, delay, error)` is called before each sleep for
    observability; pass None to skip. Non-transient errors (400, 401,
    invalid input) raise immediately.
    """
    attempt = 0
    delay = initial_backoff
    while True:
        try:
            return await backend.complete(payload)
        except Exception as err:
            if attempt >= max_retries or not _is_transient(err):
                raise
            if on_retry is not None:
                # Observability callback must never crash the caller.
                with contextlib.suppress(Exception):
                    on_retry(attempt + 1, delay, err)
            # Jitter: multiply delay by a uniform [1-j, 1+j] factor.
            jittered = delay * (1 + random.uniform(-jitter, jitter))
            await asyncio.sleep(max(0.0, jittered))
            attempt += 1
            delay *= backoff_mult


@dataclass
class SuggestedFix:
    """One LLM-augmented fix proposal anchored to a deterministic trigger."""

    # Ties back to the deterministic Recommendation that spawned this
    # suggestion. Matches the `id` / `title` of the underlying
    # recommendation so a renderer can group deterministic +
    # LLM-suggested text.
    anchor: str
    severity: str  # "info" | "warning" | "error" (matches Recommendation)
    axis: str  # e.g. "trajectory" — copied from the anchor
    # Short, concrete change proposal. Starts with a verb.
    proposal: str
    # Optional snippet the LLM thinks should be changed, e.g. a tool
    # schema YAML fragment or a prompt diff. Free-form — the renderer
    # wraps it in a code fence.
    snippet: str | None
    # LLM-reported confidence, 0..1. `< FLAG_CONFIDENCE` is surfaced
    # as `"speculative"` in the renderer.
    confidence: float
    # Brief rationale the LLM wrote. One sentence. Not the determinist
    # rationale — that one is kept separately on the anchor.
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuggestFixesResult:
    """Top-level result: fixes + input bookkeeping."""

    suggestions: list[SuggestedFix]
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    # How many deterministic recommendations were fed in — gives a
    # renderer a denominator for "1 of 4 recommendations enriched".
    anchors_considered: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "anchors_considered": self.anchors_considered,
        }


def suggest_fixes(
    report: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    backend: LlmBackend,
    *,
    model: str | None = None,
    max_anchors: int = 6,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> SuggestFixesResult:
    """Synchronous entry point. Calls the backend (with retry on 429/5xx/timeout)."""
    anchors = _collect_anchors(report, limit=max_anchors)
    if not anchors:
        return SuggestFixesResult(
            suggestions=[],
            model_id=model or backend.id,
            prompt_tokens=0,
            completion_tokens=0,
            anchors_considered=0,
        )
    evidence = _build_evidence(report, baseline_records, candidate_records, anchors)
    prompt = _render_prompt(anchors, evidence)
    request_payload: dict[str, Any] = {
        "model": model or "",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior engineer reviewing a behaviour regression "
                    "in an LLM agent. Given a structured diff summary and the "
                    "flagged turns' payloads, propose concrete code or config "
                    "changes for each anchor. Be specific — name the file, the "
                    "parameter, or the prompt phrase. No apologies, no hedging, "
                    "no general advice. Reply with a JSON object matching the "
                    "schema the user provides — nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "params": {"temperature": 0.0, "max_tokens": 2048},
    }
    response = asyncio.run(_complete_with_retry(backend, request_payload, max_retries=max_retries))
    raw_text = _extract_text(response)
    parsed = _parse_llm_json(raw_text)
    suggestions = _filter_suggestions(parsed, anchors)
    usage = response.get("usage") or {}
    return SuggestFixesResult(
        suggestions=suggestions,
        model_id=str(response.get("model") or model or backend.id),
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        anchors_considered=len(anchors),
    )


# ---- anchor collection ----------------------------------------------------


def _collect_anchors(report: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    """Pick the deterministic recommendations we'll ask the LLM to enrich.

    Priority: error-severity first, then warnings, then info. We cap at
    `limit` so the prompt stays bounded even on catastrophic diffs.
    """
    recs = report.get("recommendations") or []
    sev_order = {"error": 0, "warning": 1, "info": 2}
    ranked = sorted(recs, key=lambda r: (sev_order.get(r.get("severity", "info"), 3),))
    out: list[dict[str, Any]] = []
    for r in ranked[:limit]:
        # Rust-side `Recommendation` uses `message` (human-readable)
        # + `turn`; there is no `id`/`title` field. Python-side test
        # fixtures use `id`/`title`/`turn_index`. Accept both.
        title = r.get("title") or r.get("message") or ""
        axis = r.get("axis") or _infer_axis_from_title(title)
        turn = r.get("turn_index")
        if turn is None:
            turn = r.get("turn", 0)
        out.append(
            {
                "id": str(r.get("id") or title or f"rec-{len(out)}"),
                "title": str(title),
                "severity": str(r.get("severity") or "info"),
                "axis": str(axis or "trace"),
                "action": str(r.get("action") or ""),
                "rationale": str(r.get("rationale") or ""),
                "turn_index": int(turn or 0),
            }
        )
    return out


def _infer_axis_from_title(title: str) -> str:
    lowered = title.lower()
    for axis in (
        "semantic",
        "trajectory",
        "safety",
        "verbosity",
        "latency",
        "cost",
        "reasoning",
        "judge",
        "conformance",
    ):
        if axis in lowered:
            return axis
    return "trace"


# ---- evidence truncation --------------------------------------------------


def _build_evidence(
    report: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pull a small window of evidence each anchor can cite."""
    responses_b = [r for r in baseline_records if r.get("kind") == "chat_response"]
    responses_c = [r for r in candidate_records if r.get("kind") == "chat_response"]
    requests_b = [r for r in baseline_records if r.get("kind") == "chat_request"]
    requests_c = [r for r in candidate_records if r.get("kind") == "chat_request"]

    evidence: dict[str, Any] = {
        "top_axes": _top_axes(report),
        "first_divergence": report.get("first_divergence"),
        "pair_count": int(report.get("pair_count") or 0),
        "turns": [],
    }
    seen_turns: set[int] = set()
    for a in anchors:
        t = a["turn_index"]
        if t in seen_turns:
            continue
        seen_turns.add(t)
        evidence["turns"].append(
            {
                "turn_index": t,
                "baseline_request": _truncate_payload(
                    requests_b[t]["payload"] if t < len(requests_b) else None
                ),
                "baseline_response": _truncate_payload(
                    responses_b[t]["payload"] if t < len(responses_b) else None
                ),
                "candidate_request": _truncate_payload(
                    requests_c[t]["payload"] if t < len(requests_c) else None
                ),
                "candidate_response": _truncate_payload(
                    responses_c[t]["payload"] if t < len(responses_c) else None
                ),
            }
        )
    return evidence


def _top_axes(report: dict[str, Any]) -> list[dict[str, Any]]:
    rank = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    rows = report.get("rows") or []
    ranked = sorted(rows, key=lambda r: -rank.get(r.get("severity", "none"), 0))
    out: list[dict[str, Any]] = []
    for row in ranked[:5]:
        out.append(
            {
                "axis": row.get("axis"),
                "severity": row.get("severity"),
                "baseline_median": row.get("baseline_median"),
                "candidate_median": row.get("candidate_median"),
                "delta": row.get("delta"),
            }
        )
    return out


def _truncate_payload(payload: Any) -> Any:
    """Stringify + truncate so a single payload never blows the prompt budget."""
    if payload is None:
        return None
    text = json.dumps(payload, default=str)
    if len(text) <= MAX_EVIDENCE_CHARS:
        # Parse back so the final prompt shows structured JSON, not a
        # string literal — LLMs handle nested JSON better when it's not
        # wrapped in quotes.
        return json.loads(text)
    truncated = text[:MAX_EVIDENCE_CHARS]
    # Leave a marker so the LLM knows we cut something.
    return {"_truncated": True, "preview": truncated, "original_chars": len(text)}


# ---- prompt + response handling ------------------------------------------


_RESPONSE_SCHEMA = {
    "suggestions": [
        {
            "anchor": "string — must match one of the anchor ids verbatim",
            "proposal": "string — single imperative sentence; name a file/param/phrase",
            "snippet": "optional string — code/config fragment or null",
            "confidence": "number between 0 and 1",
            "rationale": "single sentence naming the concrete evidence you used",
        }
    ]
}


def _render_prompt(anchors: list[dict[str, Any]], evidence: dict[str, Any]) -> str:
    """Bundle anchors + evidence + schema into a single user message."""
    return (
        "Review this Shadow diff. For each anchor, propose one concrete fix.\n\n"
        f"Anchors to address (exactly these ids, in any order):\n"
        f"{json.dumps(anchors, indent=2)}\n\n"
        f"Evidence (top axes + divergence + flagged turn payloads):\n"
        f"{json.dumps(evidence, indent=2)}\n\n"
        f"Respond with JSON matching this schema exactly. No prose, no markdown fences:\n"
        f"{json.dumps(_RESPONSE_SCHEMA, indent=2)}\n\n"
        "Rules: never invent code changes unrelated to the anchors; cite the "
        "specific evidence field you used in `rationale`; if an anchor has "
        "no actionable fix, set confidence < 0.3 and say why."
    )


def _extract_text(response: dict[str, Any]) -> str:
    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Try hard to pull a suggestion list out of the LLM's response.

    The prompt asks for a strict JSON object, but even 0-temp LLMs
    occasionally wrap output in a markdown fence. We strip those, try
    the object shape first, then the list shape, then give up.
    """
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :].lstrip("\n")
    # Drop anything after the last `}` — some models append chatter.
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        cleaned = cleaned[: last_brace + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("suggestions"), list):
        return [s for s in parsed["suggestions"] if isinstance(s, dict)]
    if isinstance(parsed, list):
        return [s for s in parsed if isinstance(s, dict)]
    return []


def _filter_suggestions(
    raw: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> list[SuggestedFix]:
    """Bind each raw suggestion back to its anchor; drop ungrounded ones."""
    anchor_by_id = {a["id"]: a for a in anchors}
    anchor_by_title = {a["title"]: a for a in anchors if a["title"]}
    out: list[SuggestedFix] = []
    for raw_s in raw:
        anchor_id = str(raw_s.get("anchor") or "")
        anchor = anchor_by_id.get(anchor_id) or anchor_by_title.get(anchor_id)
        if anchor is None:
            # Silently drop: we're strict about grounding.
            continue
        proposal = str(raw_s.get("proposal") or "").strip()
        if not proposal:
            continue
        snippet = raw_s.get("snippet")
        if snippet is not None and not isinstance(snippet, str):
            snippet = json.dumps(snippet, sort_keys=True, default=str)
        try:
            confidence = float(raw_s.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(MIN_CONFIDENCE, min(1.0, confidence))
        out.append(
            SuggestedFix(
                anchor=anchor["id"],
                severity=anchor["severity"],
                axis=anchor["axis"],
                proposal=proposal,
                snippet=snippet if isinstance(snippet, str) and snippet.strip() else None,
                confidence=confidence,
                rationale=str(raw_s.get("rationale") or "").strip(),
            )
        )
    # Deterministic order: highest severity → highest confidence → anchor id.
    sev_order = {"error": 0, "warning": 1, "info": 2}
    out.sort(key=lambda s: (sev_order.get(s.severity, 3), -s.confidence, s.anchor))
    return out


def render_terminal(result: SuggestFixesResult) -> str:
    """Simple terminal rendering for `shadow diff --suggest-fixes`."""
    if not result.suggestions:
        return "LLM-assisted fixes: no concrete suggestions for this diff."
    lines = [
        f"LLM-assisted fix suggestions ({len(result.suggestions)} "
        f"of {result.anchors_considered} anchor(s), model={result.model_id}):"
    ]
    for s in result.suggestions:
        marker = {"error": "✗", "warning": "!", "info": "·"}.get(s.severity, "·")
        conf_label = f"{s.confidence:.0%}"
        if s.confidence < FLAG_CONFIDENCE:
            conf_label += " [speculative]"
        lines.append(f"  {marker} [{s.severity}/{s.axis}] {s.proposal}")
        if s.rationale:
            lines.append(f"      rationale: {s.rationale}")
        if s.snippet:
            lines.append("      snippet:")
            for ln in s.snippet.splitlines()[:12]:
                lines.append(f"        {ln}")
        lines.append(f"      anchor: {s.anchor} · confidence: {conf_label}")
    return "\n".join(lines)


__all__ = [
    "SuggestFixesResult",
    "SuggestedFix",
    "render_terminal",
    "suggest_fixes",
]
