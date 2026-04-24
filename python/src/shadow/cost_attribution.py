"""Per-session cost attribution.

Shadow's existing `cost` axis reports a per-response cost delta with
a 95% CI. That answers *whether* a PR moved cost, but not *why* or
*how much per session*. This module closes that gap:

- **Session partitioning.** A "session" is the span between two
  `metadata` records in an `.agentlog` — practically, one
  user-facing conversation including all follow-up tool calls. We
  sum input/output/cached/reasoning tokens and USD spend per
  session.

- **Attribution decomposition.** The cost delta between a baseline
  session and its candidate decomposes into three independent
  sources:

    total_delta = model_swap + token_movement + mix_residual

  where
    model_swap    = sum(candidate_tokens_i * (candidate_price_i - baseline_price_i))
    token_movement = sum((candidate_tokens_i - baseline_tokens_i) * baseline_price_i)
    mix_residual  = total_delta - model_swap - token_movement

  `mix_residual` captures non-additive interactions (e.g. a prompt
  change that both swaps model AND changes token counts). It's
  usually small; when it's > ~10% of total_delta the attribution
  is less trustworthy and we flag it.

- **Presentation.** One row per session: baseline cost, candidate
  cost, delta, and the three attribution components as
  percentages. A trace-level roll-up at the bottom sums across
  sessions.

Uses the same `Pricing` dict the Rust cost axis uses (input /
output / cached_input / cached_write_5m / cached_write_1h /
reasoning rates + batch_discount).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SessionCost:
    """Per-session cost + token breakdown."""

    session_index: int
    model: str  # modal model; sessions mixing models report "mixed"
    input_tokens: float
    output_tokens: float
    cached_input_tokens: float
    thinking_tokens: float
    total_usd: float


@dataclass
class SessionAttribution:
    """Decomposition of one session-pair's cost delta."""

    session_index: int
    baseline_usd: float
    candidate_usd: float
    delta_usd: float
    delta_pct: float  # relative to baseline; inf if baseline was 0
    model_swap_usd: float
    token_movement_usd: float
    mix_residual_usd: float
    baseline_model: str
    candidate_model: str
    n_responses: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CostAttributionReport:
    """Top-level attribution across all paired sessions."""

    per_session: list[SessionAttribution]
    total_baseline_usd: float
    total_candidate_usd: float
    total_delta_usd: float
    total_model_swap_usd: float
    total_token_movement_usd: float
    total_mix_residual_usd: float
    # True when |mix_residual| > 10% of |total_delta| — indicates
    # the two components interact and the split is less trustworthy.
    attribution_is_noisy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_session": [s.to_dict() for s in self.per_session],
            "total_baseline_usd": self.total_baseline_usd,
            "total_candidate_usd": self.total_candidate_usd,
            "total_delta_usd": self.total_delta_usd,
            "total_model_swap_usd": self.total_model_swap_usd,
            "total_token_movement_usd": self.total_token_movement_usd,
            "total_mix_residual_usd": self.total_mix_residual_usd,
            "attribution_is_noisy": self.attribution_is_noisy,
        }


# ---- session partitioning -------------------------------------------------


def partition_sessions(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a flat record list into per-session chunks.

    Primary signal: a session starts at a ``metadata`` record and runs
    until the next ``metadata`` record. This is the shape the Shadow
    SDK emits — one metadata marker per ``Session()`` context.

    Fallback: if the primary partition returns exactly one session but
    that session contains multiple user-initiated sub-sessions (imports
    from foreign tracers like A2A, MCP, or OTel often concatenate
    multiple tickets under a single metadata record), split on implicit
    boundaries inferred from request shape + terminal stop reasons.
    Without this fallback, imported multi-ticket traces would silently
    appear as one giant session in ``diff_by_session`` — the same bug
    that masked the original policy-check regressions.
    """
    sessions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("kind") == "metadata" and current:
            sessions.append(current)
            current = [rec]
        else:
            if rec.get("kind") == "metadata" and not current:
                current = [rec]
            else:
                current.append(rec)
    if current:
        sessions.append(current)

    if len(sessions) == 1:
        split = _split_by_implicit_boundaries(sessions[0])
        if len(split) > 1:
            return split
    return sessions


def _split_by_implicit_boundaries(
    session_records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Fall back to user-initiated boundary detection when a single
    metadata-delimited block contains multiple sub-sessions.

    Uses the same two-signal detector as the policy checker
    (:func:`shadow.hierarchical._compute_session_of_pair`): a new
    sub-session starts at a ``chat_request`` whose most recent
    non-system message role is ``user``, *or* whose preceding response
    had a non-``tool_use`` stop reason. The leading metadata record
    stays with the first sub-session; subsequent sub-sessions carry no
    metadata and rely on the metadata from the outer partition.
    """
    # Imported lazily to avoid a circular import with hierarchical.
    from shadow.hierarchical import _CONTINUATION_STOP_REASON, _is_session_start

    out: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prior_stop_terminal = True  # first request after metadata starts a session
    for rec in session_records:
        kind = rec.get("kind")
        if kind == "chat_request":
            payload = rec.get("payload") or {}
            starts_new = _is_session_start(payload) or prior_stop_terminal
            if starts_new and any(
                r.get("kind") in ("chat_request", "chat_response") for r in current
            ):
                # Close off the previous sub-session and begin a new one.
                out.append(current)
                current = []
            prior_stop_terminal = False
        elif kind == "chat_response":
            stop = (rec.get("payload") or {}).get("stop_reason", "")
            prior_stop_terminal = stop != _CONTINUATION_STOP_REASON
        current.append(rec)
    if current:
        out.append(current)
    return out


# ---- per-session cost rollup ---------------------------------------------


def _cost_of_response(
    payload: dict[str, Any], pricing: dict[str, Any]
) -> tuple[float, float, float, float, float]:
    """Return (usd, input_tok, output_tok, cached_input_tok, thinking_tok)
    for one chat_response payload under `pricing`."""
    usage = payload.get("usage") or {}
    input_t = float(usage.get("input_tokens") or 0)
    output_t = float(usage.get("output_tokens") or 0)
    cached_in = float(usage.get("cached_input_tokens") or 0)
    thinking = float(usage.get("thinking_tokens") or 0)
    model = str(payload.get("model") or "")
    rates = pricing.get(model)
    if rates is None:
        return 0.0, input_t, output_t, cached_in, thinking
    # Rich-dict pricing (matching Rust ModelPricing).
    if isinstance(rates, dict):
        input_rate = float(rates.get("input", 0.0))
        output_rate = float(rates.get("output", 0.0))
        cached_input_rate = float(rates.get("cached_input", input_rate))
        reasoning_rate = float(rates.get("reasoning", output_rate))
        usd = (
            (input_t - cached_in) * input_rate
            + cached_in * cached_input_rate
            + output_t * output_rate
            + thinking * reasoning_rate
        )
    else:
        # Legacy (input, output) tuple.
        input_rate, output_rate = float(rates[0]), float(rates[1])
        usd = (input_t - cached_in) * input_rate + output_t * output_rate
    return usd, input_t, output_t, cached_in, thinking


def _modal_model(models: list[str]) -> str:
    """Return the most-common model name, or `mixed` if no clear modal."""
    if not models:
        return ""
    counts: dict[str, int] = {}
    for m in models:
        counts[m] = counts.get(m, 0) + 1
    top_model = max(counts, key=lambda k: counts[k])
    return top_model if counts[top_model] > len(models) / 2 else "mixed"


def session_cost(
    session: list[dict[str, Any]], pricing: dict[str, Any], session_index: int = 0
) -> SessionCost:
    """Aggregate one session's response records into a `SessionCost`."""
    total_usd = 0.0
    input_t = 0.0
    output_t = 0.0
    cached_in = 0.0
    thinking = 0.0
    models: list[str] = []
    for rec in session:
        if rec.get("kind") != "chat_response":
            continue
        p = rec.get("payload") or {}
        usd, it, ot, ci, th = _cost_of_response(p, pricing)
        total_usd += usd
        input_t += it
        output_t += ot
        cached_in += ci
        thinking += th
        if p.get("model"):
            models.append(str(p["model"]))
    return SessionCost(
        session_index=session_index,
        model=_modal_model(models),
        input_tokens=input_t,
        output_tokens=output_t,
        cached_input_tokens=cached_in,
        thinking_tokens=thinking,
        total_usd=total_usd,
    )


# ---- attribution ---------------------------------------------------------


def _counterfactual_cost(tokens: dict[str, float], model: str, pricing: dict[str, Any]) -> float:
    """Cost of a given token bag at a given model's pricing."""
    rates = pricing.get(model)
    if rates is None:
        return 0.0
    input_t = tokens.get("input_tokens", 0.0)
    output_t = tokens.get("output_tokens", 0.0)
    cached_in = tokens.get("cached_input_tokens", 0.0)
    thinking = tokens.get("thinking_tokens", 0.0)
    if isinstance(rates, dict):
        input_rate = float(rates.get("input", 0.0))
        output_rate = float(rates.get("output", 0.0))
        cached_input_rate = float(rates.get("cached_input", input_rate))
        reasoning_rate = float(rates.get("reasoning", output_rate))
        return (
            (input_t - cached_in) * input_rate
            + cached_in * cached_input_rate
            + output_t * output_rate
            + thinking * reasoning_rate
        )
    input_rate, output_rate = float(rates[0]), float(rates[1])
    return (input_t - cached_in) * input_rate + output_t * output_rate


def _session_attribution(
    baseline: SessionCost,
    candidate: SessionCost,
    pricing: dict[str, Any],
    n_responses: int,
) -> SessionAttribution:
    """Decompose one session pair's cost delta."""
    delta = candidate.total_usd - baseline.total_usd

    # `model_swap`: if we held candidate's token bag constant but
    # priced it with the baseline's model, how much would cost move?
    candidate_tokens = {
        "input_tokens": candidate.input_tokens,
        "output_tokens": candidate.output_tokens,
        "cached_input_tokens": candidate.cached_input_tokens,
        "thinking_tokens": candidate.thinking_tokens,
    }
    counterfactual_candidate_at_baseline_price = _counterfactual_cost(
        candidate_tokens, baseline.model, pricing
    )
    model_swap = candidate.total_usd - counterfactual_candidate_at_baseline_price

    # `token_movement`: holding price constant (baseline's), how much
    # does the token count movement alone contribute?
    baseline_tokens = {
        "input_tokens": baseline.input_tokens,
        "output_tokens": baseline.output_tokens,
        "cached_input_tokens": baseline.cached_input_tokens,
        "thinking_tokens": baseline.thinking_tokens,
    }
    counterfactual_baseline_tokens_at_baseline_price = _counterfactual_cost(
        baseline_tokens, baseline.model, pricing
    )
    token_movement = (
        counterfactual_candidate_at_baseline_price
        - counterfactual_baseline_tokens_at_baseline_price
    )

    # `mix_residual`: whatever's left. Captures interaction terms
    # (simultaneous model swap + token change).
    mix_residual = delta - model_swap - token_movement

    pct = (delta / baseline.total_usd * 100.0) if baseline.total_usd > 0 else float("inf")
    return SessionAttribution(
        session_index=baseline.session_index,
        baseline_usd=baseline.total_usd,
        candidate_usd=candidate.total_usd,
        delta_usd=delta,
        delta_pct=pct,
        model_swap_usd=model_swap,
        token_movement_usd=token_movement,
        mix_residual_usd=mix_residual,
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        n_responses=n_responses,
    )


def attribute_cost(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    pricing: dict[str, Any] | None = None,
) -> CostAttributionReport:
    """Compute a cost-attribution report between two traces.

    Sessions are aligned by index — session #0 of baseline vs
    session #0 of candidate, etc. Mismatched counts: extra sessions
    on either side are flagged with `n_responses=0` on the missing
    side, so the report stays square.
    """
    pricing = pricing or {}
    base_sessions = partition_sessions(baseline_records)
    cand_sessions = partition_sessions(candidate_records)

    per_session: list[SessionAttribution] = []
    n = max(len(base_sessions), len(cand_sessions))
    for i in range(n):
        base = base_sessions[i] if i < len(base_sessions) else []
        cand = cand_sessions[i] if i < len(cand_sessions) else []
        base_cost = session_cost(base, pricing, i)
        cand_cost = session_cost(cand, pricing, i)
        n_responses = sum(1 for r in base + cand if r.get("kind") == "chat_response")
        per_session.append(_session_attribution(base_cost, cand_cost, pricing, n_responses))

    total_baseline = sum(s.baseline_usd for s in per_session)
    total_candidate = sum(s.candidate_usd for s in per_session)
    total_delta = total_candidate - total_baseline
    total_model_swap = sum(s.model_swap_usd for s in per_session)
    total_token_movement = sum(s.token_movement_usd for s in per_session)
    total_mix_residual = sum(s.mix_residual_usd for s in per_session)

    # "Noisy" if residual is > 10% of the absolute total delta.
    noisy = bool(total_delta != 0 and abs(total_mix_residual) > 0.10 * abs(total_delta))

    return CostAttributionReport(
        per_session=per_session,
        total_baseline_usd=total_baseline,
        total_candidate_usd=total_candidate,
        total_delta_usd=total_delta,
        total_model_swap_usd=total_model_swap,
        total_token_movement_usd=total_token_movement,
        total_mix_residual_usd=total_mix_residual,
        attribution_is_noisy=noisy,
    )


# ---- rendering -----------------------------------------------------------


def render_terminal(report: CostAttributionReport) -> str:
    """Rich-markup-free plain terminal rendering."""
    if not report.per_session:
        return ""
    lines: list[str] = []
    lines.append("cost attribution (per session):")
    for s in report.per_session:
        pct_str = (
            f"{s.delta_pct:+.1f}%" if s.delta_pct not in (float("inf"), float("-inf")) else "—"
        )
        swap_pct = _share(s.model_swap_usd, s.delta_usd)
        move_pct = _share(s.token_movement_usd, s.delta_usd)
        mix_pct = _share(s.mix_residual_usd, s.delta_usd)
        lines.append(
            f"  session #{s.session_index}: ${s.baseline_usd:.4f} → ${s.candidate_usd:.4f} "
            f"(Δ ${s.delta_usd:+.4f}, {pct_str})"
        )
        if abs(s.delta_usd) > 1e-9:
            lines.append(
                f"    model swap {s.baseline_model}→{s.candidate_model}: "
                f"${s.model_swap_usd:+.4f} ({swap_pct})"
            )
            lines.append(
                f"    token movement:            ${s.token_movement_usd:+.4f} ({move_pct})"
            )
            if abs(s.mix_residual_usd) > 1e-9:
                lines.append(
                    f"    mix residual:              ${s.mix_residual_usd:+.4f} ({mix_pct})"
                )
    lines.append(
        f"  total: ${report.total_baseline_usd:.4f} → ${report.total_candidate_usd:.4f} "
        f"(Δ ${report.total_delta_usd:+.4f})"
    )
    if report.attribution_is_noisy:
        lines.append(
            "  note: mix residual is > 10% of total delta — attribution split is less trustworthy"
        )
    return "\n".join(lines)


def render_markdown(report: CostAttributionReport) -> str:
    """GitHub-flavoured markdown table for PR comments."""
    if not report.per_session:
        return ""
    lines: list[str] = [
        "## Cost attribution",
        "",
        "| session | baseline | candidate | Δ | model swap | token move | mix |",
        "|--------:|---------:|----------:|--:|-----------:|-----------:|----:|",
    ]
    for s in report.per_session:
        lines.append(
            f"| #{s.session_index} | "
            f"`${s.baseline_usd:.4f}` | `${s.candidate_usd:.4f}` | "
            f"`${s.delta_usd:+.4f}` | "
            f"`${s.model_swap_usd:+.4f}` ({_share(s.model_swap_usd, s.delta_usd)}) | "
            f"`${s.token_movement_usd:+.4f}` ({_share(s.token_movement_usd, s.delta_usd)}) | "
            f"`${s.mix_residual_usd:+.4f}` ({_share(s.mix_residual_usd, s.delta_usd)}) |"
        )
    lines.append("")
    lines.append(
        f"**Total:** ${report.total_baseline_usd:.4f} → "
        f"${report.total_candidate_usd:.4f} (Δ `${report.total_delta_usd:+.4f}`)"
    )
    if report.attribution_is_noisy:
        lines.append("")
        lines.append(
            "> ⚠  Mix residual is > 10% of total delta — the split between "
            "model-swap and token-movement is less trustworthy in this PR."
        )
    return "\n".join(lines)


def _share(part: float, total: float) -> str:
    """Format `part` as a percentage of `total`, handling zero/nan safely."""
    if abs(total) < 1e-9:
        return "—"
    return f"{(part / total) * 100:+.0f}%"


__all__ = [
    "CostAttributionReport",
    "SessionAttribution",
    "SessionCost",
    "attribute_cost",
    "partition_sessions",
    "render_markdown",
    "render_terminal",
    "session_cost",
]
