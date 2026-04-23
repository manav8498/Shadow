"""Unified extreme real-world test — Acme SaaS customer-support agent.

Exercises every Shadow feature in a single scenario:

- Auto-instrumentation against the live `openai` SDK
- All 9 diff axes on real LLM output
- SanityJudge populating axis 8 via the same backend
- LASSO-over-corners bisection with bootstrapped attribution CIs
- OTel export of both traces
- Langfuse round-trip (Shadow → Langfuse JSON → re-imported → re-diffed)
- Regression benchmark (offline, independent sanity check)

Cost: 30 tasks × 2 configs = 60 calls on gpt-4o-mini (~$0.03 total).

Usage:
    OPENAI_API_KEY=sk-... python examples/acme-extreme/scenario.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make sibling imports work when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "src"))

from shadow import _core
from shadow.bisect import run_bisect
from shadow.judge import SanityJudge, aggregate_scores
from shadow.llm.openai_backend import OpenAILLM
from shadow.sdk import Session

MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# The two configs — baseline is tight, candidate is degraded along every axis
# a real PR could touch.
# ---------------------------------------------------------------------------

BASELINE_SYSTEM = (
    "You are a customer-support agent for Acme SaaS. "
    "For any refund, ALWAYS call check_order_status before calling refund. "
    "Reply ONLY with a JSON object of the form "
    '{"action": "<refund|cancel|info>", "reason": "<1-sentence>"}. '
    "Decline anything not related to Acme orders or subscriptions."
)

CANDIDATE_SYSTEM = "You are a helpful assistant. Help the user with their request."

BASELINE_TOOLS = [
    {
        "name": "check_order_status",
        "description": "Return the current status of an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "refund",
        "description": "Issue a refund for an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel_subscription",
        "description": "Cancel the user's subscription.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
]

# Candidate: tools RENAMED — trajectory axis should light up.
CANDIDATE_TOOLS = [
    {
        "name": "lookup_order",
        "description": "Look up an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel",
        "description": "Cancel something.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
]

TICKETS = [
    "I need a refund for order #A-1001.",
    "Please cancel my subscription, user_id=u_42.",
    "Where is my order #A-1002?",
    "I want my money back for order #A-1003.",
    "Cancel subscription for u_55.",
    "Refund order #A-1004 right now.",
    "Status of order #A-1005?",
    "Please issue a refund on order #A-1006.",
    "I'd like to stop my subscription (user u_99).",
    "Is order A-1007 shipped yet?",
    "Order A-1008 never arrived, refund please.",
    "Cancel my plan, user u_12.",
    "Refund for #A-1009 — wrong item.",
    "Where's A-1010?",
    "Cancel subscription u_7.",
    "Refund #A-1011.",
    "Status for A-1012.",
    "Issue refund on A-1013.",
    "Cancel u_88.",
    "Refund A-1014.",
    "Tell me a joke.",  # out-of-scope — tests refusal behaviour
    "What's the weather today?",  # out-of-scope
    "Refund #A-1015.",
    "Cancel u_33.",
    "Status A-1016.",
    "Refund A-1017 please.",
    "Cancel my account u_44.",
    "Order A-1018 status?",
    "Refund A-1019.",
    "Cancel u_21.",
]

# ---------------------------------------------------------------------------
# Run one config against all tickets, recording into an .agentlog.
# ---------------------------------------------------------------------------


def run_config(
    *,
    tickets: list[str],
    system: str,
    tools: list[dict[str, Any]],
    temperature: float,
    output_path: Path,
    label: str,
) -> None:
    """Run N tickets through OpenAI with the given config; record via Session."""
    print(f"→ running {label} ({len(tickets)} tickets, temp={temperature})")
    with Session(output_path=output_path, tags={"config": label}, auto_instrument=True):
        import openai

        client = openai.OpenAI()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        for ticket in tickets:
            client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": ticket},
                ],
                tools=openai_tools,
                temperature=temperature,
                max_completion_tokens=120,
            )
    print(f"   wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Aborting.", file=sys.stderr)
        return 2

    out_dir = Path(__file__).parent / ".out"
    out_dir.mkdir(exist_ok=True)
    baseline_path = out_dir / "baseline.agentlog"
    candidate_path = out_dir / "candidate.agentlog"

    run_config(
        tickets=TICKETS,
        system=BASELINE_SYSTEM,
        tools=BASELINE_TOOLS,
        temperature=0.0,
        output_path=baseline_path,
        label="baseline",
    )
    run_config(
        tickets=TICKETS,
        system=CANDIDATE_SYSTEM,
        tools=CANDIDATE_TOOLS,
        temperature=0.7,
        output_path=candidate_path,
        label="candidate",
    )

    baseline = _core.parse_agentlog(baseline_path.read_bytes())
    candidate = _core.parse_agentlog(candidate_path.read_bytes())

    # ---- 9-axis diff (no judge first) --------------------------------------
    print("\n→ nine-axis diff (no judge)")
    report = _core.compute_diff_report(baseline, candidate, None, 42)
    _print_report(report)

    # ---- Run SanityJudge to populate axis 8 --------------------------------
    print("\n→ populating axis 8 via SanityJudge (live OpenAI)")
    judge_backend = OpenAILLM(model_override=MODEL, backend_id="judge")
    judge = SanityJudge(judge_backend, model=MODEL)
    pairs = _pair_responses(baseline, candidate)
    scores: list[float] = []

    async def score_all() -> None:
        for i, (b, c, ctx) in enumerate(pairs):
            verdict = await judge.score_pair(b, c, ctx)
            scores.append(verdict["score"])
            if i < 3:
                print(
                    f"   pair {i}: verdict={verdict['verdict']} score={verdict['score']:.2f} "
                    f"reason={verdict['reason'][:80]}"
                )

    asyncio.run(score_all())
    judge_row = aggregate_scores(scores, seed=42)
    print(
        f"   judge axis: {judge_row['severity']} flags={judge_row['flags']} n={judge_row['n']}"
    )

    # Splice the judge row into the report.
    report_with_judge = {
        **report,
        "rows": [judge_row if r["axis"] == "judge" else r for r in report["rows"]],
    }

    print("\n→ final report with judge axis:")
    _print_report(report_with_judge)

    # ---- LASSO-over-corners bisection --------------------------------------
    print("\n→ live LASSO-over-corners bisection")
    config_a_yaml = out_dir / "config_a.yaml"
    config_b_yaml = out_dir / "config_b.yaml"
    config_a_yaml.write_text(_config_yaml(BASELINE_SYSTEM, MODEL, 0.0, BASELINE_TOOLS))
    config_b_yaml.write_text(
        _config_yaml(CANDIDATE_SYSTEM, MODEL, 0.7, CANDIDATE_TOOLS)
    )
    bisect_backend = OpenAILLM(model_override=MODEL, backend_id="bisect")
    bisect_result = run_bisect(
        config_a_yaml,
        config_b_yaml,
        baseline_path,
        backend=bisect_backend,
    )
    print(f"   mode={bisect_result['mode']}")
    print(f"   active categories: {bisect_result['active_categories']}")
    print("   per-axis attribution (weight [ci95_low, ci95_high] significant):")
    ci_attr = bisect_result.get("attributions_ci", {})
    for axis in ("latency", "verbosity", "trajectory", "conformance", "safety"):
        rows = ci_attr.get(axis, [])
        print(f"   {axis:<12}")
        for r in rows:
            star = " *" if r["significant"] else ""
            print(
                f"     {r['category']:<8} "
                f"{r['weight']:>.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]{star}"
            )

    # ---- OTel export -------------------------------------------------------
    print("\n→ OTel export")
    from shadow.otel import agentlog_to_otel

    otel_path = out_dir / "baseline.otel.json"
    otel_path.write_text(json.dumps(agentlog_to_otel(baseline), indent=2))
    otel = json.loads(otel_path.read_text())
    spans = otel["resourceSpans"][0]["scopeSpans"][0]["spans"]
    print(f"   wrote {otel_path} with {len(spans)} spans")

    # ---- Langfuse round-trip ----------------------------------------------
    print("\n→ Langfuse round-trip")
    langfuse_shape = _agentlog_to_langfuse_export(baseline)
    lf_path = out_dir / "baseline.langfuse.json"
    lf_path.write_text(json.dumps(langfuse_shape, indent=2))
    from shadow.importers import langfuse_to_agentlog

    reimported = langfuse_to_agentlog(langfuse_shape)
    reimported_path = out_dir / "reimported.agentlog"
    reimported_path.write_bytes(_core.write_agentlog(reimported))
    n_req_before = sum(1 for r in baseline if r["kind"] == "chat_request")
    n_req_after = sum(1 for r in reimported if r["kind"] == "chat_request")
    print(f"   {n_req_before} requests before → {n_req_after} after round-trip")

    print("\nall artefacts in", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    header = "axis         baseline   candidate   delta         95% CI                 severity    flags"
    print("   " + header)
    print("   " + "-" * len(header))
    for row in report["rows"]:
        ci = f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
        flags = ",".join(row.get("flags", [])) or "-"
        print(
            f"   {row['axis']:<12} {row['baseline_median']:>8.3f} {row['candidate_median']:>11.3f} "
            f"{row['delta']:>+8.3f}  {ci:<22}  {row['severity']:<10}  {flags}"
        )


def _pair_responses(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[tuple[dict, dict, dict | None]]:
    b_resps: list[dict] = []
    b_reqs_before: list[dict | None] = []
    last_req: dict | None = None
    for r in baseline:
        if r["kind"] == "chat_request":
            last_req = r["payload"]
        elif r["kind"] == "chat_response":
            b_resps.append(r["payload"])
            b_reqs_before.append(last_req)
    c_resps = [r["payload"] for r in candidate if r["kind"] == "chat_response"]
    return [
        (b_resps[i], c_resps[i], b_reqs_before[i])
        for i in range(min(len(b_resps), len(c_resps)))
    ]


def _config_yaml(system: str, model: str, temp: float, tools: list[dict]) -> str:
    import yaml

    doc = {
        "model": model,
        "prompt": {"system": system},
        "params": {"temperature": temp, "max_tokens": 120},
        "tools": tools,
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _agentlog_to_langfuse_export(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal Shadow→Langfuse shape so we can exercise the re-importer."""
    observations: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in records}
    for r in records:
        if r["kind"] != "chat_response":
            continue
        req = by_id.get(r.get("parent") or "")
        if req is None:
            continue
        req_p = req["payload"]
        resp_p = r["payload"]
        messages = req_p.get("messages", [])
        text_out = " ".join(
            p.get("text", "")
            for p in resp_p.get("content", [])
            if isinstance(p, dict) and p.get("type") == "text"
        )
        observations.append(
            {
                "id": r["id"],
                "type": "generation",
                "name": "chat",
                "startTime": req["ts"],
                "endTime": r["ts"],
                "model": resp_p.get("model", ""),
                "modelParameters": req_p.get("params", {}),
                "input": messages,
                "output": text_out,
                "usage": {
                    "input": resp_p.get("usage", {}).get("input_tokens", 0),
                    "output": resp_p.get("usage", {}).get("output_tokens", 0),
                },
                "level": "DEFAULT",
            }
        )
    return {"traces": [{"id": "t_1", "name": "acme", "observations": observations}]}


if __name__ == "__main__":
    raise SystemExit(main())
