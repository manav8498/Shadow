# Shadow vs the rest of the AI-agent tooling lane

> **Honest matrix.** Where Shadow overlaps with adjacent tools, where the
> overlap is real, and where each tool has a unique strength Shadow doesn't.
> No vaporware claims; if a Shadow column says "✓" it's verified by a
> committed test.

Last updated: 2026-05-03 against the v0.1 strategic-pivot release.

---

## TL;DR

| Tool | What it does best | Where Shadow doesn't compete |
|---|---|---|
| **EvalView** | Golden-baseline regression testing for AI agents — tool-call/parameter diffs, production traffic capture, GitHub Actions, framework adapters | Generic eval framework |
| **Microsoft AGT** | Runtime governance with sub-millisecond policy enforcement, multiple language SDKs, OPA/Rego/Cedar support, marketplace signing | Runtime guardrails / control plane |
| **Preloop** | Open-source AI-agent control plane with MCP firewall, model gateway, policy-as-code, approvals, audit trails | MCP firewall / approvals / runtime observability |
| **AgentEvals** | Behavior scoring from OpenTelemetry traces without rerunning expensive LLM calls | Generic OTel trace scoring |
| **Speedscale** | API/code production-traffic replay before merge, before/after payload diffs | Generic API/payload replay |

**Shadow's lane:** **Causal Regression Forensics for AI Agents.** Names the
exact change that broke the agent — proven against production-like traces
before merge, with bootstrap CI + E-value, plus a verified fix loop.

---

## Where Shadow is differentiated

What Shadow does that none of the above does in one tool:

| Capability | Shadow | EvalView | MS AGT | Preloop | AgentEvals | Speedscale |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Causal cause attribution** with Pearl-style ATE + bootstrap CI + E-value sensitivity | ✓ | — | — | — | — | — |
| **Single command** that names the *exact* prompt/model/tool/config change that caused the regression | ✓ | — | — | — | — | — |
| **Verify-fix** loop closing diagnose → fix → verify | ✓ | — | — | — | — | — |
| 9-axis structured behavior diff (semantic, trajectory, safety, verbosity, latency, cost, reasoning, judge, conformance) | ✓ | partial | — | — | partial | — |
| Bootstrap CI on the per-axis severities | ✓ | — | — | — | — | — |
| Reusable trace-alignment library exposed as a category primitive | ✓ (Python + TS) | — | — | — | — | — |

---

## Where each adjacent tool wins

| Capability | Shadow | EvalView | MS AGT | Preloop | AgentEvals | Speedscale |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Hosted SaaS dashboard | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Auto-suggested test cases from production traffic | — | ✓ | — | — | partial | — |
| MCP-server firewall | — | — | — | ✓ | — | — |
| Sub-millisecond runtime policy enforcement | partial | — | ✓ | ✓ | — | — |
| OPA/Rego/Cedar policy languages | — | — | ✓ | — | — | — |
| Multi-language runtime SDKs (Python + TS + Java + .NET + Go) | partial (Py + TS) | partial | ✓ | partial | partial | ✓ |
| OTLP-collector ingestion as a first-class input | partial (file-based) | partial | ✓ | ✓ | ✓ | partial |
| Marketplace + signing (AIUC-1 / Schellman) | partial (sigstore) | — | ✓ | — | — | — |
| API/payload replay (non-LLM HTTP traffic) | — | — | — | — | — | ✓ |

---

## Choose Shadow when

- A PR-time CI gate that names the **exact** change that broke the agent matters more than dashboards.
- You need bootstrap CI + E-value on causal claims; "did behavior change" alone isn't enough.
- You want the **diagnose → fix → verify** loop closed in one tool, not three.
- You've already instrumented agents with OTel and want a causal-diagnosis layer that consumes those traces.

## Choose [EvalView / AGT / Preloop / AgentEvals / Speedscale] when

- The bullet under their column above is your primary requirement.
- You want a hosted dashboard as the main UX (Shadow is CLI-first; the GitHub Action is the only first-class UI).
- You need runtime policy enforcement with sub-millisecond latency in production (Shadow's `policy_runtime` exists but isn't the headline; AGT and Preloop are purpose-built here).

---

## What Shadow doesn't try to be

(Repeating the design spec §1.3 explicit non-goals so this comparison is
truthful, not aspirational.)

- ABOM expansion beyond the existing certificate format
- Generic runtime governance suite
- A control plane competing with Microsoft AGT or Preloop
- Generic agent-eval framework competing with EvalView
- Certification marketplace
- Broad MCP firewall

The strategic-pivot lane is **causal regression forensics**. If an
adjacent tool already nails one of the above, Shadow integrates with it
(`shadow import --format otel-genai`) rather than replicating it.

---

## Sources

* [EvalView](https://evalview.com/)
* [Microsoft Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
* [Preloop on GitHub](https://github.com/preloop/preloop)
* [LangChain AgentEvals](https://github.com/langchain-ai/agentevals)
* [Speedscale](https://speedscale.com/)
* [AIUC-1 / Schellman certification](https://aiuc.com/)

This page is a living comparison; PRs welcome to update tool capabilities
as they ship features. Shadow capabilities marked "✓" are pinned by
committed tests; if a claim is wrong, file an issue and link the failing
test.
