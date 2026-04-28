<p align="center">
  <img src=".github/assets/logo.png" alt="Shadow" width="160" />
</p>

# Shadow

[![ci](https://github.com/manav8498/Shadow/actions/workflows/ci.yml/badge.svg)](https://github.com/manav8498/Shadow/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)
[![spec](https://img.shields.io/badge/.agentlog-v0.1%20%2B%20v0.2%20kinds-6f4cff.svg)](SPEC.md)
[![version](https://img.shields.io/badge/version-2.7.0-brightgreen.svg)](CHANGELOG.md)
[![rust](https://img.shields.io/badge/rust-1.95+-orange.svg)](rust-toolchain.toml)
[![python](https://img.shields.io/badge/python-3.11+-3776ab.svg)](python/pyproject.toml)

**Behavior testing for LLM agents, in the pull request.**

Shadow catches behavior regressions in AI agents before they merge. You change a prompt, swap a model, or rename a tool argument. Your agent still runs, tests still pass, but the behavior quietly shifts. Shadow replays your change against recorded agent traces and posts a behavior diff on the PR so a reviewer can see what broke and why.

## The problem

You have a working agent in production. A teammate opens a PR that tweaks the system prompt, swaps GPT-4o for a cheaper model, or adjusts a tool schema. Code review looks fine. Unit tests pass. You merge.

A week later a customer reports that the refund bot started issuing refunds without confirming the amount. It turns out the prompt edit dropped the "ask before refunding" step. The PR that caused it was merged days ago. Nobody saw it coming because the code looked harmless.

This is a common class of bug with LLM agents. The agent runs, responses look plausible, tests pass. The behavior just silently changed.

## What Shadow does

Shadow treats agent behavior as a thing you can test in CI, the same way you test code. Given a recorded set of real agent interactions (a baseline), and a candidate change (new prompt, new model, renamed tool), Shadow answers three questions on the PR:

1. **What behavior changed?** A nine-axis diff scores the candidate against the baseline on things like meaning, tool use, refusals, latency, and output structure.
2. **Why did it change?** If the PR touched multiple things at once, causal bisection estimates which specific change most likely explains each regression, then points you at the replay / counterfactual primitives to confirm it before merging.
3. **Is it safe to merge?** A policy file lets you declare rules the agent must follow (tool ordering, output shape, token budgets, forbidden outputs). Shadow reports regressions against those rules.

The report lands in the PR comment. No dashboard, no separate login, no trace upload. Traces stay on your disk.

## Install

**Step 1.** Check you have Python 3.11 or newer:

```bash
python3 --version    # 3.11.x or higher
```

**Step 2.** Install Shadow from PyPI:

```bash
pip install shadow-diff
```

That's it. `shadow --help` should now work, and `shadow quickstart` runs the demo. No clone, no Rust toolchain, no separate setup step — pre-built wheels ship for Linux x86_64, macOS arm64 (Apple Silicon), and Windows x86_64.

**On other platforms** (Intel Mac, ARM Linux, older glibc, Alpine, FreeBSD), pip falls back to the source distribution and builds the Rust core locally. Install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install shadow-diff
```

### Optional extras

Shadow's core install is intentionally lean. Optional integrations are gated behind extras so users only pay the dependency cost they actually need:

| Extra | Pulls in | Use case |
|---|---|---|
| `shadow-diff[anthropic]` | `anthropic` | Live Anthropic client wrapper for `shadow record` |
| `shadow-diff[openai]` | `openai` | Live OpenAI client wrapper for `shadow record` |
| `shadow-diff[embeddings]` | `sentence-transformers` | Paraphrase-robust semantic-similarity axis. The default lexical TF-IDF path stays fast and dependency-free; the pluggable `Embedder` trait in `shadow-core::diff::embedder` accepts any backend (this extra, ONNX runtime, HF Inference API, OpenAI embeddings, ...). |
| `shadow-diff[otel]` | `opentelemetry-sdk` | Export traces to any OTel-compatible backend |
| `shadow-diff[serve]` | `fastapi`, `uvicorn`, `websockets` | `shadow serve` HTTP dashboard |
| `shadow-diff[mcp]` | `mcp` | `shadow mcp-serve` Model Context Protocol server |
| `shadow-diff[multimodal]` | `Pillow`, `imagehash` | Image / multimodal diff |
| `shadow-diff[sign]` | `sigstore` | Sigstore keyless signing for ABOM certificates |
| `shadow-diff[langgraph]` | `langgraph` | LangGraph agent adapter |
| `shadow-diff[crewai]` | `crewai` | CrewAI agent adapter |
| `shadow-diff[ag2]` | `ag2` | AG2 / Autogen agent adapter |
| `shadow-diff[all]` | everything above | One-shot install for trying the full feature surface |
| `shadow-diff[dev]` | test/lint/type-check tooling | Contributing to Shadow itself |

Combine extras with comma-separated form:

```bash
pip install 'shadow-diff[anthropic,openai,embeddings,otel]'
```

## Try it in sixty seconds

```bash
shadow quickstart
shadow diff shadow-quickstart/fixtures/baseline.agentlog \
            shadow-quickstart/fixtures/candidate.agentlog
```

That runs a real nine-axis diff on recorded `.agentlog` fixtures. No API key, no agent code. The output looks like this:

```
axis         baseline  candidate     delta     severity
─────────────────────────────────────────────────────────
semantic        1.000      0.435    -0.565     severe
trajectory      0.000      0.000    +0.000     none
safety          0.000      0.333    +0.333     severe
verbosity      26.000     52.000   +26.000     minor
latency        98.000    412.000  +314.000     severe
cost            0.000      0.000    +0.000     none
reasoning       0.000      0.000    +0.000     none
judge           0.000      0.000    +0.000     none
conformance     1.000      0.000    -1.000     severe

top divergences (3 shown):
  #1  baseline turn #0 ↔ candidate turn #0
      kind: structural_drift  ·  axis: trajectory  ·  confidence: 56%
      tool set changed: removed `search_files(query)`,
                        added `search_files(limit,query)`
  #2  baseline turn #2 ↔ candidate turn #2
      kind: decision_drift    ·  axis: safety      ·  confidence: 32%
      stop_reason changed: `end_turn` → `content_filter`

recommendations (3):
  error   REVIEW  Review tool-schema change at turn 0: call shape diverged.
  error   REVIEW  Review refusal behaviour at turn 2: candidate may be over-refusing.
  warning REVIEW  Review response text at turn 1: semantic content shifted.
```

The severity column points the reviewer at the four axes that moved. The top-divergences list names the specific changes. The recommendations tell them what to check first.

## Writing behavior rules

The diff tells you what changed. A policy tells you what is not allowed to change. Write one YAML file that declares the agent's behavioral contract:

```yaml
# shadow-policy.yaml
rules:
  - id: confirm-before-refund
    kind: must_call_before
    params: { first: confirm_refund_amount, then: issue_refund }
    severity: error

  - id: never-leak-ssn
    kind: forbidden_text
    params: { text: "SSN:" }
    severity: error

  - id: finish-cleanly
    kind: required_stop_reason
    params: { allowed: [end_turn, tool_use] }
    severity: error

  - id: cost-ceiling
    kind: max_total_tokens
    params: { limit: 100000 }
```

Run:

```bash
shadow diff baseline.agentlog candidate.agentlog --policy shadow-policy.yaml
```

The candidate trace is checked against every rule. Violations that are new in the candidate are flagged as regressions. Violations that existed in the baseline and are now cleared are flagged as fixes. Twelve rule kinds ship today: `must_call_before`, `must_call_once`, `no_call`, `max_turns`, `required_stop_reason`, `max_total_tokens`, `must_include_text`, `forbidden_text`, `must_match_json_schema`, `must_remain_consistent`, `must_followup`, `must_be_grounded` (cheap lexical grounding gate, not NLI-backed faithfulness — see [docs/features/policy.md](docs/features/policy.md) for what it catches and what it doesn't).

`must_match_json_schema` is the structured-output assertion: every chat response is parsed as JSON and validated against a JSON Schema. Mismatches name the offending dotted path so reviewers see exactly which field broke.

```yaml
rules:
  - id: structured-output
    kind: must_match_json_schema
    params:
      schema_path: schemas/refund_decision.schema.json
    severity: error
```

Supply either an inline `schema:` dict or a `schema_path:` to a JSON Schema file. NaN / Infinity literals are rejected because they aren't valid JSON per RFC 8259 even though Python's parser accepts them.

Each rule can carry a `when:` clause that gates it on field-path conditions, so a rule fires only on the matching subset of pairs:

```yaml
rules:
  - id: confirm-large-refunds
    kind: forbidden_text
    params: { text: "refund issued" }
    when:
      - { path: "request.params.amount", op: ">", value: 500 }
      - { path: "request.model", op: "==", value: "gpt-4.1" }
```

Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`. Multiple conditions AND together. Missing paths quietly don't match (rule is skipped on that pair) instead of crashing the whole check.

This is the part that makes Shadow feel like CI for agents instead of monitoring. See [docs/features/policy.md](docs/features/policy.md) for the full rule reference, conditional gating semantics, and severity → `--fail-on` mapping.

## Block bad behavior at runtime

The same policy file can run inside the SDK to block or replace a violating model response at record time, not just after the fact:

```python
from shadow.policy_runtime import EnforcedSession, PolicyEnforcer

enforcer = PolicyEnforcer.from_policy_file("shadow-policy.yaml")
with EnforcedSession(enforcer=enforcer, output_path="run.agentlog") as s:
    s.record_chat(request=..., response=...)
```

When a recorded turn introduces a new violation, the session swaps the response for a refusal payload by default (`stop_reason: "policy_blocked"`) so downstream code keeps running. Set `on_violation="raise"` for hard failure, `"warn"` for log-only. The enforcer is incremental — whole-trace rules fire once when crossed, not once per recorded record.

For dangerous tools (`issue_refund`, `send_email`, `execute_sql`, `delete_user`), wrap the tool registry to enforce BEFORE the function runs:

```python
guarded = s.wrap_tools({
    "issue_refund": issue_refund,
    "delete_user": delete_user,
})
result = guarded["delete_user"](user_id="u-42")
# → blocked by no_call rule, real delete_user never called
```

The wrapper probes the enforcer with a synthesised candidate `tool_call` record. Tool-sequence rules (`no_call`, `must_call_before`, `must_call_once`) all work pre-dispatch. Response-text rules stay on `record_chat`. See [docs/features/runtime-enforcement.md](docs/features/runtime-enforcement.md) for the full surface, including standalone `wrap_tools(..., records_provider=...)` for framework-adapter integrations.

## Recording real agent traces

Shadow's SDK auto-instruments the Anthropic and OpenAI SDKs. No code changes to the agent itself:

```bash
shadow record -o baseline.agentlog -- python your_agent.py

# change a prompt, swap a model, re-record
shadow record -o candidate.agentlog -- python your_agent.py

shadow diff baseline.agentlog candidate.agentlog
```

If you want more control (custom tags, a non-default redactor, nested sessions), use the `Session` context manager:

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog", tags={"env": "prod"}):
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

Secrets (API keys, emails, credit cards) are redacted by default.

The TypeScript SDK covers the recording side of this same workflow plus a CI-gating decision surface. Numerical analyses that depend on the Rust core (replay, diff, bisect, certify, MCP server) stay on the Python/CLI side:

| Feature | Python | TypeScript |
|---|:---:|:---:|
| `.agentlog` write / parse / canonicalisation | ✅ | ✅ |
| `Session` context manager | ✅ | ✅ |
| Redaction | ✅ | ✅ |
| Distributed-trace (W3C) propagation | ✅ | ✅ |
| OpenAI Chat Completions + Anthropic Messages auto-instrument | ✅ | ✅ |
| OpenAI Responses API auto-instrument | ✅ | ✅ |
| Streaming aggregation in auto-instrument | ✅ | ✅ |
| LTLf evaluator (bottom-up DP, all 10 operators) | ✅ | ✅ |
| Policy gating (`no_call`, `must_call_before`, `must_call_once`, `forbidden_text`, `must_include_text`) | ✅ | ✅ |
| `gate(records, { rules, ltlFormulas })` CI decision | ✅ (via `shadow.policy_runtime`) | ✅ |
| Runtime policy enforcement (`EnforcedSession`, pre-dispatch tool guards) | ✅ | ❌ |
| `shadow certify` / `--sign` / `verify-cert` | ✅ (CLI) | ❌ |
| `shadow diff` / `bisect` / `replay` / `mine` | ✅ (CLI) | ❌ |
| MCP server (`shadow mcp-serve`) | ✅ (CLI) | ❌ |

The Python SDK and TypeScript SDK ship lockstep at the same version. The `.agentlog` format itself is the contract — TS-recorded traces feed into Python's `shadow diff`, `shadow certify`, and the MCP server without translation. The TS gate decisions are byte-identical to the Python equivalents on the same fixtures (cross-validated by `python/tests/test_typescript_parity.py`). For deeper analyses (multi-axis diff, bisect, certify), run those from the Python CLI against the TS-recorded trace.

If your agent is built on LangGraph, CrewAI, or AG2, prefer the matching adapter (next section) over auto-instrumentation. Auto-instrument patches `.create` on the underlying provider SDK, which is a moving target across SDK majors. The framework adapters hook each framework's documented extension surface, which is the more stable contract.

## Record from agent frameworks

If your agent runs on a framework, Shadow has a direct hook for each of the three most common ones. Install the matching extra and drop the handler in; no monkey-patch, nothing to rewrite in the agent.

**LangGraph / LangChain**

```python
from shadow.sdk import Session
from shadow.adapters.langgraph import ShadowLangChainHandler

with Session(output_path="trace.agentlog") as s:
    handler = ShadowLangChainHandler(s)
    graph.invoke(
        {"messages": [HumanMessage("...")]},
        config={"callbacks": [handler],
                "configurable": {"thread_id": "t-42"}},
    )
```

`pip install 'shadow-diff[langgraph]'`. Works under `invoke` and `ainvoke`. The `thread_id` from the config carries through as the session boundary, so one `invoke` is one session even across tool loops and fan-outs.

**CrewAI**

```python
from shadow.sdk import Session
from shadow.adapters.crewai import ShadowCrewAIListener

with Session(output_path="trace.agentlog") as s:
    ShadowCrewAIListener(s)
    crew.kickoff(inputs={"topic": "..."})
```

`pip install 'shadow-diff[crewai]'`. One `Crew.kickoff()` is one session, even when it triggers many LLM calls; the adapter marks the boundary on `CrewKickoffStartedEvent`.

**AG2 (formerly AutoGen)**

```python
from shadow.sdk import Session
from shadow.adapters.ag2 import ShadowAG2Adapter

with Session(output_path="trace.agentlog") as s:
    adapter = ShadowAG2Adapter(s)
    adapter.install_all([planner, executor])
    planner.initiate_chat(executor, message="...")
```

`pip install 'shadow-diff[ag2]'`. Captures the message bodies that `autogen.opentelemetry` redacts by default, so semantic diffs have something to compare against.

## Replay the candidate, end-to-end, without touching production

For a candidate change to a prompt or model, `shadow diff` shows what's different between two recorded traces. Sandboxed replay drives the candidate's agent loop *forward* against a baseline and produces a candidate trace without making any real LLM calls or running any real tool side effects:

```bash
shadow replay candidate.yaml \
  --baseline baseline.agentlog \
  --agent-loop \
  --tool-backend replay \
  --novel-tool-policy stub \
  --output candidate.agentlog
```

`--tool-backend replay` resolves every tool call against the baseline's recorded results. `--novel-tool-policy` decides what happens when the candidate calls a tool the baseline didn't (`strict` aborts, `stub` returns a placeholder, `fuzzy` matches the nearest same-tool call by arg shape). For real tool functions with side effects you'd otherwise hit, the programmatic API exposes `SandboxedToolBackend` which patches `socket.connect`, `subprocess.run`, and write-mode `open()` calls during execution. Counterfactual primitives (`branch_at_turn`, `replace_tool_result`, `replace_tool_args`) let you isolate one variable at a time. See [docs/features/sandboxed-replay.md](docs/features/sandboxed-replay.md).

## Import traces from any OpenTelemetry backend

If you already export OTLP to Datadog, Honeycomb, or any OTel collector, pipe that same export into Shadow:

```bash
shadow import traces.json --format otel --output my.agentlog
```

Reads the full GenAI semantic convention v1.40 surface: structured `gen_ai.input.messages` / `gen_ai.output.messages`, `gen_ai.provider.name`, cache tokens, tool definitions, agent spans, evaluation events. Also accepts the older v1.28-v1.36 flat indexed attributes, so traces from OpenLLMetry and similar implementers that haven't tracked the v1.37 restructure still round-trip cleanly.

## Wire it into every pull request

```bash
shadow init --github-action
```

Drops a ready-to-commit workflow at `.github/workflows/shadow-diff.yml`. Point the `BASELINE` and `CANDIDATE` paths at fixtures you commit, and every PR gets a behavior-diff comment.

To gate the merge, add `--fail-on severe` (or `moderate` / `minor`) to the `shadow diff` step. The PR comment posts first; the gate runs as a separate step so a blocked PR still has the explanation.

```bash
shadow diff baseline.agentlog candidate.agentlog \
  --policy shadow-policy.yaml \
  --fail-on severe
```

Exits 1 when the worst axis severity or policy regression hits the threshold; 0 otherwise.

## Sign every release with an Agent Behavior Certificate

```bash
shadow certify candidate.agentlog \
  --agent-id refund-agent@2.3.0 \
  --policy shadow-policy.yaml \
  --baseline baseline.agentlog \
  --output release.cert.json

shadow verify-cert release.cert.json
```

Produces a content-addressed JSON release artifact (Agent Behavior Bill of Materials) that captures the trace's content-id, all distinct models observed, content-ids of system prompts and tool schemas, the policy file hash, and an optional baseline-vs-candidate nine-axis regression-suite rollup. The certificate is self-verifying: `verify-cert` recomputes the body hash and exits 1 on tamper, so it works as a release gate.

Add `--sign` to layer cosign / sigstore keyless signing on top:

```bash
pip install 'shadow-diff[sign]'

shadow certify candidate.agentlog \
  --agent-id refund-agent@2.3.0 \
  --output release.cert.json \
  --sign

shadow verify-cert release.cert.json \
  --verify-signature \
  --cert-identity 'https://github.com/org/repo/.github/workflows/release.yml@refs/tags/v2.3.0'
```

The signed payload is the canonical certificate body, so tampering breaks both `cert_id` and the signature. The signature is bound to a specific signer identity (a workflow URL or email) — a leaked Bundle signed by another identity won't verify even if the crypto is otherwise valid. See [docs/features/certificate.md](docs/features/certificate.md) for the full format, signing details, and MCP integration.

## Use Shadow from an agentic CLI (MCP server)

Shadow speaks the Model Context Protocol. Any MCP-aware client (Claude Desktop, Claude Code, Cursor, Zed, Windsurf, and others) can invoke Shadow as a tool:

```json
{
  "mcpServers": {
    "shadow": {
      "command": "shadow",
      "args": ["mcp-serve"]
    }
  }
}
```

Tools exposed: `shadow_diff`, `shadow_check_policy`, `shadow_token_diff`, `shadow_schema_watch`, `shadow_summarise`, `shadow_certify`, `shadow_verify_cert`. Install the extra first: `pip install 'shadow-diff[mcp]'`. See [docs/features/mcp-server.md](docs/features/mcp-server.md) for the per-tool reference.

## Mine production traces into a regression suite

Most teams never write eval sets because it's tedious. Let Shadow do it from your production traces:

```bash
shadow mine production.agentlog --output suite.agentlog --max-cases 50
```

Clusters every turn-pair by tool sequence, stop reason, and verbosity, picks the most interesting example from each cluster (errors, refusals, high cost, heavy reasoning, very long or empty responses), and writes a new `.agentlog` you can commit as your CI baseline.

## Why regressions happened, not just that they happened

When a PR changes three things at once (prompt + model + tool schema), a diff alone cannot tell you which one broke the agent. `shadow bisect` fits a sparse linear model (LASSO over corners with Meinshausen-Bühlmann stability selection) that attributes each behavioral axis's regression to specific config deltas:

```bash
shadow bisect config_a.yaml config_b.yaml \
  --traces baseline.agentlog --candidate-traces candidate.agentlog
```

Output:

```
attribution:
  trajectory   ← search_files.arguments.limit added     (weight 0.72)
  semantic     ← system_prompt line 42 changed          (weight 0.19)
  latency      ← model: claude-haiku → gpt-4o-mini      (weight 0.61)
```

The review comment tells you: "72% of the trajectory regression is explained by the tool-schema change. Revert that line and the agent should behave."

## The nine behavioral dimensions

Each dimension is measured independently with a bootstrap 95% confidence interval. Severity is one of none, minor, moderate, severe:

| # | Dimension | What it measures |
|--:|---|---|
| 1 | `semantic` | How different are the outputs' meanings? |
| 2 | `trajectory` | Did the agent use a different sequence of tools? |
| 3 | `safety` | Did refusal rates change? |
| 4 | `verbosity` | Are outputs longer or shorter? |
| 5 | `latency` | Is it slower or faster? |
| 6 | `cost` | Are token costs up or down? |
| 7 | `reasoning` | Is the agent thinking less or more? |
| 8 | `judge` | Your own LLM-judge rubric (optional). |
| 9 | `conformance` | Does the output match the expected structure? |

Per-axis math, severity bands, and bootstrap details: [`docs/features/nine-axis.md`](docs/features/nine-axis.md). The on-disk trace format is in [`SPEC.md`](SPEC.md).

## Where Shadow fits among existing tools

| | Langfuse | Braintrust | LangSmith | **Shadow** |
|---|:---:|:---:|:---:|:---:|
| Trace logging | ✅ | ✅ | ✅ | ✅ |
| Dashboard UI | ✅ | ✅ | ✅ | no |
| Local-first / repo-native | partial (self-host) | partial (self-host) | no | ✅ |
| PR comment from CI | partial | partial | partial | ✅ |
| Declarative YAML behavior policy | partial via evals | partial via evals | partial via evals | ✅ |
| Merge-blocking PR check | partial via webhooks | partial via webhooks | partial via webhooks | ✅ |
| Content-addressed release certificate | no | no | no | ✅ |
| Cosign / sigstore signing on certificate | no | no | no | ✅ |
| Estimated causal attribution (LASSO + bootstrap CI) | no | no | no | ✅ |
| Nine pre-built behavior axes | partial | partial | partial | ✅ |
| Open content-addressed trace format | no | no | no | ✅ |

The "partial" cells reflect that all three platforms support evals + webhooks + custom CI integrations that a determined team can build into a PR-comment / gate workflow. Shadow's claim isn't that those tools can't be wired up — it's that Shadow ships the workflow as a single command, and ships an open trace format, declarative policy language, and signed release certificate as primitives. Pair Shadow with any of these tools for the dashboard side.

## Examples

Every example runs offline from committed fixtures. No API key required:

| Example | What it shows |
|---|---|
| [`examples/demo/`](examples/demo/) | The fastest working example. `just demo`. |
| [`examples/customer-support/`](examples/customer-support/) | Refund bot that regresses after a well-meaning prompt edit |
| [`examples/devops-agent/`](examples/devops-agent/) | Database agent with a tool-ordering bug that unit tests would miss |
| [`examples/er-triage/`](examples/er-triage/) | High-stakes clinical scenario with safety rules |
| [`examples/edge-cases/`](examples/edge-cases/) | 20 adversarial probes used as a regression guard |
| [`examples/refund-agent-audit/`](examples/refund-agent-audit/) | Statistical safety audit on a model upgrade (Hotelling T² + SPRT + LTL + conformal) |
| [`examples/canary-monitor/`](examples/canary-monitor/) | Production canary with always-valid mSPRT and Bonferroni-corrected family-wise error |
| [`examples/harmful-content-judge/`](examples/harmful-content-judge/) | Domain-aware harm detection where the safety axis isn't enough |
| [`examples/production-incident-suite/`](examples/production-incident-suite/) | Five public-incident patterns (Air Canada, Avianca, NEDA, McDonald's, Replit) caught by the v2.5+ pipeline |
| [`examples/integrations/`](examples/integrations/) | Push traces to Datadog, Splunk, or any OTel collector |

## Statistical, formal, and causal primitives (v2.5+)

Shadow ships a layer most LLM-eval tools don't have — empirically-validated statistical and formal-verification primitives that give the certificates real meaning. These compose with the nine-axis diff above.

| Module | What it does | Reference |
|---|---|---|
| [`shadow.statistical`](python/src/shadow/statistical/) | Behavioral fingerprinting, Hotelling T² (with OAS shrinkage and permutation p-values), Wald + mixture SPRT, variance-adaptive `MSPRTtDetector` | [docs/theory/sprt.md](docs/theory/sprt.md), [docs/theory/hotelling.md](docs/theory/hotelling.md) |
| [`shadow.ltl`](python/src/shadow/ltl/) | Finite-trace LTLf model checking with bottom-up DP (O(\|π\|×\|φ\|)); `WeakUntil` for "must-call-before" rules; YAML compiler | [docs/theory/ltl.md](docs/theory/ltl.md) |
| [`shadow.conformal`](python/src/shadow/conformal.py) | Distribution-free split conformal (`conformal_calibrate`); Adaptive Conformal Inference (`ACIDetector`, Gibbs & Candès 2021) for distribution shift | [docs/theory/conformal.md](docs/theory/conformal.md) |
| [`shadow.causal`](python/src/shadow/causal/) | Pearl-style do-calculus attribution: intervention-based ATE, percentile-bootstrap CIs (Efron 1979), back-door adjustment for named confounders, VanderWeele-Ding (2017) E-value sensitivity to unmeasured confounding | [docs/theory/causal.md](docs/theory/causal.md) |
| [`shadow.diff_py`](python/src/shadow/diff_py/) | Scenario-aware multi-case diff: partition by `meta.scenario_id`, run per-scenario diffs without spurious "dropped turns" |  |
| [`shadow.policy_suggest`](python/src/shadow/policy_suggest/) | Mine baseline traces for `must_call_before` patterns; suggest policies the operator approves before adding |  |
| [`shadow.storage`](python/src/shadow/storage/) | Pluggable Storage interface (FileStore + InMemoryStore in OSS; cloud backends plug in) |  |

The validation suite at [`python/tests/test_statistical_validation.py`](python/tests/test_statistical_validation.py) (run with `pytest -m slow`) empirically verifies Type-I rate, power across an effect-size × n grid, the always-valid bound under continuous peeking, and conformal coverage on heavy-tailed held-out data.

## CLI reference

| Command | Does |
|---|---|
| `shadow quickstart` | Drop a working demo scenario. No API key needed. |
| `shadow init` | Scaffold a `.shadow/` folder. `--github-action` drops a CI workflow. |
| `shadow record -- <cmd>` | Run `<cmd>`, auto-capture its LLM calls. Zero code changes. |
| `shadow replay <cfg> --baseline <trace>` | Replay baseline through a new config. `--partial --branch-at N` locks a prefix, replays only the suffix. |
| `shadow diff <baseline> <candidate>` | Nine-axis behavior diff. `--policy <f>` to enforce rules. `--fail-on {minor,moderate,severe}` to gate the merge. `--token-diff` for per-turn token distribution. `--suggest-fixes` for LLM-assisted fix proposals. |
| `shadow bisect <cfg-a> <cfg-b> --traces <set>` | Attribute each axis regression to specific config deltas. |
| `shadow schema-watch <cfg-a> <cfg-b>` | Classify tool-schema changes before replaying. |
| `shadow import <src> --format <fmt>` | Import foreign traces (langfuse, braintrust, langsmith, openai-evals, otel, mcp, a2a, vercel-ai, pydantic-ai). |
| `shadow mine <traces...>` | Cluster production traces and pick representative cases as a regression suite. |
| `shadow mcp-serve` | Run Shadow as a Model Context Protocol server so agentic CLIs can invoke it as a tool. |
| `shadow report <report.json>` | Re-render a diff as terminal, markdown, or PR-comment. |
| `shadow certify <trace>` | Generate an Agent Behavior Certificate (ABOM) for a release. `--baseline` folds in a regression-suite rollup; `--policy` records its hash. `--sign` adds a sigstore keyless signature (requires `[sign]` extra). |
| `shadow verify-cert <cert>` | Verify a certificate's content-addressed `cert_id` matches the body. Exits 1 on tamper. `--verify-signature --cert-identity <id>` also verifies the sigstore signature against the canonical body and a specific signer identity. |

## Project layout

```
Shadow/
├── crates/shadow-core/         Rust core: parser, differ, replay, bisect
├── python/                     Python SDK + CLI (maturin-built, ships as shadow-diff on PyPI)
│   ├── src/shadow/
│   └── tests/
├── typescript/                 TypeScript SDK
├── docs/                       mkdocs site (published at manav8498.github.io/Shadow)
├── examples/                   Runnable scenarios (demo, customer-support, devops-agent, er-triage, etc.)
├── benchmarks/                 Scale and correctness benchmarks
├── scripts/                    One-off build and release helpers
├── .github/
│   ├── actions/shadow-action/  Reusable composite action for PR comments
│   ├── workflows/              ci.yml, docs.yml, release.yml
│   └── ISSUE_TEMPLATE/
├── SPEC.md                     The .agentlog format specification (Apache-2.0)
├── CHANGELOG.md                Release notes
├── SECURITY.md                 Security policy and vulnerability reporting
├── CONTRIBUTING.md             How to contribute
├── GOVERNANCE.md               Project governance
├── Cargo.toml                  Rust workspace manifest
├── justfile                    Common dev tasks (just setup, just test, just demo)
├── mkdocs.yml                  Docs site config
└── pricing.json                Per-model token pricing for cost attribution
```

## License

- **Code and spec**: [Apache License 2.0](LICENSE).
- **Name "Shadow" and logo**: see [TRADEMARK.md](TRADEMARK.md).
- **Contributions**: every commit must carry a Developer Certificate of Origin sign-off (`git commit -s`). See [CONTRIBUTING.md](CONTRIBUTING.md#developer-certificate-of-origin-dco).

## Community

- [GitHub Discussions](https://github.com/manav8498/Shadow/discussions) for questions and help
- [GitHub Issues](https://github.com/manav8498/Shadow/issues) for bugs and feature requests
- [SECURITY.md](SECURITY.md) to report vulnerabilities privately
- [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- [Contributor Covenant v2.1](CODE_OF_CONDUCT.md)

## Citing

If you use Shadow in academic work, see [`CITATION.cff`](CITATION.cff) or click "Cite this repository" on the GitHub page.
