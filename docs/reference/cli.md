# CLI reference

## `shadow quickstart [PATH]`

Scaffold a working Shadow scenario in `PATH` (default
`shadow-quickstart`). No API keys required. See
[Install and first diff](../quickstart/install.md).

## `shadow init [PATH]`

Scaffold `.shadow/` in `PATH`. `--github-action` also drops
`.github/workflows/shadow-diff.yml`. Path-traversal hardened -
refuses system directories (`/etc`, `/usr`, etc.).

## `shadow record -- <cmd>`

Run `<cmd>` with zero-config auto-instrumentation. Writes to
`-o path.agentlog`. Flags:

- `--tags KEY=V,K=V`, metadata tags
- `--no-auto-instrument`, skip the sitecustomize shim
- Fail-fast writability preflight on the output path

## `shadow replay <config> --baseline <trace>`

Replay `<trace>` through `<config>` via
`--backend {mock,positional,anthropic,openai}`.

### Partial replay (v1.2)

Lock a baseline prefix verbatim and replay only the suffix through the
backend:

- `--partial`, enable partial-replay mode
- `--branch-at N`, 0-based turn index where live replay begins
  (`0` = fully live, same as default; `>= len(turns)` = full-baseline
  copy)

Useful for "what would have happened from turn 3 onward if the model had
stayed on the baseline path through turn 2?" experiments.

## `shadow diff <baseline> <candidate>`

Nine-axis behavioural diff. Key flags:

- `--judge {none,auto,sanity,pairwise,llm,procedure,schema,factuality,refusal,tone,correctness,format}`
- `--judge-config <file.yaml>` for rubric-based judges
- `--judge-backend {mock,anthropic,openai}` for live judges
- `--explain` for LLM-sourced paragraph summary
- `--hierarchical` for session-level breakdown
- `--pricing <file.json>` for cost attribution
- `--output-json <file>` to write the full report

### v1.2 additions

- `--token-diff`, per-dimension token distribution (input / output /
  thinking) with median + p25 + p75 + p95 + max + total; plus the
  top-k worst per-pair deltas. See
  [Hierarchical diff, token-level](../features/hierarchical.md#token-level-v12).
- `--policy path/to/rules.yaml`, check a declarative YAML policy
  overlay against both traces and classify rule violations as
  regressions vs fixes. Supports 12 rule kinds:
  `must_call_before`, `must_call_once`, `no_call`, `max_turns`,
  `required_stop_reason`, `max_total_tokens`, `must_include_text`,
  `forbidden_text`, `must_match_json_schema`, `must_remain_consistent`,
  `must_followup`, `must_be_grounded`. Each rule can carry a
  `when:` clause that gates it on field-path conditions (operators:
  `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`,
  `not_contains`). See [Behavior policy](../features/policy.md).
- `--fail-on {minor,moderate,severe}`, exit non-zero when the worst
  axis severity or policy regression hits the threshold. Default is
  `never` (post the report, exit 0). Use `--fail-on severe` to gate
  a PR merge on agent regressions.
- `--suggest-fixes`, layer an LLM pass on top of the deterministic
  recommendation engine to produce concrete code-level fix proposals.
  Each suggestion is grounded on a deterministic anchor (ungrounded
  model output is rejected). Requires a live backend
  (`--judge-backend anthropic|openai` or `--judge auto` with the
  corresponding env var set). Retry/backoff on 429/5xx/timeout.

## `shadow bisect <config_a> <config_b> --traces <trace>`

LASSO-over-corners causal attribution. `--backend anthropic|openai`
enables live-replay mode; default (`none`) uses the heuristic
allocator. `--candidate-traces <trace>` supplies a candidate trace
when the backend is `none`.

## `shadow schema-watch <config_a> <config_b>`

Tool-schema change detection. `--format {terminal,markdown,json}`.
`--fail-on {breaking,risky,additive,neutral,none}`.

## `shadow report <report.json>`

Re-render a saved JSON report. `--format {terminal,markdown,github-pr}`.

## `shadow import <source> --format <fmt>`

Import foreign traces to `.agentlog`. Supported formats (v1.2):

- `langfuse`, Langfuse `traces` export
- `braintrust`, Braintrust experiment row export (JSONL or array)
- `langsmith`, LangSmith runs export (top-level array)
- `openai-evals`, OpenAI Evals JSONL
- `otel`, OpenTelemetry OTLP/JSON with GenAI semconv attributes
- `mcp`, Model Context Protocol session log (JSONL, JSON array, or
  wrapped `{messages: [...]}`)
- `vercel-ai` *(new in v1.2)*, Vercel AI SDK telemetry export
  (OTLP-style `{spans: [...]}` or dashboard-style `{events: [...]}`)
- `pydantic-ai` *(new in v1.2)*, PydanticAI `all_messages_json()`
  output or Logfire span export

## `shadow export <trace>`

Export to `otel` (OTLP/JSON) for OpenTelemetry collectors.

## `shadow join <logs...>`

Merge multiple `.agentlog` files into one logical trace via
`meta.trace_id`.

## `shadow mine <traces...>`

Cluster a corpus of production traces by tool sequence, stop reason,
response length, and latency, then surface representative cases as a
regression suite. Output is a list of `(trace_id, cluster_id, why)`
triples that you can commit alongside the agent as your golden test
set.

## `shadow mcp-serve`

Run Shadow as a Model Context Protocol server over stdio. Any
MCP-aware agentic CLI (Claude Code, Cursor, Zed, Claude Desktop,
Windsurf) can invoke Shadow as a tool. Tools exposed:

- `shadow_diff`
- `shadow_check_policy`
- `shadow_token_diff`
- `shadow_schema_watch`
- `shadow_summarise`
- `shadow_certify` (v1.7.2+)
- `shadow_verify_cert` (v1.7.2+)

Install the extra first: `pip install 'shadow-diff[mcp]'`. See
[MCP importer](../features/mcp.md) for the reverse direction
(importing MCP traces into Shadow).

## `shadow certify <trace>`

Generate an Agent Behavior Certificate (ABOM) for a release. The
certificate is a content-addressed JSON release artefact capturing
the trace's content-id, all distinct models, content-ids of system
prompts, content-ids of tool schemas, optional policy hash, and an
optional baseline-vs-candidate nine-axis regression-suite rollup.

Required: `--agent-id <id>` and `--output <path>`. Optional:
`--policy <file>` (records its hash), `--baseline <trace>` (folds
in a regression-suite rollup), `--pricing <file>` (for the
regression-suite cost axis), `--seed <int>`.

The certificate is self-verifying via `shadow verify-cert`. See
[Release certificate](../features/certificate.md).

## `shadow verify-cert <cert>`

Verify a certificate's content-addressed `cert_id` matches the
body. Exits 0 when consistent, 1 on tamper, malformed payload, or
unsupported `cert_version`. Designed to run as a release-pipeline
gate.

## `shadow serve`

Start the live diff dashboard (requires the `serve` extra:
`pip install 'shadow-diff[serve]'`).

## `shadow version`

Prints the installed Shadow version + `.agentlog` spec version.
