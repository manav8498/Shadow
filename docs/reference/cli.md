# CLI reference

## `shadow quickstart [PATH]`

Scaffold a working Shadow scenario in `PATH` (default
`shadow-quickstart`). No API keys required. See
[Install and first diff](../quickstart/install.md).

## `shadow init [PATH]`

Scaffold `.shadow/` in `PATH`. `--github-action` also drops
`.github/workflows/shadow-diff.yml`. Path-traversal hardened —
refuses system directories (`/etc`, `/usr`, etc.).

## `shadow record -- <cmd>`

Run `<cmd>` with zero-config auto-instrumentation. Writes to
`-o path.agentlog`. Flags:

- `--tags KEY=V,K=V` — metadata tags
- `--no-auto-instrument` — skip the sitecustomize shim
- Fail-fast writability preflight on the output path

## `shadow replay <config> --baseline <trace>`

Replay `<trace>` through `<config>` via
`--backend {mock,positional,anthropic,openai}`.

### Partial replay (v1.2)

Lock a baseline prefix verbatim and replay only the suffix through the
backend:

- `--partial` — enable partial-replay mode
- `--branch-at N` — 0-based turn index where live replay begins
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

- `--token-diff` — per-dimension token distribution (input / output /
  thinking) with median + p25 + p75 + p95 + max + total; plus the
  top-k worst per-pair deltas. See
  [Hierarchical diff — token-level](../features/hierarchical.md#token-level-v12).
- `--policy path/to/rules.yaml` — check a declarative YAML policy
  overlay against both traces and classify rule violations as
  regressions vs fixes. Supports 8 rule kinds:
  `must_call_before`, `must_call_once`, `no_call`, `max_turns`,
  `required_stop_reason`, `max_total_tokens`, `must_include_text`,
  `forbidden_text`. See
  [Hierarchical diff — policy-level](../features/hierarchical.md#policy-level-v12).
- `--suggest-fixes` — layer an LLM pass on top of the deterministic
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

- `langfuse` — Langfuse `traces` export
- `braintrust` — Braintrust experiment row export (JSONL or array)
- `langsmith` — LangSmith runs export (top-level array)
- `openai-evals` — OpenAI Evals JSONL
- `otel` — OpenTelemetry OTLP/JSON with GenAI semconv attributes
- `mcp` — Model Context Protocol session log (JSONL, JSON array, or
  wrapped `{messages: [...]}`)
- `vercel-ai` *(new in v1.2)* — Vercel AI SDK telemetry export
  (OTLP-style `{spans: [...]}` or dashboard-style `{events: [...]}`)
- `pydantic-ai` *(new in v1.2)* — PydanticAI `all_messages_json()`
  output or Logfire span export

## `shadow export <trace>`

Export to `otel` (OTLP/JSON) for OpenTelemetry collectors.

## `shadow join <logs...>`

Merge multiple `.agentlog` files into one logical trace via
`meta.trace_id`.

## `shadow serve`

Start the live diff dashboard (requires the `serve` extra:
`pip install 'shadow-diff[serve]'`).

## `shadow version`

Prints the installed Shadow version + `.agentlog` spec version.
