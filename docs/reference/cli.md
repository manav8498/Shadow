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

Replay `<trace>` through `<config>` via `--backend {mock,positional,anthropic,openai}`.

## `shadow diff <baseline> <candidate>`

Nine-axis behavioural diff. Key flags:

- `--judge {none,auto,sanity,pairwise,llm,procedure,schema,factuality,refusal,tone}`
- `--judge-config <file.yaml>` for rubric-based judges
- `--judge-backend {mock,anthropic,openai}` for live judges
- `--explain` for LLM-sourced paragraph summary
- `--hierarchical` for session-level breakdown
- `--pricing <file.json>` for cost attribution
- `--output-json <file>` to write the full report

## `shadow bisect <config_a> <config_b> --traces <trace>`

LASSO-over-corners causal attribution. `--backend anthropic|openai`
enables live-replay mode; default (`none`) uses the heuristic
allocator.

## `shadow schema-watch <config_a> <config_b>`

Tool-schema change detection. `--format {terminal,markdown,json}`.
`--fail-on {breaking,risky,additive,neutral,none}`.

## `shadow report <report.json>`

Re-render a saved JSON report. `--format {terminal,markdown,github-pr}`.

## `shadow import <source> --format <fmt>`

Import foreign traces to `.agentlog`. Formats: `langfuse`,
`braintrust`, `langsmith`, `openai-evals`, `otel`, `mcp`.

## `shadow export <trace>`

Export to `otel` (OTLP/JSON) for OpenTelemetry collectors.

## `shadow join <logs...>`

Merge multiple `.agentlog` files into one logical trace via
`meta.trace_id`.

## `shadow version`

Prints the installed Shadow version + spec version.
