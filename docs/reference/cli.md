# CLI reference

## `shadow diagnose-pr`

Names the exact change that broke the agent and emits a markdown PR
comment. Composes the parser, mining, the nine-axis differ, the policy
checker, and the causal-attribution module into one PR-time surface.

Required flags:

- `--traces PATH ...`, baseline production-like `.agentlog` files
  (or directories). Repeatable.
- `--baseline-config FILE`, baseline agent config YAML (same schema
  as `shadow replay`).
- `--candidate-config FILE`, candidate agent config YAML.
- `--out PATH`, where to write the JSON report (`diagnose-pr/v0.1`
  schema).

Common optional flags:

- `--candidate-traces PATH ...`, candidate `.agentlog` files paired
  by filename to baseline (omit to skip per-trace diff).
- `--policy FILE`, behavior-policy YAML overlay.
- `--pr-comment FILE`, write the markdown PR comment.
- `--changed-files PATH ...`, files changed in the PR (e.g. from
  `git diff --name-only`); used to attach human-readable filenames
  to prompt deltas.
- `--baseline-ref REF`, git ref or SHA to diff prompt files against
  (typically `${{ github.event.pull_request.base.sha }}` in CI). When
  set with `--changed-files`, prompt deltas pick up file:line +
  removed-text blame so the PR comment cites
  `prompts/refund.md:17 removed: "Always confirm…"` instead of the
  flattened config path.
- `--max-traces N`, cap on traces fed to the per-trace loop. Mining
  selects representatives above this. Default 200.
- `--fail-on {none,probe,hold,stop}`, exit non-zero on this verdict
  floor. Default `none`.
- `--n-bootstrap N`, bootstrap resample count for causal CIs. Used
  only when `--backend != recorded`. Default 500.
- `--backend {recorded,mock,live}`, cause-attribution backend:
  - `recorded` (default): simple attribution from observed deltas;
    no CI; one-delta runs still get conf=1.0.
  - `mock`: deterministic per-delta intervention for tests/demos;
    real bootstrap CIs but synthetic ATE magnitudes (PR comment
    surfaces the disclosure).
  - `live`: real OpenAI replay per baseline trace; divergence is
    the corpus mean.
- `--max-cost USD`, hard ceiling on total spend during `--backend
  live` runs. Aborts before exceeding the cap; ignored otherwise.

The verdict is one of `ship` / `probe` / `hold` / `stop`. The
`affected_trace_ids` field of the JSON report drives `shadow
verify-fix`. See [Causal PR diagnosis](../features/causal-pr-diagnosis.md)
for the underlying flow.

## `shadow verify-fix`

Closes the diagnose → fix → verify loop. Reads a previous
`diagnose-pr` report, re-diffs only the affected traces against a
candidate-with-patch, and reports pass/fail with explicit
fail-reasons. Exit 0 pass, 1 fail.

Required flags:

- `--report FILE`, path to the diagnose-pr `report.json`.
- `--traces PATH ...`, original baseline `.agentlog` files.
- `--fixed-traces PATH ...`, candidate-with-patch `.agentlog` files,
  paired by filename to baseline.
- `--out PATH`, where to write the verify-fix JSON report.

Optional flags:

- `--policy FILE`, behavior-policy YAML overlay (same schema as
  diagnose-pr).
- `--affected-threshold FLOAT`, minimum fraction of affected traces
  that must be reversed. Default 0.90.
- `--safe-ceiling FLOAT`, maximum fraction of previously-safe traces
  that may regress. Default 0.02.

## `shadow gate-pr`

CI-friendly wrapper around `shadow diagnose-pr` with verdict-mapped
exit codes:

| Verdict | Exit | Meaning |
|---|---:|---|
| `ship` | 0 | No behavior regression — merge clear. |
| `probe` / `hold` | 1 | Regression detected — merge held. |
| `stop` | 2 | Critical violation — do not merge. |
| (internal/tooling error) | 3 | Treat as failure in CI. |

Always writes the JSON report (to a tempfile if `--out` omitted) and
the PR comment markdown so a follow-up CI step can post it. Flags
mirror `diagnose-pr` (`--traces`, `--candidate-traces`,
`--baseline-config`, `--candidate-config`, `--policy`,
`--changed-files`, `--max-traces`, `--backend`, `--n-bootstrap`,
`--max-cost`, `--pr-comment`, `--out`); see above for full
descriptions.

## `shadow dashboard --report FILE`

Serve a `diagnose-pr` `report.json` as a browsable HTML page —
verdict, top causes, and per-trace diagnoses. Single-process,
single-report, no auth. Meant for local review of CI artefacts or
sharing with someone who can't run the CLI.

Flags:

- `--report FILE` (required), the report to render.
- `--port N`, default 8080.
- `--host HOST`, default `127.0.0.1`. Pass `0.0.0.0` to expose on
  the network — but **don't expose to the public internet without
  a reverse proxy doing TLS + auth**.
- `--open`, launch the URL in the default browser on startup.

Routes: `/` (HTML), `/report.json` (raw JSON), `/healthz` (liveness
probe). User-controlled fields are HTML-escaped. Requires the
`serve` extra: `pip install 'shadow-diff[serve]'`.

## `shadow demo`

Run a nine-axis diff against bundled fixtures. Single command, no
API key, no files written. The fastest "does Shadow work on my box?"
check.

## `shadow quickstart [PATH]`

Scaffold a working Shadow scenario in `PATH` (default
`shadow-quickstart`). No API keys required. See
[Install and first diff](../quickstart/install.md).

## `shadow inspect <trace.agentlog> [<candidate>]`

One-screen terminal view of a trace — turn / role / tokens / latency
/ cost / redactions / first divergence. The daily-debug surface;
what `pytest -v` is to test runs, this is to recorded agent traces.

Flags:

- `--full`, expand truncated summaries to the full text.
- Two positional args = comparison mode: the first divergent turn is
  highlighted in red.

## `shadow scan <paths...>`

Scan committed `.agentlog` files for credentials / PII / custom
patterns. Exits non-zero on any hit, so it composes into CI before
`shadow gate-pr`.

Flags:

- `--patterns FILE`, load extra `name=regex` rules per line (`#`
  comments allowed; the loader names the offending line on syntax
  errors).
- `--only NAMES`, restrict to a comma-separated subset of pattern
  names (e.g. `--only email,credit_card`).
- `--redact-snippets`, replace the literal match with the pattern
  marker in stdout/stderr — safe for CI logs without leaking the
  credential to the build's log archive.
- `--json`, emit a single JSON object on stdout for piping into
  `jq` or annotating PR comments.

Built-in patterns: `private_key`, `jwt`, `anthropic_api_key`,
`openai_api_key`, `aws_access_key_id`, `github_token`, `email`,
`phone`, `credit_card` (Luhn-validated). Same set used by the
`Redactor` so the scanner is the second line of defence behind the
SDK's write-time redaction.

## `shadow baseline create / update / approve / verify`

Frozen-baseline workflow modeled on Insta + Jest snapshots — the
hash of the baseline trace corpus is pinned into `shadow.yaml`, and
PRs that change the baseline show up in `git diff` as a single line.

| Subcommand | Effect |
|---|---|
| `baseline create <dir>` | Hash the directory and pin it into `shadow.yaml`. Errors if a pin already exists. |
| `baseline update --force` | Re-pin the existing baseline directory after a deliberate regeneration. The `--force` is the friction flag — without it, a typo can't approve a regression silently. |
| `baseline approve <candidate-dir> --force` | Copy `.agentlog` files from the candidate directory into the baseline directory and re-pin. The end state: this trace set is now the gold standard. |
| `baseline verify` | Re-hash the baseline directory and compare against the pin. Exits non-zero on drift; the message names both digests so a reviewer can decide quickly whether the change is intentional. |

The hash is invariant under filename rename — same record bytes,
different filenames, same digest. Mutating any record flips the
digest.

## `shadow init [PATH]`

Scaffold `.shadow/` and `shadow.yaml` in `PATH`. Auto-detects the
project type (Python / Node / Rust). With `--github-action`, also
drops a CI workflow:

- default: `.github/workflows/shadow-diagnose-pr.yml` (runs
  `shadow diagnose-pr` on every PR; the recommended path)
- `--legacy-diff`: `.github/workflows/shadow-diff.yml` (runs the
  older `shadow diff` flow without causal attribution)

Path-traversal hardened — refuses system directories (`/etc`,
`/usr`, etc.).

## `shadow record -- <cmd>`

Run `<cmd>` with zero-config auto-instrumentation. Writes to
`-o path.agentlog`. Flags:

- `--tags KEY=V,K=V`, metadata tags
- `--no-auto-instrument`, skip the sitecustomize shim
- Fail-fast writability preflight on the output path

## `shadow replay <config> --baseline <trace>`

Replay `<trace>` through `<config>` via `--backend {mock,positional}`.
`mock` returns the baseline response verbatim; `positional` uses a
recorded reference trace (`--reference <path>`) and replays the
candidate against it. Live LLM backends (anthropic / openai) live
on the diff path through `--judge-backend`, not on replay.

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

Nine-axis behavior diff. Key flags:

- `--judge {none,auto,sanity,pairwise,llm,procedure,schema,factuality,refusal,tone}`
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

### v2.4 additions

- `--harness-diff`, render a per-(category, name) diff over
  `harness_event` records (retry, rate_limit, model_switch,
  context_trim, cache, guardrail, budget, stream_interrupt,
  tool_lifecycle). Regressions appear before fixes, ordered by
  severity then absolute count delta.
- `--multimodal-diff`, render a per-blob diff over `blob_ref`
  records. Cheap tier uses 64-bit dHash Hamming distance; semantic
  tier uses cosine similarity over recorded embeddings when both
  sides have them. Identical `blob_id` short-circuits.

## `shadow gate <report.json>`

Apply `--fail-on` to a saved `report.json` (produced by `shadow diff
--output-json`) without re-running the diff. Designed for CI flows
that already produced the report for the PR comment and want to
gate the merge as a separate, cheap step:

```bash
shadow diff base.agentlog cand.agentlog --output-json report.json
shadow gate report.json --fail-on severe
```

With `--policy <yaml>`, the gate also recomputes policy regressions
from the original traces (passed via `--baseline` / `--candidate`)
and counts them toward the threshold. Without `--policy`, it gates
purely on axis severity and is fast.

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
