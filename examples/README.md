# Examples

Each subdirectory is a self-contained scenario that exercises Shadow against
a realistic baseline-vs-candidate pair. Fixtures are committed so everything
runs offline, no API keys needed.

| Directory | Scenario | Key stress |
|---|---|---|
| [`refund-causal-diagnosis/`](refund-causal-diagnosis/) | **The wedge demo for `shadow diagnose-pr`.** Refund agent's prompt loses "always confirm before refunding" — Shadow names `prompt.system` as the dominant cause with bootstrap CI + E-value + suggested fix. | **`diagnose-pr → verify-fix` end-to-end** in one command. Default mock backend; opt into `--backend live` for real OpenAI. |
| [`demo/`](demo/) | Minimal 3-turn toy agent + 3 atomic config deltas. The `just demo` target. | Fast end-to-end smoke test (<2s). |
| [`customer-support/`](customer-support/) | Acme Widgets refund-handling bot. PR drops the confirm-before-refund protocol + renames a tool param + loses JSON output. | **Catches a "faster but wrong" tool-call regression** that unit tests can't see. |
| [`er-triage/`](er-triage/) | Clinical decision support for ER nurses across 5 heterogeneous patients. PR drops mandatory safety steps + downgrades ESI levels + loses JSON for EHR integration. | **High-stakes multi-tool scenario** showing what Shadow catches vs what needs a domain Judge. |
| [`devops-agent/`](devops-agent/) | Autonomous prod-DB agent with 10 tools across 5 scenarios. Scenario 4 is a deliberate **tool-order reversal** (`pause_replication` and `restore_database` swapped → would corrupt replicas). | **Ordering regressions, tool-schema renames, negative-direction latency/verbosity regressions.** |
| [`edge-cases/`](edge-cases/) | 20-case adversarial probe suite: identical traces, empty traces, Unicode NFC collisions, deeply nested payloads, order-only divergence, monotone signals, n=1 stats, etc. | **Permanent regression guard** against silent misbehaviour on unusual inputs. |
| [`acme-extreme/`](acme-extreme/) | End-to-end stress scenario exercising every Shadow feature at once: multi-session trace, bisect, cost attribution, span diff, schema-watch, judge rubrics. | **Feature-integration proof** that the whole pipeline composes. |
| [`production-incident-suite/`](production-incident-suite/) | Five public production-incident patterns from the past 18 months, encoded as fixtures. Exercises Shadow's statistical, formal, and causal primitives against real-world drift signatures. | **Real-incident regression coverage** — Air Canada / Avianca / NEDA / McDonald's / Replit reproductions. |
| [`harmful-content-judge/`](harmful-content-judge/) | Custom domain judge for prompt-injection / harmful-output detection, layered on top of the built-in safety axis. | **When the safety axis isn't enough** — building a domain-specific Judge. |
| [`refund-agent-audit/`](refund-agent-audit/) | Standalone audit script over a refund-agent trace bundle (no baseline/candidate). | **Single-trace forensic walkthrough** without a regression target. |
| [`canary-monitor/`](canary-monitor/) | Long-running monitor that watches a `.agentlog` directory and emits alerts on policy violations. | **Continuous canary** pattern for staging/prod environments. |
| [`sandboxed-replay/`](sandboxed-replay/) | Minimal `shadow replay` invocation showing how real tool functions run with network/subprocess/FS-write blocked. | **Sandboxing proof** — replay a candidate config without touching production. |
| [`standalone-align/`](standalone-align/) | One-file demo of the reusable trace-comparison primitives (`trajectory_distance`, `tool_arg_delta`) without the full Shadow pipeline. | **Eval-framework / linter integration** without taking a Shadow dependency on everything. |
| [`stress/`](stress/) | Synthetic high-volume input generator for stress-testing the differ. | **Throughput / scale** sanity check. |
| [`judges/`](judges/) | Ready-to-copy rubric files for the four data-driven judges: `factuality`, `llm`, `procedure`, `refusal`. | **Judge templates**, drop one into your repo and point `--judge-config` at it. |
| [`mcp-session/`](mcp-session/) | MCP (Model Context Protocol) server-session log → `shadow import --format mcp` → diff. Walkthrough shows what MCP captures vs what it doesn't. | **Bridge from MCP tooling** (Claude Desktop, Cursor, Zed) into Shadow. |
| [`integrations/`](integrations/) | Push Shadow traces to Datadog, Splunk, or any OTel collector. Three sub-dirs: `datadog/`, `splunk/`, `otel-collector/`. | **Observability-platform hand-off** for teams that already have a dashboard. |

## Running the baseline-vs-candidate scenarios

The five baseline/candidate scenarios (`demo/`, `customer-support/`,
`devops-agent/`, `er-triage/`, `mcp-session/`) all follow the same
pattern. Use the in-repo venv (created by `just setup`):

```bash
# Regenerate the committed.agentlog fixtures (idempotent; skip for
# mcp-session/ which uses an importer instead of a generator):
.venv/bin/python examples/<scenario>/generate_fixtures.py

# Compute the nine-axis diff:
.venv/bin/shadow diff \
  examples/<scenario>/fixtures/baseline.agentlog \
  examples/<scenario>/fixtures/candidate.agentlog

# Causal attribution across config deltas:
.venv/bin/shadow bisect \
  examples/<scenario>/config_a.yaml \
  examples/<scenario>/config_b.yaml \
  --traces examples/<scenario>/fixtures/baseline.agentlog \
  --candidate-traces examples/<scenario>/fixtures/candidate.agentlog
```

### The other directories

- `refund-causal-diagnosis/` runs `shadow diagnose-pr` end-to-end via
  its own `demo.sh`; see the directory README.
- `edge-cases/` is a probe runner, not a baseline/candidate scenario:
  `python examples/edge-cases/probe.py` exercises 20 adversarial
  inputs against the differ.
- `acme-extreme/` has its own top-level scripts (`scenario.py`,
  `bisect_small.py`); read them for usage.
- `production-incident-suite/` and `harmful-content-judge/` each have
  their own top-level README + scripts.
- `canary-monitor/`, `refund-agent-audit/`, `sandboxed-replay/`,
  `standalone-align/`, `stress/` are single-script scenarios that
  document themselves in-code; run with `.venv/bin/python <path>`.
- `integrations/` has three sub-directories (`datadog/`, `splunk/`,
  `otel-collector/`), each with its own README.
- `judges/` is a set of ready-to-copy rubric YAMLs, not runnable
  on its own; see `judges/README.md`.

The main scenarios (`customer-support/`, `devops-agent/`, `er-triage/`,
`mcp-session/`) each ship a `WALKTHROUGH.md` that narrates what the
scenario demonstrates, what Shadow catches, and what it doesn't.

## Reading order for a new user

1. **[refund-causal-diagnosis/](refund-causal-diagnosis/)** —
   the wedge demo for `shadow diagnose-pr`. Run `./demo.sh`; see the
   full diagnose → fix → verify loop in under five seconds.
2. **[demo/](demo/)** — same pipeline, smaller scenario; the `just
   demo` target.
3. **[customer-support/WALKTHROUGH.md](customer-support/WALKTHROUGH.md)** —
   a realistic e-commerce PR. Shows the "before merging this, read the
   Shadow diff" value prop at its simplest.
4. **[er-triage/WALKTHROUGH.md](er-triage/WALKTHROUGH.md)** — the same
   tool applied to a high-stakes domain, with a clear-eyed discussion of
   where the default axes stop and where a `Judge` takes over.
5. **[devops-agent/WALKTHROUGH.md](devops-agent/WALKTHROUGH.md)** —
   biggest scenario. Tests ordering sensitivity + negative-direction
   regressions (candidate faster/shorter but worse).
6. **[edge-cases/probe.py](edge-cases/probe.py)** — 20 weird-input
   probes. Run it as a regression check when modifying `diff/` or
   `bisect/`.

## Writing your own scenario

Copy one of the existing `generate_fixtures.py` files as a template.
The shape for a baseline/candidate scenario is:

1. Two YAML configs (`config_a.yaml`, `config_b.yaml`) that differ
   along a handful of atomic deltas.
2. A Python generator that hand-authors representative baseline and
   candidate responses per user turn.
3. A `fixtures/` directory with the two committed `.agentlog` files
   (regenerated by running the generator; idempotent).
4. Optionally, a `WALKTHROUGH.md` narrating what the scenario
   demonstrates, recommended for non-trivial scenarios so a reviewer
   can skim the intent in 60 seconds.

Scenarios are the fastest way to contribute. If you have a real
regression you caught (or wish you'd caught), turning it into an example
here benefits every future user.
