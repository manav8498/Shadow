# Shadow

[![ci](https://github.com/manav8498/Shadow/actions/workflows/ci.yml/badge.svg)](https://github.com/manav8498/Shadow/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![spec](https://img.shields.io/badge/.agentlog-v0.1-6f4cff.svg)](SPEC.md)
[![version](https://img.shields.io/badge/version-1.2.4-brightgreen.svg)](CHANGELOG.md)
[![rust](https://img.shields.io/badge/rust-1.95+-orange.svg)](rust-toolchain.toml)
[![python](https://img.shields.io/badge/python-3.11+-3776ab.svg)](python/pyproject.toml)

**Behavioural code review for LLM agents.** Shadow records the calls your agent makes to Claude or GPT, replays them against a new config, and tells you on the pull request what changed, why, and what to do about it.

## What problem it solves

A code diff tool breaks down when the change is a prompt edit, a model swap, or a tool-schema rename. The agent still runs. It just behaves differently. The regression ships, the customer complains, nobody knows which line of the PR caused it.

Shadow reviews the PR like a senior engineer would. It compares before and after behaviour on a fixed set of traces, ranks the differences by severity across nine dimensions with statistical confidence, names the first point the candidate diverged from the baseline, attributes each regression to the specific config delta that caused it, and ends with a short list of concrete fixes.

## How it works

1. **Record.** Your agent talks to Claude or GPT as usual. Shadow's SDK saves every request and response to a `.agentlog` file. Your code does not change.
2. **Replay.** In CI, Shadow runs those recorded requests through your new config (new prompt, new model, whatever changed) and collects the new responses.
3. **Diff.** Shadow compares the two sets across nine behavioural dimensions: meaning, tool use, refusals, verbosity, speed, cost, reasoning depth, an LLM-judge score, and output format. It posts the report as a pull-request comment.

If multiple things changed at once, Shadow names which change caused which part of the regression. That is the part that makes it useful in real PR review. You do not have to guess.

## Five-minute adoption

Three commands. Nothing else to configure:

```bash
pip install shadow-diff          # PyPI name is shadow-diff; import + CLI stay "shadow"
shadow quickstart                # scaffolds a working scenario in ./shadow-quickstart/
shadow diff shadow-quickstart/fixtures/baseline.agentlog \
            shadow-quickstart/fixtures/candidate.agentlog
```

That is a real nine-axis diff on pre-recorded `.agentlog` fixtures. No API keys, no agent code. Then run it on your own agent:

```bash
# Zero code changes. shadow auto-instruments anthropic + openai:
shadow record -o baseline.agentlog -- python your_agent.py

# Make a change (new prompt, swap model, edit a tool) and re-record:
shadow record -o candidate.agentlog -- python your_agent.py

# Diff:
shadow diff baseline.agentlog candidate.agentlog
```

To wire it into every pull request, one more command:

```bash
shadow init --github-action      # drops .github/workflows/shadow-diff.yml
```

Edit the `BASELINE` and `CANDIDATE` paths in that workflow to point at fixtures you commit. Every PR gets a behavioural-diff comment.

<details>
<summary>Developing Shadow itself (contributors only)</summary>

```bash
git clone https://github.com/manav8498/Shadow && cd Shadow
just setup    # installs Rust + Python deps, builds the native extension
just demo     # end-to-end diff in under 10 seconds, no API key
just ci       # full test + lint + coverage replica
```
</details>

You will see a table like this:

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
  #3  baseline turn #1 ↔ candidate turn #1
      kind: decision_drift    ·  axis: semantic    ·  confidence: 9%
      response text diverged (text similarity 0.21); same tool sequence

recommendations (3):
  error   REVIEW  Review tool-schema change at turn 0: call shape diverged.
  error   REVIEW  Review refusal behaviour at turn 2: candidate may be over-refusing.
  warning REVIEW  Review response text at turn 1: semantic content shifted.
```

Each row is one behavioural dimension. The severity column shows where to look. The top-divergences list names the exact changes responsible. The recommendations list tells you what to do about them. One pass, under 30 seconds, no clicking through traces. Root-cause attribution plus fixes, no dashboard needed.

## Instrument your own agent

Two ways. Both are zero-code-change for the agent itself.

**Zero-config.** Wrap any Python script with `shadow record`. Shadow's auto-instrumentor patches `anthropic.*` and `openai.*` via a PYTHONPATH-injected sitecustomize at interpreter startup:

```bash
shadow record -o trace.agentlog -- python your_agent.py
```

**Session context manager.** For custom tags, a non-default redactor, or nested sessions:

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog", tags={"env": "prod"}):
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

Either way Shadow captures every Anthropic / OpenAI request and response (Python and TypeScript SDKs). Secrets are redacted by default.

Then in CI:

```bash
shadow replay new-config.yaml --baseline trace.agentlog
shadow diff trace.agentlog candidate.agentlog
shadow bisect old-config.yaml new-config.yaml --traces trace.agentlog
```

## What's different about Shadow

|  | Langfuse | Braintrust | LangSmith | **Shadow** |
|---|:---:|:---:|:---:|:---:|
| Raw trace logging | ✅ | ✅ | ✅ | ✅ |
| Dashboard UI | ✅ | ✅ | ✅ | no |
| Self-hostable | ✅ | no | no | ✅ |
| PR comment from CI | partial | partial | partial | ✅ |
| Nine pre-built behavioural axes | no | no | no | ✅ |
| Causal bisection | no | no | no | ✅ |
| Content-addressed open trace format | no | no | no | ✅ |

Shadow lives in your pull request, not a separate dashboard. It runs locally. Traces stay on your disk, the diff runs in your CI, the comment posts to your PR. The `.agentlog` format is an open spec ([`SPEC.md`](SPEC.md)) that any tool can read or write.

## The nine axes

Each axis is measured independently with a bootstrap 95% confidence interval and a severity rating (none, minor, moderate, severe):

| # | Axis | What it measures |
|--:|---|---|
| 1 | `semantic` | How different are the outputs' meanings? |
| 2 | `trajectory` | Did the agent use a different sequence of tools? |
| 3 | `safety` | Did refusal rates change? |
| 4 | `verbosity` | Are outputs longer or shorter? |
| 5 | `latency` | Is it slower or faster? |
| 6 | `cost` | Are token costs up or down? |
| 7 | `reasoning` | Is the agent thinking less or more? |
| 8 | `judge` | Your own LLM-judge rubric (optional). |
| 9 | `conformance` | Does the output still match the expected structure? |

Full details in [`SPEC.md`](SPEC.md).

## Examples

Every example runs offline from committed fixtures. No API key required:

| Example | What it shows |
|---|---|
| [`examples/demo/`](examples/demo/) | The fastest working example. `just demo`. |
| [`examples/customer-support/`](examples/customer-support/) | Refund bot that regresses after a prompt edit |
| [`examples/devops-agent/`](examples/devops-agent/) | Production database agent with a tool-ordering bug |
| [`examples/er-triage/`](examples/er-triage/) | High-stakes clinical-style agent |
| [`examples/edge-cases/`](examples/edge-cases/) | 20 adversarial cases used as a permanent regression guard |
| [`examples/acme-extreme/`](examples/acme-extreme/) | End-to-end scenario exercising every Shadow feature |
| [`examples/integrations/`](examples/integrations/) | Push traces to Datadog, Splunk, or any OTel collector |

## CLI reference

| Command | Does |
|---|---|
| `shadow quickstart` | Drop a working demo scenario into `./shadow-quickstart/`. No API key needed. |
| `shadow init` | Scaffold a `.shadow/` folder. `--github-action` also drops a ready-to-commit CI workflow. |
| `shadow record -- <cmd>` | Run `<cmd>`, auto-capture its LLM calls. Zero code changes to the wrapped script. |
| `shadow replay <cfg> --baseline <trace>` | Replay baseline through a new config |
| `shadow diff <baseline> <candidate>` | Nine-axis behavioural diff |
| `shadow bisect <cfg-a> <cfg-b> --traces <set>` | Which config delta moved which axis |
| `shadow schema-watch <cfg-a> <cfg-b>` | Classify tool-schema changes (renames, breaking edits) before replaying |
| `shadow report <report.json>` | Re-render a diff as terminal, markdown, or PR-comment |

## Project layout

```
Shadow/
├── SPEC.md                    Open spec for the.agentlog format
├── crates/shadow-core/        Rust: parser, differ, replay, bisect
├── python/src/shadow/         Python SDK + CLI
├── typescript/                TypeScript SDK
├── examples/                  Runnable scenarios
└── .github/actions/           Reusable GitHub Action for PR comments
```

## License

- **Code** (Rust, Python, TypeScript): dual [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Pick either.
- **Spec** (`SPEC.md`): [Apache-2.0](LICENSE-SPEC) only.
- **Name "Shadow" and logo**: see [TRADEMARK.md](TRADEMARK.md).

## Community

- [GitHub Discussions](https://github.com/manav8498/Shadow/discussions) for questions and help
- [GitHub Issues](https://github.com/manav8498/Shadow/issues) for bugs and feature requests
- [SECURITY.md](SECURITY.md) to report vulnerabilities privately
- [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- [Contributor Covenant v2.1](CODE_OF_CONDUCT.md)

## Citing

If you use Shadow in academic work, see [`CITATION.cff`](CITATION.cff) or click "Cite this repository" on the GitHub page.
