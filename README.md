# Shadow

[![ci](https://github.com/manav8498/Shadow/actions/workflows/ci.yml/badge.svg)](https://github.com/manav8498/Shadow/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![spec](https://img.shields.io/badge/.agentlog-v0.1-6f4cff.svg)](SPEC.md)
[![version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)](CHANGELOG.md)
[![rust](https://img.shields.io/badge/rust-1.95+-orange.svg)](rust-toolchain.toml)
[![python](https://img.shields.io/badge/python-3.11+-3776ab.svg)](python/pyproject.toml)

> **Catch AI-agent regressions before they hit production.**
> Shadow is a PR-native diff tool for LLM agents — think "Codecov, but for
> your Claude / GPT agents."

## The problem

You change a prompt. Or upgrade the model. Or tweak a tool schema.
Everything still looks fine in staging. Then a week later a customer
complains the agent is acting weird.

Sound familiar? That's because most LLM regressions are **behavioural,
not functional** — the agent still runs, it just decides differently.
Accuracy dashboards don't catch this. By the time they do, you've
already shipped the bug.

**Shadow catches it in the pull request, before merge.**

## How it works

Shadow has three simple steps:

1. **Record** — your agent talks to OpenAI / Anthropic normally.
   Shadow's SDK silently saves every request and response to a
   `.agentlog` file. Nothing in your code changes.
2. **Replay** — in CI, Shadow takes those saved requests and runs them
   through your new config (new prompt, new model, whatever changed).
   Gets new responses.
3. **Diff** — Shadow compares old vs new responses across **nine
   behavioural dimensions** — meaning, tool use, refusals, verbosity,
   speed, cost, reasoning depth, LLM-judge score, and output format.
   Posts the report as a PR comment.

If anything moved in a statistically meaningful way, you see it
**before** merging. If multiple things changed at once, Shadow can
tell you **which change caused which regression** (we call this
"bisection" — it's the bit no other tool does).

## Try it

```bash
git clone https://github.com/manav8498/Shadow && cd Shadow
just setup    # installs rust + python deps, builds the native extension
just demo     # runs an end-to-end diff in < 10 seconds, no API key needed
```

You'll see a table like this:

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
```

Four severe regressions — meaning the agent is behaving very
differently, even if both versions still technically "work."

## Instrument your own agent

```python
from shadow.sdk import Session

with Session(output_path="trace.agentlog"):
    # Your existing Anthropic / OpenAI code. No changes.
    client.messages.create(model="claude-sonnet-4-6", messages=[...])
```

That's it. Shadow automatically patches the Anthropic and OpenAI Python
SDKs (and their TypeScript equivalents) to capture every request and
response. Secrets are redacted by default.

Then in CI:

```bash
shadow replay new-config.yaml --baseline trace.agentlog
shadow diff trace.agentlog candidate.agentlog
shadow bisect old-config.yaml new-config.yaml --traces trace.agentlog
```

## How it compares

|  | Langfuse | Braintrust | LangSmith | **Shadow** |
|---|:---:|:---:|:---:|:---:|
| Raw trace logging | ✅ | ✅ | ✅ | ✅ |
| Dashboard UI | ✅ | ✅ | ✅ | — *(by design)* |
| Self-hostable | ✅ | — | — | ✅ |
| PR comment from CI | ~ | ~ | ~ | ✅ |
| **9 pre-built behavioural axes** | — | — | — | ✅ |
| **Causal bisection** (which change broke which axis) | — | — | — | ✅ |
| **Content-addressed trace format** (open spec) | — | — | — | ✅ |

Shadow is narrower than a full observability platform — no hosted UI,
no cross-org trace sharing. It's focused on the specific question:
"did this PR make my agent worse?"

## The nine axes

Each axis is measured independently with a bootstrap 95% confidence
interval and a severity (none / minor / moderate / severe):

| # | Axis | What it measures |
|--:|---|---|
| 1 | `semantic` | How different are the outputs' meanings? |
| 2 | `trajectory` | Did the agent use a different sequence of tools? |
| 3 | `safety` | Did refusal rates change? |
| 4 | `verbosity` | Are outputs longer or shorter? |
| 5 | `latency` | Is it slower or faster? |
| 6 | `cost` | Are token costs up or down? |
| 7 | `reasoning` | Is the agent thinking less / more? |
| 8 | `judge` | Your own LLM-judge rubric (optional). |
| 9 | `conformance` | Does the output still match the expected structure? |

Full details in [`SPEC.md`](SPEC.md).

## Examples to learn from

Every example runs offline from committed fixtures. No API key required:

| Example | What it shows |
|---|---|
| [`examples/demo/`](examples/demo/) | Fastest working example — `just demo` |
| [`examples/customer-support/`](examples/customer-support/) | Refund bot that regresses after a prompt edit |
| [`examples/devops-agent/`](examples/devops-agent/) | Prod-DB agent with a tool-ordering bug |
| [`examples/er-triage/`](examples/er-triage/) | High-stakes clinical-style agent |
| [`examples/edge-cases/`](examples/edge-cases/) | 20 adversarial cases — permanent regression guard |
| [`examples/acme-extreme/`](examples/acme-extreme/) | End-to-end scenario exercising every Shadow feature |
| [`examples/integrations/`](examples/integrations/) | Push traces to Datadog, Splunk, and any OTel collector |

## Current limitations (v0.1)

Up-front honesty is better than surprises later:

- **Local-only** — traces live on your disk. No cloud, no cross-team sharing.
- **Embeddings are optional** — the semantic axis uses a fast TF-IDF
  default; for real embeddings install `shadow[embeddings]`.
- **Judge axis is opt-in** — you bring your own rubric. Shadow doesn't
  ship a default one (it'd be domain-specific and therefore wrong).
- **CI tested on Linux + macOS only** — Windows isn't tested in v0.1.

## CLI reference

| Command | Does |
|---|---|
| `shadow init` | Scaffold a `.shadow/` folder in the current repo |
| `shadow record -- <cmd>` | Run `<cmd>`, auto-capture its LLM calls |
| `shadow replay <cfg> --baseline <trace>` | Replay baseline through new config |
| `shadow diff <baseline> <candidate>` | Nine-axis behavioural diff |
| `shadow bisect <cfg-a> <cfg-b> --traces <set>` | Which config delta caused which axis to move |
| `shadow report <report.json>` | Re-render a diff as terminal / markdown / PR-comment |

## Project layout

```
Shadow/
├── SPEC.md                    Open spec for the .agentlog format
├── crates/shadow-core/        Rust: parser, differ, replay, bisect
├── python/src/shadow/         Python SDK + CLI (wraps the Rust core)
├── typescript/                TypeScript SDK (same wire protocol)
├── examples/                  Runnable scenarios
└── .github/actions/           Reusable GitHub Action for PR comments
```

## License

- **Code** (Rust + Python + TypeScript): dual **[MIT](LICENSE-MIT) OR
  [Apache-2.0](LICENSE-APACHE)** — pick either, matching the Rust
  ecosystem default.
- **Spec** (`SPEC.md`): **[Apache-2.0](LICENSE-SPEC)** only — so anyone
  can build a compatible `.agentlog` reader or writer without patent
  risk.
- **Name "Shadow" and logo**: see [TRADEMARK.md](TRADEMARK.md).

## Community

- **Questions / help**: [GitHub Discussions](https://github.com/manav8498/Shadow/discussions)
- **Bugs / features**: [GitHub Issues](https://github.com/manav8498/Shadow/issues)
- **Security**: [SECURITY.md](SECURITY.md) (private disclosure via GitHub)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Governance**: [GOVERNANCE.md](GOVERNANCE.md) · [MAINTAINERS.md](MAINTAINERS.md)
- **Code of Conduct**: [Contributor Covenant v2.1](CODE_OF_CONDUCT.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md) · **Roadmap**: [ROADMAP.md](ROADMAP.md)

## Citing

If you use Shadow in academic work, see [`CITATION.cff`](CITATION.cff) or
click "Cite this repository" at the top of the GitHub page.
