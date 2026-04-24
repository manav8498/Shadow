# Shadow

**Behavioural code review for LLM agents.** Shadow records the calls your
agent makes to Claude or GPT, replays them against a new config, and
tells you — right on the pull request — *what changed, why it changed,
and what to do about it.*

## Why it exists

A diff tool for code breaks when the change is a prompt edit, a model
swap, or a tool-schema rename. The agent still runs. It just behaves
differently. The regression ships, the customer complains, and nobody
knows which line of the PR caused it.

Shadow reviews the PR the way a senior engineer would: compares the
before-and-after behaviour on a fixed set of traces, ranks divergences
by severity across **nine dimensions** with statistical confidence,
names the first point the candidate diverged from the baseline,
attributes each dimension's regression to the specific config delta
that caused it, and ends with a short list of concrete fixes.

## 60-second adoption

```bash
pip install shadow-diff
shadow quickstart
shadow diff shadow-quickstart/fixtures/baseline.agentlog \
            shadow-quickstart/fixtures/candidate.agentlog
```

That's a real nine-axis diff on pre-recorded `.agentlog` fixtures —
no API keys, no agent code. See [Install and first diff](quickstart/install.md)
for the full walkthrough.

## Highlights

- **Nine axes**: semantic, trajectory, safety, verbosity, latency,
  cost, reasoning, LLM-judge, format conformance
- **Ten built-in judges**, including a configurable `LlmJudge` —
  all verified against live Claude Haiku 4.5 and GPT-4o-mini
- **Hardened causal bisection**: pairwise interactions + bootstrap
  95% CIs on attribution weights
- **Session-cost attribution**: decomposes cost delta into model-swap,
  token-movement, mix-residual
- **Six-level hierarchical diff** — trace, session, turn, span,
  **token** *(v1.2)*, **policy** *(v1.2)*
- **Eight importers** for bridging from existing observability
  tooling: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP, MCP,
  **Vercel AI SDK** *(v1.2)*, **PydanticAI** *(v1.2)*
- **Partial replay** *(v1.2)* — lock a baseline prefix, replay only
  the suffix through a live backend
- **LLM-assisted prescriptive fixes** *(v1.2)* — grounded, anchored
  to deterministic recommendations; ungrounded model output rejected
- **Zero-config instrumentation**: `shadow record -- python agent.py`
  works on any agent, zero code changes
- **Cross-OS**: Ubuntu, macOS, Windows × Python 3.11, 3.12, 3.13

## Where to next

- New user? → [Quickstart: Install and first diff](quickstart/install.md)
- Understanding the output? → [Nine-axis diff](features/nine-axis.md)
- Wiring Shadow into CI? → [Wire into CI](quickstart/ci.md)
- Need a specific feature? → feature index in the sidebar

## Links

- **PyPI**: [shadow-diff](https://pypi.org/project/shadow-diff/)
- **Source**: [manav8498/Shadow](https://github.com/manav8498/Shadow)
- **Spec**: [.agentlog v0.1](https://github.com/manav8498/Shadow/blob/main/SPEC.md)
- **License**: MIT OR Apache-2.0
