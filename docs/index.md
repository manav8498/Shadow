# Shadow

**Behavioural code review for LLM agents.** Shadow records the calls your agent makes to Claude or GPT, replays them against a new config, and tells you on the pull request what changed, why, and what to do about it.

## Why it exists

A code diff tool breaks down when the change is a prompt edit, a model swap, or a tool-schema rename. The agent still runs. It just behaves differently. The regression ships, the customer complains, nobody knows which line of the PR caused it.

Shadow reviews the PR like a senior engineer would. It compares before and after behaviour on a fixed set of traces, ranks the differences by severity across **nine dimensions** with statistical confidence, names the first point the candidate diverged from the baseline, attributes each regression to the specific config delta that caused it, and ends with a short list of concrete fixes.

## Sixty-second adoption

```bash
pip install shadow-diff
shadow quickstart
shadow diff shadow-quickstart/fixtures/baseline.agentlog \
            shadow-quickstart/fixtures/candidate.agentlog
```

That is a real nine-axis diff on pre-recorded `.agentlog` fixtures. No API keys, no agent code. See [Install and first diff](quickstart/install.md) for the full walkthrough.

## Highlights

- **Nine axes**: semantic, trajectory, safety, verbosity, latency, cost, reasoning, LLM-judge, format conformance.
- **Ten built-in judges**, including a configurable `LlmJudge`. All verified against live Claude Haiku 4.5 and GPT-4o-mini.
- **Causal bisection** with pairwise interactions and bootstrap 95% CIs on attribution weights.
- **Session-cost attribution** that decomposes cost delta into model-swap, token-movement, and mix-residual.
- **Six-level hierarchical diff**: trace, session, turn, span, token, policy.
- **Eight importers** to bridge from existing observability tooling: Langfuse, Braintrust, LangSmith, OpenAI Evals, OTLP, MCP, Vercel AI SDK, PydanticAI.
- **Partial replay**: lock a baseline prefix, replay only the suffix through a live backend.
- **LLM-assisted prescriptive fixes** grounded on deterministic recommendations. Ungrounded model output is rejected.
- **Zero-config instrumentation**: `shadow record -- python agent.py` works on any agent. No code changes.
- **Cross-OS**: Ubuntu, macOS, and Windows on Python 3.11, 3.12, and 3.13.

## Where to next

- New to Shadow? [Install and first diff](quickstart/install.md).
- Understanding the output? [Nine-axis diff](features/nine-axis.md).
- Wiring Shadow into CI? [Wire into CI](quickstart/ci.md).
- Looking for a specific feature? See the feature index in the sidebar.

## Links

- **PyPI**: [shadow-diff](https://pypi.org/project/shadow-diff/)
- **Source**: [manav8498/Shadow](https://github.com/manav8498/Shadow)
- **Spec**: [.agentlog v0.1](https://github.com/manav8498/Shadow/blob/main/SPEC.md)
- **License**: MIT OR Apache-2.0
