# Shadow

> **Git-native behavioral diff and shadow deployment for LLM agents.**
> Codecov for AI agents — replay production traces against a proposed
> config change, get a nine-axis behavioral diff in your PR, and
> automatically bisect which change caused which regression.

<!-- TODO(v0.2): replace with a real recorded demo GIF.
     Generate with: bash examples/demo/demo.sh | asciinema rec | agg out.gif -->
![demo — TODO: record and embed](.github/assets/demo.gif)

*(Demo GIF placeholder — the `just demo` command reproduces the terminal
output shown below. See `examples/demo/` for the exact script.)*

## Why Shadow?

Every AI team hits this problem weekly: a model drops
(`claude-opus-4-6` → `4-7`), a prompt gets edited, or a tool schema
changes. Nobody knows what silently regressed in production until a user
complains.

Existing observability products are dashboards you log into *after*
something is broken. **Shadow lives in the PR** — it tells you, before
you merge, that the candidate config produces 4× the latency, 2× the
verbosity, and a 33% increase in safety refusals, all attributable to
your prompt edit and not the model bump.

### What makes it different

| | Langfuse | Braintrust | LangSmith | **Shadow** |
|---|---|---|---|---|
| Dashboard to visit | ✓ | ✓ | ✓ | PR comment |
| Self-hostable | ✓ | — | — | ✓ (local-only in v0.1) |
| Replay prod against new config | ✓ | ✓ | ✓ | ✓ |
| Behavioral diff across N axes | — | partial (custom scorers) | partial | **9 axes, bootstrapped CIs** |
| Causal bisection | — | — | — | **LASSO + Plackett-Burman** |
| Content-addressed trace format (OSS spec) | — | — | — | **SPEC.md (Apache-2.0)** |
| CI-native | partial | partial | partial | **composite GitHub Action** |

Shadow is narrower in scope (no hosted UI, no cross-org trace
sharing) but sharper on the "does this PR regress things?" question
that owns you at 3 AM.

## Quickstart (60 seconds)

```bash
git clone https://github.com/shadow-dev/shadow
cd shadow
just setup          # user-local rustup if needed, then uv venv, maturin develop
just demo           # runs examples/demo/demo.sh — MockLLM, no network
```

Expected: the 9-axis diff table prints in your terminal in ≈1 s.

<details>
<summary>Sample output (abbreviated)</summary>

```
Shadow diff — 3 response pair(s)
axis              baseline  candidate  delta        95% CI            severity
semantic             1.000      0.732  -0.268   [-0.31, -0.25]        moderate
trajectory           0.000      0.000  +0.000   [+0.00, +1.00]            none
safety               0.000      0.333  +0.333   [+0.00, +1.00]            none
verbosity           26.000     52.000 +26.000   [-17.00, +40.00]        severe
latency             98.000    412.000 +314.000  [+314.00, +419.00]      severe
cost                 0.000      0.000  +0.000   [+0.00, +0.00]            none
reasoning            0.000      0.000  +0.000   [+0.00, +0.00]            none
judge                0.000      0.000  +0.000   [+0.00, +0.00]            none
conformance          0.000      0.000  +0.000   [+0.00, +0.00]            none
worst severity: severe
```
</details>

Then point it at your own code:

```bash
cd path/to/your/agent/repo
shadow init
# instrument your agent with shadow.sdk.Session (see examples/demo/agent.py)
python my_agent.py   # produces .shadow/traces/*.agentlog
shadow replay candidate-config.yaml --baseline path/to/baseline.agentlog
shadow diff path/to/baseline.agentlog .shadow/replays/latest.agentlog
```

## How it works

### Architecture

```mermaid
flowchart LR
    A[agent.py] -->|anthropic / openai client| B[shadow.sdk<br/>Session]
    B -->|redact + canonical JSON<br/>+ SHA-256| C[.agentlog<br/>records]
    C --> D[(content-addressed<br/>blob store)]
    C --> E[(SQLite index)]
    D --> F[replay engine]
    F -->|new config| G[candidate<br/>.agentlog]
    D --> H[9-axis differ]
    G --> H
    H --> I[DiffReport]
    I --> J[LASSO bisect]
    J --> K[attribution]
    I --> L[terminal / markdown /<br/>GitHub Action PR comment]
    K --> L
```

Five stages — record (Python SDK), store (content-addressed FS +
SQLite), replay (pluggable backend), diff (nine axes, bootstrap CI,
severity classification), bisect (LASSO over a Plackett-Burman design).

### Record → Replay → Diff

```mermaid
sequenceDiagram
    participant Agent
    participant Session as shadow.sdk.Session
    participant LLM as Anthropic/OpenAI
    participant Replay as shadow replay
    participant Diff as shadow diff

    Note over Session: redaction on by default
    Agent->>LLM: request (config A)
    LLM-->>Agent: response
    Agent->>Session: record_chat(request, response)
    Session->>Session: canonical JSON + SHA-256 → .agentlog
    Note over Replay: later, in CI
    Replay->>Replay: load baseline + candidate config B
    Replay->>LLM: replay requests under B (or MockLLM)
    LLM-->>Replay: new responses
    Replay->>Diff: baseline.agentlog, candidate.agentlog
    Diff->>Diff: 9 axes × bootstrap CI × severity
    Diff-->>Agent: PR comment (severity table)
```

### Bisection

```mermaid
flowchart TB
    A[config A vs config B] --> B[typed deltas<br/>model, params, prompt,<br/>tool schemas]
    B --> C{k ≤ 6?}
    C -->|yes| D[2^k full factorial]
    C -->|no| E[Plackett-Burman<br/>next multiple of 4 ≥ k+1]
    D --> F[replay at each corner]
    E --> F
    F --> G[per-axis divergence<br/>matrix runs × 9]
    G --> H[LASSO fit per axis<br/>sklearn]
    H --> I[ranked attribution<br/>86% trajectory ← tool schema<br/>12% ← system prompt<br/>2% ← model swap]
```

## CLI

| Command | What it does |
|---|---|
| `shadow init` | Scaffold `.shadow/` with config, sharded trace dir, SQLite index |
| `shadow record -- <cmd>` | Run `<cmd>` with `SHADOW_SESSION_OUTPUT` set, subprocess captures |
| `shadow replay <cfg> --baseline <trace>` | Replay baseline requests against `<cfg>`; writes a candidate `.agentlog` |
| `shadow diff <baseline> <candidate>` | Nine-axis diff with bootstrap CIs; optional `--output-json` |
| `shadow bisect <cfg-a> <cfg-b> --traces <set>` | LASSO attribution of axis movements to atomic config deltas |
| `shadow report <report.json> --format {terminal,markdown,github-pr}` | Re-render a saved DiffReport |

See `CLAUDE.md` §Coding conventions for the exact error-message format
(every error ends with a `hint:` line).

## The `.agentlog` format

Open spec in [`SPEC.md`](SPEC.md) — Apache-2.0, dual-licensed separately
from the MIT-licensed implementation. Headlines:

- **JSON Lines**, one record per line, streaming-safe.
- **Content-addressed**: `id = sha256(canonical_json(payload))`
  (RFC 8785 JCS + Unicode NFC). Two identical requests dedupe to one
  blob (§6.1).
- **OpenTelemetry GenAI compatible** — every field maps onto the OTel
  semantic conventions (§7).
- **Redaction-by-default** at record boundaries, with a per-key
  allowlist (§9).
- **Known-vector conformance test** (§5.6): `{"hello":"world"}` →
  `sha256:93a23971…681588`.

## GitHub Action

Composite action under [`.github/actions/shadow-action/`](.github/actions/shadow-action/).
Post a nine-axis PR comment on every PR that touches your prompts or
configs:

```yaml
- uses: shadow-dev/shadow/.github/actions/shadow-action@v0.1.0
  with:
    baseline: path/to/baseline.agentlog
    candidate: path/to/candidate.agentlog   # produced earlier in the workflow
    pricing: path/to/pricing.json           # optional
```

Sample PR comment in [`docs/sample-pr-comment.md`](docs/sample-pr-comment.md).

## Limitations (v0.1)

Deliberate scope cuts — honesty up front beats discovering them after
you've adopted:

- **Local-only.** Traces live on disk + SQLite; no cloud, no remote
  sharing. Exporters for Langfuse / Braintrust / LangSmith map cleanly
  from the `.agentlog` format but aren't shipped in v0.1 (SPEC §11).
- **Semantic axis uses a hash surrogate in pure-Rust tests.**
  Production embeddings (`sentence-transformers/all-MiniLM-L6-v2`) live
  in the Python layer and require the `[embeddings]` extra.
- **Bisection divergence is a placeholder in v0.1.** The delta-detection
  and LASSO-attribution plumbing are correct and tested against a
  synthetic ground-truth recovery case (≥0.9 attribution to the true
  driver); but the per-corner replay scorer — the step that turns
  "here are the corners" into "here's how much each corner diverged" —
  defaults to zeros until the v0.2 live-replay wiring lands.
- **No auto-instrumentation** of `anthropic` / `openai` Python clients.
  Users instrument manually with `shadow.sdk.Session.record_chat()`.
  The two SDKs' streaming surfaces are too divergent to unify cleanly
  in a first cut.
- **Judge axis is a trait.** Shadow doesn't ship a default Judge; users
  plug in their own LLM-judge via the `Judge` protocol.
- **CI: Ubuntu + macOS only.** Windows isn't tested in v0.1.
- **Redaction is record-boundary, not token-level.** See SPEC §1.2.

## Contributing

1. `just setup` to bootstrap.
2. `just ci` must pass locally before pushing.
3. All changes follow [Conventional Commits](https://www.conventionalcommits.org/)
   (`type(scope): subject`).
4. Every behavioural change lands with a test; strict TDD is preferred.
5. CLAUDE.md §Workflow is the source of truth for process conventions.

## License

- Implementation code: **MIT** (see [`LICENSE`](LICENSE)).
- `SPEC.md` (the `.agentlog` format): **Apache-2.0** (see
  [`LICENSE-SPEC`](LICENSE-SPEC)).

Dual-licensing is deliberate: anyone can re-implement the `.agentlog`
format under any license, while the reference implementation stays
MIT.
