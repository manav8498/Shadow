# Shadow: Causal Regression Forensics for AI Agents

**Status:** approved design (2026-05-03)
**Owner:** manav8498
**Supersedes:** the broad "Agent Behavior Control Plane" framing
**Implementation plan:** to be generated via `superpowers:writing-plans`

---

## 1. Strategic pivot

Shadow narrows from a broad "behavior control plane" to a single, defensible
technical wedge:

> **Shadow tells you which prompt, model, tool schema, or config change broke
> your AI agent — proven against production-like traces before merge.**

### 1.1 Why narrow

The broad lane is crowded. EvalView, Microsoft Agent Governance Toolkit,
Preloop, AIUC-1 / Schellman, and AgentEvals already occupy adjacent
positions — eval frameworks, runtime governance, control planes,
certification, and OTel-based trace scoring respectively. Shadow loses on
breadth and wins on a question none of them answer directly:

> **Which exact change caused the regression, how many real traces does it
> affect, with what confidence, and which fix reverses it?**

### 1.2 Five questions the wedge product answers

`shadow diagnose-pr` answers, in one PR comment:

1. Did agent behavior change?
2. How many real or production-like traces are affected?
3. Which exact candidate change caused the regression?
4. How confident are we?
5. What fix should be verified before merge?

### 1.3 Non-goals

The following are explicitly *not* the headline. They may exist as supporting
infrastructure but are demoted in messaging and roadmap priority:

- ABOM expansion
- generic policy packs
- approval workflows
- dashboards
- generic runtime governance
- generic agent eval framework
- certification marketplace
- broad MCP firewall

Microsoft AGT, Preloop, and AIUC-1 already cover these spaces; competing
there is not the wedge.

---

## 2. Existing assets (verified by recon, 2026-05-03)

The pivot composes existing Shadow internals rather than reinventing them.

| Capability | Module | Public API |
|---|---|---|
| Pearl-style causal attribution (ATE, bootstrap CI, back-door, E-value) | `python/src/shadow/causal/attribution.py` | `causal_attribution(baseline_config, candidate_config, replay_fn, n_replays, n_bootstrap, confounders, sensitivity, ...) -> CausalAttribution` |
| Live LLM replayer (deterministic seed-from-hash, config-hash cache, env-only API key, retry, divergence) | `python/src/shadow/causal/replay/openai_replayer.py` | `OpenAIReplayer(...).__call__(config) -> ReplayResult` |
| Production trace mining (clusters by `{tool seq, stop reason, length bucket, latency bucket}`, score-based representative selection) | `python/src/shadow/mine.py` | `mine(traces, max_cases, per_cluster, pricing) -> MiningResult` |
| 9-axis behavioral diff | `crates/shadow-core/src/diff/` (Rust) via PyO3 | `shadow._core.compute_diff_report(baseline, candidate, pricing, seed) -> dict` |
| Policy DSL + checker (`must_call_before`, `must_call_once`, `no_call`, `max_turns`, etc.) | `python/src/shadow/hierarchical.py` | `load_policy(data)`, `check_policy(records, rules)`, `policy_diff(baseline, candidate, rules)` |
| `.agentlog` parser/writer | `crates/shadow-core/src/agentlog/` via PyO3 | `shadow._core.parse_agentlog(bytes) -> list[Record]` |
| Existing CLI commands | `python/src/shadow/cli/app.py` (Typer) | `init`, `diff`, `report`, `gate`, `mine`, `replay`, `bisect`, `certify`, etc. |
| GitHub Action | `.github/actions/shadow-action/action.yml` + `comment.py` | composite action: install → run diff → render → post comment |

### 2.1 Two divergence systems

A subtle but architecturally important fact:

- **9-axis structured diff** (Rust core) — produces per-axis `AxisStat`
  with `{baseline_median, candidate_median, delta, ci95_low, ci95_high,
  severity, n, flags}` for human reports. Axes: semantic, trajectory,
  safety, verbosity, latency, cost, reasoning, judge, conformance.

- **5-axis scalar divergence** (Python replayer) — produces a single
  `dict[axis_name, float]` for causal attribution input. Axes: semantic,
  trajectory, safety, verbosity, latency.

`diagnose-pr` uses **both**: the 9-axis differ for "is this trace affected?"
and human report rendering; the 5-axis scalar for `causal_attribution`'s
`ReplayFn` contract. Do not force the causal engine to consume the full
9-axis structured report.

### 2.2 Recon refinements

Three points the original memo missed that are folded into the design:

1. **Mining is already done.** `shadow.mine.mine()` already implements the
   clustering + scoring described in the memo. `diagnose-pr` calls it
   when `len(traces) > --max-traces`; no new mining module is needed.

2. **Replay backend defaults to pre-baked, live LLM is opt-in.** Real LLM
   calls cost money and are slow. The default backend is `recorded`
   (loads candidate traces from disk). `live` (OpenAIReplayer) and
   `mock` (deterministic stub) are opt-in via `--backend`.

3. **Causal `ReplayFn` is per-config, not per-trace.** The signature is
   `replay_fn(config) -> dict[axis, float]` — a single divergence vector
   per config. To turn a corpus of traces into that scalar input, we
   wrap with `diagnose_pr.attribution.build_corpus_replay_fn(traces) ->
   ReplayFn` that runs the per-trace replay loop and means the per-axis
   divergences across the corpus. The causal engine itself stays
   untouched.

4. **Policy module is `shadow.hierarchical`.** Naming is historical;
   `diagnose_pr/policy.py` will be a thin adapter that imports from
   `shadow.hierarchical` and exposes `evaluate_policy_diff(...)`.

5. **`shadow gate` already exists.** `gate-pr` is a *new* CI-friendly
   wrapper around `diagnose-pr`, not a replacement of `gate`. Both
   coexist.

---

## 3. The flagship product

### 3.1 Command surface (Phase 1)

```bash
shadow diagnose-pr \
  --traces .shadow/prod-sample/ \
  --baseline-config .shadow/baseline.yaml \
  --candidate-config .shadow/candidate.yaml \
  --policy shadow-policy.yaml \
  --changed-files system_prompt.md tools.yaml model.yaml \
  --max-traces 200 \
  --n-replays 1 \
  --n-bootstrap 500 \
  --confidence 0.95 \
  --backend recorded \
  --out .shadow/diagnose-pr/report.json \
  --pr-comment .shadow/diagnose-pr/comment.md \
  --fail-on hold
```

**Flags:**

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--traces` | path | required | dir or file list of `.agentlog` |
| `--baseline-config` | path | required | YAML, same schema as `shadow replay` |
| `--candidate-config` | path | required | YAML |
| `--policy` | path | optional | YAML, `shadow.hierarchical` schema |
| `--changed-files` | path... | optional | files changed in PR (for delta extraction) |
| `--max-traces` | int | 200 | mining cap |
| `--n-replays` | int | 1 | per-config replay count for noise reduction |
| `--n-bootstrap` | int | 500 | bootstrap resample count for CI |
| `--confidence` | float | 0.95 | CI width |
| `--backend` | enum | `recorded` | `recorded` \| `live` \| `mock` |
| `--out` | path | required | JSON report destination |
| `--pr-comment` | path | optional | markdown comment destination |
| `--fail-on` | enum | `none` | `none` \| `probe` \| `hold` \| `stop` |

### 3.2 Algorithm

```
diagnose_pr.run(...) =
  1. Load baseline + candidate configs (YAML)
  2. Extract config deltas (deltas.py)
  3. Load all .agentlog traces from --traces
  4. If len(traces) > max_traces: traces = mine(traces, max_traces)
  5. For each selected trace:
     a. Get baseline records (already in trace)
     b. Get candidate records: load from disk OR replay candidate config
     c. Run 9-axis diff (compute_diff_report)
     d. Classify affected/not-affected (see §3.4)
     e. Run policy check if --policy provided
     f. Store TraceDiagnosis
  6. Build corpus_replay_fn over selected traces
  7. Run causal_attribution(baseline_config, candidate_config,
                           corpus_replay_fn, n_bootstrap, sensitivity=True)
  8. Convert CausalAttribution into list[CauseEstimate], rank by
     |ate| * confidence_multiplier * blast_radius
  9. Compute blast_radius = affected / total
 10. Compute verdict (see §3.5)
 11. Emit report.json (see §3.6)
 12. Emit PR-comment markdown (see §3.7)
 13. Exit code from --fail-on threshold
```

### 3.3 Data models (`diagnose_pr/models.py`)

```python
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["ship", "probe", "hold", "stop"]

DeltaKind = Literal[
    "prompt", "model", "tool_schema", "retriever",
    "temperature", "policy", "unknown",
]

@dataclass(frozen=True)
class ConfigDelta:
    id: str               # "system_prompt.md:47" or "model:gpt-4.1->gpt-4.1-mini"
    kind: DeltaKind
    path: str             # config-key path or file path
    old_hash: str | None
    new_hash: str | None
    display: str          # human-readable

@dataclass(frozen=True)
class TraceDiagnosis:
    trace_id: str
    affected: bool
    risk: float           # 0..100
    worst_axis: str | None
    first_divergence: dict[str, Any] | None
    policy_violations: list[dict[str, Any]]

@dataclass(frozen=True)
class CauseEstimate:
    delta_id: str
    axis: str
    ate: float
    ci_low: float | None
    ci_high: float | None
    e_value: float | None
    confidence: float     # 0.0 or 1.0 in v1

@dataclass(frozen=True)
class DiagnosePrReport:
    schema_version: str            # "diagnose-pr/v0.1"
    verdict: Verdict
    total_traces: int
    affected_traces: int
    blast_radius: float            # 0.0..1.0
    dominant_cause: CauseEstimate | None
    top_causes: list[CauseEstimate]
    trace_diagnoses: list[TraceDiagnosis]
    affected_trace_ids: list[str]  # subset of trace_diagnoses where affected=True
    new_policy_violations: int
    worst_policy_rule: str | None
    suggested_fix: str | None
    flags: list[str]               # e.g. ["low_power"] when n < 30
```

### 3.4 Affected-trace classification

```python
affected = (
    any axis severity in {"moderate", "severe"}
    OR len(new_policy_violations) > 0
    OR (first_divergence is not None
        AND first_divergence is not "trivial")
)
```

"trivial" means delta below per-axis threshold; thresholds are pulled from
the existing 9-axis differ (already calibrated in `crates/shadow-core/
src/diff/axes.rs`).

### 3.5 Risk and verdict

```python
ci_excludes_zero = (
    cause.ci_low is not None
    and cause.ci_high is not None
    and (cause.ci_low > 0 or cause.ci_high < 0)
)
confidence_multiplier = 1.0 if ci_excludes_zero else 0.5
impact = abs(cause.ate)
risk = min(100.0, 100.0 * impact * blast_radius * confidence_multiplier)
```

**Verdict thresholds (v1):**

| Verdict | Condition |
|---|---|
| `ship` | `affected_traces == 0` and no new policy violations |
| `probe` | affected traces exist but `not ci_excludes_zero` and no dangerous policy violation |
| `hold` | affected traces exist and `ci_excludes_zero` |
| `stop` | dangerous-tool policy violation OR severe safety/trajectory regression |

"Dangerous tool" detection (v1):

- Policy rule has `severity: error` or `severity: critical`, **and**
- One of:
  - Rule has explicit `tags: [dangerous]` in its YAML (preferred path,
    forward-compatible).
  - Rule's `params.tool` (or `params.first` / `params.then`) name
    contains a v1 keyword: `refund`, `pay`, `transfer`, `wire`,
    `delete`, `drop`, `shutdown`, `revoke`, `grant`, `escalate`.

The keyword fallback is a v1 stopgap; v2 makes `tags: [dangerous]` the
only path. The keyword list lives in `diagnose_pr/risk.py:_DANGEROUS_KEYWORDS`
so it's easy to find and review.

### 3.6 JSON report schema

```json
{
  "schema_version": "diagnose-pr/v0.1",
  "verdict": "hold",
  "total_traces": 1247,
  "affected_traces": 84,
  "blast_radius": 0.067,
  "dominant_cause": {
    "delta_id": "system_prompt.md:47",
    "axis": "trajectory",
    "ate": 0.31,
    "ci_low": 0.22,
    "ci_high": 0.44,
    "e_value": 2.8,
    "confidence": 1.0
  },
  "top_causes": [
    {
      "delta_id": "system_prompt.md:47",
      "axis": "trajectory",
      "ate": 0.31,
      "ci_low": 0.22,
      "ci_high": 0.44,
      "e_value": 2.8,
      "confidence": 1.0
    },
    {
      "delta_id": "params.temperature:0.2->0.7",
      "axis": "verbosity",
      "ate": 0.04,
      "ci_low": -0.02,
      "ci_high": 0.10,
      "e_value": 1.1,
      "confidence": 0.5
    }
  ],
  "trace_diagnoses": [
    {
      "trace_id": "sha256:aeaa25c8...",
      "affected": true,
      "risk": 78.4,
      "worst_axis": "trajectory",
      "first_divergence": {
        "pair_index": 2,
        "axis": "trajectory",
        "detail": "tool order: confirm_refund_amount missing"
      },
      "policy_violations": [
        {"rule_id": "confirm-before-refund", "severity": "error", "pair_index": 2}
      ]
    }
  ],
  "affected_trace_ids": ["sha256:aeaa25c8...", "sha256:b1c2d3e4..."],
  "new_policy_violations": 6,
  "worst_policy_rule": "confirm-before-refund",
  "suggested_fix": "Restore the refund confirmation instruction.",
  "flags": []
}
```

### 3.7 PR comment (markdown)

Plain English first, metrics second:

```markdown
## Shadow verdict: HOLD

This PR changes agent behavior on **84 / 1,247** production-like traces.

### Dominant cause

`system_prompt.md:47` appears to be the main cause.

- Axis: `trajectory`
- ATE: `+0.31`
- 95% CI: `[0.22, 0.44]`
- E-value: `2.8`

### Why it matters

6 traces now call `issue_refund` before `confirm_refund_amount`.

### Suggested fix

Restore the confirmation instruction or add this policy:

```yaml
- id: confirm-before-refund
  kind: must_call_before
  params:
    first: confirm_refund_amount
    then: issue_refund
  severity: error
```

### Verify the fix

```bash
shadow verify-fix --report .shadow/diagnose-pr/report.json
```
```

The renderer lives in `diagnose_pr/render.py` and is **plain Python** (no
Rust) so we can iterate on the human voice without recompiling.

---

## 4. File plan

### 4.1 New files (Phase 1)

```
python/src/shadow/diagnose_pr/__init__.py
python/src/shadow/diagnose_pr/models.py        # dataclasses (§3.3)
python/src/shadow/diagnose_pr/loaders.py       # YAML + .agentlog loaders
python/src/shadow/diagnose_pr/deltas.py        # ConfigDelta extraction
python/src/shadow/diagnose_pr/diffing.py       # 9-axis diff per trace
python/src/shadow/diagnose_pr/attribution.py   # corpus replay_fn + cause ranking
python/src/shadow/diagnose_pr/policy.py        # adapter to shadow.hierarchical
python/src/shadow/diagnose_pr/risk.py          # verdict / risk / blast_radius
python/src/shadow/diagnose_pr/report.py        # DiagnosePrReport assembly + JSON
python/src/shadow/diagnose_pr/render.py        # PR-comment markdown
python/tests/test_diagnose_pr.py               # end-to-end tests
python/tests/test_diagnose_pr_render.py        # snapshot tests for markdown
python/tests/test_diagnose_pr_risk.py          # verdict matrix
python/tests/test_diagnose_pr_deltas.py        # delta extractor unit tests
```

### 4.2 Modified files

- `python/src/shadow/cli/app.py` — register `diagnose-pr` (and later
  `verify-fix`, `gate-pr`).
- `README.md` — hero rewrite per §6.
- `docs/features/causal-pr-diagnosis.md` — new feature page.
- `.github/actions/shadow-action/action.yml` — add `diagnose-pr` step.

### 4.3 Phase 2 deliverable

```
examples/refund-causal-diagnosis/
├── README.md
├── demo.sh
├── prompts/baseline.md
├── prompts/candidate.md
├── config/baseline.yaml
├── config/candidate.yaml
├── policy.yaml
├── traces/                      # pre-baked .agentlog fixtures
└── expected/comment.md          # snapshot of expected PR output
```

### 4.4 Phase 3 files (verify-fix)

```
python/src/shadow/verify_fix/__init__.py
python/src/shadow/verify_fix/runner.py
python/src/shadow/verify_fix/report.py
python/tests/test_verify_fix.py
```

---

## 5. Delivery plan

### 5.1 30-day plan

| Week | Deliverable | Definition of done |
|---|---|---|
| 1 | `models.py`, `loaders.py`, `deltas.py`, `report.py`, `render.py`, CLI registration | `shadow diagnose-pr --help` works; running against `examples/demo/` produces `report.json` + `comment.md` with `verdict`, `total_traces`, `affected_traces`, `blast_radius`. No causal attribution yet. |
| 2 | `diffing.py` (9-axis integration), `policy.py` (hierarchical adapter), `risk.py` (verdict mapping) | Candidate with known tool-order policy violation returns HOLD/STOP and PR comment names the violated rule. |
| 3 | `attribution.py` (causal integration), dominant cause + ATE/CI/E-value rendering | Synthetic two-delta test: prompt_delta causes regression, model_delta does not → `dominant_cause == prompt_delta` and CI excludes zero. |
| 4 | `examples/refund-causal-diagnosis/`, `docs/features/causal-pr-diagnosis.md`, `shadow gate-pr`, GH Action update | `cd examples/refund-causal-diagnosis && ./demo.sh` outputs HOLD/STOP, names prompt delta as dominant cause, names `confirm-before-refund` policy, suggests fix. |

### 5.2 60-day plan

| Days | Deliverable | DoD |
|---|---|---|
| 31–45 | `shadow verify-fix` (Phase 3) | Bad candidate fails; fixed candidate passes; safe traces don't regress. |
| 46–60 | OTel GenAI importer/exporter (Phase 5) | `shadow import --format otel-genai` round-trips through `diagnose-pr`. |

### 5.3 90-day plan

| Days | Deliverable | DoD |
|---|---|---|
| 61–75 | `shadow-align` standalone library (Phase 6) | Rust crate + Python + TS wrappers, alignment algorithm reusable outside Shadow CLI. |
| 76–90 | Public comparison pages (Shadow vs EvalView / Microsoft AGT / Preloop / AgentEvals / Speedscale) | Honest diff matrix, no vaporware claims. |

---

## 6. Repositioning (Phase 0)

### 6.1 README hero (replaces current top block)

```markdown
# Shadow

**Find the exact change that broke your AI agent.**

Shadow tells you which prompt, model, tool schema, or config change broke
your AI agent — proven against production-like traces before merge.

```bash
shadow diagnose-pr --traces prod-traces/ --baseline-config baseline.yaml --candidate-config candidate.yaml
```

Three steps:

1. **Replay** production-like traces against a candidate change.
2. **Diagnose** the exact behavior regression and dominant cause, with confidence.
3. **Verify** the fix before merge.
```

### 6.2 Demoted (still documented, lower in README)

- ABOM
- runtime guardrails
- behavior control plane framing
- generic 9-axis diff messaging
- policy packs

---

## 7. Verify-fix (Phase 3 preview)

```bash
shadow verify-fix \
  --report .shadow/diagnose-pr/report.json \
  --candidate-config .shadow/candidate.yaml \
  --fixed-config .shadow/fixed.yaml \
  --traces .shadow/prod-sample/ \
  --out .shadow/verify-fix/report.json
```

**Behavior:**

1. Read diagnose-pr report → extract `affected_trace_ids`.
2. Replay only affected traces against fixed config.
3. Confirm regression reversed (delta on dominant axis ≤ threshold).
4. Sample N safe traces (default 100), confirm no new regression.
5. Re-check policy violations.
6. Emit verify-fix report.

**Pass criteria:**

- `affected_reversed_rate >= 0.90`
- `new_policy_violations == 0`
- `safe_trace_regression_rate <= 0.02`
- `dominant_axis_delta <= per-axis threshold`

**Output:**

```
Shadow verify-fix: PASS

Regression reversed:
  82 / 84 affected traces now match the baseline-safe path.

Safe trace check:
  0 / 100 sampled safe traces regressed.

No new policy violations.
```

---

## 8. Gate-pr (Phase 4 preview)

`gate-pr` is a thin CI-friendly wrapper around `diagnose-pr`. It does not
replace `shadow gate` (which is a generic post-hoc report gate); it is a
new command optimized for GitHub Action use.

```bash
shadow gate-pr \
  --traces .shadow/prod-sample/ \
  --baseline-config .shadow/baseline.yaml \
  --candidate-config .shadow/candidate.yaml \
  --policy shadow-policy.yaml \
  --pr-comment shadow-comment.md
```

**Exit codes:**

- 0 = ship
- 1 = probe / hold
- 2 = stop
- 3 = internal/tooling error

The existing GitHub Action (`.github/actions/shadow-action/action.yml`)
gets a new step that calls `gate-pr` instead of (or alongside) `shadow
diff` + `shadow gate`.

---

## 9. Architectural risks and mitigations

| Risk | Mitigation |
|---|---|
| **Live-replay cost.** Real LLM calls × N traces × M deltas × bootstrap can run into real money. | Default backend is `recorded`. Live replay is opt-in via `--backend live`. Config-hash cache on `OpenAIReplayer` already short-circuits repeated identical calls. |
| **Causal attribution noise on small N.** Bootstrap CI on tiny corpora is wide, leading to no `ci_excludes_zero` and stuck `probe` verdicts. | Surface `low_power` flag in report when n < 30. Document the `--n-replays` and `--n-bootstrap` knobs. |
| **Delta extraction false positives.** Reformatting a YAML file shows as a delta even though semantics didn't change. | Compute hashes over **canonical** form (sorted keys, normalized whitespace) — same canonicalisation `shadow.certify` uses. |
| **The 9-axis differ severity calibration may not match what "affected" should mean for diagnose-pr.** | Make affected-trace classification a thin policy in `risk.py`, not buried in the differ. Easy to tune with feedback. |
| **`shadow gate` overlap.** Users may be confused by `gate` vs `gate-pr`. | Document the split clearly: `gate` consumes a saved report.json, `gate-pr` orchestrates the full PR-time flow. |
| **Suggested-fix generation is non-trivial.** v1 cannot generate full patches. | v1 produces a *fix hint* (one sentence + optional policy YAML snippet). Real patch generation is its own future workstream. |
| **`PR comment` voice drift over time.** Renderer is in Python by design; snapshot tests pin the format. | `test_diagnose_pr_render.py` uses goldens; updating the goldens is a deliberate review step. |

---

## 10. Out of scope (this design)

- Web dashboard for diagnose-pr reports.
- Multi-PR diff aggregation.
- Per-user / per-org policy templates.
- Patch generation via LLM.
- Real-time runtime enforcement (lives in `shadow.policy_runtime`, not
  this surface).
- Cross-language SDKs for `diagnose-pr` itself (Python is enough; the
  *trace format* and *alignment library* will go cross-language in
  Phase 6).

---

## 11. Open questions for review

1. **Default `--max-traces`.** 200 is a round number from the memo; do we
   want to calibrate from the demo / customer-support / devops-agent
   examples instead?
2. **Verdict gradient.** v1 has 4 levels (`ship` / `probe` / `hold` /
   `stop`). Consider whether `probe` should bin into `probe-low` /
   `probe-high` based on blast radius.
3. **Policy adapter naming.** `diagnose_pr/policy.py` adapts
   `shadow.hierarchical`. Should we also rename the underlying module
   to `shadow.policy` for clarity? (Out of scope for this design but
   worth flagging as a follow-up.)
4. **`gate-pr` exit code 3.** Should "internal error" actually fall back
   to exit 1 to avoid silent CI passes? Conservative answer: yes —
   default to fail-closed.

---

## 12. Acceptance criteria (Phase 1 done)

- `shadow diagnose-pr --help` shows the documented flags.
- Running against `examples/demo/` produces a JSON report with
  `verdict`, `total_traces`, `affected_traces`, `blast_radius`,
  `dominant_cause` (with non-null ATE/CI/E-value), and a markdown PR
  comment matching the §3.7 format (snapshot test).
- Running against `examples/refund-causal-diagnosis/` produces:
  - `verdict ∈ {hold, stop}`
  - `dominant_cause.delta_id` references the prompt file
  - `worst_policy_rule == "confirm-before-refund"`
  - PR comment includes "Suggested fix" section
- Synthetic two-delta unit test (`test_diagnose_pr.py`):
  prompt_delta causes regression, model_delta does not →
  `dominant_cause == prompt_delta` and `ci_excludes_zero == True`.
- All existing tests still pass (1701 + new tests).
- `cargo fmt --check`, `cargo clippy -D warnings`, `ruff check`, `ruff
  format --check` all clean.

---

*End of design.*
