# Judges

Shadow's nine-axis diff reserves axis 8 (`judge`) for a "did the
candidate do the job at least as well as the baseline?" signal.
Because any default rubric would be domain hardcoding, the Rust core
leaves this axis empty and exposes a `Judge` Protocol for user-supplied
evaluators.

The Python package ships **ten ready-made judges** that cover common
domains, plus a `LlmJudge` base for custom rubrics.

## The ten

| Judge | Catches |
|---|---|
| `SanityJudge` | Generic "better / equal / worse" regression floor |
| `PairwiseJudge` | Position-bias-free A/B preference (double-evaluates with flipped order) |
| `CorrectnessJudge` | Matches candidate against a reference-answer rubric |
| `FormatJudge` | Mechanical JSON schema conformance, no LLM calls |
| `LlmJudge` | User-configurable: rubric prompt + score map |
| `SchemaConformanceJudge` | Semantic schema review (shape + meaning) |
| `ProcedureAdherenceJudge` | Catches dropped safety-procedure steps |
| `FactualityJudge` | Contradictions with a known-fact set |
| `RefusalAppropriateJudge` | Over-/under-refusal vs explicit policy |
| `ToneJudge` | Tone / persona drift vs target |

All ten are verified end-to-end against real Anthropic Claude Haiku 4.5
and OpenAI gpt-4o-mini. `SanityJudge` and `PairwiseJudge` are
domain-agnostic; the rest take rubric data via `--judge-config file.yaml`.

## `--judge auto`

If you set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your env, run:

```bash
shadow diff baseline.agentlog candidate.agentlog --judge auto
```

`auto` picks `sanity` against whichever backend key is set
(Anthropic preferred, cheaper). Turns axis 8 from an empty row into
a real signal. Budget: ~$0.0003 per diff run.

If neither key is set, `auto` falls through to `none` cleanly with a
one-line hint naming the env vars to set.

## `--explain`

Opt-in LLM-generated paragraph summarising the diff for a tech lead:

```bash
shadow diff baseline.agentlog candidate.agentlog --judge auto --explain
```

Emits a tight ~60-word narrative after the axis table:

> Candidate agent regressed severely: tool set shrunk (4→1 tools),
> format compliance failed (-1.0), semantic similarity collapsed
> (-0.94), response bloated then truncated (-226 tokens), latency
> spiked (+1110ms). Root cause: structural drift at turn 0-critical
> database/notification tools removed, `run_migration` signature
> changed. Priority: restore removed tools and verify parameter
> signatures match baseline.

Requires `--judge-backend anthropic|openai` (same keys auto-detected
for `--judge auto`). ~300-500 tokens per run.

## Writing custom rubrics

`LlmJudge` is the generic base:

```python
from shadow.judge import LlmJudge
from shadow.llm import AnthropicLLM

judge = LlmJudge(
    backend=AnthropicLLM(),
    rubric="""Rate whether CANDIDATE correctly answered TASK relative to BASELINE.
    Reply with JSON: {{"verdict": "great"|"ok"|"poor", "confidence": 0-1, "reason": "..."}}

    TASK: {task}
    BASELINE: {baseline}
    CANDIDATE: {candidate}
    """,
    score_map={"great": 1.0, "ok": 0.5, "poor": 0.0},
    model="claude-haiku-4-5-20251001",
)
```

Allowed placeholders: `{task}`, `{baseline}`, `{candidate}`. Any
other `{name}` in the rubric raises `ValueError` at construction
time, fail fast.

Or drop a YAML config at `examples/judges/my-judge.yaml`:

```yaml
rubric: |
  Rate on a three-tier scale.
  TASK: {task}
  BASELINE: {baseline}
  CANDIDATE: {candidate}
  Reply with JSON: {{"verdict": "great"|"ok"|"poor", "confidence": 0-1, "reason": "..."}}
score_map:
  great: 1.0
  ok: 0.5
  poor: 0.0
```

And use via CLI:

```bash
shadow diff baseline.agentlog candidate.agentlog \
  --judge llm --judge-config examples/judges/my-judge.yaml \
  --judge-backend anthropic
```
