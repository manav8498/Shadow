"""Default LLM judges for Shadow's `judge` axis.

Shadow's 9-axis diff reserves axis 8 for a "did the candidate do the
job at least as well as the baseline?" signal. Because any default
rubric would be domain hardcoding, the Rust core leaves this axis
empty and exposes a Judge trait for user-supplied evaluators. This
Python package ships ten ready-made judges callers can plug in
without writing their own rubric.

Inventory (10 judges):

- **Coarse / generic:** `SanityJudge` — "is the candidate at least as
  good as the baseline?" Use as a regression floor, not an oracle.
- **Coarse / generic, position-bias-resistant:** `PairwiseJudge` —
  A/B preference, double-evaluated with flipped order.
- **User-configurable:** `LlmJudge` — generic LLM-as-judge with a
  rubric and score-map supplied by the caller. Use when the five
  rubric-driven judges below don't cover the case.
- **Reference-matching:** `CorrectnessJudge` — score against a
  known-answer rubric.
- **Mechanical schema:** `FormatJudge` — tight JSON-schema
  conformance, no LLM calls.
- **Semantic schema:** `SchemaConformanceJudge` — shape + meaning
  check via LLM review.
- **Procedure adherence:** `ProcedureAdherenceJudge` — catches dropped
  safety steps (the devops-agent use case).
- **Factuality:** `FactualityJudge` — flags contradictions with a
  known-fact set.
- **Refusal policy:** `RefusalAppropriateJudge` — over-/under-refusal
  against an explicit policy.
- **Tone/persona:** `ToneJudge` — tone drift against a target.

Every LLM-backed judge has a deterministic `MockLLM` path for tests;
none require network access at import time.
"""

from shadow.judge.aggregate import aggregate_scores
from shadow.judge.base import Judge, JudgeVerdict
from shadow.judge.correctness import CorrectnessJudge
from shadow.judge.factuality import FactualityJudge
from shadow.judge.format import FormatJudge
from shadow.judge.llm import LlmJudge
from shadow.judge.pairwise import PairwiseJudge
from shadow.judge.procedure import ProcedureAdherenceJudge
from shadow.judge.refusal import RefusalAppropriateJudge
from shadow.judge.sanity import SanityJudge
from shadow.judge.schema import SchemaConformanceJudge
from shadow.judge.tone import ToneJudge

__all__ = [
    "CorrectnessJudge",
    "FactualityJudge",
    "FormatJudge",
    "Judge",
    "JudgeVerdict",
    "LlmJudge",
    "PairwiseJudge",
    "ProcedureAdherenceJudge",
    "RefusalAppropriateJudge",
    "SanityJudge",
    "SchemaConformanceJudge",
    "ToneJudge",
    "aggregate_scores",
]
