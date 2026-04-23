"""Default LLM judges for Shadow's `judge` axis.

Shadow's 9-axis diff reserves axis 8 for a "did the candidate do the
job at least as well as the baseline?" signal. Because any default
rubric would be domain hardcoding, the Rust core leaves this axis
empty and exposes a Judge trait for user-supplied evaluators. This
Python package ships one ready-made judge — `SanityJudge` — that
callers can plug in to populate axis 8 without writing their own
rubric.

`SanityJudge` is honest about what it is: a *sanity check*, not a
correctness oracle. It asks a judge LLM "is the candidate at least as
good as the baseline for this task?" and returns a score in `[0, 1]`
per pair. Use it as a floor; swap in a domain-specific rubric for
production.
"""

from shadow.judge.aggregate import aggregate_scores
from shadow.judge.base import Judge, JudgeVerdict
from shadow.judge.correctness import CorrectnessJudge
from shadow.judge.format import FormatJudge
from shadow.judge.pairwise import PairwiseJudge
from shadow.judge.sanity import SanityJudge

__all__ = [
    "CorrectnessJudge",
    "FormatJudge",
    "Judge",
    "JudgeVerdict",
    "PairwiseJudge",
    "SanityJudge",
    "aggregate_scores",
]
