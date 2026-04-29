"""shadow.autopr — synthesise policy rules from a regression.

Given a (baseline, candidate) trace pair where the candidate has
regressed, this module emits a Shadow policy YAML whose rules pin
the regression. The rules use Shadow's existing 12-kind policy
language; no new vocabulary is introduced.

Pure deterministic. No LLM. No network. Safe in CI.

The closed loop is:

    1. CI catches a regression on the candidate trace
    2. `shadow autopr baseline.agentlog candidate.agentlog` writes
       a policy YAML
    3. The next PR runs `shadow diff ... --policy <generated>.yaml`
       and the same regression is now a hard-fail gate.

Reference: see the strategy notes in this repo's design discussion;
the surface is intentionally a thin layer on top of the existing
`shadow.hierarchical` policy module so the generated YAML is
guaranteed to round-trip through `load_policy()`.
"""

from shadow.autopr.synthesis import (
    SynthesizedPolicy,
    synthesize_policy,
    verify_policy,
)

__all__ = ["SynthesizedPolicy", "synthesize_policy", "verify_policy"]
