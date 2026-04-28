"""Tiny policy suggestion engine.

Looks at recorded ``.agentlog`` traces and surfaces candidate
``must_call_before`` rules — pairs of tools where one tool consistently
appears before another across every session in the trace. The output is
a list of suggestions; humans review and ``--accept`` them, never
auto-applied.

This is **deliberately narrow**. The external evaluator's "Risk Lab"
proposal asked for full policy synthesis (auto-detect "issue_refund
requires confirmation", "execute_sql cannot mutate prod"). That needs
production semantics that aren't in the trace data and would require
a content-business layer. This module ships only the part that *is*
derivable from observed traces: ordering invariants like "tool A is
always called before tool B".

Usage
-----
    from shadow.policy_suggest import suggest_policies

    suggestions = suggest_policies(records, scenario_aware=True)
    for s in suggestions:
        print(s.rule_id, s.confidence, s.kind, s.params)

CLI:
    shadow suggest-policies trace.agentlog --accept policies.yaml

The CLI surface (in shadow.cli.app) is added in a follow-up commit
to keep this module focused on the analysis logic.
"""

from shadow.policy_suggest.suggest import (
    PolicySuggestion,
    suggest_policies,
)

__all__ = [
    "PolicySuggestion",
    "suggest_policies",
]
