"""shadow.heal — causal-classifier-only auto-heal (audit mode).

Given a diff report, classifies the regression into one of three tiers:

    * ``heal``    — every gate passed; the change is provably cosmetic
                    or below the noise floor. An auto-mode (not yet
                    enabled) would accept the candidate as the new
                    anchor.
    * ``propose`` — the change is real but not blocking; the gates that
                    matter for safety all passed. An auto-mode would
                    save the candidate as a named variant for the user
                    to approve explicitly.
    * ``hold``    — at least one hard-refusal gate failed. No automatic
                    action is safe; the reviewer needs to look at it.

This phase ships *classification only*. There is no ``--apply`` flag,
no retry, no variant write. The output is the decision plus the full
list of evidence each gate produced. Users can act on the
recommendation manually; the classifier earns trust before any
automation is enabled.

Design philosophy follows the QA Wolf "self-healing is causal, not
probabilistic" principle: heal only when the math proves implementation
changed but behaviour didn't. Every gate is a one-way refusal — the
default is ``hold``, and a tier upgrade requires every check to pass.
"""

from shadow.heal.classify import (
    HealCheck,
    HealDecision,
    HealTier,
    classify,
)
from shadow.heal.render import render_decision

__all__ = [
    "HealCheck",
    "HealDecision",
    "HealTier",
    "classify",
    "render_decision",
]
