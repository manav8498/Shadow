"""Tiered statistical-power classification for diff and diagnose-pr reports.

Background
----------

Bootstrap confidence intervals on per-axis severities get tighter as
the paired-response count `n` grows. The reviewer-driven tiering is:

* ``n < 5`` — *low*. Directional only. Bootstrap CIs are wider than
  the deltas, so severity classifications are advisory, not stable.
* ``5 <= n < 30`` — *moderate*. CIs are computable, but a single noisy
  pair shifts them visibly. Useful for early-iteration debugging, not
  for merge-gating.
* ``30 <= n < 100`` — *adequate*. Standard eval practice. CIs are
  stable enough to gate routine PRs.
* ``n >= 100`` — *robust*. Enterprise-grade. Causal attribution and
  per-axis severity are both trustworthy.

Why these thresholds
--------------------

* n=5 is the textbook minimum for percentile bootstrap. Below it the
  resample distribution is essentially the data, not an estimate of
  the sampling distribution.
* n=30 is the conventional threshold for the central-limit
  approximation to kick in for mean-like statistics. It's the same
  threshold most A/B testing platforms use.
* n=100 is what every external reviewer of Shadow has named when
  asked "what would you require for Fortune-100 adoption?" — it's
  not a deep statistical claim, just the practitioner consensus.

The thresholds are documented here in one place so renderers, the
diff JSON serializer, and the diagnose-pr JSON serializer all agree.
"""

from __future__ import annotations

from typing import Literal

PowerTier = Literal["low", "moderate", "adequate", "robust"]

_LOW_THRESHOLD = 5
_MODERATE_THRESHOLD = 30
_ADEQUATE_THRESHOLD = 100


def classify_power(pair_count: int) -> PowerTier:
    """Map a paired-response count to a tiered power classification."""
    if pair_count < _LOW_THRESHOLD:
        return "low"
    if pair_count < _MODERATE_THRESHOLD:
        return "moderate"
    if pair_count < _ADEQUATE_THRESHOLD:
        return "adequate"
    return "robust"


def recommended_sample_size(pair_count: int) -> int | None:
    """The next-tier target. ``None`` once the count is robust."""
    tier = classify_power(pair_count)
    if tier == "low":
        return _MODERATE_THRESHOLD
    if tier == "moderate":
        return _ADEQUATE_THRESHOLD
    if tier == "adequate":
        # The 100-trace bar is the practitioner consensus for enterprise
        # use; recommend it again so a 70-trace caller sees "aim for 100"
        # rather than "you're done."
        return _ADEQUATE_THRESHOLD
    return None


def power_blurb(pair_count: int) -> str:
    """One-line human-readable description of the current tier and
    the recommended next step. Used by terminal + markdown renderers.
    """
    tier = classify_power(pair_count)
    recommended = recommended_sample_size(pair_count)
    if tier == "low":
        return (
            f"Statistical power: low ({pair_count} pair(s)). Severities below "
            f"are directional. Record {recommended}+ pairs for stable confidence "
            "intervals."
        )
    if tier == "moderate":
        return (
            f"Statistical power: moderate ({pair_count} pairs). CIs are "
            f"computable but unstable; routine merge-gating recommends "
            f"{recommended}+ pairs."
        )
    if tier == "adequate":
        return (
            f"Statistical power: adequate ({pair_count} pairs). Suitable for "
            f"CI gating on routine PRs; enterprise reviewers recommend "
            f"{recommended}+ pairs."
        )
    return f"Statistical power: robust ({pair_count} pairs)."


__all__ = [
    "PowerTier",
    "classify_power",
    "power_blurb",
    "recommended_sample_size",
]
