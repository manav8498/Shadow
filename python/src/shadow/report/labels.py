"""User-facing labels for the nine diff axes.

The internal axis names (``semantic``, ``trajectory``, ``conformance``,
``verbosity`` …) come straight from the SPEC and the ``.agentlog``
record schema. They're terse and useful for engineers — but in PR
comments and terminal reports they read as math jargon to someone
who hasn't read the spec.

This module ships **plain-English display labels** that replace the
internal axis name in user-facing surfaces. The internal name stays
unchanged in the on-disk format, the JSON ``rows[].axis`` field, the
SPEC, and any tooling that consumes the report programmatically — only
the rendered presentation switches.

Two functions:

- :func:`axis_label` — get the display label for one axis. Falls back
  to the input string when no mapping is registered, so unknown axes
  pass through unchanged rather than raising.
- :func:`axis_blurb` — get a one-line description of what the axis
  measures. Used in the diff table when the renderer wants to spell
  out the meaning under the column.
"""

from __future__ import annotations

# Plain-English display labels.
#
# Keep these short (≤ 22 chars) so they fit a terminal table column
# and a markdown table cell without wrapping.
_AXIS_LABELS: dict[str, str] = {
    "semantic": "response meaning",
    "trajectory": "tool calls",
    "safety": "refusals",
    "verbosity": "response length",
    "latency": "response time",
    "cost": "token cost",
    "reasoning": "reasoning depth",
    "judge": "LLM-judge score",
    "conformance": "output format",
}

# One-line descriptions — what each axis actually measures. Used by
# the markdown / terminal renderer to spell out meaning under the
# column header for first-time readers.
_AXIS_BLURBS: dict[str, str] = {
    "semantic": "how close the response meaning is to baseline (1 = identical, 0 = unrelated)",
    "trajectory": "how the agent's tool-call sequence diverged (0 = same calls, 1 = different)",
    "safety": "fraction of responses that refused or were content-filtered",
    "verbosity": "median response length in tokens",
    "latency": "median end-to-end response time in milliseconds",
    "cost": "median per-call cost in USD (zero when no pricing table is supplied)",
    "reasoning": "depth of intermediate reasoning steps emitted before the final answer",
    "judge": "score from an LLM-as-judge over response quality (0 to 1)",
    "conformance": "fraction of responses that matched the declared output schema",
}


def axis_label(axis: str) -> str:
    """Return the user-facing display label for an internal axis name.

    Falls back to the input when no mapping is registered, so unknown
    axes (e.g. third-party extensions or future spec additions) pass
    through unchanged.
    """
    return _AXIS_LABELS.get(axis, axis)


def axis_blurb(axis: str) -> str:
    """Return the one-line description of what an axis measures.

    Empty string when no description is registered.
    """
    return _AXIS_BLURBS.get(axis, "")
