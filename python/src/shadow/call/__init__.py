"""shadow.call — decide whether a candidate is safe to ship.

Given a ``DiffReport`` (the output of
:func:`shadow._core.compute_diff_report`), this module produces a
single-line decision — ``ship``, ``hold``, ``probe``, or ``stop`` — plus
the dominant driver of any regression and a short list of next-step
commands the reviewer can run.

The module has two halves:

    * :mod:`shadow.call.decide` — pure logic over a diff report. Inputs
      are dicts; outputs are :class:`CallResult` and the enumerated
      tier and confidence types. No I/O, no rendering. Trivially
      unit-testable.

    * :mod:`shadow.call.render` — the visual layer. Imports Rich
      lazily so callers that only need the data don't pay the cost.

Public surface:

    >>> from shadow.call import compute_call, render_call, Tier, Confidence
    >>> result = compute_call(report_dict)
    >>> render_call(result)
"""

from shadow.call.decide import (
    CallResult,
    Confidence,
    Tier,
    compute_call,
)
from shadow.call.render import render_call

__all__ = [
    "CallResult",
    "Confidence",
    "Tier",
    "compute_call",
    "render_call",
]
