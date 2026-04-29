"""Replay backends for :func:`shadow.causal.causal_attribution`.

The causal-attribution API takes an opaque ``replay_fn`` callable so
the user can wire any backend that produces a per-axis divergence
vector for a given config. This subpackage ships two production-grade
implementations:

* :class:`OpenAIReplayer` — calls the live OpenAI Chat Completions API
  with deterministic seeding, retry on rate limits, and per-config
  caching keyed by canonical-JSON content hash. Reads
  ``OPENAI_API_KEY`` from the environment; never accepts the key as a
  parameter (keys must not pass through call sites that get logged
  in tracebacks or test fixtures).

* :class:`RecordedReplayer` — replays from a previously-recorded
  ``.agentlog`` file. Used in CI / unit tests so the causal
  attribution pipeline can be validated end-to-end without API spend.

Both replayers share a small ``Replayer`` Protocol so callers can
swap implementations transparently. The signature matches the
``ReplayFn`` type alias in ``shadow.causal.attribution``:

    replayer(config: dict) -> dict[axis, float]

For the live replayer, "config" is a dict that includes at minimum
``model`` and ``system_prompt``; the divergence vector is computed
from comparing the live response to a recorded baseline response on
the same config.
"""

from __future__ import annotations

from shadow.causal.replay.openai_replayer import OpenAIReplayer
from shadow.causal.replay.recorded import RecordedReplayer
from shadow.causal.replay.types import Replayer, ReplayResult

__all__ = [
    "OpenAIReplayer",
    "RecordedReplayer",
    "ReplayResult",
    "Replayer",
]
