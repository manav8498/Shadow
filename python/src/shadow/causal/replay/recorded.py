"""Offline replay backend backed by a recorded ``.agentlog``.

The recorded replayer plays back a previously-captured trace by
config hash. Used in CI and unit tests to validate the causal
attribution pipeline end-to-end without API spend.

Construct with a mapping of config-hash → recorded ``ReplayResult``.
The mapping can be built by running the live OpenAIReplayer once,
serialising its cache to disk, and loading on subsequent runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from shadow.causal.replay.types import ReplayResult


def _canonical_config_hash(config: dict[str, Any]) -> str:
    """Stable hash for cache keying. Same hash function as the live
    OpenAIReplayer so cache files port across replayers."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class RecordedReplayer:
    """Replay from a pre-recorded result table.

    The class is a small Protocol-conforming callable. It looks up
    the config hash in the table and returns the matching
    :class:`ReplayResult`. If the config isn't in the table, it
    raises :class:`KeyError` — silently returning a zero divergence
    would mask "you forgot to record this stratum" bugs in causal
    attribution.
    """

    def __init__(self, table: dict[str, ReplayResult]) -> None:
        self._table = dict(table)

    @classmethod
    def from_results(cls, results: list[tuple[dict[str, Any], ReplayResult]]) -> RecordedReplayer:
        """Build from a list of (config, result) pairs."""
        return cls({_canonical_config_hash(c): r for c, r in results})

    @property
    def size(self) -> int:
        return len(self._table)

    def __call__(self, config: dict[str, Any]) -> ReplayResult:
        key = _canonical_config_hash(config)
        if key not in self._table:
            raise KeyError(
                f"RecordedReplayer has no recording for config hash {key!r}; "
                f"either run OpenAIReplayer once to populate, or add this "
                f"config to the recorded table"
            )
        return self._table[key]
