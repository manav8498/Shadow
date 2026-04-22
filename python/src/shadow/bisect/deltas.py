"""Parse two YAML configs into a list of typed deltas.

A config has the shape:

```yaml
model: claude-opus-4-7
params:
  temperature: 0.2
  top_p: 1.0
prompt:
  system: "You are a careful code reviewer."
tools:
  - name: search_files
    description: "Search for files matching a glob."
    input_schema:
      type: object
      properties:
        query: { type: string }
```

`diff_configs(a, b)` yields `Delta` rows for every leaf that differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Delta:
    """One atomic difference between two configs."""

    path: str
    """Dotted path into the config, e.g. `params.temperature`."""
    old_value: Any
    new_value: Any

    @property
    def kind(self) -> str:
        """Top-level category used by the attribution report."""
        return self.path.split(".", 1)[0]


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for k, v in value.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_walk(v, child))
        return out
    if isinstance(value, list):
        out = []
        for i, v in enumerate(value):
            child = f"{prefix}[{i}]"
            out.extend(_walk(v, child))
        return out
    return [(prefix, value)]


def diff_configs(a: dict[str, Any], b: dict[str, Any]) -> list[Delta]:
    """Return a list of Deltas for every leaf that differs between a and b."""
    a_flat = dict(_walk(a))
    b_flat = dict(_walk(b))
    all_paths = sorted(set(a_flat) | set(b_flat))
    sentinel = object()
    deltas: list[Delta] = []
    for path in all_paths:
        av = a_flat.get(path, sentinel)
        bv = b_flat.get(path, sentinel)
        if av != bv:
            deltas.append(
                Delta(
                    path=path,
                    old_value=None if av is sentinel else av,
                    new_value=None if bv is sentinel else bv,
                )
            )
    return deltas


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file into a dict."""
    return yaml.safe_load(Path(path).read_text()) or {}
