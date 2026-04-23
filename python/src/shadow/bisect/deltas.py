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

`diff_configs(a, b)` yields `Delta` rows for every meaningful difference.

### Coalescing (default ON)

Without coalescing, a single "tool schema edit" would produce dozens of
atomic leaf-level deltas — every field in the schema tree becomes its
own `Delta`. For bisection, that's a disaster: LASSO over 2**k corners
can't decorrelate tens of correlated micro-deltas with only 16 or 32 runs.

With `coalesce=True` (default):

  - `tools[*]`: one `Delta` per tool (matched by `name` across both
    configs). If tool `verify_extraction` exists in A but not B, that's
    a single `Delta("tools.verify_extraction", {...}, None)`. Schema
    edits to the same-named tool collapse to a single `Delta` for that
    tool, not per-field deltas.
  - Other nested mappings: leaf-level (unchanged — these are usually
    small and meaningful, e.g. `params.temperature`).

Users who need leaf-level granularity can pass `coalesce=False`.
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
    """Dotted path into the config, e.g. `params.temperature` or
    `tools.verify_extraction` (coalesced)."""
    old_value: Any
    new_value: Any

    @property
    def kind(self) -> str:
        """Top-level category used by the attribution report.

        Strips array-index brackets so e.g. `tools[0].description`
        categorises as `tools`, not `tools[0]`. Coalesced tool paths
        (`tools.<name>`) also return `tools`.
        """
        head = self.path.split(".", 1)[0]
        bracket = head.find("[")
        return head[:bracket] if bracket >= 0 else head


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


def _tools_by_name(tools: Any) -> dict[str, dict[str, Any]]:
    """Index a YAML `tools:` list by `name`. Tools without a name get a
    positional key like `_pos_0` so they still participate in the diff."""
    if not isinstance(tools, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        key = str(name) if isinstance(name, str) and name else f"_pos_{i}"
        out[key] = t
    return out


def _coalesce_tools_deltas(a_tools: Any, b_tools: Any) -> list[Delta]:
    """Emit exactly one Delta per tool-name that differs between a and b.

    Added/removed/modified tools all map to a single `tools.<name>` row,
    collapsing potentially dozens of schema-leaf differences into one
    attributable unit.
    """
    a_map = _tools_by_name(a_tools)
    b_map = _tools_by_name(b_tools)
    all_names = sorted(set(a_map) | set(b_map))
    out: list[Delta] = []
    for name in all_names:
        av = a_map.get(name)
        bv = b_map.get(name)
        if av != bv:
            out.append(Delta(path=f"tools.{name}", old_value=av, new_value=bv))
    return out


def diff_configs(a: dict[str, Any], b: dict[str, Any], coalesce: bool = True) -> list[Delta]:
    """Return a list of Deltas for meaningful differences between a and b.

    Parameters
    ----------
    a, b:
        Parsed YAML configs.
    coalesce:
        If True (default), tool-subtree edits collapse to a single Delta
        per tool-name. If False, every differing leaf becomes its own
        Delta (legacy v0.1 behaviour — useful for fine-grained inspection
        but unusable for LASSO bisect with typical corner counts).
    """
    if not coalesce:
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

    # Coalesced path: diff non-tools keys leaf-wise, tools key by tool-name.
    a_no_tools = {k: v for k, v in (a or {}).items() if k != "tools"}
    b_no_tools = {k: v for k, v in (b or {}).items() if k != "tools"}
    deltas = diff_configs(a_no_tools, b_no_tools, coalesce=False)
    deltas.extend(_coalesce_tools_deltas((a or {}).get("tools"), (b or {}).get("tools")))
    return deltas


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file into a dict."""
    return yaml.safe_load(Path(path).read_text()) or {}
