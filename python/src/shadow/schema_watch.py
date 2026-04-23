"""Tool-schema change watcher.

Compares the `tools:` section of two agent configs (or two arbitrary
schema sets) and classifies each change by its expected runtime impact
on an LLM agent:

- **breaking** — the new schema can no longer be invoked the way the
  old one could (removed tool, required param added, param renamed,
  type change, enum narrowed). An agent, downstream audit system, or
  dependent tool wiring will fail silently or loudly.
- **risky** — invocation still works but observable behaviour shifts
  (required dropped to optional, default changed, description edit
  that changes an imperative verb).
- **additive** — strictly-new capability that can't break existing
  callers (new tool, new optional param, enum value added).
- **neutral** — cosmetic edits (description edits preserving verbs,
  example additions).

Rename detection: when a param is dropped from one tool and another
appears on the same tool with the same type and `required` status,
they're treated as a rename. This is the subtlest — and most
common — breaking schema change in practice (see the `devops-agent`
example: every tool renamed `database` to `db`).

This is the "cheap, proactive check" pass that catches schema
regressions before a full behavioural diff would even be needed.
Intended to run in CI on every PR that touches tool configs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from shadow.errors import ShadowConfigError

__all__ = [
    "ChangeKind",
    "SchemaChange",
    "SchemaWatchReport",
    "Severity",
    "render_markdown",
    "render_terminal",
    "watch",
    "watch_configs",
    "watch_files",
]


class Severity(str, Enum):
    """Impact tier. Ordered high-to-low for exit-code math."""

    BREAKING = "breaking"
    RISKY = "risky"
    ADDITIVE = "additive"
    NEUTRAL = "neutral"


class ChangeKind(str, Enum):
    """Concrete change types. Stable for JSON output consumers."""

    TOOL_ADDED = "tool_added"
    TOOL_REMOVED = "tool_removed"
    PARAM_ADDED = "param_added"
    PARAM_REMOVED = "param_removed"
    PARAM_RENAMED = "param_renamed"
    TYPE_CHANGED = "type_changed"
    REQUIRED_ADDED = "required_added"
    REQUIRED_REMOVED = "required_removed"
    ENUM_NARROWED = "enum_narrowed"
    ENUM_BROADENED = "enum_broadened"
    DESCRIPTION_EDITED = "description_edited"


@dataclass(frozen=True)
class SchemaChange:
    """One classified schema change."""

    tool: str
    kind: ChangeKind
    severity: Severity
    path: str
    summary: str
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["severity"] = self.severity.value
        return d


@dataclass
class SchemaWatchReport:
    """Top-level schema-diff report."""

    changes: list[SchemaChange]
    tools_old: int
    tools_new: int

    @property
    def breaking(self) -> int:
        return sum(1 for c in self.changes if c.severity is Severity.BREAKING)

    @property
    def risky(self) -> int:
        return sum(1 for c in self.changes if c.severity is Severity.RISKY)

    @property
    def additive(self) -> int:
        return sum(1 for c in self.changes if c.severity is Severity.ADDITIVE)

    @property
    def neutral(self) -> int:
        return sum(1 for c in self.changes if c.severity is Severity.NEUTRAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools_old": self.tools_old,
            "tools_new": self.tools_new,
            "breaking": self.breaking,
            "risky": self.risky,
            "additive": self.additive,
            "neutral": self.neutral,
            "changes": [c.to_dict() for c in self.changes],
        }


# ---- rename heuristic ------------------------------------------------------

# A rename is detected when both type and required-status match. Description
# similarity is NOT required — renames in the wild often change descriptions
# at the same time (see devops-agent fixture).
_RENAME_TYPE_MATCH_WEIGHT = 0.7
_RENAME_REQUIRED_MATCH_WEIGHT = 0.3
_RENAME_THRESHOLD = 0.6


def _rename_score(
    old_param: dict[str, Any], new_param: dict[str, Any], old_req: bool, new_req: bool
) -> float:
    score = 0.0
    if _type_of(old_param) == _type_of(new_param) and _type_of(old_param) is not None:
        score += _RENAME_TYPE_MATCH_WEIGHT
    if old_req == new_req:
        score += _RENAME_REQUIRED_MATCH_WEIGHT
    return score


def _detect_renames(
    removed: dict[str, dict[str, Any]],
    added: dict[str, dict[str, Any]],
    old_required: set[str],
    new_required: set[str],
) -> list[tuple[str, str, float]]:
    """Greedy best-match: pair each removed param with the highest-scoring
    added param above threshold, at most once per side.

    Returns `[(old_name, new_name, score), ...]`.
    """
    candidates: list[tuple[float, str, str]] = []
    for old_name, old_p in removed.items():
        for new_name, new_p in added.items():
            s = _rename_score(old_p, new_p, old_name in old_required, new_name in new_required)
            if s >= _RENAME_THRESHOLD:
                candidates.append((s, old_name, new_name))
    candidates.sort(reverse=True)
    matched_old: set[str] = set()
    matched_new: set[str] = set()
    out: list[tuple[str, str, float]] = []
    for score, old_name, new_name in candidates:
        if old_name in matched_old or new_name in matched_new:
            continue
        matched_old.add(old_name)
        matched_new.add(new_name)
        out.append((old_name, new_name, score))
    return out


# ---- classification -------------------------------------------------------


def _type_of(param: dict[str, Any]) -> str | None:
    t = param.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        # JSON Schema draft-7+ allows a list of types; canonicalise
        return "|".join(sorted(str(x) for x in t))
    return None


def _enum_of(param: dict[str, Any]) -> list[Any] | None:
    e = param.get("enum")
    if isinstance(e, list):
        return e
    return None


def _tools_list(config: dict[str, Any]) -> list[dict[str, Any]]:
    tools = config.get("tools", [])
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)]


def _tools_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in tools if isinstance(t.get("name"), str)}


def _params_of(tool: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema = tool.get("input_schema") or tool.get("parameters") or {}
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    return {k: v if isinstance(v, dict) else {} for k, v in props.items()}


def _required_of(tool: dict[str, Any]) -> set[str]:
    schema = tool.get("input_schema") or tool.get("parameters") or {}
    if not isinstance(schema, dict):
        return set()
    req = schema.get("required") or []
    if not isinstance(req, list):
        return set()
    return {str(x) for x in req}


def _classify_type_change(old_t: str | None, new_t: str | None) -> bool:
    """Return True if a type change is breaking (widening-only is additive).

    JSON Schema union-type widening (e.g. string → string|null) is still
    treated as breaking for agent runtimes: downstream consumers parsing
    strictly into a native type will choke.
    """
    if old_t is None or new_t is None:
        return old_t != new_t
    return old_t != new_t


def _enum_direction(old: list[Any] | None, new: list[Any] | None) -> ChangeKind | None:
    if old is None and new is None:
        return None
    if old is None:
        # adding an enum constraint where there was none = narrowing
        return ChangeKind.ENUM_NARROWED
    if new is None:
        # removing the enum constraint = broadening
        return ChangeKind.ENUM_BROADENED
    old_s, new_s = set(map(_hashable, old)), set(map(_hashable, new))
    if old_s == new_s:
        return None
    if old_s.issubset(new_s):
        return ChangeKind.ENUM_BROADENED
    if new_s.issubset(old_s):
        return ChangeKind.ENUM_NARROWED
    # Overlap with values both gained and lost → net-narrowing (some old
    # valid calls no longer valid).
    return ChangeKind.ENUM_NARROWED


def _hashable(x: Any) -> Any:
    """Make enum values hashable for set comparison."""
    if isinstance(x, list):
        return tuple(_hashable(y) for y in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    return x


def _compare_descriptions(old: str, new: str) -> Severity | None:
    """Heuristic: a description edit that drops an imperative verb is
    RISKY; pure cosmetic edits are NEUTRAL; no change returns None."""
    if old == new:
        return None
    # Imperative verbs that change tool-use semantics if dropped.
    imperative_markers = {
        "ONLY",
        "MUST",
        "NEVER",
        "DO NOT",
        "REQUIRED",
        "FIRST",
        "BEFORE",
        "AFTER",
    }
    old_has = any(m in old.upper() for m in imperative_markers)
    new_has = any(m in new.upper() for m in imperative_markers)
    if old_has and not new_has:
        return Severity.RISKY
    return Severity.NEUTRAL


# ---- top-level watcher ----------------------------------------------------


def watch_configs(old_config: dict[str, Any], new_config: dict[str, Any]) -> SchemaWatchReport:
    """Diff the `tools:` section of two parsed config dicts."""
    old_tools = _tools_list(old_config)
    new_tools = _tools_list(new_config)
    return _compare_tool_lists(old_tools, new_tools)


def watch(
    old_schemas: Sequence[dict[str, Any]], new_schemas: Sequence[dict[str, Any]]
) -> SchemaWatchReport:
    """Diff two tool-schema lists directly.

    Each schema is `{"name": str, "description": str,
    "input_schema": {...}}` (Anthropic shape) or
    `{"name": str, "description": str, "parameters": {...}}` (OpenAI
    shape — `parameters` is accepted as an alias for `input_schema`).
    """
    return _compare_tool_lists(list(old_schemas), list(new_schemas))


def watch_files(old_path: Path, new_path: Path) -> SchemaWatchReport:
    """Diff tool schemas loaded from two YAML config files."""
    try:
        old_cfg = yaml.safe_load(old_path.read_text()) or {}
        new_cfg = yaml.safe_load(new_path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ShadowConfigError(f"could not parse config YAML: {e}") from e
    if not isinstance(old_cfg, dict) or not isinstance(new_cfg, dict):
        raise ShadowConfigError("config files must be YAML mappings at the top level")
    return watch_configs(old_cfg, new_cfg)


# ---- core comparator ------------------------------------------------------


def _compare_tool_lists(
    old_tools: list[dict[str, Any]], new_tools: list[dict[str, Any]]
) -> SchemaWatchReport:
    old_map = _tools_by_name(old_tools)
    new_map = _tools_by_name(new_tools)
    changes: list[SchemaChange] = []

    # Tool-level add/remove.
    for name in sorted(old_map.keys() - new_map.keys()):
        changes.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.TOOL_REMOVED,
                severity=Severity.BREAKING,
                path=f"tools.{name}",
                summary=f"tool `{name}` removed",
                rationale=(
                    "Any agent turn that historically called this tool will "
                    "now fall back to prose or a different tool."
                ),
            )
        )
    for name in sorted(new_map.keys() - old_map.keys()):
        changes.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.TOOL_ADDED,
                severity=Severity.ADDITIVE,
                path=f"tools.{name}",
                summary=f"tool `{name}` added",
                rationale="New capability; existing traces unaffected.",
            )
        )

    # Per-tool diff.
    for name in sorted(old_map.keys() & new_map.keys()):
        changes.extend(_compare_tool(name, old_map[name], new_map[name]))

    # Deterministic global ordering: severity (BREAKING first), then
    # tool name, then path.
    severity_rank = {
        Severity.BREAKING: 0,
        Severity.RISKY: 1,
        Severity.ADDITIVE: 2,
        Severity.NEUTRAL: 3,
    }
    changes.sort(key=lambda c: (severity_rank[c.severity], c.tool, c.path))
    return SchemaWatchReport(changes=changes, tools_old=len(old_map), tools_new=len(new_map))


def _compare_tool(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[SchemaChange]:
    out: list[SchemaChange] = []

    # Description (tool-level) edits.
    old_desc = str(old.get("description") or "")
    new_desc = str(new.get("description") or "")
    desc_sev = _compare_descriptions(old_desc, new_desc)
    if desc_sev is not None:
        out.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.DESCRIPTION_EDITED,
                severity=desc_sev,
                path=f"tools.{name}.description",
                summary=(
                    "description rewritten — imperative verbs removed"
                    if desc_sev is Severity.RISKY
                    else "description rewritten"
                ),
                rationale=(
                    "Tool descriptions guide when the agent picks this tool. "
                    "Imperative verbs ('ONLY', 'MUST', 'BEFORE') shape the "
                    "agent's action sequencing."
                )
                if desc_sev is Severity.RISKY
                else (
                    "Cosmetic description edit. May still nudge the agent's "
                    "tool-selection under temperature > 0."
                ),
                details={"old": old_desc, "new": new_desc},
            )
        )

    # Param-level diff.
    old_params = _params_of(old)
    new_params = _params_of(new)
    old_req = _required_of(old)
    new_req = _required_of(new)

    removed = {k: v for k, v in old_params.items() if k not in new_params}
    added = {k: v for k, v in new_params.items() if k not in old_params}
    renames = _detect_renames(removed, added, old_req, new_req)
    rename_old = {o for o, _, _ in renames}
    rename_new = {n for _, n, _ in renames}

    for old_name, new_name, score in renames:
        out.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.PARAM_RENAMED,
                severity=Severity.BREAKING,
                path=f"tools.{name}.properties.{old_name}",
                summary=f"parameter renamed `{old_name}` → `{new_name}`",
                rationale=(
                    "Downstream consumers reading the old key by name will "
                    "see missing data. Agents trained on the old schema may "
                    "emit the old name. Rename detected via same type "
                    f"+ same required status (confidence {score:.2f})."
                ),
                details={
                    "old_name": old_name,
                    "new_name": new_name,
                    "score": score,
                    "old_type": _type_of(old_params[old_name]),
                    "new_type": _type_of(new_params[new_name]),
                },
            )
        )

    for pname in sorted(k for k in removed if k not in rename_old):
        out.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.PARAM_REMOVED,
                severity=Severity.BREAKING,
                path=f"tools.{name}.properties.{pname}",
                summary=f"parameter `{pname}` removed",
                rationale=(
                    "Agent calls carrying this parameter will be rejected "
                    "by strict schema validators."
                ),
                details={"type": _type_of(removed[pname])},
            )
        )

    for pname in sorted(k for k in added if k not in rename_new):
        was_required = pname in new_req
        out.append(
            SchemaChange(
                tool=name,
                kind=ChangeKind.PARAM_ADDED,
                severity=Severity.BREAKING if was_required else Severity.ADDITIVE,
                path=f"tools.{name}.properties.{pname}",
                summary=(
                    f"parameter `{pname}` added (required)"
                    if was_required
                    else f"parameter `{pname}` added (optional)"
                ),
                rationale=(
                    "New required param — old agent invocations will fail " "validation."
                    if was_required
                    else "New optional param — old invocations remain valid."
                ),
                details={
                    "type": _type_of(added[pname]),
                    "required": was_required,
                },
            )
        )

    # Params present on both sides: type / enum / required changes.
    for pname in sorted(set(old_params) & set(new_params)):
        op = old_params[pname]
        np_ = new_params[pname]
        # Type change.
        ot, nt = _type_of(op), _type_of(np_)
        if _classify_type_change(ot, nt):
            out.append(
                SchemaChange(
                    tool=name,
                    kind=ChangeKind.TYPE_CHANGED,
                    severity=Severity.BREAKING,
                    path=f"tools.{name}.properties.{pname}.type",
                    summary=f"parameter `{pname}` type changed: {ot} → {nt}",
                    rationale=(
                        "Type transitions break strict parsers and may "
                        "silently coerce at the model boundary, yielding "
                        "unexpected downstream values."
                    ),
                    details={"old": ot, "new": nt},
                )
            )
        # Enum change.
        enum_kind = _enum_direction(_enum_of(op), _enum_of(np_))
        if enum_kind is ChangeKind.ENUM_NARROWED:
            out.append(
                SchemaChange(
                    tool=name,
                    kind=enum_kind,
                    severity=Severity.BREAKING,
                    path=f"tools.{name}.properties.{pname}.enum",
                    summary=f"parameter `{pname}` enum narrowed",
                    rationale=(
                        "Prior agent outputs selecting removed enum values "
                        "are no longer valid invocations."
                    ),
                    details={"old": _enum_of(op), "new": _enum_of(np_)},
                )
            )
        elif enum_kind is ChangeKind.ENUM_BROADENED:
            out.append(
                SchemaChange(
                    tool=name,
                    kind=enum_kind,
                    severity=Severity.ADDITIVE,
                    path=f"tools.{name}.properties.{pname}.enum",
                    summary=f"parameter `{pname}` enum broadened",
                    rationale=(
                        "New enum values available to the agent; all prior "
                        "valid values still accepted."
                    ),
                    details={"old": _enum_of(op), "new": _enum_of(np_)},
                )
            )
        # Required flip.
        was_req = pname in old_req
        is_req = pname in new_req
        if not was_req and is_req:
            out.append(
                SchemaChange(
                    tool=name,
                    kind=ChangeKind.REQUIRED_ADDED,
                    severity=Severity.BREAKING,
                    path=f"tools.{name}.required",
                    summary=f"parameter `{pname}` is now required",
                    rationale=(
                        "Prior agent invocations that omitted this param " "will fail validation."
                    ),
                )
            )
        elif was_req and not is_req:
            out.append(
                SchemaChange(
                    tool=name,
                    kind=ChangeKind.REQUIRED_REMOVED,
                    severity=Severity.RISKY,
                    path=f"tools.{name}.required",
                    summary=f"parameter `{pname}` is no longer required",
                    rationale=(
                        "Agent may now omit this field; downstream consumers "
                        "that treated it as always-present can break."
                    ),
                )
            )

    return out


# ---- renderers ------------------------------------------------------------


_SEVERITY_ICON = {
    Severity.BREAKING: "✖",
    Severity.RISKY: "!",
    Severity.ADDITIVE: "+",
    Severity.NEUTRAL: "·",
}

_SEVERITY_COLOR = {
    Severity.BREAKING: "red",
    Severity.RISKY: "yellow",
    Severity.ADDITIVE: "green",
    Severity.NEUTRAL: "dim",
}


def render_terminal(report: SchemaWatchReport) -> str:
    """Rich-markup string for console display (ANSI via rich, or plain)."""
    if not report.changes:
        return f"No tool-schema changes across " f"{report.tools_old}→{report.tools_new} tools.\n"
    lines: list[str] = []
    lines.append(f"Tool-schema changes ({report.tools_old}→{report.tools_new} tools):\n")
    for c in report.changes:
        icon = _SEVERITY_ICON[c.severity]
        tag = f"[{_SEVERITY_COLOR[c.severity]}]{icon} {c.severity.value.upper():<9}[/]"
        lines.append(f"  {tag} {c.tool}: {c.summary}")
        lines.append(f"      [dim]{c.rationale}[/]")
    lines.append("")
    lines.append(
        f"Summary: {report.breaking} breaking, {report.risky} risky, "
        f"{report.additive} additive, {report.neutral} neutral."
    )
    return "\n".join(lines) + "\n"


def render_markdown(report: SchemaWatchReport) -> str:
    """GitHub-flavoured markdown for PR comments."""
    lines: list[str] = []
    lines.append("## Tool-schema changes")
    lines.append("")
    lines.append(
        f"**{report.breaking}** breaking · **{report.risky}** risky · "
        f"**{report.additive}** additive · **{report.neutral}** neutral "
        f"*(across {report.tools_old}→{report.tools_new} tools)*"
    )
    lines.append("")
    if not report.changes:
        lines.append("_No changes detected._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| Severity | Tool | Change |")
    lines.append("|---|---|---|")
    for c in report.changes:
        sev = c.severity.value
        icon = _SEVERITY_ICON[c.severity]
        lines.append(f"| {icon} **{sev}** | `{c.tool}` | {c.summary} |")
    lines.append("")
    # Expandable rationale section.
    lines.append("<details><summary>Why each of these matters</summary>")
    lines.append("")
    for c in report.changes:
        lines.append(f"- **{c.tool}** — {c.summary}")
        lines.append(f"  - {c.rationale}")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)
