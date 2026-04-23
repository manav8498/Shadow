"""Ground-truth tests for shadow.schema_watch.

Each test constructs two tool-schema inputs with a single known change
and asserts the watcher classifies it correctly.
"""

from __future__ import annotations

from shadow.schema_watch import (
    ChangeKind,
    Severity,
    render_markdown,
    render_terminal,
    watch,
    watch_configs,
)


def _tool(
    name: str,
    *,
    description: str = "x",
    props: dict[str, dict[str, object]] | None = None,
    required: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": props or {},
            "required": required or [],
        },
    }


# ---- tool add / remove ---------------------------------------------------


def test_tool_added_is_additive() -> None:
    r = watch([_tool("a")], [_tool("a"), _tool("b")])
    assert r.breaking == 0
    added = [c for c in r.changes if c.kind is ChangeKind.TOOL_ADDED]
    assert len(added) == 1
    assert added[0].tool == "b"
    assert added[0].severity is Severity.ADDITIVE


def test_tool_removed_is_breaking() -> None:
    r = watch([_tool("a"), _tool("b")], [_tool("a")])
    removed = [c for c in r.changes if c.kind is ChangeKind.TOOL_REMOVED]
    assert len(removed) == 1
    assert removed[0].tool == "b"
    assert removed[0].severity is Severity.BREAKING


# ---- param add / remove --------------------------------------------------


def test_required_param_added_is_breaking() -> None:
    old = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    new = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x", "y"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.PARAM_ADDED)
    assert c.severity is Severity.BREAKING
    assert "y" in c.summary


def test_optional_param_added_is_additive() -> None:
    old = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    new = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.PARAM_ADDED)
    assert c.severity is Severity.ADDITIVE


def test_param_removed_is_breaking() -> None:
    old = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "integer"}},
        required=["x"],
    )
    new = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.PARAM_REMOVED)
    assert c.severity is Severity.BREAKING


# ---- rename detection ----------------------------------------------------


def test_param_rename_detected_when_type_and_required_match() -> None:
    old = _tool("foo", props={"database": {"type": "string"}}, required=["database"])
    new = _tool("foo", props={"db": {"type": "string"}}, required=["db"])
    r = watch([old], [new])
    renames = [c for c in r.changes if c.kind is ChangeKind.PARAM_RENAMED]
    assert len(renames) == 1
    assert renames[0].severity is Severity.BREAKING
    assert "database" in renames[0].summary
    assert "db" in renames[0].summary
    # Rename consumes both the removed and added param — no ADD/REMOVE
    # leaks through.
    assert not any(c.kind in (ChangeKind.PARAM_ADDED, ChangeKind.PARAM_REMOVED) for c in r.changes)


def test_rename_not_claimed_when_types_differ() -> None:
    old = _tool("foo", props={"a": {"type": "string"}}, required=["a"])
    new = _tool("foo", props={"b": {"type": "integer"}}, required=["b"])
    r = watch([old], [new])
    # Neither required-match nor type-match → falls below threshold (0.6)
    # because only required-match (0.3) would survive.
    assert not any(c.kind is ChangeKind.PARAM_RENAMED for c in r.changes)
    # Seen as a remove + add pair instead.
    kinds = {c.kind for c in r.changes}
    assert ChangeKind.PARAM_REMOVED in kinds
    assert ChangeKind.PARAM_ADDED in kinds


def test_rename_detected_across_both_tools_independently() -> None:
    """Each tool's renames are scoped to that tool, not cross-tool."""
    old = [
        _tool("a", props={"x": {"type": "string"}}, required=["x"]),
        _tool("b", props={"y": {"type": "integer"}}, required=["y"]),
    ]
    new = [
        _tool("a", props={"xx": {"type": "string"}}, required=["xx"]),
        _tool("b", props={"yy": {"type": "integer"}}, required=["yy"]),
    ]
    r = watch(old, new)
    renames = [c for c in r.changes if c.kind is ChangeKind.PARAM_RENAMED]
    assert len(renames) == 2
    tools_renamed = {c.tool for c in renames}
    assert tools_renamed == {"a", "b"}


# ---- type / enum / required changes --------------------------------------


def test_type_change_is_breaking() -> None:
    old = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    new = _tool("foo", props={"x": {"type": "integer"}}, required=["x"])
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.TYPE_CHANGED)
    assert c.severity is Severity.BREAKING


def test_enum_narrowed_is_breaking() -> None:
    old = _tool(
        "foo",
        props={"mode": {"type": "string", "enum": ["a", "b", "c"]}},
        required=["mode"],
    )
    new = _tool(
        "foo",
        props={"mode": {"type": "string", "enum": ["a", "b"]}},
        required=["mode"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.ENUM_NARROWED)
    assert c.severity is Severity.BREAKING


def test_enum_broadened_is_additive() -> None:
    old = _tool(
        "foo",
        props={"mode": {"type": "string", "enum": ["a"]}},
        required=["mode"],
    )
    new = _tool(
        "foo",
        props={"mode": {"type": "string", "enum": ["a", "b"]}},
        required=["mode"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.ENUM_BROADENED)
    assert c.severity is Severity.ADDITIVE


def test_required_dropped_is_risky_not_breaking() -> None:
    old = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x", "y"],
    )
    new = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.REQUIRED_REMOVED)
    assert c.severity is Severity.RISKY


def test_required_added_is_breaking() -> None:
    old = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x"],
    )
    new = _tool(
        "foo",
        props={"x": {"type": "string"}, "y": {"type": "string"}},
        required=["x", "y"],
    )
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.REQUIRED_ADDED)
    assert c.severity is Severity.BREAKING


# ---- description-edit severity -------------------------------------------


def test_description_drops_imperative_is_risky() -> None:
    old = _tool(
        "refund",
        description="Issue a refund. ONLY use after customer confirmation.",
    )
    new = _tool("refund", description="Issue a refund.")
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.DESCRIPTION_EDITED)
    assert c.severity is Severity.RISKY


def test_description_cosmetic_edit_is_neutral() -> None:
    old = _tool("refund", description="Issue a refund against an order.")
    new = _tool("refund", description="Process a refund against an order.")
    r = watch([old], [new])
    c = next(c for c in r.changes if c.kind is ChangeKind.DESCRIPTION_EDITED)
    assert c.severity is Severity.NEUTRAL


def test_no_change_has_no_changes() -> None:
    t = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    r = watch([t], [t])
    assert r.changes == []


# ---- openai `parameters` alias -------------------------------------------


def test_openai_parameters_alias_accepted() -> None:
    old = {
        "name": "foo",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {"database": {"type": "string"}},
            "required": ["database"],
        },
    }
    new = {
        "name": "foo",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {"db": {"type": "string"}},
            "required": ["db"],
        },
    }
    r = watch([old], [new])
    renames = [c for c in r.changes if c.kind is ChangeKind.PARAM_RENAMED]
    assert len(renames) == 1


# ---- watch_configs (YAML-style dicts) ------------------------------------


def test_watch_configs_extracts_tools_section() -> None:
    old_cfg = {
        "model": "m",
        "tools": [_tool("a", props={"x": {"type": "string"}}, required=["x"])],
    }
    new_cfg = {
        "model": "m",
        "tools": [_tool("a", props={"y": {"type": "string"}}, required=["y"])],
    }
    r = watch_configs(old_cfg, new_cfg)
    # x → y detected as rename (same type, same required).
    assert any(c.kind is ChangeKind.PARAM_RENAMED for c in r.changes)


def test_watch_configs_empty_tools_is_noop() -> None:
    r = watch_configs({"model": "m"}, {"model": "m"})
    assert r.changes == []
    assert r.tools_old == 0
    assert r.tools_new == 0


# ---- report ordering / renderers -----------------------------------------


def test_changes_sorted_breaking_first() -> None:
    old = [
        _tool("a", props={"x": {"type": "string"}}, required=["x"]),
        _tool("b", description="ONLY on confirm."),
    ]
    new = [
        _tool("a", props={"x": {"type": "integer"}}, required=["x"]),
        _tool("b", description="On confirm."),
    ]
    r = watch(old, new)
    # Both a (breaking type change) and b (risky description) are flagged.
    severities = [c.severity for c in r.changes]
    # BREAKING must precede RISKY in the sorted output.
    assert severities.index(Severity.BREAKING) < severities.index(Severity.RISKY)


def test_render_terminal_includes_every_change() -> None:
    r = watch(
        [_tool("a", props={"x": {"type": "string"}})],
        [_tool("a", props={"y": {"type": "string"}})],
    )
    out = render_terminal(r)
    assert "parameter renamed" in out
    assert "breaking" in out.lower()


def test_render_markdown_has_table_and_rationale_section() -> None:
    r = watch(
        [_tool("a", props={"x": {"type": "string"}})],
        [_tool("a", props={"y": {"type": "string"}})],
    )
    md = render_markdown(r)
    assert "| Severity | Tool | Change |" in md
    assert "<details>" in md
    assert "parameter renamed" in md


def test_render_terminal_empty_report() -> None:
    t = _tool("foo", props={"x": {"type": "string"}}, required=["x"])
    r = watch([t], [t])
    out = render_terminal(r)
    assert "No tool-schema changes" in out


# ---- counts --------------------------------------------------------------


def test_counts_are_accurate() -> None:
    old = [
        _tool("a", props={"x": {"type": "string"}}, required=["x"]),
        _tool("c", description="ONLY on confirm."),
    ]
    new = [
        _tool("a", props={"x": {"type": "integer"}}, required=["x"]),
        _tool("c", description="On confirm."),
        _tool("d"),  # new tool added
    ]
    r = watch(old, new)
    assert r.breaking == 1  # type change
    assert r.risky == 1  # description edit dropping ONLY
    assert r.additive == 1  # tool added
    assert r.tools_old == 2
    assert r.tools_new == 3
