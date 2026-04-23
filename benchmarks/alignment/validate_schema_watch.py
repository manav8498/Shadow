"""Real-world validation for shadow.schema_watch.

Runs the watcher against committed example fixtures whose schema
changes are intentional and documented, and asserts the classifier
picks them up.

The two fixtures have known ground-truth:

- `examples/devops-agent/config_{a,b}.yaml` — every tool with a
  `database` parameter has it renamed to `db`. Eight tools, one rename
  each (plus `request_human_approval` and `send_notification`, which
  don't carry a `database` param). Expected: 8 PARAM_RENAMED changes,
  all BREAKING. No other schema changes.

- `examples/customer-support/config_{a,b}.yaml` — `lookup_order` has
  `order_id` renamed to `id` AND a new optional `include_shipping`
  parameter added. `refund_order` has its imperative
  "ONLY use after customer confirmation" line dropped from the
  description. Expected: 1 rename (BREAKING), 1 additive param, 1
  risky description edit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from shadow.schema_watch import ChangeKind, Severity, watch_files

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert(condition: bool, message: str) -> None:
    marker = "✓" if condition else "✗"
    print(f"  {marker} {message}")
    if not condition:
        raise AssertionError(message)


def validate_devops_agent() -> None:
    print("\n=== devops-agent (config_a vs config_b) ===")
    old = REPO_ROOT / "examples/devops-agent/config_a.yaml"
    new = REPO_ROOT / "examples/devops-agent/config_b.yaml"
    report = watch_files(old, new)

    renames = [c for c in report.changes if c.kind is ChangeKind.PARAM_RENAMED]
    _assert(
        len(renames) == 8,
        f"detected 8 renamed parameters (got {len(renames)})",
    )
    _assert(
        all(c.severity is Severity.BREAKING for c in renames),
        "all renames classified BREAKING",
    )
    rename_pairs = {(c.details["old_name"], c.details["new_name"]) for c in renames}
    _assert(
        rename_pairs == {("database", "db")},
        f"every rename is database→db (got {rename_pairs})",
    )
    # The two tools without a `database` param should have NO
    # structural schema changes — only description edits are allowed.
    for tool_name in ("request_human_approval", "send_notification"):
        structural = [
            c
            for c in report.changes
            if c.tool == tool_name
            and c.kind
            in (
                ChangeKind.PARAM_ADDED,
                ChangeKind.PARAM_REMOVED,
                ChangeKind.PARAM_RENAMED,
                ChangeKind.TYPE_CHANGED,
                ChangeKind.REQUIRED_ADDED,
                ChangeKind.REQUIRED_REMOVED,
                ChangeKind.ENUM_NARROWED,
                ChangeKind.ENUM_BROADENED,
            )
        ]
        _assert(
            len(structural) == 0,
            f"{tool_name} has no structural schema changes",
        )
    # No spurious PARAM_ADDED / PARAM_REMOVED leaked through from the
    # database→db pairings.
    leaked = [
        c
        for c in report.changes
        if c.kind in (ChangeKind.PARAM_ADDED, ChangeKind.PARAM_REMOVED)
    ]
    _assert(
        len(leaked) == 0,
        "no add/remove leak-through — every database/db pairing detected as a rename",
    )
    print(
        f"  summary: {report.breaking}✖ {report.risky}! {report.additive}+ {report.neutral}·"
    )


def validate_customer_support() -> None:
    print("\n=== customer-support (config_a vs config_b) ===")
    old = REPO_ROOT / "examples/customer-support/config_a.yaml"
    new = REPO_ROOT / "examples/customer-support/config_b.yaml"
    report = watch_files(old, new)

    # 1. `lookup_order.order_id` → `id` rename
    lookup_renames = [
        c
        for c in report.changes
        if c.kind is ChangeKind.PARAM_RENAMED and c.tool == "lookup_order"
    ]
    _assert(
        len(lookup_renames) == 1,
        "lookup_order has exactly one rename",
    )
    _assert(
        lookup_renames[0].details["old_name"] == "order_id",
        f"old name is 'order_id' (got {lookup_renames[0].details['old_name']})",
    )
    _assert(
        lookup_renames[0].details["new_name"] == "id",
        f"new name is 'id' (got {lookup_renames[0].details['new_name']})",
    )
    _assert(
        lookup_renames[0].severity is Severity.BREAKING,
        "rename is BREAKING",
    )

    # 2. include_shipping added as optional
    added_optional = [
        c
        for c in report.changes
        if c.kind is ChangeKind.PARAM_ADDED
        and c.tool == "lookup_order"
        and not c.details.get("required")
    ]
    _assert(
        len(added_optional) == 1,
        "lookup_order gained one optional parameter",
    )
    _assert(
        added_optional[0].severity is Severity.ADDITIVE,
        "added optional param is ADDITIVE",
    )

    # 3. refund_order description dropped "ONLY use after customer confirmation"
    desc_edits = [
        c
        for c in report.changes
        if c.kind is ChangeKind.DESCRIPTION_EDITED and c.tool == "refund_order"
    ]
    _assert(
        len(desc_edits) == 1,
        "refund_order description was edited",
    )
    _assert(
        desc_edits[0].severity is Severity.RISKY,
        "description edit is RISKY (ONLY dropped)",
    )
    print(
        f"  summary: {report.breaking}✖ {report.risky}! {report.additive}+ {report.neutral}·"
    )


def main() -> int:
    try:
        validate_devops_agent()
        validate_customer_support()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print("\nAll real-world schema-watch assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
