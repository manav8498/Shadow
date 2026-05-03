"""Smoke test: the diagnose_pr package is importable and exposes its
v0.1 schema constant. Anchors the public surface so future renames
break tests, not consumers."""

from __future__ import annotations


def test_package_imports() -> None:
    import shadow.diagnose_pr as dp

    assert hasattr(dp, "SCHEMA_VERSION")
    assert dp.SCHEMA_VERSION == "diagnose-pr/v0.1"
