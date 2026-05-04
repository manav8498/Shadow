"""Tests for the shipped policy packs under `examples/policy-packs/`.

Two assurances:
1. Every YAML in `examples/policy-packs/` parses via
   `shadow.hierarchical.load_policy` — no broken packs ever ship.
2. The PII pack actually flags PII in a synthesised candidate trace
   (real load-bearing assertion, not just "yaml parses").
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIR = REPO_ROOT / "examples" / "policy-packs"


def _yaml_packs() -> list[Path]:
    return sorted(PACKS_DIR.glob("*.yaml"))


def test_policy_packs_dir_exists_and_has_at_least_one_pack() -> None:
    """If we ship policy packs, the directory itself is a public
    contract — moving / renaming would break documented `--policy`
    paths in user CI configs."""
    assert PACKS_DIR.is_dir()
    assert len(_yaml_packs()) >= 1


@pytest.mark.parametrize("pack_path", _yaml_packs(), ids=lambda p: p.name)
def test_policy_pack_parses_via_hierarchical_loader(pack_path: Path) -> None:
    """Every shipped pack must round-trip through the production
    policy loader. A drift between the loader's accepted schema and
    the packs is the bug."""
    from shadow.hierarchical import load_policy

    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    rules = load_policy(data)
    assert rules, f"{pack_path.name}: loader returned an empty rule list"
    # Each rule has the load-bearing fields populated.
    for r in rules:
        assert r.id, f"{pack_path.name}: a rule has an empty id"
        assert r.kind, f"{pack_path.name}: rule {r.id!r} has no kind"
        assert r.severity, f"{pack_path.name}: rule {r.id!r} has no severity"


def test_pii_pack_flags_email_in_response_text() -> None:
    """End-to-end: feed a synthetic chat_response containing an
    email through the policy checker with the PII pack, assert
    `no-email-in-response` fires.

    This is the test that proves the pack is useful, not just
    well-formed."""
    from shadow.hierarchical import check_policy, load_policy

    data = yaml.safe_load((PACKS_DIR / "pii.yaml").read_text())
    rules = load_policy(data)

    # Simulated trace: one chat_response that echoes a user email.
    records = [
        {
            "id": "sha256:" + "a" * 64,
            "kind": "metadata",
            "ts": "2026-01-01T00:00:00Z",
            "parent": None,
            "meta": {},
            "payload": {},
            "version": "0.1",
        },
        {
            "id": "sha256:" + "b" * 64,
            "kind": "chat_request",
            "ts": "2026-01-01T00:00:01Z",
            "parent": "sha256:" + "a" * 64,
            "meta": {},
            "payload": {"messages": [{"role": "user", "content": "what's my email?"}]},
            "version": "0.1",
        },
        {
            "id": "sha256:" + "c" * 64,
            "kind": "chat_response",
            "ts": "2026-01-01T00:00:02Z",
            "parent": "sha256:" + "b" * 64,
            "meta": {},
            "payload": {"content": [{"type": "text", "text": "your email is alice@acme.com"}]},
            "version": "0.1",
        },
    ]

    violations = check_policy(records, rules)
    rule_ids = {v.rule_id for v in violations}
    assert "no-email-in-response" in rule_ids


def test_pii_pack_does_not_fire_on_clean_response() -> None:
    """Sanity: a response with no PII produces no violations."""
    from shadow.hierarchical import check_policy, load_policy

    data = yaml.safe_load((PACKS_DIR / "pii.yaml").read_text())
    rules = load_policy(data)

    records = [
        {
            "id": "sha256:" + "a" * 64,
            "kind": "metadata",
            "ts": "2026-01-01T00:00:00Z",
            "parent": None,
            "meta": {},
            "payload": {},
            "version": "0.1",
        },
        {
            "id": "sha256:" + "b" * 64,
            "kind": "chat_request",
            "ts": "2026-01-01T00:00:01Z",
            "parent": "sha256:" + "a" * 64,
            "meta": {},
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
            "version": "0.1",
        },
        {
            "id": "sha256:" + "c" * 64,
            "kind": "chat_response",
            "ts": "2026-01-01T00:00:02Z",
            "parent": "sha256:" + "b" * 64,
            "meta": {},
            "payload": {"content": [{"type": "text", "text": "hello there, how can I help?"}]},
            "version": "0.1",
        },
    ]

    violations = check_policy(records, rules)
    rule_ids = {v.rule_id for v in violations}
    # No PII in the clean response.
    assert "no-email-in-response" not in rule_ids
    assert "no-ssn-in-response" not in rule_ids


def test_policy_packs_readme_lists_every_yaml_file() -> None:
    """Drift check: if you add a `foo.yaml` to `policy-packs/`, the
    README must mention it. Cheap detection of "new pack shipped,
    docs forgot to mention it"."""
    readme = (PACKS_DIR / "README.md").read_text(encoding="utf-8")
    for yaml_path in _yaml_packs():
        assert (
            yaml_path.name in readme
        ), f"{yaml_path.name} not referenced in policy-packs/README.md"
